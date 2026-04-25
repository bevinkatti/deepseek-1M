"""
DeepSeek-1M Client
~~~~~~~~~~~~~~~~~~
A production-grade async/sync client for DeepSeek-V4 with native 1M context support.
Handles streaming, thinking mode, retry logic, token estimation, and cost tracking.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Literal, Optional

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    console = None  # type: ignore

try:
    from openai import AsyncOpenAI, OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# DeepSeek V4 pricing per million tokens (as of April 2026)
PRICING = {
    "deepseek-v4-pro": {"input": 0.27, "output": 1.10, "cache_hit": 0.07},
    "deepseek-v4-flash": {"input": 0.07, "output": 0.28, "cache_hit": 0.018},
}

# Approximate tokens per character for estimation before API call
TOKENS_PER_CHAR = 0.25
MAX_CONTEXT_TOKENS = 1_000_000


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    model: str = "deepseek-v4-flash"
    elapsed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        p = PRICING.get(self.model, PRICING["deepseek-v4-flash"])
        input_cost = (self.prompt_tokens / 1_000_000) * p["input"]
        output_cost = (self.completion_tokens / 1_000_000) * p["output"]
        cache_cost = (self.cache_hit_tokens / 1_000_000) * p["cache_hit"]
        return round(input_cost + output_cost + cache_cost, 6)

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return round(self.completion_tokens / self.elapsed_seconds, 1)

    def render(self) -> str:
        return (
            f"[dim]Tokens:[/dim] {self.prompt_tokens:,} in + {self.completion_tokens:,} out "
            f"| [dim]Cost:[/dim] ${self.estimated_cost_usd:.4f} "
            f"| [dim]Speed:[/dim] {self.tokens_per_second} tok/s "
            f"| [dim]Time:[/dim] {self.elapsed_seconds:.1f}s"
        )


@dataclass
class DeepSeekResponse:
    content: str
    thinking: Optional[str]
    usage: UsageStats
    model: str


class ContextWindowError(Exception):
    """Raised when content exceeds the 1M token limit."""
    pass


class DeepSeekClient:
    """
    Unified sync/async client for DeepSeek-V4 API.

    Features:
    - Native 1M context window support with pre-flight size checks
    - Streaming with live Rich terminal rendering
    - Thinking mode (chain-of-thought) support
    - Automatic retry with exponential backoff
    - Real-time cost and token tracking
    - Context caching awareness
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Literal["deepseek-v4-pro", "deepseek-v4-flash"] = "deepseek-v4-flash",
        thinking: bool = False,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        max_retries: int = 3,
        timeout: float = 300.0,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY env var "
                "or pass api_key= to DeepSeekClient()."
            )
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.timeout = timeout
        self._base_url = "https://api.deepseek.com"

        self._sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self._base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self._base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Token estimation (fast, no API call)
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Fast local token count estimate using character heuristic."""
        return int(len(text) * TOKENS_PER_CHAR)

    def check_context_size(self, content: str, label: str = "content") -> int:
        """
        Estimate token count and raise ContextWindowError if it exceeds 1M.
        Returns estimated token count.
        """
        estimated = self.estimate_tokens(content)
        if estimated > MAX_CONTEXT_TOKENS:
            raise ContextWindowError(
                f"'{label}' is ~{estimated:,} tokens, which exceeds the "
                f"1M context limit ({MAX_CONTEXT_TOKENS:,}). "
                f"Consider using chunking or summarization."
            )
        utilization = (estimated / MAX_CONTEXT_TOKENS) * 100
        console.print(
            f"[dim]Context check:[/dim] ~{estimated:,} tokens "
            f"({utilization:.1f}% of 1M window)[/dim]"
        )
        return estimated

    # ------------------------------------------------------------------
    # Sync chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        stream: bool = True,
        show_thinking: bool = False,
    ) -> DeepSeekResponse:
        """
        Send a chat request. Streams by default with live Rich output.
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        extra_body = {}
        if self.thinking:
            extra_body["thinking"] = {"type": "enabled"}

        start = time.perf_counter()

        if stream:
            return self._stream_chat(full_messages, extra_body, start, show_thinking)
        else:
            return self._blocking_chat(full_messages, extra_body, start)

    def _blocking_chat(self, messages, extra_body, start) -> DeepSeekResponse:
        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            reasoning_effort=self.reasoning_effort if self.thinking else None,
            extra_body=extra_body if extra_body else None,
        )
        elapsed = time.perf_counter() - start
        msg = response.choices[0].message
        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=self.model,
            elapsed_seconds=elapsed,
        )
        return DeepSeekResponse(
            content=msg.content or "",
            thinking=getattr(msg, "reasoning_content", None),
            usage=usage,
            model=self.model,
        )

    def _stream_chat(self, messages, extra_body, start, show_thinking) -> DeepSeekResponse:
        stream = self._sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            reasoning_effort=self.reasoning_effort if self.thinking else None,
            extra_body=extra_body if extra_body else None,
        )

        full_content = []
        full_thinking = []
        in_thinking = False
        prompt_tokens = 0
        completion_tokens = 0

        with Live(console=console, refresh_per_second=15) as live:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    # usage chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens or 0
                        completion_tokens = chunk.usage.completion_tokens or 0
                    continue

                # Handle thinking blocks
                thinking_delta = getattr(delta, "reasoning_content", None)
                if thinking_delta and show_thinking:
                    full_thinking.append(thinking_delta)
                    thinking_text = "".join(full_thinking)
                    live.update(
                        Panel(
                            Markdown(thinking_text),
                            title="[yellow]🧠 Thinking...[/yellow]",
                            border_style="yellow",
                        )
                    )
                    continue

                if delta.content:
                    full_content.append(delta.content)
                    content_text = "".join(full_content)
                    live.update(Markdown(content_text))

        elapsed = time.perf_counter() - start
        usage = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
            elapsed_seconds=elapsed,
        )
        console.print(f"\n[dim]{usage.render()}[/dim]")

        return DeepSeekResponse(
            content="".join(full_content),
            thinking="".join(full_thinking) if full_thinking else None,
            usage=usage,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Async chat
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        stream: bool = True,
        show_thinking: bool = False,
    ) -> DeepSeekResponse:
        """Async version of chat()."""
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        extra_body = {}
        if self.thinking:
            extra_body["thinking"] = {"type": "enabled"}

        start = time.perf_counter()

        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            stream=False,
            reasoning_effort=self.reasoning_effort if self.thinking else None,
            extra_body=extra_body if extra_body else None,
        )
        elapsed = time.perf_counter() - start
        msg = response.choices[0].message
        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=self.model,
            elapsed_seconds=elapsed,
        )
        return DeepSeekResponse(
            content=msg.content or "",
            thinking=getattr(msg, "reasoning_content", None),
            usage=usage,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Convenience: single-turn quick chat
    # ------------------------------------------------------------------

    def ask(self, question: str, system: Optional[str] = None, **kwargs) -> str:
        """One-liner chat. Returns just the string response."""
        resp = self.chat(
            messages=[{"role": "user", "content": question}],
            system=system,
            **kwargs,
        )
        return resp.content