"""
DeepSeek-1M Session
~~~~~~~~~~~~~~~~~~~
Stateful multi-turn conversation manager that keeps your entire 1M context
alive across turns. Supports context injection, conversation export,
and session replay.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.text import Text
    console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    console = None  # type: ignore
    class Panel:  # type: ignore
        @staticmethod
        def fit(*a, **kw): return ""
        def __init__(self, *a, **kw): pass
    Markdown = Rule = Prompt = Text = None  # type: ignore

from .client import DeepSeekClient, DeepSeekResponse, UsageStats

def _sprint(*args, **kwargs):
    if console is not None:
        console.print(*args, **kwargs)
    else:
        # Fallback: strip rich markup and print plain
        import re as _re
        text = " ".join(str(a) for a in args)
        text = _re.sub(r'\[.*?\]', '', text)
        print(text)


from .loader import LoadResult



@dataclass
class Turn:
    role: str  # 'user' | 'assistant'
    content: str
    thinking: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    tokens_in: int = 0
    tokens_out: int = 0


class Session:
    """
    A stateful 1M-context chat session.

    The session keeps a running message history and an optional
    "context block" (your codebase / document / email archive) that
    is prepended to every request — staying resident in DeepSeek-V4's
    1M context window for the entire conversation.

    Usage::

        from deepseek_1m import Session, load

        ctx = load("https://github.com/fastapi/fastapi")
        session = Session(context=ctx)
        session.chat()   # starts interactive REPL
    """

    def __init__(
        self,
        client: Optional[DeepSeekClient] = None,
        context: Optional[LoadResult] = None,
        system: Optional[str] = None,
        show_thinking: bool = False,
    ):
        self.client = client or DeepSeekClient()
        self.context = context
        self.system = system or self._default_system()
        self.show_thinking = show_thinking
        self.history: list[Turn] = []
        self._context_string: Optional[str] = None

        if context:
            self._context_string = context.to_context_string()
            est = context.total_tokens_estimate
            _sprint(
                Panel(
                    f"[green]✓ Context loaded[/green] — "
                    f"[cyan]{context.total_files}[/cyan] files, "
                    f"[cyan]~{est:,}[/cyan] tokens "
                    f"([cyan]{(est/1_000_000)*100:.1f}%[/cyan] of 1M window)\n"
                    f"Source: [dim]{context.source_label}[/dim]",
                    title="[bold]DeepSeek-1M Session[/bold]",
                    border_style="blue",
                )
            )

    def _default_system(self) -> str:
        return (
            "You are an expert AI assistant with access to a large context window "
            "of up to 1 million tokens. When the user provides source code, documents, "
            "or data, you analyze it thoroughly and give precise, accurate answers. "
            "You cite specific file names and line references when relevant. "
            "You are concise, technical, and deeply helpful."
        )

    # ------------------------------------------------------------------
    # Single-turn ask
    # ------------------------------------------------------------------

    def ask(self, question: str, stream: bool = True) -> str:
        """Send a question, return the response string."""
        messages = self._build_messages(question)
        response = self.client.chat(
            messages=messages,
            system=self.system,
            stream=stream,
            show_thinking=self.show_thinking,
        )
        self.history.append(
            Turn(
                role="user",
                content=question,
                tokens_in=response.usage.prompt_tokens,
            )
        )
        self.history.append(
            Turn(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tokens_out=response.usage.completion_tokens,
            )
        )
        return response.content

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------

    def chat(self):
        """
        Launch an interactive terminal chat session.

        Commands:
          /clear     — clear conversation history (keeps context)
          /save      — save session to JSON
          /tokens    — show current token usage
          /thinking  — toggle thinking mode display
          /exit      — quit
        """
        _sprint(Rule("[bold blue]DeepSeek-1M Chat[/bold blue]"))
        _sprint(
            "[dim]Commands: /clear /save /tokens /thinking /exit[/dim]\n"
        )

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                _sprint("\n[dim]Session ended.[/dim]")
                break

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            _sprint(Rule(style="dim"))
            _sprint("[bold green]DeepSeek-V4[/bold green]")
            self.ask(user_input)
            _sprint()

    def _handle_command(self, cmd: str):
        cmd = cmd.lower().strip()
        if cmd == "/clear":
            self.history.clear()
            _sprint("[yellow]✓ Conversation history cleared (context preserved)[/yellow]")
        elif cmd == "/tokens":
            total_in = sum(t.tokens_in for t in self.history)
            total_out = sum(t.tokens_out for t in self.history)
            ctx_est = self.context.total_tokens_estimate if self.context else 0
            _sprint(
                f"[cyan]Context tokens:[/cyan] ~{ctx_est:,}\n"
                f"[cyan]Conversation tokens in:[/cyan] {total_in:,}\n"
                f"[cyan]Conversation tokens out:[/cyan] {total_out:,}\n"
                f"[cyan]Total estimated:[/cyan] ~{ctx_est + total_in + total_out:,} / 1,000,000"
            )
        elif cmd == "/thinking":
            self.show_thinking = not self.show_thinking
            state = "ON" if self.show_thinking else "OFF"
            _sprint(f"[yellow]Thinking mode display: {state}[/yellow]")
        elif cmd.startswith("/save"):
            parts = cmd.split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else "session.json"
            self.save(path)
        elif cmd == "/exit":
            raise KeyboardInterrupt
        else:
            _sprint(f"[red]Unknown command: {cmd}[/red]")

    # ------------------------------------------------------------------
    # Message builder — injects context on first turn
    # ------------------------------------------------------------------

    def _build_messages(self, question: str) -> list[dict]:
        messages = []

        # Add conversation history
        for turn in self.history:
            messages.append({"role": turn.role, "content": turn.content})

        # On the very first message, inject the context block
        if self._context_string and not self.history:
            content = (
                f"I'm going to provide you with a large context to analyze.\n\n"
                f"{self._context_string}\n\n"
                f"---\n\nWith the above context fully loaded, please answer:\n\n{question}"
            )
        elif self._context_string and len(self.history) == 0:
            content = question
        else:
            content = question

        messages.append({"role": "user", "content": content})
        return messages

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "session.json"):
        """Save session history to JSON for replay or analysis."""
        data = {
            "model": self.client.model,
            "source": self.context.source_label if self.context else None,
            "turns": [asdict(t) for t in self.history],
        }
        Path(path).write_text(json.dumps(data, indent=2))
        _sprint(f"[green]✓ Session saved to {path}[/green]")

    @classmethod
    def load(cls, path: str, client: Optional[DeepSeekClient] = None) -> "Session":
        """Restore a session from a saved JSON file."""
        data = json.loads(Path(path).read_text())
        session = cls(client=client)
        session.history = [Turn(**t) for t in data["turns"]]
        _sprint(f"[green]✓ Session restored: {len(session.history)} turns[/green]")
        return session
