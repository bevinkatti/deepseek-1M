"""
Example: Chat with an Entire GitHub Codebase
============================================
This demo loads a full GitHub repository into DeepSeek-V4's 1M context
window and lets you chat with it interactively.

DeepSeek-V4 holds the ENTIRE codebase in memory — no chunking, no embeddings,
no retrieval. Just pure, complete understanding.

Usage:
    export DEEPSEEK_API_KEY=your_key_here
    python examples/codebase_chat.py
    python examples/codebase_chat.py --repo https://github.com/tiangolo/fastapi
    python examples/codebase_chat.py --repo . --model deepseek-v4-pro
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from deepseek_1m import DeepSeekClient, Session, load

console = Console()

SYSTEM_PROMPT = """You are an expert software engineer with complete knowledge of the
provided codebase. You have read every single file in the repository.

When answering questions:
- Reference specific file paths and line numbers when relevant
- Explain architectural decisions you can infer from the code
- Identify patterns, anti-patterns, and potential improvements
- Be precise and technical — the user is a developer

You have access to the COMPLETE source code. Never say you "don't have access"
to a file — if it's in the repo, it's in your context."""


def main():
    parser = argparse.ArgumentParser(description="Chat with any GitHub repo or local codebase")
    parser.add_argument(
        "--repo",
        default="https://github.com/tiangolo/fastapi",
        help="GitHub URL or local path (default: fastapi)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        choices=["deepseek-v4-flash", "deepseek-v4-pro"],
        help="Model to use (flash=fast+cheap, pro=most capable)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="GitHub branch (default: main)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable chain-of-thought thinking mode",
    )
    parser.add_argument(
        "--github-token",
        help="GitHub token for private repos or higher rate limits",
    )
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold blue]DeepSeek-1M · Codebase Chat[/bold blue]\n"
            f"[dim]Loading: {args.repo}[/dim]",
            border_style="blue",
        )
    )

    # ── Load the codebase ────────────────────────────────────────────
    kwargs = {}
    if args.github_token:
        kwargs["token"] = args.github_token
    if args.branch != "main":
        kwargs["branch"] = args.branch

    console.print(f"\n[cyan]Loading codebase...[/cyan]")
    ctx = load(args.repo, **kwargs)

    if ctx.total_tokens_estimate > 950_000:
        console.print(
            "[yellow]⚠ Context is very large. Consider using --model deepseek-v4-pro "
            "for best results with long context.[/yellow]"
        )

    # ── Set up client ────────────────────────────────────────────────
    client = DeepSeekClient(
        model=args.model,
        thinking=args.thinking,
        reasoning_effort="high" if args.thinking else "medium",
    )

    # ── Start session ────────────────────────────────────────────────
    session = Session(
        client=client,
        context=ctx,
        system=SYSTEM_PROMPT,
        show_thinking=args.thinking,
    )

    console.print("\n[bold green]Codebase loaded! Ask anything about it.[/bold green]")
    console.print("[dim]Examples:[/dim]")
    console.print("  [dim]→ How is authentication implemented?[/dim]")
    console.print("  [dim]→ What does the main entry point do?[/dim]")
    console.print("  [dim]→ Find all database queries[/dim]")
    console.print("  [dim]→ What dependencies does this project use?[/dim]")
    console.print("  [dim]→ Explain the overall architecture[/dim]\n")

    session.chat()


if __name__ == "__main__":
    main()
