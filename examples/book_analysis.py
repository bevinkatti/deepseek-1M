"""
Example: Analyze a Full Book or PDF with 1M Context
====================================================
Load an entire PDF (research paper, book, legal doc) into DeepSeek-V4's
1M context window. No page limits. No chunking. The entire document at once.

Use cases:
  - Research papers: cross-reference findings across sections
  - Legal contracts: find conflicting clauses across a 200-page doc
  - Technical manuals: answer questions that span multiple chapters
  - Academic books: deep analysis with full context retention

Usage:
    python examples/book_analysis.py --pdf path/to/book.pdf
    python examples/book_analysis.py --pdf research_paper.pdf --thinking
"""

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from deepseek_1m import DeepSeekClient, Session, load_pdf

console = Console()

SYSTEM_PROMPT = """You are an expert analyst with the complete text of a document
in your context. You have read every page.

Your capabilities:
- Answer questions that require understanding across multiple sections
- Identify themes, patterns, and contradictions in the document
- Provide precise quotes and page references when available
- Summarize any section, chapter, or the entire work
- Compare and contrast ideas from different parts of the document

Never claim you don't have access to any part of the document.
The entire text is available to you."""


def demo_questions(source_label: str) -> list[str]:
    """Return demo questions appropriate for the document type."""
    is_legal = any(w in source_label.lower() for w in ["contract", "agreement", "legal", "law"])
    is_research = any(w in source_label.lower() for w in ["paper", "arxiv", "research", "study"])

    if is_legal:
        return [
            "What are the key obligations of each party?",
            "Are there any termination clauses? What triggers them?",
            "Identify any clauses that could be considered unfavorable to the buyer.",
            "What is the total liability cap mentioned in the document?",
            "Summarize the entire contract in plain English.",
        ]
    elif is_research:
        return [
            "What is the main thesis or contribution of this paper?",
            "What methodology did the authors use?",
            "What were the key findings and results?",
            "What limitations do the authors acknowledge?",
            "How does this compare to prior work in the field?",
        ]
    else:
        return [
            "What is this document about? Give me a high-level summary.",
            "What are the most important concepts covered?",
            "Are there any contradictions or inconsistencies in the text?",
            "What questions does this document answer? What does it leave open?",
            "Extract all key facts, figures, and statistics mentioned.",
        ]


def main():
    parser = argparse.ArgumentParser(description="Analyze any PDF with DeepSeek-V4's 1M context")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument(
        "--model",
        default="deepseek-v4-pro",
        choices=["deepseek-v4-flash", "deepseek-v4-pro"],
        help="Model (pro recommended for long documents)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode for deeper analysis",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo questions automatically without interactive mode",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        console.print(f"[red]Error: File not found: {args.pdf}[/red]")
        sys.exit(1)

    console.print(
        Panel.fit(
            "[bold blue]DeepSeek-1M · Book & Document Analysis[/bold blue]\n"
            f"[dim]File: {args.pdf}[/dim]",
            border_style="blue",
        )
    )

    # ── Load PDF ─────────────────────────────────────────────────────
    ctx = load_pdf(args.pdf)

    # ── Set up client ────────────────────────────────────────────────
    client = DeepSeekClient(
        model=args.model,
        thinking=args.thinking,
        reasoning_effort="high",
    )
    session = Session(
        client=client,
        context=ctx,
        system=SYSTEM_PROMPT,
        show_thinking=args.thinking,
    )

    if args.demo:
        # Auto-run demo questions
        questions = demo_questions(ctx.source_label)
        console.print("\n[bold]Running demo analysis...[/bold]\n")
        for i, q in enumerate(questions, 1):
            console.print(Rule(f"[cyan]Question {i}/{len(questions)}[/cyan]"))
            console.print(f"[bold cyan]Q:[/bold cyan] {q}\n")
            session.ask(q)
            console.print()
    else:
        console.print("\n[bold green]Document loaded! Start asking questions.[/bold green]")
        console.print("[dim]The entire document is in DeepSeek-V4's memory.[/dim]\n")

        # Show suggested questions
        suggestions = demo_questions(ctx.source_label)
        console.print("[dim]Suggested questions:[/dim]")
        for q in suggestions[:3]:
            console.print(f"  [dim]→ {q}[/dim]")
        console.print()

        session.chat()


if __name__ == "__main__":
    main()
