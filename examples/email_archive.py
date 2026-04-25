"""
Example: Chat With a Year of Your Emails
=========================================
Load your entire email archive (mbox format) into DeepSeek-V4's 1M context.
Ask questions across thousands of emails at once — no search, no filters,
just pure understanding of your entire email history.

This is the demo that breaks people's brains.

Use cases:
  - "Who have I emailed most in the last year?"
  - "What project discussions happened in Q3?"
  - "Did anyone follow up on the contract we discussed in March?"
  - "Summarize all emails related to the acquisition"
  - "What recurring issues does our team deal with?"

Export your email archive:
  Gmail:   Settings → See all settings → Accounts → Google Takeout → Mail → mbox
  Outlook: File → Open & Export → Import/Export → Export to a file → Outlook .pst → convert
  Apple:   Mailbox → Export Mailbox → .mbox

Usage:
    python examples/email_archive.py --mbox ~/Downloads/mail.mbox
    python examples/email_archive.py --mbox mail.mbox --max-emails 2000
"""

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from deepseek_1m import DeepSeekClient, Session, load_mbox

console = Console()

SYSTEM_PROMPT = """You are an expert email analyst. You have access to the user's
complete email archive. Every email in their mailbox is available to you.

Your job is to help the user understand patterns, find information, and extract
insights from their email history.

Capabilities:
- Identify frequent contacts and communication patterns
- Summarize email threads and ongoing projects
- Find specific information across thousands of emails
- Detect action items, commitments, and follow-ups that were never completed
- Analyze communication tone and relationship dynamics
- Track the evolution of projects over time

Be specific when referencing emails (date, subject, sender).
The user's privacy is paramount — only share information from their own archive."""


def main():
    parser = argparse.ArgumentParser(description="Chat with your email archive using DeepSeek-V4")
    parser.add_argument("--mbox", required=True, help="Path to .mbox email archive file")
    parser.add_argument(
        "--max-emails",
        type=int,
        default=3000,
        help="Max emails to load (default: 3000)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-pro",
        choices=["deepseek-v4-flash", "deepseek-v4-pro"],
    )
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    if not os.path.exists(args.mbox):
        console.print(f"[red]Error: File not found: {args.mbox}[/red]")
        console.print(
            "\n[dim]Export your Gmail archive from:[/dim]\n"
            "[cyan]https://takeout.google.com[/cyan] → Mail → .mbox format"
        )
        sys.exit(1)

    console.print(
        Panel.fit(
            "[bold blue]DeepSeek-1M · Email Archive Analysis[/bold blue]\n"
            f"[dim]{args.mbox} — up to {args.max_emails:,} emails[/dim]",
            border_style="blue",
        )
    )

    # ── Load email archive ────────────────────────────────────────────
    ctx = load_mbox(args.mbox, max_emails=args.max_emails)

    # ── Set up session ────────────────────────────────────────────────
    client = DeepSeekClient(model=args.model, thinking=args.thinking)
    session = Session(client=client, context=ctx, system=SYSTEM_PROMPT)

    console.print(
        "\n[bold green]Archive loaded! Your emails are in DeepSeek-V4's memory.[/bold green]"
    )
    console.print("\n[dim]Try asking:[/dim]")
    console.print("  [dim]→ Who are my top 10 most frequent contacts?[/dim]")
    console.print("  [dim]→ What projects have I been working on this year?[/dim]")
    console.print("  [dim]→ Are there any unanswered emails I should follow up on?[/dim]")
    console.print("  [dim]→ Summarize all emails about [topic][/dim]")
    console.print("  [dim]→ What recurring problems keep coming up in my inbox?[/dim]\n")

    session.chat()


if __name__ == "__main__":
    main()
