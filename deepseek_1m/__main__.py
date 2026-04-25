"""
DeepSeek-1M CLI
~~~~~~~~~~~~~~~
Command-line interface for deepseek-1m.

Usage:
    deepseek-1m chat --repo https://github.com/owner/repo
    deepseek-1m chat --pdf document.pdf
    deepseek-1m chat --folder ./my-project
    deepseek-1m ask "What is the capital of France?"
    deepseek-1m demo
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    parser = argparse.ArgumentParser(
        prog="deepseek-1m",
        description="Chat with your codebase, docs, or emails using DeepSeek-V4's 1M context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  deepseek-1m chat --repo https://github.com/fastapi/fastapi
  deepseek-1m chat --pdf my_contract.pdf --model deepseek-v4-pro
  deepseek-1m chat --folder ./my-project --thinking
  deepseek-1m ask "Who wrote the Zen of Python?"
  deepseek-1m demo        # launch web UI
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── chat ─────────────────────────────────────────────────────────
    chat_parser = subparsers.add_parser("chat", help="Load a source and start chatting")
    source_group = chat_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--repo", help="GitHub URL or local directory")
    source_group.add_argument("--pdf", help="Path to PDF file")
    source_group.add_argument("--folder", help="Path to local folder")
    source_group.add_argument("--mbox", help="Path to .mbox email archive")
    source_group.add_argument("--url", help="Web URL to load")
    chat_parser.add_argument("--model", default="deepseek-v4-flash",
                              choices=["deepseek-v4-flash", "deepseek-v4-pro"])
    chat_parser.add_argument("--thinking", action="store_true")
    chat_parser.add_argument("--github-token", help="GitHub PAT for private repos")

    # ── ask ──────────────────────────────────────────────────────────
    ask_parser = subparsers.add_parser("ask", help="Ask a one-off question")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument("--model", default="deepseek-v4-flash",
                            choices=["deepseek-v4-flash", "deepseek-v4-pro"])

    # ── demo ─────────────────────────────────────────────────────────
    demo_parser = subparsers.add_parser("demo", help="Launch the Streamlit web demo")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "ask":
        from deepseek_1m import ask
        result = ask(args.question, model=args.model, stream=True)

    elif args.command == "demo":
        import subprocess
        demo_path = str(__import__("pathlib").Path(__file__).parent.parent / "demo" / "app.py")
        console.print(
            Panel.fit(
                "[bold blue]Launching DeepSeek-1M Web Demo[/bold blue]\n"
                "[dim]Opening http://localhost:8501[/dim]",
                border_style="blue",
            )
        )
        subprocess.run(["streamlit", "run", demo_path], check=True)

    elif args.command == "chat":
        from deepseek_1m import DeepSeekClient, Session, load, load_pdf, load_mbox, load_url

        source = args.repo or args.folder or args.pdf or args.mbox or args.url

        kwargs = {}
        if args.github_token and args.repo:
            kwargs["token"] = args.github_token

        if args.pdf:
            ctx = load_pdf(args.pdf)
        elif args.mbox:
            ctx = load_mbox(args.mbox)
        elif args.url:
            ctx = load_url(args.url)
        else:
            ctx = load(source, **kwargs)

        client = DeepSeekClient(model=args.model, thinking=args.thinking)
        session = Session(client=client, context=ctx, show_thinking=args.thinking)
        session.chat()


if __name__ == "__main__":
    main()
