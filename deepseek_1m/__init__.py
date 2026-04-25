"""
deepseek-1M
~~~~~~~~~~~
Unlock DeepSeek-V4's 1,000,000-token context window.
Load entire codebases, books, email archives, and more — then chat with them.

Quick start::

    from deepseek_1m import load, Session

    # Load a GitHub repo
    ctx = load("https://github.com/fastapi/fastapi")

    # Start chatting with it
    session = Session(context=ctx)
    session.chat()

Or in one line::

    from deepseek_1m import ask
    print(ask("What is 2+2?"))
"""

from .client import DeepSeekClient, DeepSeekResponse, UsageStats, ContextWindowError
from .loader import load, load_local, load_github, load_pdf, load_mbox, load_url, LoadResult
from .session import Session

__version__ = "1.0.0"
__author__ = "deepseek-1M contributors"
__license__ = "MIT"

__all__ = [
    # Client
    "DeepSeekClient",
    "DeepSeekResponse",
    "UsageStats",
    "ContextWindowError",
    # Loaders
    "load",
    "load_local",
    "load_github",
    "load_pdf",
    "load_mbox",
    "load_url",
    "LoadResult",
    # Session
    "Session",
]


def ask(question: str, model: str = "deepseek-v4-flash", **kwargs) -> str:
    """One-liner: ask DeepSeek-V4 a question without any setup."""
    client = DeepSeekClient(model=model)
    return client.ask(question, **kwargs)
