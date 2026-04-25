"""
DeepSeek-1M Loader
~~~~~~~~~~~~~~~~~~
Ingest any source into a single context-ready string for DeepSeek-V4's 1M window.

Supports:
  - GitHub repositories (public, via raw content API)
  - Local directories (full recursive file tree)
  - PDF files (text extraction)
  - Plain text / Markdown files
  - Email archives (mbox format)
  - URLs (webpage text extraction)
"""

from __future__ import annotations

import email
import mailbox
import mimetypes
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )
    from rich.table import Table
    console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    console = None  # type: ignore

    class _NoOpProgress:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def add_task(self, *a, **kw): return 0
        def advance(self, *a): pass

    Progress = _NoOpProgress  # type: ignore
    SpinnerColumn = BarColumn = TextColumn = TimeElapsedColumn = lambda *a, **kw: None  # type: ignore

    class Table:  # type: ignore
        def __init__(self, *a, **kw): pass
        def add_column(self, *a, **kw): pass
        def add_row(self, *a, **kw): pass

def _cprint(*args, **kwargs):
    if console is not None:
        console.print(*args, **kwargs)



# File extensions we know how to read as text
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json", ".xml",
    ".html", ".css", ".scss", ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".proto", ".tf", ".hcl", ".dockerfile",
    ".env", ".ini", ".cfg", ".conf", ".gitignore", ".editorconfig",
    "makefile", "dockerfile", "procfile",
}

# Folders to always skip
SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".next", ".nuxt",
    "venv", ".venv", "env", ".env", ".tox", "coverage", ".coverage",
    "htmlcov", ".eggs", "*.egg-info",
}

# Files to always skip
SKIP_FILES = {
    ".DS_Store", "Thumbs.db", ".gitkeep", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
}

MAX_FILE_SIZE_BYTES = 500_000  # 500KB per file max


@dataclass
class LoadedFile:
    path: str
    content: str
    size_bytes: int
    source_type: str  # 'local', 'github', 'pdf', 'email', 'url'

    @property
    def token_estimate(self) -> int:
        return int(len(self.content) * 0.25)


@dataclass
class LoadResult:
    files: list[LoadedFile] = field(default_factory=list)
    source_label: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(f.content) for f in self.files)

    @property
    def total_tokens_estimate(self) -> int:
        return int(self.total_chars * 0.25)

    @property
    def total_files(self) -> int:
        return len(self.files)

    def to_context_string(self) -> str:
        """
        Serialize all loaded files into a single structured string
        optimized for DeepSeek-V4's 1M context window.
        """
        parts = [
            f"# Source: {self.source_label}",
            f"# Files loaded: {self.total_files}",
            f"# Estimated tokens: ~{self.total_tokens_estimate:,}",
            "",
        ]
        for lf in self.files:
            ext = Path(lf.path).suffix.lstrip(".") or "txt"
            parts.append(f"## File: {lf.path}")
            parts.append(f"```{ext}")
            parts.append(lf.content.rstrip())
            parts.append("```")
            parts.append("")
        return "\n".join(parts)

    def print_summary(self):
        table = Table(title=f"📦 Loaded: {self.source_label}", show_header=True)
        table.add_column("Stat", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Files loaded", str(self.total_files))
        table.add_row("Total characters", f"{self.total_chars:,}")
        table.add_row("Est. tokens", f"~{self.total_tokens_estimate:,}")
        table.add_row(
            "Context utilization",
            f"{(self.total_tokens_estimate / 1_000_000) * 100:.1f}% of 1M",
        )
        if self.errors:
            table.add_row("Skipped/errors", str(len(self.errors)))
        _cprint(table)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def load(source: str, **kwargs) -> LoadResult:
    """
    Universal loader. Auto-detects source type from the input string.

    Args:
        source: One of:
            - A GitHub URL:      "https://github.com/owner/repo"
            - A local path:      "/path/to/project" or "."
            - A PDF path:        "/path/to/doc.pdf"
            - An mbox path:      "/path/to/mail.mbox"
            - A web URL:         "https://example.com/article"

    Returns:
        LoadResult with all files and a .to_context_string() method.
    """
    source = source.strip()

    if source.startswith("https://github.com") or source.startswith("http://github.com"):
        return load_github(source, **kwargs)
    elif source.endswith(".pdf") and os.path.isfile(source):
        return load_pdf(source)
    elif source.endswith(".mbox") and os.path.isfile(source):
        return load_mbox(source, **kwargs)
    elif os.path.isdir(source) or os.path.isfile(source):
        return load_local(source, **kwargs)
    elif source.startswith("http://") or source.startswith("https://"):
        return load_url(source)
    else:
        raise ValueError(
            f"Cannot determine source type for: '{source}'\n"
            "Pass a GitHub URL, local path, PDF path, mbox path, or web URL."
        )


# ──────────────────────────────────────────────────────────────────────
# Local directory / file loader
# ──────────────────────────────────────────────────────────────────────

def load_local(path: str, max_files: int = 2000) -> LoadResult:
    """Load a local directory or single file recursively."""
    root = Path(path).resolve()
    result = LoadResult(source_label=str(root))

    if root.is_file():
        lf = _read_file(root, str(root), "local")
        if lf:
            result.files.append(lf)
        return result

    all_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for fname in filenames:
            if fname in SKIP_FILES:
                continue
            fpath = Path(dirpath) / fname
            all_paths.append(fpath)

    all_paths = all_paths[:max_files]

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Loading files..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("loading", total=len(all_paths))
        for fpath in all_paths:
            rel = str(fpath.relative_to(root))
            lf = _read_file(fpath, rel, "local")
            if lf:
                result.files.append(lf)
            else:
                result.errors.append(rel)
            progress.advance(task)

    result.print_summary()
    return result


# ──────────────────────────────────────────────────────────────────────
# GitHub loader
# ──────────────────────────────────────────────────────────────────────

def load_github(
    url: str,
    branch: str = "main",
    token: Optional[str] = None,
    max_files: int = 1000,
) -> LoadResult:
    """
    Load a public (or private with token) GitHub repo via the API.

    Args:
        url:      Full GitHub URL, e.g. "https://github.com/owner/repo"
        branch:   Branch to load (default: 'main', falls back to 'master')
        token:    GitHub personal access token for private repos / rate limits
        max_files: Cap on number of files to load
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")
    owner, repo = parts[0], parts[1]

    result = LoadResult(source_label=f"github.com/{owner}/{repo}@{branch}")
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "deepseek-1M"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def api_get(endpoint: str) -> dict | list:
        req = urllib.request.Request(
            f"https://api.github.com{endpoint}", headers=headers
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            import json
            return json.loads(r.read().decode())

    # Get default branch if not specified
    try:
        repo_info = api_get(f"/repos/{owner}/{repo}")
        default_branch = repo_info.get("default_branch", "main")
        if branch == "main" and default_branch != "main":
            branch = default_branch
            result.source_label = f"github.com/{owner}/{repo}@{branch}"
    except Exception:
        pass

    # Get full file tree
    _cprint(f"[cyan]Fetching file tree from {owner}/{repo}...[/cyan]")
    try:
        tree_data = api_get(
            f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch repo tree. Is the repo public? Error: {e}\n"
            "For private repos, pass token=<your_github_token>"
        )

    blobs = [
        item for item in tree_data.get("tree", [])
        if item["type"] == "blob"
        and _is_readable_extension(item["path"])
        and item.get("size", 0) < MAX_FILE_SIZE_BYTES
    ][:max_files]

    _cprint(f"[green]Found {len(blobs)} readable files[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Downloading..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("dl", total=len(blobs))
        for item in blobs:
            try:
                raw_url = (
                    f"https://raw.githubusercontent.com/"
                    f"{owner}/{repo}/{branch}/{item['path']}"
                )
                req = urllib.request.Request(raw_url, headers={"User-Agent": "deepseek-1M"})
                if token:
                    req.add_header("Authorization", f"Bearer {token}")
                with urllib.request.urlopen(req, timeout=10) as r:
                    content = r.read().decode("utf-8", errors="replace")
                result.files.append(
                    LoadedFile(
                        path=item["path"],
                        content=content,
                        size_bytes=item.get("size", len(content)),
                        source_type="github",
                    )
                )
            except Exception as ex:
                result.errors.append(f"{item['path']}: {ex}")
            finally:
                progress.advance(task)

    result.print_summary()
    return result


# ──────────────────────────────────────────────────────────────────────
# PDF loader
# ──────────────────────────────────────────────────────────────────────

def load_pdf(path: str) -> LoadResult:
    """Extract text from a PDF file."""
    try:
        import pypdf
    except ImportError:
        raise ImportError("pip install pypdf  # required for PDF loading")

    result = LoadResult(source_label=path)
    reader = pypdf.PdfReader(path)
    pages = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Extracting PDF pages..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("pdf", total=len(reader.pages))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"--- Page {i + 1} ---\n{text}")
            progress.advance(task)

    full_text = "\n\n".join(pages)
    result.files.append(
        LoadedFile(
            path=Path(path).name,
            content=full_text,
            size_bytes=os.path.getsize(path),
            source_type="pdf",
        )
    )
    result.print_summary()
    return result


# ──────────────────────────────────────────────────────────────────────
# Mbox / email archive loader
# ──────────────────────────────────────────────────────────────────────

def load_mbox(path: str, max_emails: int = 5000) -> LoadResult:
    """Load an mbox email archive."""
    result = LoadResult(source_label=f"mbox:{Path(path).name}")
    mbox = mailbox.mbox(path)
    emails_text = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Parsing emails..."),
    ) as progress:
        task = progress.add_task("mbox", total=None)
        for i, msg in enumerate(mbox):
            if i >= max_emails:
                break
            subject = msg.get("subject", "(no subject)")
            from_ = msg.get("from", "unknown")
            date = msg.get("date", "")
            body = _extract_email_body(msg)
            emails_text.append(
                f"=== Email {i + 1} ===\n"
                f"From: {from_}\nDate: {date}\nSubject: {subject}\n\n{body}"
            )
            progress.advance(task)

    result.files.append(
        LoadedFile(
            path=Path(path).name,
            content="\n\n".join(emails_text),
            size_bytes=os.path.getsize(path),
            source_type="email",
        )
    )
    result.print_summary()
    return result


# ──────────────────────────────────────────────────────────────────────
# URL / webpage loader
# ──────────────────────────────────────────────────────────────────────

def load_url(url: str) -> LoadResult:
    """Fetch and extract readable text from a web URL."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("pip install beautifulsoup4  # required for URL loading")

    result = LoadResult(source_label=url)
    _cprint(f"[cyan]Fetching: {url}[/cyan]")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; deepseek-1M/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    result.files.append(
        LoadedFile(
            path=url,
            content=text,
            size_bytes=len(text),
            source_type="url",
        )
    )
    result.print_summary()
    return result


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _read_file(path: Path, rel: str, source_type: str) -> Optional[LoadedFile]:
    """Try to read a file as UTF-8 text. Returns None if not readable."""
    if not _is_readable_extension(str(path)):
        return None
    try:
        size = path.stat().st_size
        if size == 0 or size > MAX_FILE_SIZE_BYTES:
            return None
        content = path.read_text(encoding="utf-8", errors="replace")
        return LoadedFile(path=rel, content=content, size_bytes=size, source_type=source_type)
    except Exception:
        return None


def _is_readable_extension(filepath: str) -> bool:
    ext = Path(filepath).suffix.lower()
    name = Path(filepath).name.lower()
    return ext in TEXT_EXTENSIONS or name in TEXT_EXTENSIONS


def _extract_email_body(msg: email.message.Message) -> str:
    """Extract plain text body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                try:
                    return part.get_payload(decode=True).decode("utf-8", errors="replace")
                except Exception:
                    pass
        return ""
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode("utf-8", errors="replace")
        except Exception:
            pass
    return str(msg.get_payload())
