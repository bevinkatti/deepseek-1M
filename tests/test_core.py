"""
Tests for deepseek-1m
~~~~~~~~~~~~~~~~~~~~~
Run with: pytest tests/ -v
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseek_1m.client import DeepSeekClient, UsageStats, ContextWindowError
from deepseek_1m.loader import (
    LoadResult, LoadedFile,
    load_local, _is_readable_extension, _read_file,
)
from deepseek_1m.session import Session, Turn


# ──────────────────────────────────────────────────────────────────────
# UsageStats
# ──────────────────────────────────────────────────────────────────────

class TestUsageStats:
    def test_total_tokens(self):
        u = UsageStats(prompt_tokens=1000, completion_tokens=500)
        assert u.total_tokens == 1500

    def test_cost_calculation_flash(self):
        u = UsageStats(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            model="deepseek-v4-flash",
        )
        # 0.07 + 0.28 = 0.35 per 1M tokens
        assert abs(u.estimated_cost_usd - 0.35) < 0.001

    def test_cost_calculation_pro(self):
        u = UsageStats(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            model="deepseek-v4-pro",
        )
        # 0.27 + 1.10 = 1.37 per 1M tokens
        assert abs(u.estimated_cost_usd - 1.37) < 0.001

    def test_tokens_per_second(self):
        u = UsageStats(completion_tokens=1000, elapsed_seconds=2.0)
        assert u.tokens_per_second == 500.0

    def test_tokens_per_second_zero_elapsed(self):
        u = UsageStats(completion_tokens=1000, elapsed_seconds=0.0)
        assert u.tokens_per_second == 0.0

    def test_render_returns_string(self):
        u = UsageStats(prompt_tokens=100, completion_tokens=50, elapsed_seconds=1.0)
        rendered = u.render()
        assert "100" in rendered
        assert "50" in rendered


# ──────────────────────────────────────────────────────────────────────
# LoadResult
# ──────────────────────────────────────────────────────────────────────

class TestLoadResult:
    def make_result(self, files: list[tuple[str, str]]) -> LoadResult:
        result = LoadResult(source_label="test")
        for path, content in files:
            result.files.append(
                LoadedFile(path=path, content=content, size_bytes=len(content), source_type="local")
            )
        return result

    def test_total_chars(self):
        r = self.make_result([("a.py", "hello"), ("b.py", "world!")])
        assert r.total_chars == 11

    def test_total_tokens_estimate(self):
        # 100 chars * 0.25 = 25 tokens
        content = "x" * 100
        r = self.make_result([("a.txt", content)])
        assert r.total_tokens_estimate == 25

    def test_total_files(self):
        r = self.make_result([("a.py", "a"), ("b.py", "b"), ("c.py", "c")])
        assert r.total_files == 3

    def test_to_context_string_structure(self):
        r = self.make_result([("src/main.py", "print('hello')")])
        ctx = r.to_context_string()
        assert "## File: src/main.py" in ctx
        assert "```py" in ctx
        assert "print('hello')" in ctx
        assert "Source: test" in ctx

    def test_to_context_string_multiple_files(self):
        r = self.make_result([
            ("readme.md", "# Hello"),
            ("app.py", "import os"),
        ])
        ctx = r.to_context_string()
        assert "readme.md" in ctx
        assert "app.py" in ctx
        assert "# Hello" in ctx
        assert "import os" in ctx


# ──────────────────────────────────────────────────────────────────────
# File extension detection
# ──────────────────────────────────────────────────────────────────────

class TestIsReadableExtension:
    def test_python_files(self):
        assert _is_readable_extension("main.py")
        assert _is_readable_extension("utils/helper.py")

    def test_js_ts_files(self):
        assert _is_readable_extension("index.js")
        assert _is_readable_extension("app.ts")
        assert _is_readable_extension("Component.tsx")

    def test_config_files(self):
        assert _is_readable_extension("config.yaml")
        assert _is_readable_extension("pyproject.toml")
        assert _is_readable_extension(".env")

    def test_binary_files_excluded(self):
        assert not _is_readable_extension("image.png")
        assert not _is_readable_extension("archive.zip")
        assert not _is_readable_extension("binary.exe")
        assert not _is_readable_extension("data.pkl")

    def test_dockerfile_no_extension(self):
        assert _is_readable_extension("dockerfile")
        assert _is_readable_extension("makefile")


# ──────────────────────────────────────────────────────────────────────
# Local loader
# ──────────────────────────────────────────────────────────────────────

class TestLoadLocal:
    def test_load_single_file(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("print('hello world')")
        result = load_local(str(f))
        assert result.total_files == 1
        assert "print('hello world')" in result.files[0].content

    def test_load_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("import os")
        (tmp_path / "utils.py").write_text("def add(a, b): return a + b")
        (tmp_path / "readme.md").write_text("# My Project")
        result = load_local(str(tmp_path))
        assert result.total_files == 3
        paths = [f.path for f in result.files]
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)

    def test_skips_node_modules(self, tmp_path):
        (tmp_path / "index.js").write_text("const x = 1")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lodash.js").write_text("// lodash")
        result = load_local(str(tmp_path))
        assert result.total_files == 1

    def test_skips_pycache(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "app.cpython-311.pyc").write_bytes(b"\x00\x01\x02")
        result = load_local(str(tmp_path))
        assert result.total_files == 1

    def test_skips_binary_files(self, tmp_path):
        (tmp_path / "script.py").write_text("pass")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        result = load_local(str(tmp_path))
        assert result.total_files == 1

    def test_source_label_is_set(self, tmp_path):
        (tmp_path / "x.py").write_text("pass")
        result = load_local(str(tmp_path))
        assert str(tmp_path) in result.source_label

    def test_nested_dirs(self, tmp_path):
        src = tmp_path / "src" / "core"
        src.mkdir(parents=True)
        (src / "engine.py").write_text("class Engine: pass")
        (tmp_path / "main.py").write_text("from src.core.engine import Engine")
        result = load_local(str(tmp_path))
        assert result.total_files == 2


# ──────────────────────────────────────────────────────────────────────
# DeepSeekClient — mocked to avoid real API calls
# ──────────────────────────────────────────────────────────────────────

class TestDeepSeekClientInit:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            DeepSeekClient(api_key=None)

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
        client = DeepSeekClient()
        assert client.api_key == "test-key-123"

    def test_explicit_api_key(self):
        client = DeepSeekClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_default_model(self):
        client = DeepSeekClient(api_key="key")
        assert client.model == "deepseek-v4-flash"

    def test_context_check_passes(self):
        client = DeepSeekClient(api_key="key")
        # 10 chars = ~2.5 tokens, well within 1M
        tokens = client.check_context_size("hello world", label="test")
        assert tokens < 10

    def test_context_check_fails_on_huge_input(self):
        client = DeepSeekClient(api_key="key")
        huge_content = "x" * 4_200_000  # ~1.05M tokens
        with pytest.raises(ContextWindowError, match="exceeds the 1M context limit"):
            client.check_context_size(huge_content)

    def test_estimate_tokens(self):
        client = DeepSeekClient(api_key="key")
        # 400 chars * 0.25 = 100 tokens
        assert client.estimate_tokens("x" * 400) == 100


# ──────────────────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────────────────

class TestSession:
    def make_session(self):
        client = DeepSeekClient(api_key="test-key")
        return Session(client=client)

    def test_session_init_no_context(self):
        s = self.make_session()
        assert s.history == []
        assert s._context_string is None

    def test_session_with_context(self):
        result = LoadResult(source_label="test-repo")
        result.files.append(
            LoadedFile(path="main.py", content="x = 1", size_bytes=5, source_type="local")
        )
        client = DeepSeekClient(api_key="test-key")
        s = Session(client=client, context=result)
        assert s._context_string is not None
        assert "main.py" in s._context_string

    def test_build_messages_no_context(self):
        s = self.make_session()
        msgs = s._build_messages("What is 2+2?")
        assert len(msgs) == 1
        assert msgs[-1]["role"] == "user"
        assert "What is 2+2?" in msgs[-1]["content"]

    def test_build_messages_with_history(self):
        s = self.make_session()
        s.history = [
            Turn(role="user", content="Hello"),
            Turn(role="assistant", content="Hi there!"),
        ]
        msgs = s._build_messages("Follow up question")
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"
        assert msgs[-1]["content"] == "Follow up question"

    def test_save_and_load(self, tmp_path):
        s = self.make_session()
        s.history = [
            Turn(role="user", content="test question"),
            Turn(role="assistant", content="test answer"),
        ]
        path = str(tmp_path / "session.json")
        s.save(path)
        assert Path(path).exists()

        # Load it back
        client = DeepSeekClient(api_key="test-key")
        loaded = Session.load(path, client=client)
        assert len(loaded.history) == 2
        assert loaded.history[0].content == "test question"
        assert loaded.history[1].content == "test answer"

    def test_thinking_toggle(self):
        s = self.make_session()
        assert s.show_thinking is False
        s.show_thinking = True
        assert s.show_thinking is True


# ──────────────────────────────────────────────────────────────────────
# Integration: context string round-trip
# ──────────────────────────────────────────────────────────────────────

class TestContextRoundTrip:
    def test_load_to_context_string_round_trip(self, tmp_path):
        """Verify that a loaded codebase becomes a valid context string."""
        (tmp_path / "main.py").write_text("def hello(): return 'world'")
        (tmp_path / "utils.py").write_text("import os\nimport sys")
        (tmp_path / "config.yaml").write_text("key: value\ndebug: true")

        result = load_local(str(tmp_path))
        ctx_str = result.to_context_string()

        # All files should appear in context
        assert "main.py" in ctx_str
        assert "utils.py" in ctx_str
        assert "config.yaml" in ctx_str
        assert "def hello(): return 'world'" in ctx_str
        assert "import os" in ctx_str
        assert "key: value" in ctx_str

        # Structure should be valid
        assert ctx_str.startswith("# Source:")
        assert "## File:" in ctx_str
        assert "```" in ctx_str
