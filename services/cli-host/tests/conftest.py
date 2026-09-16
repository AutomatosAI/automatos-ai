"""Shared fixtures for the CLI host suite — stdlib + pytest only."""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAKE_CLAUDE = Path(__file__).with_name("fake_claude.py")
FAKE_CODEX = Path(__file__).with_name("fake_codex.py")


@pytest.fixture(scope="session", autouse=True)
def _fake_claude_executable():
    for fake in (FAKE_CLAUDE, FAKE_CODEX):
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def short_tmp():
    """A short temp dir: AF_UNIX socket paths are capped (~104 bytes on macOS)."""
    d = tempfile.mkdtemp(prefix="acli-", dir="/tmp")
    yield Path(d)


@pytest.fixture
def fake_home(short_tmp, monkeypatch):
    """A HOME with a Claude Code that has completed onboarding; nothing else."""
    home = short_tmp / "home"
    home.mkdir()
    (home / ".claude.json").write_text('{"hasCompletedOnboarding": true, "projects": {}}')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return home


@pytest.fixture
def fake_codex_home(short_tmp, monkeypatch):
    """A HOME whose ~/.codex is logged in with a ChatGPT plan (design §9.1) and
    carries the operator's own config; nothing else."""
    home = short_tmp / "home"
    home.mkdir(exist_ok=True)
    codex = home / ".codex"
    codex.mkdir(exist_ok=True)
    (codex / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "not-a-secret-fixture"}}))
    (codex / "config.toml").write_text('model = "gpt-fake"\nmodel_reasoning_effort = "medium"\n\n[projects."/somewhere/else"]\ntrust_level = "trusted"\n')
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def env_clean(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_ENTRYPOINT", "CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FAKE_CLAUDE_SCENARIO", raising=False)
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_HOME", "FAKE_CODEX_SCENARIO"):
        monkeypatch.delenv(key, raising=False)
