"""Shared fixtures for the CLI host suite — stdlib + pytest only."""
from __future__ import annotations

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


@pytest.fixture(scope="session", autouse=True)
def _fake_claude_executable():
    FAKE_CLAUDE.chmod(FAKE_CLAUDE.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


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
def env_clean(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_ENTRYPOINT", "CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FAKE_CLAUDE_SCENARIO", raising=False)
