"""Shared by the agent-opt worker's tests (conftest.py sets the environment from these)."""
from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
TOKEN = "ci-agent-opt-worker-token"
