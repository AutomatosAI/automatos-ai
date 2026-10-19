"""The agent-opt worker's own suite (PRD-262 US-006).

It runs INSIDE the worker image, built with `--target test` by the
`agent-opt-worker` job in .github/workflows/test.yml: `python -m pytest` from /app.
main.py reads its token at import, so the environment is set before any test
imports it. No test reaches FutureAGI or OpenAI: the scoring and the
optimisation job are replaced where a request gets past validation.
"""
from __future__ import annotations

import os
import sys

import pytest

from helpers import APP_DIR, TOKEN

os.environ["AGENT_OPT_WORKER_TOKEN"] = TOKEN
os.environ["LOG_RELAY_ENABLED"] = "false"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture(scope="session")
def worker_main():
    import main

    return main


@pytest.fixture()
def client(worker_main):
    from fastapi.testclient import TestClient

    with TestClient(worker_main.app) as test_client:
        yield test_client


@pytest.fixture()
def authed():
    return {"X-Internal-Token": TOKEN}


@pytest.fixture()
def fake_scoring(worker_main, monkeypatch):
    """Every template scores 1.0 without calling FutureAGI; records the templates run."""
    calls: list[str] = []

    def score(template, inputs, model="turing_flash"):
        calls.append(template)
        return {"score": 1.0, "passed": True, "reason": "stub"}

    monkeypatch.setattr(worker_main, "_run_single_template", score)
    return calls
