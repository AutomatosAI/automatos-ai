"""PRD-262 US-006: the worker refuses to start without its token.

uvicorn imports main:app at start-up, so an import that raises is a start-up
that fails. Each case runs in a fresh interpreter, as the container's CMD would.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from helpers import APP_DIR

TOKEN_ENV = "AGENT_OPT_WORKER_TOKEN"


def _import_main(token: str | None) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key != TOKEN_ENV}
    env["LOG_RELAY_ENABLED"] = "false"
    if token is not None:
        env[TOKEN_ENV] = token
    return subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=APP_DIR, env=env, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("token", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_start_up_fails_without_a_token(token):
    result = _import_main(token)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"{TOKEN_ENV} is not set" in result.stderr


def test_start_up_succeeds_with_a_token():
    result = _import_main("a-real-token")
    assert result.returncode == 0, result.stderr


def test_the_dependency_refuses_an_empty_expected_token():
    from worker_auth import require_internal_token

    with pytest.raises(ValueError):
        require_internal_token("")
