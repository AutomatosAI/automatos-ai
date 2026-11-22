"""PRD-203 C·S8 — the canvas worker SDK subprocess carries a model credential.

Without a committed credential the streaming SDK client connects and idles
forever. These prove start_session threads the worker-config credential into
ClaudeAgentOptions.env, and fails fast + greppable (never a silent idle) when
none is configured on the real SDK path.

Container-free: the SDK client is mocked (injected factory); the missing-auth
case uses the DEFAULT factory but the guard fires BEFORE it is invoked, so no
claude_agent_sdk import happens.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

_WORKER_DIR = Path(__file__).resolve().parents[2] / "services" / "workspace-worker"
sys.path.insert(0, str(_WORKER_DIR))
try:
    import canvas_session_service as css
finally:
    sys.path.remove(str(_WORKER_DIR))


class _FakeSDKClient:
    def __init__(self, option_kwargs: Dict[str, Any], sid: str = "sdk-cs8") -> None:
        self.option_kwargs = option_kwargs
        self.sid = sid
        self._closed = asyncio.Event()

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        self._closed.set()

    async def receive_messages(self):
        yield SimpleNamespace(subtype="init", data={"session_id": self.sid})
        await self._closed.wait()


class _FactorySpy:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def __call__(self, option_kwargs: Dict[str, Any]) -> _FakeSDKClient:
        self.calls.append(option_kwargs)
        return _FakeSDKClient(option_kwargs)


def test_canvas_options_carry_model_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-cs8")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    async def scenario():
        factory = _FactorySpy()
        mgr = css.CanvasSessionManager(
            str(tmp_path), sdk_client_factory=factory, init_timeout=2.0
        )
        result = await mgr.start_session("ws-auth")
        assert result["success"] is True

        env = factory.calls[0]["env"]
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-cs8"
        # The credential is MERGED into the SDK env, not replacing CLAUDE_CONFIG_DIR.
        assert "CLAUDE_CONFIG_DIR" in env

        await mgr.stop_session("ws-auth")

    asyncio.run(scenario())


def test_canvas_prefers_oauth_token_when_both_set(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")

    async def scenario():
        factory = _FactorySpy()
        mgr = css.CanvasSessionManager(
            str(tmp_path), sdk_client_factory=factory, init_timeout=2.0
        )
        await mgr.start_session("ws-both")
        env = factory.calls[0]["env"]
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok-123"
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-x"
        await mgr.stop_session("ws-both")

    asyncio.run(scenario())


def test_missing_model_auth_fails_clearly_not_silently(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    async def scenario():
        # DEFAULT (real SDK) factory — the guard fires BEFORE it is called, so no
        # claude_agent_sdk import is attempted.
        mgr = css.CanvasSessionManager(str(tmp_path), init_timeout=2.0)
        result = await mgr.start_session("ws-noauth")

        assert result["success"] is False
        assert "model credential" in result["error"].lower()

        # State persisted FAILED (a clear terminal signal, not a hung "running").
        on_disk = json.loads(
            (tmp_path / "ws-noauth" / ".canvas" / "session.json").read_text()
        )
        assert on_disk["status"] == css.STATUS_FAILED

    asyncio.run(scenario())


# The coding agent's environment: the worker's own variables are blanked for the agent, the
# model credential and what a CLI needs to run are kept.
_WORKER_SECRETS = {
    "DATABASE_URL": "postgresql://worker:pw@db:5432/automatos",
    "REDIS_URL": "redis://:pw@redis:6379/0",
    "WORKER_INTERNAL_TOKEN": "worker-internal-token",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE": "/var/run/secrets/pod-identity/token",
    "OTEL_EXPORTER_OTLP_HEADERS": "authorization=Bearer otel",
}


def test_the_agent_never_sees_the_workers_own_variables(tmp_path, monkeypatch):
    for name, value in _WORKER_SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-agent")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    async def scenario():
        factory = _FactorySpy()
        mgr = css.CanvasSessionManager(str(tmp_path), sdk_client_factory=factory, init_timeout=2.0)
        await mgr.start_session("ws-env")
        env = factory.calls[0]["env"]
        # Each is laid over the inherited environment as empty, so the CLI and its commands get "".
        for name in _WORKER_SECRETS:
            assert env[name] == "", name
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-agent"
        assert env["CLAUDE_CONFIG_DIR"].startswith(str(tmp_path))
        await mgr.stop_session("ws-env")

    asyncio.run(scenario())


def test_what_a_cli_needs_to_run_is_left_as_the_worker_has_it():
    inherited = {
        "PATH": "/usr/bin", "HOME": "/home/worker", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "TZ": "UTC", "HTTPS_PROXY": "http://proxy:3128", "NODE_EXTRA_CA_CERTS": "/ca.pem",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_OAUTH_TOKEN": "tok", "ANTHROPIC_API_KEY": "sk", "DATABASE_URL": "postgresql://x",
    }
    blanked = css.agent_environment(inherited)
    # Kept names are left out of the overlay, so the agent sees the worker's own value.
    for kept in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "HTTPS_PROXY", "NODE_EXTRA_CA_CERTS",
                 "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DISABLE_TELEMETRY"):
        assert kept not in blanked, kept
    # The credentials are blanked here and added back only by model_auth_env, when configured.
    assert blanked == {"CLAUDE_CODE_OAUTH_TOKEN": "", "ANTHROPIC_API_KEY": "", "DATABASE_URL": ""}


def test_the_blanked_environment_adds_no_value_of_its_own():
    assert css.agent_environment({}) == {}
    assert all(value == "" for value in css.agent_environment({"A": "1", "B": "2"}).values())
