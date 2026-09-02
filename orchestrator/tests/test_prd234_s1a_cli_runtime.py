"""PRD-234 S1a — runtime kind rules, the boot gate, the claim predicate, the factory guard.

Pure/DB-free (the blessed fake-POSTGRES preamble): every assertion here is a
behaviour the session-mode backend contract relies on:

* ``core.cli_runtime`` — ``api`` is the default for every agent that exists
  today; ``cli`` needs the flag, a known provider and a CLI-shaped model.
* ``config.validate_auth_edition`` — ``CLI_RUNTIME_ENABLED`` outside the local
  edition aborts boot (the SaaS path stays byte-identical).
* ``board_dispatcher.claim_tasks`` — the dispatcher's claim excludes ``cli``
  tickets and a host's claim takes only them, scoped to its workspace.
* ``AgentFactory.execute_with_prompt`` — the choke point every execution lane
  passes through refuses a ``cli`` agent with a typed, actionable error result
  before any activation or model client (review §B3: eight lanes, one guard).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.cli_runtime import (  # noqa: E402
    RUNTIME_API,
    RUNTIME_CLI,
    RuntimeMismatchError,
    is_valid_cli_model,
    runtime_kind_of,
    validate_runtime_configuration,
)


# ── core.cli_runtime ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "configuration, expected",
    [
        (None, RUNTIME_API),
        ({}, RUNTIME_API),
        ({"llm_config": {"temperature": 0.2}}, RUNTIME_API),
        ({"runtime": "api"}, RUNTIME_API),
        ({"runtime": "cli"}, RUNTIME_CLI),
        ({"runtime": " CLI "}, RUNTIME_CLI),
        ({"runtime": "session"}, RUNTIME_API),  # unknown never becomes cli
        ({"runtime": 3}, RUNTIME_API),
        ("not-a-mapping", RUNTIME_API),
    ],
)
def test_runtime_kind_defaults_to_api_for_every_existing_agent(configuration, expected):
    assert runtime_kind_of(configuration) == expected


def test_api_configuration_is_never_rejected():
    assert validate_runtime_configuration({"runtime": "api"}, cli_enabled=False) == []
    assert validate_runtime_configuration({"anything": 1}, cli_enabled=False) == []
    assert validate_runtime_configuration(None, cli_enabled=False) == []


def test_cli_requires_the_flag_a_provider_and_a_cli_model():
    errs = validate_runtime_configuration({"runtime": "cli"}, cli_enabled=False)
    assert any("CLI_RUNTIME_ENABLED" in e for e in errs)
    assert any("provider" in e for e in errs)

    ok = {"runtime": "cli", "provider": "claude", "model": "opus"}
    assert validate_runtime_configuration(ok, cli_enabled=True) == []

    bad_model = {"runtime": "cli", "provider": "claude", "model": "openrouter/deepseek-chat"}
    errs = validate_runtime_configuration(bad_model, cli_enabled=True)
    assert len(errs) == 1 and "model" in errs[0]

    bad_runtime = validate_runtime_configuration({"runtime": "pty"}, cli_enabled=True)
    assert len(bad_runtime) == 1 and "runtime must be one of" in bad_runtime[0]


@pytest.mark.parametrize(
    "provider, model, valid",
    [
        ("claude", None, True),
        ("claude", "", True),
        ("claude", "sonnet", True),
        ("claude", "claude-opus-4-8", True),
        ("claude", "claude-sonnet-4-6[1m]", True),
        ("claude", "gpt-5", False),
        ("claude", "anthropic/claude-opus-4", False),
        ("codex", "gpt-5-codex", True),
        ("codex", "openai/gpt-5", False),
        ("gemini", "gemini-pro", False),  # not a v1 provider
    ],
)
def test_model_shapes_per_provider(provider, model, valid):
    assert is_valid_cli_model(provider, model) is valid


def test_runtime_mismatch_error_has_the_factory_error_shape():
    err = RuntimeMismatchError(42, RUNTIME_CLI)
    result = err.as_result()
    assert result["status"] == "error"
    assert result["error_code"] == "runtime_mismatch"
    assert result["agent_id"] == 42 and result["runtime"] == "cli"
    assert "ticket" in result["error"]  # the actionable half of the message


# ── the boot gate ────────────────────────────────────────────────────────────

def test_boot_gate_refuses_cli_runtime_outside_the_local_edition(monkeypatch):
    from config import config as cfg

    monkeypatch.setattr(cfg, "AUTH_EDITION", "saas")
    monkeypatch.setattr(cfg, "CLI_RUNTIME_ENABLED", True)
    monkeypatch.setattr(cfg, "CLERK_JWKS_URL", "https://x/.well-known/jwks.json", raising=False)
    monkeypatch.setattr(cfg, "CLERK_SECRET_KEY", "sk_test", raising=False)
    with pytest.raises(RuntimeError) as exc:
        cfg.validate_auth_edition()
    assert "CLI_RUNTIME_ENABLED" in str(exc.value) and "AUTH_EDITION=local" in str(exc.value)


def test_boot_gate_allows_cli_runtime_in_the_local_edition(monkeypatch):
    from config import config as cfg

    monkeypatch.setattr(cfg, "AUTH_EDITION", "local")
    monkeypatch.setattr(cfg, "CLI_RUNTIME_ENABLED", True)
    monkeypatch.setattr(cfg, "DEFAULT_WORKSPACE_ID", str(uuid4()), raising=False)
    cfg.validate_auth_edition()  # must not raise


def test_boot_gate_default_is_off_and_saas_is_unchanged(monkeypatch):
    from config import config as cfg

    assert cfg.CLI_RUNTIME_ENABLED is False or os.getenv("CLI_RUNTIME_ENABLED")
    monkeypatch.setattr(cfg, "AUTH_EDITION", "saas")
    monkeypatch.setattr(cfg, "CLI_RUNTIME_ENABLED", False)
    monkeypatch.setattr(cfg, "CLERK_JWKS_URL", "https://x/.well-known/jwks.json", raising=False)
    monkeypatch.setattr(cfg, "CLERK_SECRET_KEY", "sk_test", raising=False)
    cfg.validate_auth_edition()  # the saas path is byte-identical with the flag off


# ── the claim predicate (SQL text, DB-free) ──────────────────────────────────

class _CapturingDb:
    """Records every statement ``claim_tasks`` issues; returns no rows."""

    def __init__(self):
        self.statements = []

    def execute(self, stmt, params=None):
        self.statements.append((str(getattr(stmt, "text", stmt)), dict(params or {})))
        return SimpleNamespace(fetchall=lambda: [])

    def commit(self):
        pass


def _claim_sql(runtime, workspace_id=None, slots=None):
    from services import board_dispatcher as bd

    db = _CapturingDb()
    out = bd.claim_tasks(
        db, worker_id="w", limit=5, lease_seconds=60,
        max_slots_per_agent=slots, runtime=runtime, workspace_id=workspace_id,
    )
    assert out == []
    return db.statements[0]


def test_dispatcher_claim_excludes_cli_agents_by_default():
    from services import board_dispatcher as bd

    sql, params = _claim_sql(bd.RUNTIME_API)
    assert "<> 'cli'" in sql and "= 'cli'" not in sql
    assert "COALESCE" in sql and "a.configuration->>'runtime'" in sql
    assert "ws" not in params


def test_host_claim_takes_only_cli_agents_in_its_workspace():
    ws = uuid4()
    sql, params = _claim_sql("cli", workspace_id=ws)
    assert "= 'cli'" in sql and "<> 'cli'" not in sql
    assert "CAST(:ws AS uuid)" in sql and params["ws"] == str(ws)


def test_slot_aware_claim_carries_the_same_predicates():
    ws = uuid4()
    sql, params = _claim_sql("cli", workspace_id=ws, slots=2)
    assert "= 'cli'" in sql and "t.assigned_agent_id" in sql
    assert "t.workspace_id = CAST(:ws AS uuid)" in sql and params["ws"] == str(ws)
    api_sql, api_params = _claim_sql("api", slots=2)
    assert "<> 'cli'" in api_sql and "ws" not in api_params


def test_claim_tasks_default_arguments_are_backward_compatible():
    """Callers that pass nothing new (the dispatch loop, PRD-161 tests) get the
    API predicate and no workspace filter — behaviour unchanged for API agents."""
    from services import board_dispatcher as bd

    db = _CapturingDb()
    bd.claim_tasks(db, worker_id="w", limit=1, lease_seconds=1)
    sql, params = db.statements[0]
    assert "<> 'cli'" in sql and "ws" not in params


# ── the factory guard (the single choke point) ───────────────────────────────

class _FakeConfigQuery:
    def __init__(self, configuration):
        self._configuration = configuration

    def filter(self, *a, **k):
        return self

    def first(self):
        return (self._configuration,)


class _FakeSession:
    def __init__(self, configuration):
        self.configuration = configuration
        self.queries = 0

    def query(self, *a, **k):
        self.queries += 1
        return _FakeConfigQuery(self.configuration)


def _bare_factory(configuration):
    """An AgentFactory with no __init__ side effects: only what the guard reads."""
    import modules.agents.factory.agent_factory as af

    factory = object.__new__(af.AgentFactory)
    factory.db_session = _FakeSession(configuration)
    factory.logger = logging.getLogger("prd234-test")
    factory.active_agents = {}

    async def _never_activate(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("activate_agent must never run for a cli agent")

    factory.activate_agent = _never_activate
    return factory


@pytest.mark.parametrize("agent_ref", [7, SimpleNamespace(agent_id=7)])
def test_factory_refuses_a_cli_agent_before_activation(agent_ref):
    import asyncio

    factory = _bare_factory({"runtime": "cli", "provider": "claude"})
    result = asyncio.run(factory.execute_with_prompt(agent=agent_ref, prompt="do it"))
    assert result["status"] == "error"
    assert result["error_code"] == "runtime_mismatch"
    assert result["agent_id"] == 7
    assert factory.db_session.queries == 1  # one configuration read, nothing else


def test_factory_guard_is_silent_for_api_agents():
    factory = _bare_factory({"runtime": "api"})
    assert factory._runtime_mismatch(7) is None
    factory = _bare_factory(None)
    assert factory._runtime_mismatch(SimpleNamespace(agent_id=9)) is None


def test_factory_guard_skips_when_it_cannot_read_a_configuration():
    """No session / no agent id → not the guard's call; the existing paths decide."""
    import modules.agents.factory.agent_factory as af

    factory = object.__new__(af.AgentFactory)
    factory.db_session = None
    factory.logger = logging.getLogger("prd234-test")
    assert factory._runtime_mismatch(7) is None
    factory.db_session = _FakeSession({"runtime": "cli"})
    assert factory._runtime_mismatch(SimpleNamespace()) is None  # no agent_id attribute
