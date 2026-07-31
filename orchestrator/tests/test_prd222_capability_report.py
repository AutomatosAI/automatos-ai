"""PRD-222 W1 (US-007) — the onboarding capability report.

Pure tests: no Postgres. Config presence is monkeypatched; the workspace LLM-key
check runs against a fake session. The report is booleans-only by contract, so a
leak test asserts no secret string ever rides through.
"""
from __future__ import annotations

from pathlib import Path

from config import config
from services.capability_report import onboarding_capabilities

REPO = Path(__file__).resolve().parents[1]

FAKE_SECRET = "fc-fake-DO-NOT-LEAK-0000000000"


# --------------------------------------------------------------------------- #
# Fakes for the workspace LLM-key lookup
# --------------------------------------------------------------------------- #


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result


class _FakeDB:
    def __init__(self, result):
        self._result = result

    def query(self, _model):
        return _FakeQuery(self._result)


# --------------------------------------------------------------------------- #
# AC1 — four booleans, secrets never surfaced
# --------------------------------------------------------------------------- #


def test_returns_exactly_the_four_capability_booleans():
    caps = onboarding_capabilities()
    assert set(caps) == {
        "llm_key_valid",
        "firecrawl_configured",
        "composio_configured",
        "redis_configured",
    }
    # Every value is a plain bool — never a string/secret.
    assert all(isinstance(v, bool) for v in caps.values())


def test_no_secret_value_ever_appears_in_the_report(monkeypatch):
    monkeypatch.setattr(config, "FIRECRAWL_API_KEY", FAKE_SECRET)
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", FAKE_SECRET)
    caps = onboarding_capabilities()
    assert caps["firecrawl_configured"] is True
    assert caps["composio_configured"] is True
    # The presence booleans are True but the KEY VALUE is nowhere in the payload.
    assert FAKE_SECRET not in caps.values()
    assert all(isinstance(v, bool) for v in caps.values())


# --------------------------------------------------------------------------- #
# AC3 — unset FIRECRAWL_API_KEY → False, no raise
# --------------------------------------------------------------------------- #


def test_unset_firecrawl_is_false_without_raising(monkeypatch):
    monkeypatch.setattr(config, "FIRECRAWL_API_KEY", None)
    caps = onboarding_capabilities()
    assert caps["firecrawl_configured"] is False


def test_configured_firecrawl_is_true(monkeypatch):
    monkeypatch.setattr(config, "FIRECRAWL_API_KEY", FAKE_SECRET)
    assert onboarding_capabilities()["firecrawl_configured"] is True


def test_composio_presence_maps_to_bool(monkeypatch):
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", None)
    assert onboarding_capabilities()["composio_configured"] is False
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", FAKE_SECRET)
    assert onboarding_capabilities()["composio_configured"] is True


def test_redis_presence_maps_to_bool(monkeypatch):
    # REDIS_URL is a config property — patch it at the class so bool() maps it.
    monkeypatch.setattr(type(config), "REDIS_URL", None)
    assert onboarding_capabilities()["redis_configured"] is False
    monkeypatch.setattr(type(config), "REDIS_URL", "redis://fake-host:6379/0")
    assert onboarding_capabilities()["redis_configured"] is True


# --------------------------------------------------------------------------- #
# AC1 — llm_key_valid is workspace-scoped from US-006 truth (is_active)
# --------------------------------------------------------------------------- #


def test_llm_key_valid_true_when_active_key_present():
    db = _FakeDB(result=object())  # an active UserApiKey row exists
    assert onboarding_capabilities(db, workspace_id="ws-1")["llm_key_valid"] is True


def test_llm_key_valid_false_when_no_active_key():
    db = _FakeDB(result=None)
    assert onboarding_capabilities(db, workspace_id="ws-1")["llm_key_valid"] is False


def test_llm_key_valid_false_without_db_or_workspace():
    assert onboarding_capabilities(None, workspace_id="ws-1")["llm_key_valid"] is False
    assert onboarding_capabilities(_FakeDB(object()), workspace_id=None)["llm_key_valid"] is False


# --------------------------------------------------------------------------- #
# AC2 — surfaced on the existing admin system-health endpoint (grep-proof)
# --------------------------------------------------------------------------- #


def test_system_health_response_carries_capabilities_field():
    from core.models.core import SystemHealthResponse

    assert "capabilities" in SystemHealthResponse.model_fields


def test_health_endpoint_wires_the_capability_report():
    src = (REPO / "api" / "system.py").read_text()
    assert "from services.capability_report import onboarding_capabilities" in src
    assert "capabilities=capabilities" in src  # passed into SystemHealthResponse
