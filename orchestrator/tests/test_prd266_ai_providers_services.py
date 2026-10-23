"""PRD-266 US-017b: the other AI services a workspace's words or voice reach.

``services.ai_providers.workspace_ai_providers`` adds, to the chat providers of
US-017a: (c) the embeddings provider, (d) the RAG reranker (Cohere), (e) on a
local install with session mode on, the CLI each session agent runs on, and
(f) Retell when the workspace can use Auto Live voice: the platform switch,
Retell armed, the workspace's own switch and, on saas only, a plan whose
families allow voice (or a known tier that declares none). The monthly cap is
not one of them.

System settings are a table in each test (both readers of the canonical
accessor are pointed at it); the workspace and agent rows are real Postgres
(CI), inside the test's rolled-back transaction. Sentinels stand where the keys
would be, and no sentinel reaches the result or any log record.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import core.llm.manager as llm_manager_mod  # noqa: E402
import core.llm.rerank_manager as rerank_manager_mod  # noqa: E402
from config import config  # noqa: E402
from core.llm import providers as registry  # noqa: E402
from core.models.core import Agent  # noqa: E402
from core.models.credentials import Credential  # noqa: E402
from core.models.voice_calls import VoiceCall  # noqa: E402
from modules.voice import live_settings, voice_meter  # noqa: E402
from services.ai_providers import CREDENTIAL_NAME_FORMS, workspace_ai_providers  # noqa: E402

# Stored where a key would be. Low-entropy on purpose: no secret scanner reads it as a key.
RETELL_KEY_SENTINEL = "retell-key-sentinel-never-read"
RETELL_SECRET_SENTINEL = "retell-secret-sentinel-never-read"
RERANK_KEY_SENTINEL = "rerank-key-sentinel-never-read"
ENV_SENTINEL = "env-key-sentinel-never-read"
SENTINELS = (RETELL_KEY_SENTINEL, RETELL_SECRET_SENTINEL, RERANK_KEY_SENTINEL, ENV_SENTINEL)

RETELL = {"id": "retell", "name": "Retell AI"}
CLAUDE_CODE = {"id": "claude_code", "name": "Claude Code"}


# ---------------------------------------------------------------------------
# Fixtures: every source off, a settings table, a reranker with no key
# ---------------------------------------------------------------------------


@pytest.fixture
def floor(db_session, seed_workspace, monkeypatch):
    """A saas workspace with every source off: no keys, no settings, no session mode, no voice."""
    for spec in registry.REGISTRY.values():
        if spec.env_key:
            monkeypatch.setattr(config, spec.env_key, "", raising=False)
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", "")
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", None)
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", False)
    # The platform lane reads the credential store unscoped: a row another test
    # committed is removed inside this test's transaction (its rollback restores it).
    every_name = [form.format(slug=slug) for slug in registry.all_slugs() for form in CREDENTIAL_NAME_FORMS]
    db_session.query(Credential).filter(Credential.name.in_(every_name)).delete(synchronize_session=False)

    settings = {}

    def _setting(category, key, default=None):
        return settings.get((category, key)) or default

    monkeypatch.setattr(llm_manager_mod, "get_system_setting", _setting)
    monkeypatch.setattr(live_settings, "get_system_setting", _setting)
    reranker = rerank_manager_mod.RerankManager()
    reranker._loaded = True  # its settings read found no key
    monkeypatch.setattr(rerank_manager_mod, "get_rerank_manager", lambda: reranker)
    return SimpleNamespace(db=db_session, ws=seed_workspace(), settings=settings, reranker=reranker)


def _ids(floor) -> list:
    return [entry["id"] for entry in workspace_ai_providers(floor.db, floor.ws)]


def _workspace(floor, *, plan=None, voice_live=None, monthly_cap_minutes=None) -> None:
    voice = {"enabled": voice_live} if voice_live is not None else {}
    if monthly_cap_minutes is not None:
        voice = {**voice, "monthly_cap_minutes": monthly_cap_minutes}
    floor.db.execute(
        text("UPDATE workspaces SET settings = CAST(:s AS jsonb), plan = :plan WHERE id = CAST(:id AS uuid)"),
        {"s": json.dumps({"voice_live": voice} if voice else {}), "plan": plan, "id": floor.ws},
    )
    floor.db.expire_all()


def _platform_voice(floor, *, switch=True, api_key=RETELL_KEY_SENTINEL,
                    webhook_secret=RETELL_SECRET_SENTINEL, agent_id="retell-agent-sentinel") -> None:
    floor.settings.update({
        (live_settings.VOICE_SETTINGS_CATEGORY, live_settings.KEY_LIVE_ENABLED): "true" if switch else "false",
        (live_settings.VOICE_SETTINGS_CATEGORY, live_settings.KEY_RETELL_API_KEY): api_key,
        (live_settings.VOICE_SETTINGS_CATEGORY, live_settings.KEY_RETELL_WEBHOOK_SECRET): webhook_secret,
        (live_settings.VOICE_SETTINGS_CATEGORY, live_settings.KEY_RETELL_AGENT_ID): agent_id,
    })


def _agent(db, workspace_id: str, configuration) -> None:
    db.add(Agent(name=f"agent-{uuid.uuid4().hex[:8]}", agent_type="custom",
                 workspace_id=uuid.UUID(workspace_id), configuration=configuration))
    db.flush()


def test_the_floor_lists_nothing(floor):
    assert _ids(floor) == []


# ---------------------------------------------------------------------------
# (c) the embeddings provider
# ---------------------------------------------------------------------------


def test_the_embeddings_provider_appears_once_when_it_is_also_a_chat_route(floor, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", ENV_SENTINEL)
    floor.settings[("embeddings", "provider")] = "openrouter"

    assert workspace_ai_providers(floor.db, floor.ws) == [{"id": "openrouter", "name": "OpenRouter"}]


@pytest.mark.parametrize("provider", ["disabled", "huggingface_local", ""])
def test_an_embeddings_provider_that_sends_nothing_out_adds_none(floor, monkeypatch, provider):
    floor.settings[("embeddings", "provider")] = provider
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", provider or None)

    assert _ids(floor) == []


def test_the_system_setting_wins_over_the_config_and_the_config_is_the_fallback(floor, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "openai")
    assert workspace_ai_providers(floor.db, floor.ws) == [{"id": "openai", "name": "OpenAI"}]

    floor.settings[("embeddings", "provider")] = "disabled"
    assert _ids(floor) == []


def test_a_key_only_embeddings_provider_is_named_by_its_registry_label(floor):
    floor.settings[("embeddings", "provider")] = "cohere"

    assert workspace_ai_providers(floor.db, floor.ws) == [{"id": "cohere", "name": "Cohere"}]


def test_the_hugging_face_inference_api_is_named_as_the_registrys_hugging_face(floor):
    floor.settings[("embeddings", "provider")] = "huggingface_api"

    assert workspace_ai_providers(floor.db, floor.ws) == [{"id": "huggingface", "name": "HuggingFace"}]


# ---------------------------------------------------------------------------
# (d) the reranker
# ---------------------------------------------------------------------------


def test_the_reranker_counts_when_reranking_is_on_and_it_has_a_key(floor):
    floor.reranker._api_key = RERANK_KEY_SENTINEL

    assert workspace_ai_providers(floor.db, floor.ws) == [{"id": "cohere", "name": "Cohere"}]


def test_the_reranker_does_not_count_without_a_key(floor):
    assert floor.reranker.is_available() is False

    assert _ids(floor) == []


def test_the_reranker_does_not_count_when_reranking_is_off(floor):
    floor.reranker._api_key = RERANK_KEY_SENTINEL
    floor.settings[("rag", "rerank_enabled")] = "false"

    assert _ids(floor) == []


# ---------------------------------------------------------------------------
# (e) session-mode runtimes
# ---------------------------------------------------------------------------


def test_a_session_agent_on_a_local_install_adds_claude_code_once(floor, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True)
    _agent(floor.db, floor.ws, {"runtime": "cli"})
    _agent(floor.db, floor.ws, {"runtime": "CLI", "provider": "claude"})
    _agent(floor.db, floor.ws, {"runtime": "api", "provider": "codex"})
    _agent(floor.db, floor.ws, None)

    assert workspace_ai_providers(floor.db, floor.ws) == [CLAUDE_CODE]


def test_each_session_cli_is_named_by_its_preset_label(floor, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True)
    _agent(floor.db, floor.ws, {"runtime": "cli", "provider": "codex"})
    _agent(floor.db, floor.ws, {"runtime": "cli", "provider": "copilot"})

    assert workspace_ai_providers(floor.db, floor.ws) == [
        {"id": "codex", "name": "Codex"},
        {"id": "copilot_cli", "name": "GitHub Copilot"},
    ]


def test_a_session_agent_on_a_cli_with_no_preset_adds_nothing(floor, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True)
    _agent(floor.db, floor.ws, {"runtime": "cli", "provider": "Not A Real CLI <b>"})

    assert _ids(floor) == []


def test_a_session_agent_adds_nothing_on_saas(floor, monkeypatch):
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True)
    _agent(floor.db, floor.ws, {"runtime": "cli"})

    assert _ids(floor) == []


def test_a_session_agent_adds_nothing_while_session_mode_is_off(floor, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    _agent(floor.db, floor.ws, {"runtime": "cli"})

    assert _ids(floor) == []


def test_another_workspaces_session_agent_does_not_count(floor, seed_workspace, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True)
    _agent(floor.db, seed_workspace(), {"runtime": "cli"})

    assert _ids(floor) == []


# ---------------------------------------------------------------------------
# (f) Retell, when the workspace can use Auto Live voice
# ---------------------------------------------------------------------------


def test_retell_is_listed_for_a_business_workspace_with_voice_on_and_retell_armed(floor):
    _platform_voice(floor)
    _workspace(floor, plan="business", voice_live=True)

    assert workspace_ai_providers(floor.db, floor.ws) == [RETELL]


def test_retell_is_listed_on_a_local_install_whose_workspace_has_no_plan(floor, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    _platform_voice(floor)
    _workspace(floor, plan=None, voice_live=True)

    assert _ids(floor) == ["retell"]


def test_retell_is_not_listed_when_the_platform_switch_is_off(floor):
    _platform_voice(floor, switch=False)
    _workspace(floor, plan="business", voice_live=True)

    assert _ids(floor) == []


def test_retell_is_not_listed_when_retell_is_not_armed(floor):
    _platform_voice(floor, agent_id="")
    _workspace(floor, plan="business", voice_live=True)

    assert _ids(floor) == []


def test_retell_is_not_listed_when_the_workspaces_own_switch_is_off(floor):
    _platform_voice(floor)
    _workspace(floor, plan="business", voice_live=False)

    assert _ids(floor) == []


@pytest.mark.parametrize("plan", ["basic", "pro", None])
def test_retell_is_not_listed_on_saas_for_a_plan_without_voice(floor, plan):
    _platform_voice(floor)
    _workspace(floor, plan=plan, voice_live=True)

    assert _ids(floor) == []


def test_retell_is_listed_for_an_enterprise_workspace_whose_tier_declares_no_families(floor):
    _platform_voice(floor)
    _workspace(floor, plan="enterprise", voice_live=True)

    assert _ids(floor) == ["retell"]


def test_retell_is_listed_for_a_workspace_over_its_monthly_cap(floor):
    _platform_voice(floor)
    _workspace(floor, plan="business", voice_live=True, monthly_cap_minutes=1)
    workspace_uuid = uuid.UUID(floor.ws)
    floor.db.add(VoiceCall(call_id=f"call-{uuid.uuid4().hex}", workspace_id=workspace_uuid, status="ended",
                           ended_at=datetime.now(timezone.utc).replace(tzinfo=None), duration_seconds=600))
    floor.db.flush()
    allowed, _reason = voice_meter.cap_allows_mint(voice_meter.monthly_meter(floor.db, workspace_uuid), 1)
    assert allowed is False  # the mint would refuse this workspace a call

    assert _ids(floor) == ["retell"]


# ---------------------------------------------------------------------------
# What never leaks
# ---------------------------------------------------------------------------


def test_no_retell_key_or_any_other_key_value_appears_in_the_result_or_in_any_log_record(
    floor, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", ENV_SENTINEL)
    floor.settings[("embeddings", "provider")] = "openai"
    floor.reranker._api_key = RERANK_KEY_SENTINEL
    _agent(floor.db, floor.ws, {"runtime": "cli"})
    _platform_voice(floor)
    _workspace(floor, voice_live=True)

    result = workspace_ai_providers(floor.db, floor.ws)

    assert [entry["id"] for entry in result] == ["anthropic", "claude_code", "cohere", "openai", "retell"]
    for sentinel in SENTINELS:
        assert sentinel not in json.dumps(result)
        assert all(sentinel not in record.getMessage() for record in caplog.records)
