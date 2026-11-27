"""#1678: with only an OpenAI key saved, Settings → Orchestrator said "Auto's model has no key to
run with" and never which key.

A fresh install seeds Auto's model as ``anthropic`` / ``claude-haiku-5-5``
(``core/seeds/llm_tier_route.py``). A key saved in Settings → Model keys counts for it only when it
is that provider's (``core.llm.workspace_keys.get_platform_workspace_key``), so an OpenAI key leaves
it without one. The readiness line now names the provider and model that need the key, and says
both ways out. Real rows on Postgres: one workspace, the platform-key workspace as the local
edition makes it (``envs/api.defaults``), holding one active OpenAI key; removed afterwards.
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402
from cryptography.fernet import Fernet  # noqa: E402
from sqlalchemy import text  # noqa: E402

from config import config  # noqa: E402

FRESH_INSTALL = {"provider": "anthropic", "model": "claude-haiku-5-5"}  # core/llm/defaults.py
OPENAI_KEY = "sk-test-1678-not-a-real-key"
ENV_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY")


def _database_or_skip():
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — no Postgres here: skipped, as the other real-DB tests are
        db.close()
        pytest.skip(f"no test database: {type(exc).__name__}")
    return db


@pytest.fixture
def install(monkeypatch):
    """The local edition, Auto's model as a fresh install seeds it, no provider key in the
    environment, and one active OpenAI key in the platform-key workspace."""
    from core.credentials import encryption
    from core.llm import manager

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode(), raising=False)
    monkeypatch.setattr(encryption, "_encryption_service", None)
    for name in ENV_KEYS:
        monkeypatch.setattr(config, name, "", raising=False)
    settings = {"orchestrator_llm": dict(FRESH_INSTALL)}
    monkeypatch.setattr(manager, "get_system_setting",
                        lambda category, key, default=None: settings.get(category, {}).get(key, default))
    manager._credential_cache.clear()

    db = _database_or_skip()
    ws = str(uuid.uuid4())
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", ws)
    try:
        db.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'issue-1678')"), {"id": ws})
        db.execute(text("INSERT INTO user_api_keys (workspace_id, provider, encrypted_key, is_active) "
                        "VALUES (CAST(:ws AS uuid), 'openai', :key, true)"),
                   {"ws": ws, "key": encryption.get_encryption_service().encrypt(OPENAI_KEY)})
        db.commit()
        yield uuid.UUID(ws), settings
    finally:
        db.rollback()
        db.execute(text("DELETE FROM user_api_keys WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws})
        db.execute(text("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)"), {"ws": ws})
        db.commit()
        db.close()


def test_an_openai_key_alone_leaves_auto_s_seeded_model_without_one_and_the_line_names_it(install):
    from services.readiness import probes, strings

    ws, _ = install
    finding = probes.model_route(None, ws)
    assert (finding.state, finding.detail, finding.fix) == ("needs_setup", "model_no_key", "fix_add_key")
    assert finding.where == "Anthropic claude-haiku-5-5"
    assert "or choose a model whose provider has one" in strings.say("en", finding.detail)


def test_the_same_key_runs_auto_once_auto_s_model_is_an_openai_one(install):
    from core.llm import manager
    from services.proactive.runner import heartbeat_llm
    from services.readiness import probes

    ws, settings = install
    settings["orchestrator_llm"] = {"provider": "openai", "model": "gpt-4o-mini"}
    manager._credential_cache.clear()
    finding = probes.model_route(None, ws)
    assert (finding.state, finding.where) == ("ready", "OpenAI gpt-4o-mini")
    assert heartbeat_llm(ws, None).config.api_key == OPENAI_KEY
