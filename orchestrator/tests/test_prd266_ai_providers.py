"""PRD-266 US-017a: the LLM providers a workspace's Auto and agents can call.

``services.ai_providers.workspace_ai_providers`` lists, as ``{id, name}`` sorted
by id, (a) the chat providers whose own key the workspace holds active with its
``byok_overrides`` switch on, and (b) the chat providers with a platform key this
edition allows: an operator-workspace row, a credential-store row under one of
the resolver's four names, or the config's env key. NVIDIA's platform key is the
operator's own only on a local install; Cohere has no chat client, so it is
never listed from a key.

It reads key PRESENCE only: the tests make every decrypting path raise, store
sentinel strings where the keys would be, and check that no sentinel reaches the
result or any log record. Real Postgres (CI) for the key and credential rows,
inside the test's rolled-back transaction.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402
from sqlalchemy import event, text  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import core.credentials.encryption as encryption_mod  # noqa: E402
import core.credentials.resolver as credential_resolver_mod  # noqa: E402
import core.llm.key_resolver as key_resolver_mod  # noqa: E402
import core.llm.workspace_keys as workspace_keys_mod  # noqa: E402
from config import config  # noqa: E402
from core.credentials.service import CredentialStore  # noqa: E402
from core.llm import providers as registry  # noqa: E402
from core.models.core import UserApiKey  # noqa: E402
from core.models.credentials import Credential, CredentialType  # noqa: E402
from services import ai_providers  # noqa: E402
from services.ai_providers import CREDENTIAL_NAME_FORMS, workspace_ai_providers  # noqa: E402

# Stored where a key would be. Low-entropy on purpose: no secret scanner reads it as a key.
ROW_SENTINEL = "stored-key-sentinel-never-read"
ENV_SENTINEL = "env-key-sentinel-never-read"


# ---------------------------------------------------------------------------
# Fixtures: a clean key floor, and every decrypting path made to fail
# ---------------------------------------------------------------------------


@pytest.fixture
def floor(db_session, seed_workspace, monkeypatch):
    """A workspace and an operator workspace, no env keys, saas, no stray credential rows."""
    for spec in registry.REGISTRY.values():
        if spec.env_key:
            monkeypatch.setattr(config, spec.env_key, "", raising=False)
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", "")
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    # The platform lane reads the credential store unscoped, so a row another test
    # committed would count here: removed inside this test's transaction (its
    # rollback restores it).
    every_name = [form.format(slug=slug) for slug in registry.all_slugs() for form in CREDENTIAL_NAME_FORMS]
    db_session.query(Credential).filter(Credential.name.in_(every_name)).delete(synchronize_session=False)
    return SimpleNamespace(db=db_session, ws=seed_workspace(), operator_ws=seed_workspace())


@pytest.fixture(autouse=True)
def nothing_decrypts(monkeypatch):
    """Any path that would decrypt or resolve a key fails the test."""

    def _refuse(*_args, **_kwargs):
        raise AssertionError("ai_providers must read key presence only")

    monkeypatch.setattr(encryption_mod.EncryptionService, "decrypt", _refuse)
    monkeypatch.setattr(encryption_mod.EncryptionService, "decrypt_dict", _refuse)
    monkeypatch.setattr(key_resolver_mod, "resolve_provider_key", _refuse)
    monkeypatch.setattr(workspace_keys_mod, "get_platform_workspace_key", _refuse)
    monkeypatch.setattr(credential_resolver_mod.CredentialResolver, "get_dict", _refuse)
    monkeypatch.setattr(CredentialStore, "get_decrypted_credential_by_name", _refuse)
    monkeypatch.setattr(CredentialStore, "get_decrypted_credential", _refuse)


def _key(db, workspace_id: str, provider: str, *, active: bool = True) -> UserApiKey:
    row = UserApiKey(workspace_id=uuid.UUID(workspace_id), provider=provider,
                     encrypted_key=ROW_SENTINEL, is_active=active)
    db.add(row)
    db.flush()
    return row


def _overrides(db, workspace_id: str, **switches: bool) -> None:
    db.execute(
        text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:id AS uuid)"),
        {"s": json.dumps({"byok_overrides": switches}), "id": workspace_id},
    )
    db.expire_all()


def _credential(db, workspace_id: str, name: str, *, active: bool = True) -> None:
    kind = CredentialType(name=f"api-{uuid.uuid4().hex[:10]}", display_name="API key", schema_definition=[])
    db.add(kind)
    db.flush()
    db.add(Credential(name=name, workspace_id=uuid.UUID(workspace_id), credential_type_id=kind.id,
                      encrypted_data=ROW_SENTINEL, is_active=active))
    db.flush()


def _ids(floor) -> list:
    return [entry["id"] for entry in workspace_ai_providers(floor.db, floor.ws)]


# ---------------------------------------------------------------------------
# (b) platform routes
# ---------------------------------------------------------------------------


def test_saas_lists_the_platform_openrouter_and_anthropic_keys_with_their_registry_labels(floor, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", ENV_SENTINEL)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", ENV_SENTINEL)

    assert workspace_ai_providers(floor.db, floor.ws) == [
        {"id": "anthropic", "name": "Anthropic"},
        {"id": "openrouter", "name": "OpenRouter"},
    ]


def test_a_byok_only_nvidia_env_key_is_not_listed_on_saas_and_is_listed_on_local(floor, monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", ENV_SENTINEL)

    assert "nvidia" not in _ids(floor)

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    assert {"id": "nvidia", "name": "NVIDIA"} in workspace_ai_providers(floor.db, floor.ws)


def test_an_operator_workspace_row_counts_and_an_inactive_one_does_not(floor, monkeypatch):
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", floor.operator_ws)
    _key(floor.db, floor.operator_ws, "google")
    _key(floor.db, floor.operator_ws, "deepseek", active=False)

    assert _ids(floor) == ["google"]


def test_an_operator_workspace_id_that_is_not_a_uuid_counts_nothing_and_breaks_nothing(floor, monkeypatch):
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", "not-a-workspace")
    monkeypatch.setattr(config, "OPENAI_API_KEY", ENV_SENTINEL)

    assert _ids(floor) == ["openai"]


@pytest.mark.parametrize("form", CREDENTIAL_NAME_FORMS)
def test_a_credential_store_row_under_each_of_the_resolver_names_counts(floor, form):
    _credential(floor.db, floor.operator_ws, form.format(slug="grok"))

    assert workspace_ai_providers(floor.db, floor.ws) == [{"id": "grok", "name": "Grok / xAI"}]


def test_an_inactive_credential_store_row_does_not_count(floor):
    _credential(floor.db, floor.operator_ws, "openai_api", active=False)

    assert _ids(floor) == []


# ---------------------------------------------------------------------------
# (a) the workspace's own keys
# ---------------------------------------------------------------------------


def test_the_workspaces_own_deepseek_key_counts_only_while_active_with_its_override_on(floor):
    db = floor.db
    row = _key(db, floor.ws, "deepseek")

    _overrides(db, floor.ws, deepseek=True)
    assert workspace_ai_providers(db, floor.ws) == [{"id": "deepseek", "name": "DeepSeek"}]

    _overrides(db, floor.ws, deepseek=False)
    assert _ids(floor) == []

    _overrides(db, floor.ws, deepseek=True)
    row.is_active = False
    db.flush()
    assert _ids(floor) == []

    row.is_active = True
    db.flush()
    assert _ids(floor) == ["deepseek"]
    db.delete(row)
    db.flush()
    assert _ids(floor) == []


def test_another_workspaces_own_key_does_not_count(floor, seed_workspace):
    elsewhere = seed_workspace()
    _key(floor.db, elsewhere, "deepseek")
    _overrides(floor.db, elsewhere, deepseek=True)
    _overrides(floor.db, floor.ws, deepseek=True)

    assert _ids(floor) == []


def test_the_workspaces_own_nvidia_key_counts_on_saas(floor):
    _key(floor.db, floor.ws, "nvidia")
    _overrides(floor.db, floor.ws, nvidia=True)

    assert _ids(floor) == ["nvidia"]


# ---------------------------------------------------------------------------
# What never counts, and what never leaks
# ---------------------------------------------------------------------------


def test_a_key_only_provider_adds_nothing_through_either_source(floor, monkeypatch):
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", floor.operator_ws)
    monkeypatch.setattr(config, "COHERE_API_KEY", ENV_SENTINEL)
    _key(floor.db, floor.ws, "cohere")
    _overrides(floor.db, floor.ws, cohere=True)
    _key(floor.db, floor.operator_ws, "cohere")
    _credential(floor.db, floor.operator_ws, "cohere_api")

    assert _ids(floor) == []


def test_no_key_value_appears_in_the_result_or_in_any_log_record(floor, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", floor.operator_ws)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", ENV_SENTINEL)
    _key(floor.db, floor.ws, "deepseek")
    _overrides(floor.db, floor.ws, deepseek=True)
    _key(floor.db, floor.operator_ws, "anthropic")
    _credential(floor.db, floor.operator_ws, "openai_api")

    result = workspace_ai_providers(floor.db, floor.ws)

    assert [entry["id"] for entry in result] == ["anthropic", "deepseek", "openai", "openrouter"]
    for sentinel in (ROW_SENTINEL, ENV_SENTINEL):
        assert sentinel not in json.dumps(result)
        assert all(sentinel not in record.getMessage() for record in caplog.records)


def test_the_list_is_sorted_and_unique_when_every_source_names_the_same_provider(floor, monkeypatch):
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", floor.operator_ws)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", ENV_SENTINEL)
    monkeypatch.setattr(config, "GOOGLE_API_KEY", ENV_SENTINEL)
    _key(floor.db, floor.ws, "openrouter")
    _key(floor.db, floor.ws, "openrouter")
    _overrides(floor.db, floor.ws, openrouter=True)
    _key(floor.db, floor.operator_ws, "openrouter")
    _credential(floor.db, floor.operator_ws, "development_openrouter_api")
    _credential(floor.db, floor.operator_ws, "openrouter")

    ids = _ids(floor)

    assert ids == ["google", "openrouter"]
    assert ids == sorted(set(ids))


def test_each_store_is_read_once_whatever_the_number_of_providers(floor, monkeypatch):
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", floor.operator_ws)
    for slug in ("openai", "anthropic", "deepseek", "grok"):
        _key(floor.db, floor.ws, slug)
        _key(floor.db, floor.operator_ws, slug)
    _overrides(floor.db, floor.ws, openai=True, anthropic=True, deepseek=True, grok=True)
    statements = []

    def _record(_conn, _cursor, statement, *_args):
        statements.append(statement)

    connection = floor.db.connection()
    event.listen(connection, "before_cursor_execute", _record)
    try:
        assert _ids(floor) == ["anthropic", "deepseek", "grok", "openai"]
    finally:
        event.remove(connection, "before_cursor_execute", _record)

    assert sum("FROM user_api_keys" in s for s in statements) == 2  # (a) and the operator workspace
    assert sum("FROM credentials" in s for s in statements) == 1
    assert not any("encrypted_key" in s or "encrypted_data" in s for s in statements)


def test_an_unknown_slug_is_named_by_itself_and_a_known_one_by_its_label():
    assert ai_providers._registry_names(["voyage", "aws_bedrock"]) == {
        "voyage": "voyage",
        "bedrock": "AWS Bedrock",
    }
