"""An organization-level Anthropic key names the workspace it bills (9 Oct 2026).

Gerard's Anthropic key failed in Settings → API Keys: "Key saved but failed validation:
... This API key is not scoped to a workspace, so this request must include the
anthropic-workspace-id header". A key created at organization level must name a
workspace on every request, and the dialog had nowhere to put one. The workspace ID
(``wrkspc_…``) is now saved with the key (``user_api_keys.provider_workspace_id``),
sent with the key check, and added by every Anthropic client built from that key
(``core.llm.anthropic_workspace``). A key scoped to its workspace is sent as before.

Pure tests: the Anthropic SDK, httpx, the database and encryption are stubbed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import api.user_api_keys as uak
import core.llm.anthropic_workspace as aw
import core.llm.providers as registry
from api.user_api_keys import ApiKeyCreate, ApiKeyValidation, _validate_provider_key

WORKSPACE = "wrkspc_01JwQvzr7rXLA5AGx3HKfFUJ"
ORG_KEY = "sk-ant-api03-org-level-key"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    aw.clear_cache()
    monkeypatch.setattr(aw.config, "ANTHROPIC_WORKSPACE_ID", None)
    monkeypatch.setattr(aw.config, "ANTHROPIC_API_KEY", None)
    yield
    aw.clear_cache()


class _FakeAnthropic:
    """Stands in for ``anthropic.Anthropic``: records how it was built."""

    built = []

    def __init__(self, **kwargs):
        _FakeAnthropic.built.append(kwargs)
        self.models = SimpleNamespace(list=lambda: [])


@pytest.fixture
def fake_sdk(monkeypatch):
    import sys
    _FakeAnthropic.built = []
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=_FakeAnthropic))
    return _FakeAnthropic.built


# ── the ID itself ──────────────────────────────────────────────────────────


def test_a_workspace_id_is_kept_trimmed_and_a_blank_one_is_none():
    assert aw.clean_workspace_id("anthropic", f"  {WORKSPACE}  ") == WORKSPACE
    assert aw.clean_workspace_id("anthropic", "  ") is None
    assert aw.clean_workspace_id("anthropic", None) is None


@pytest.mark.parametrize("raw", ["workspace-1", "wrkspc_", "wrkspc_abc", "wrkspc_01J/../x", "sk-ant-api03-xyz"])
def test_anything_that_is_not_a_workspace_id_is_refused(raw):
    with pytest.raises(aw.WorkspaceIdRefused):
        aw.clean_workspace_id("anthropic", raw)


def test_only_an_anthropic_key_takes_a_workspace_id():
    with pytest.raises(aw.WorkspaceIdRefused):
        aw.clean_workspace_id("openai", WORKSPACE)
    with pytest.raises(HTTPException) as err:
        uak._key_workspace("openai", WORKSPACE)
    assert err.value.status_code == 400


def test_the_registry_offers_the_workspace_field_for_anthropic_only():
    assert registry.to_public_dict(registry.get_spec("anthropic"))["workspace_id_placeholder"].startswith("wrkspc_")
    assert registry.to_public_dict(registry.get_spec("openai"))["workspace_id_placeholder"] is None


# ── the key check ──────────────────────────────────────────────────────────


def test_the_key_check_names_the_workspace(fake_sdk):
    result = asyncio.run(_validate_provider_key("anthropic", ORG_KEY, workspace_id=WORKSPACE))

    assert result.valid
    assert fake_sdk[-1]["default_headers"] == {"anthropic-workspace-id": WORKSPACE}


def test_a_scoped_key_is_checked_without_the_header(fake_sdk):
    assert asyncio.run(_validate_provider_key("anthropic", ORG_KEY)).valid
    assert fake_sdk[-1]["default_headers"] == {}


class _KeysDB:
    def __init__(self):
        self.added = []

    def query(self, _model):
        return SimpleNamespace(get=lambda _id: None)

    def add(self, row):
        row.id, row.created_at = 1, datetime.utcnow()
        self.added.append(row)

    def commit(self):
        pass

    def refresh(self, _row):
        pass


def test_saving_a_key_stores_its_workspace_and_checks_it_there(monkeypatch):
    seen = {}

    async def _check(provider, key, base_url=None, workspace_id=None):
        seen.update(provider=provider, workspace_id=workspace_id)
        return ApiKeyValidation(valid=False, message="Invalid key: 401", tested_at=datetime.utcnow())

    monkeypatch.setattr(uak, "_validate_provider_key", _check)
    monkeypatch.setattr(uak, "get_encryption_service", lambda: SimpleNamespace(
        encrypt=lambda s: f"enc::{s}", decrypt=lambda s: s[5:]))
    db = _KeysDB()
    body = ApiKeyCreate(provider="anthropic", api_key=ORG_KEY, workspace_id=f" {WORKSPACE} ")

    asyncio.run(uak.add_api_key(body, ctx=SimpleNamespace(workspace_id="ws-1"), db=db))

    assert db.added[0].provider_workspace_id == WORKSPACE
    assert db.added[0].key_fingerprint == aw.key_fingerprint(ORG_KEY)
    assert ORG_KEY not in db.added[0].key_fingerprint
    assert seen == {"provider": "anthropic", "workspace_id": WORKSPACE}


def test_a_key_saved_without_a_workspace_stores_no_fingerprint(monkeypatch):
    async def _check(provider, key, base_url=None, workspace_id=None):
        return ApiKeyValidation(valid=False, message="Invalid key: 401", tested_at=datetime.utcnow())

    monkeypatch.setattr(uak, "_validate_provider_key", _check)
    monkeypatch.setattr(uak, "get_encryption_service", lambda: SimpleNamespace(
        encrypt=lambda s: f"enc::{s}", decrypt=lambda s: s[5:]))
    db = _KeysDB()

    asyncio.run(uak.add_api_key(ApiKeyCreate(provider="anthropic", api_key=ORG_KEY),
                                ctx=SimpleNamespace(workspace_id="ws-1"), db=db))

    assert db.added[0].provider_workspace_id is None and db.added[0].key_fingerprint is None


# ── every client built from the key ────────────────────────────────────────


def test_the_workspace_saved_with_a_key_is_found_and_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(aw, "_stored_workspace", lambda fingerprint: calls.append(fingerprint) or WORKSPACE)

    assert aw.workspace_for_key(ORG_KEY) == WORKSPACE
    assert aw.workspace_for_key(ORG_KEY) == WORKSPACE
    assert calls == [aw.key_fingerprint(ORG_KEY)]
    assert aw.workspace_for_key("") is None


class _FingerprintDB:
    """Answers the one fingerprint query; records what it was asked."""

    def __init__(self, answer):
        self.answer, self.filters, self.closed = answer, None, False

    def query(self, *_columns):
        return self

    def filter(self, *criteria):
        self.filters = criteria
        return self

    def first(self):
        return self.answer

    def close(self):
        self.closed = True


def test_the_lookup_reads_one_row_by_fingerprint_and_decrypts_nothing(monkeypatch):
    import core.credentials.encryption as encryption
    import core.database.database as database

    db = _FingerprintDB((WORKSPACE,))
    monkeypatch.setattr(database, "SessionLocal", lambda: db)
    monkeypatch.setattr(encryption, "get_encryption_service", lambda: pytest.fail("no key is decrypted"))

    assert aw._stored_workspace(aw.key_fingerprint(ORG_KEY)) == WORKSPACE
    assert any("key_fingerprint" in str(c) for c in db.filters) and db.closed


def test_the_cache_stays_bounded(monkeypatch):
    monkeypatch.setattr(aw, "_stored_workspace", lambda fingerprint: None)
    for i in range(aw.CACHE_MAX_KEYS + 10):
        aw.workspace_for_key(f"sk-ant-api03-key-{i}")
    assert len(aw._cache) <= aw.CACHE_MAX_KEYS


def test_the_operator_env_key_takes_its_workspace_from_config(monkeypatch):
    monkeypatch.setattr(aw.config, "ANTHROPIC_API_KEY", ORG_KEY)
    monkeypatch.setattr(aw.config, "ANTHROPIC_WORKSPACE_ID", WORKSPACE)
    monkeypatch.setattr(aw, "_stored_workspace", lambda key: pytest.fail("the env key needs no lookup"))

    assert aw.workspace_for_key(ORG_KEY) == WORKSPACE


def test_a_failed_lookup_sends_the_key_without_a_header(monkeypatch):
    import core.database.database as database

    def _broken():
        raise RuntimeError("database down")

    monkeypatch.setattr(database, "SessionLocal", _broken)
    assert aw._stored_workspace(aw.key_fingerprint(ORG_KEY)) is None


def test_the_anthropic_client_names_the_workspace(monkeypatch):
    import core.llm.clients.anthropic_client as client_module
    from core.llm.clients.base import LLMConfig, LLMProvider

    _FakeAnthropic.built = []
    monkeypatch.setattr(client_module, "anthropic", SimpleNamespace(Anthropic=_FakeAnthropic))
    monkeypatch.setattr(client_module, "workspace_for_key", lambda key: WORKSPACE if key == ORG_KEY else None)

    client_module.AnthropicProvider(LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-5", api_key=ORG_KEY))
    client_module.AnthropicProvider(LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-5", api_key="sk-scoped"))

    assert _FakeAnthropic.built[0]["default_headers"] == {"anthropic-workspace-id": WORKSPACE}
    assert "default_headers" not in _FakeAnthropic.built[1]


def test_the_model_catalogue_sync_names_the_workspace(monkeypatch):
    import core.services.anthropic_catalog_sync as sync

    seen = {}

    def _page(_client, headers, _after):
        seen.update(headers)
        return {"data": [], "has_more": False}

    monkeypatch.setattr(sync, "_fetch_page", _page)
    monkeypatch.setattr(sync, "workspace_for_key", lambda key: WORKSPACE)

    sync.fetch_anthropic_models(ORG_KEY)

    assert seen["anthropic-workspace-id"] == WORKSPACE
    assert seen["x-api-key"] == ORG_KEY
