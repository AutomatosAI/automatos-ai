"""PRD-222 W1 (US-006) — credential-validation truth + converted hook.

Pure tests: no Postgres, no live network. The provider SDK is stubbed into
``sys.modules`` so ``_validate_provider_key`` exercises its real branches; the
``add_api_key`` handler runs against fake session/encryption/validation so the
validate-on-save wiring, the is_active persistence, and the trial→converted flip
are all asserted without a DB. Real-Postgres + live-provider coverage is CI's job.

The seam under test: a BYOK key save (``POST /api/keys`` → ``add_api_key`` on the
``UserApiKey`` store) is the ONLY save that flips ``is_byok=True`` at the LLM
key-resolution choke point, so it is the only save that can bypass/convert the
trial. ``UserApiKey`` has no ``test_status``/``tested_at`` column and this branch
spends its ONE migration on US-001, so the resolver-visible truth is persisted in
``is_active`` (what ``_resolve_api_key`` filters on and the BYOK badge reads).
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime
from types import SimpleNamespace

import pytest

import api.user_api_keys as uak
from api.user_api_keys import ApiKeyCreate, ApiKeyValidation, _validate_provider_key
from services.trial_ledger import (
    TRIAL_ACTIVE,
    TRIAL_CONVERTED,
    TRIAL_EXHAUSTED,
    TRIAL_WARNED,
    mark_trial_converted,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeQuery:
    def __init__(self, ws):
        self._ws = ws

    def get(self, _id):
        return self._ws


class _FakeEnc:
    def encrypt(self, s):
        return f"enc::{s}"

    def decrypt(self, s):
        return s[5:] if isinstance(s, str) and s.startswith("enc::") else "********"


class _ConvDB:
    """Session stand-in for mark_trial_converted: query→ws, records trial UPDATEs."""

    def __init__(self, ws):
        self._ws = ws
        self.trial_writes = []

    def query(self, _model):
        return _FakeQuery(self._ws)

    def execute(self, clause, params=None):
        s = str(clause).strip().upper()
        if s.startswith("UPDATE") and "WORKSPACES" in s:
            self.trial_writes.append(json.loads(params["trial"]))
        return _FakeResult(None)


class _KeysDB(_ConvDB):
    """add_api_key session stand-in: also stamps id/created_at on add(), commits."""

    def __init__(self, ws):
        super().__init__(ws)
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            obj.id = 1
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime.utcnow()

    def commit(self):
        self.commits += 1

    def refresh(self, _obj):
        pass


def _ws(trial=None, *, settings=None):
    doc = {"stage": "questions", "stages": {}, "segment": {}}
    if trial is not None:
        doc["trial"] = trial
    return SimpleNamespace(onboarding=doc, settings=dict(settings or {}))


def _trial(state, *, granted=5.0, spent=1.0):
    return {"granted_usd": granted, "spent_usd": spent, "state": state}


def _install_fake_openai(monkeypatch, *, raises=None):
    """Stub the ``openai`` SDK so _validate_provider_key hits a controllable call."""
    mod = types.ModuleType("openai")

    class _Models:
        def list(self):
            if raises:
                raise Exception(raises)
            return SimpleNamespace(data=["m"])

    class OpenAI:  # noqa: N801 - mirror the real class name the code imports
        def __init__(self, **_kwargs):
            self.models = _Models()

    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)


# =========================================================================== #
# _validate_provider_key — the LIVE call (AC1)
# =========================================================================== #


def test_validate_supported_key_success(monkeypatch):
    _install_fake_openai(monkeypatch)
    v = asyncio.run(_validate_provider_key("openai", "sk-fake-000000000000"))
    assert v.valid is True
    assert v.message == "API key is valid"
    assert v.tested_at is not None


def test_validate_failure_carries_provider_error(monkeypatch):
    _install_fake_openai(monkeypatch, raises="401 Unauthorized: invalid_api_key")
    v = asyncio.run(_validate_provider_key("openai", "sk-fake-deleted"))
    assert v.valid is False
    # The provider's own error text rides through so the UI can render it in-flow.
    assert "401" in v.message and "invalid_api_key" in v.message
    assert v.tested_at is not None


def test_validate_unsupported_provider_is_honest():
    # No live check available → valid=True but we NEVER claim a validation we
    # didn't run; the message says so.
    v = asyncio.run(_validate_provider_key("cohere", "key-fake-0000"))
    assert v.valid is True
    assert "not available" in v.message.lower()


def test_validate_never_raises(monkeypatch):
    _install_fake_openai(monkeypatch, raises="connection reset by peer")
    v = asyncio.run(_validate_provider_key("openrouter", "sk-or-fake-0000"))
    assert v.valid is False and "connection reset" in v.message


# =========================================================================== #
# mark_trial_converted — the CONVERT hook (AC2)
# =========================================================================== #


@pytest.mark.parametrize("state", [TRIAL_ACTIVE, TRIAL_WARNED, TRIAL_EXHAUSTED])
def test_mark_converted_flips_on_trial_states(state):
    ws = _ws(_trial(state, spent=2.0))
    db = _ConvDB(ws)
    assert mark_trial_converted(db, "ws-1") is True
    assert db.trial_writes == [
        {"granted_usd": 5.0, "spent_usd": 2.0, "state": TRIAL_CONVERTED}
    ]


def test_mark_converted_is_noop_for_already_converted():
    ws = _ws(_trial(TRIAL_CONVERTED))
    db = _ConvDB(ws)
    assert mark_trial_converted(db, "ws-2") is False
    assert db.trial_writes == []  # a paying customer is never dragged back on


def test_mark_converted_is_noop_for_no_trial():
    db = _ConvDB(_ws(None))
    assert mark_trial_converted(db, "ws-3") is False
    assert db.trial_writes == []


def test_mark_converted_guards_none_inputs():
    assert mark_trial_converted(None, "ws-4") is False
    assert mark_trial_converted(_ConvDB(_ws(_trial(TRIAL_ACTIVE))), None) is False


# =========================================================================== #
# add_api_key — validate-on-save end to end (AC1 + AC2 + AC3)
# =========================================================================== #


def _patch_handler(monkeypatch, *, valid, message="API key is valid"):
    monkeypatch.setattr(uak, "get_encryption_service", lambda: _FakeEnc())
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda *a, **k: None)

    async def _stub(provider, key):
        return ApiKeyValidation(valid=valid, message=message, tested_at=datetime.utcnow())

    monkeypatch.setattr(uak, "_validate_provider_key", _stub)


def _ctx():
    return SimpleNamespace(workspace_id="ws-1", user=SimpleNamespace(id="u1"))


def test_add_valid_key_activates_persists_and_converts(monkeypatch):
    _patch_handler(monkeypatch, valid=True)
    ws = _ws(_trial(TRIAL_ACTIVE, spent=1.0))
    db = _KeysDB(ws)
    body = ApiKeyCreate(provider="OpenAI", api_key="sk-fake-000000000000")

    out = asyncio.run(uak.add_api_key(body, ctx=_ctx(), db=db))

    # AC1: live-validation outcome persisted onto the row + echoed in the response.
    assert out.validation.valid is True
    assert db.added[0].is_active is True
    assert db.added[0].last_used_at is not None
    # AC2: valid save enables BYOK for the provider and converts the trial.
    assert ws.settings["byok_overrides"]["openai"] is True
    assert db.trial_writes == [
        {"granted_usd": 5.0, "spent_usd": 1.0, "state": TRIAL_CONVERTED}
    ]


def test_add_invalid_key_inactive_no_byok_no_convert(monkeypatch):
    _patch_handler(monkeypatch, valid=False, message="Invalid key: 401 invalid_api_key")
    ws = _ws(_trial(TRIAL_ACTIVE, spent=1.0))
    db = _KeysDB(ws)
    body = ApiKeyCreate(provider="OpenAI", api_key="sk-fake-deleted-000")

    out = asyncio.run(uak.add_api_key(body, ctx=_ctx(), db=db))

    # The badge must never lie: a failed key is saved inactive and untrusted.
    assert out.validation.valid is False
    assert "401" in out.validation.message
    assert db.added[0].is_active is False
    assert db.added[0].last_used_at is None
    assert "byok_overrides" not in ws.settings  # BYOK never enabled
    assert db.trial_writes == []                 # trial NOT converted


def test_add_valid_key_on_non_trial_workspace_does_not_convert(monkeypatch):
    _patch_handler(monkeypatch, valid=True)
    ws = _ws(None)  # never-granted → no trial to convert
    db = _KeysDB(ws)
    body = ApiKeyCreate(provider="anthropic", api_key="sk-ant-fake-00000000")

    out = asyncio.run(uak.add_api_key(body, ctx=_ctx(), db=db))

    assert out.validation.valid is True
    assert ws.settings["byok_overrides"]["anthropic"] is True  # BYOK still enabled
    assert db.trial_writes == []  # nothing to convert — non-trial unaffected


def test_add_valid_key_on_converted_workspace_is_idempotent(monkeypatch):
    _patch_handler(monkeypatch, valid=True)
    ws = _ws(_trial(TRIAL_CONVERTED, spent=5.0))
    db = _KeysDB(ws)
    body = ApiKeyCreate(provider="openai", api_key="sk-fake-second-key00")

    out = asyncio.run(uak.add_api_key(body, ctx=_ctx(), db=db))

    assert out.validation.valid is True
    assert db.trial_writes == []  # already converted — never rewritten
