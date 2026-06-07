"""Unit tests for the per-workspace autonomy service.

``core.services.auto_autonomy`` is the single canonical reader/writer for
``workspace.settings.autonomy``. These tests pin its contract:

  * read merges onto the supervised default and validates the stored level;
  * an unknown / corrupt stored level fails *safe* to ``standard``;
  * the writer reassigns the whole ``settings`` dict (so SQLAlchemy sees the
    JSON mutation) and preserves sibling settings keys — never mutates in place;
  * an invalid level or a missing workspace raises ``ValueError``.

No real DB: a tiny fake ``Session`` returns a fake ``Workspace`` (or ``None``),
so the behaviour is deterministic and the suite runs in a lean venv.
"""
from __future__ import annotations

import uuid

import pytest

from core.services import auto_autonomy as svc


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeSession:
    """Minimal Session double: query() returns a preset workspace; flush() counts."""

    def __init__(self, workspace):
        self._workspace = workspace
        self.flushes = 0

    def query(self, _model):
        return _FakeQuery(self._workspace)

    def flush(self):
        self.flushes += 1


class _FakeWorkspace:
    def __init__(self, settings):
        self.id = uuid.uuid4()
        self.settings = settings


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

def test_load_missing_workspace_returns_default():
    db = _FakeSession(None)
    assert svc.load_autonomy(db, uuid.uuid4()) == {"level": svc.STANDARD}


def test_load_no_autonomy_key_returns_standard():
    db = _FakeSession(_FakeWorkspace({"auto_reporting": {"enabled": True}}))
    assert svc.get_autonomy_level(db, uuid.uuid4()) == svc.STANDARD


def test_load_full_level_round_trips():
    db = _FakeSession(_FakeWorkspace({"autonomy": {"level": "full"}}))
    assert svc.get_autonomy_level(db, uuid.uuid4()) == svc.FULL
    assert svc.is_full_autonomy(db, uuid.uuid4()) is True


@pytest.mark.parametrize("bad", ["FULL", "max", "", None, 3, {"x": 1}])
def test_load_invalid_level_fails_safe_to_standard(bad):
    db = _FakeSession(_FakeWorkspace({"autonomy": {"level": bad}}))
    assert svc.get_autonomy_level(db, uuid.uuid4()) == svc.STANDARD
    assert svc.is_full_autonomy(db, uuid.uuid4()) is False


def test_load_none_settings_returns_standard():
    db = _FakeSession(_FakeWorkspace(None))
    assert svc.get_autonomy_level(db, uuid.uuid4()) == svc.STANDARD


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def test_set_full_persists_and_flushes():
    ws = _FakeWorkspace({})
    db = _FakeSession(ws)

    result = svc.set_autonomy_level(db, ws.id, "full")

    assert result == {"level": "full"}
    assert ws.settings["autonomy"] == {"level": "full"}
    assert db.flushes == 1


def test_set_preserves_sibling_settings_keys():
    """Writing autonomy must not clobber other settings (immutability/merge)."""
    original = {"auto_reporting": {"enabled": True}, "branding": {"name": "X"}}
    ws = _FakeWorkspace(original)
    db = _FakeSession(ws)

    svc.set_autonomy_level(db, ws.id, "full")

    assert ws.settings["auto_reporting"] == {"enabled": True}
    assert ws.settings["branding"] == {"name": "X"}
    assert ws.settings["autonomy"] == {"level": "full"}
    # The whole dict is reassigned (new object), not mutated in place.
    assert ws.settings is not original


def test_set_invalid_level_raises_value_error():
    ws = _FakeWorkspace({})
    db = _FakeSession(ws)
    with pytest.raises(ValueError):
        svc.set_autonomy_level(db, ws.id, "ludicrous")
    assert db.flushes == 0


def test_set_missing_workspace_raises_value_error():
    db = _FakeSession(None)
    with pytest.raises(ValueError):
        svc.set_autonomy_level(db, uuid.uuid4(), "full")
