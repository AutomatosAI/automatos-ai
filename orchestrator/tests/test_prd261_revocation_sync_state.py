"""PRD-261 W-C3 seam fix A: one validated reader of the keys the hosted feed revoked.

``services/themes/revocation_sync_state.py`` reads ``theme_revocation_sync.revoked_key_ids`` for
the signature check on load (``signature_state``) and the sync status (``revocation_pull``), over
SQLite. Pins:

* local edition only: the hosted edition never reads the row; no row reads as no key;
* only string entries count; a value that isn't a list reads as no key; malformed data is logged
  once per value, and again after a well-formed read; the answer is a frozenset;
* ``signature_state.revoked_key_ids`` is that reader, and ``revocation_pull.stored_revoked_keys``
  is its sorted list (the sync status's ``revokedKeys``);
* the reader is a leaf (the models, ``config`` and the standard library), so the revocation rule
  (``revocations``) never imports the pull or the feed: no import cycle.
"""
from __future__ import annotations

import ast
import logging
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from config import config  # noqa: E402
from core.models.themes import THEME_REVOCATION_SYNC_ROW_ID, ThemeRevocationSync  # noqa: E402
from services.themes import revocation_pull, revocation_sync_state, signature_state  # noqa: E402

KEY_A, KEY_B = "ed25519:" + "a" * 16, "ed25519:" + "b" * 16
THEMES = _ORCH / "services" / "themes"
THEMES_PACKAGE = "services.themes"
# What the leaf may import beyond the standard library.
LEAF_FIRST_PARTY = {"config", "core.models.themes"}
FIRST_PARTY_ROOTS = {"api", "config", "core", "modules", "services"}


@pytest.fixture
def db(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemeRevocationSync.metadata.create_all(engine, tables=[ThemeRevocationSync.__table__])
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    session = sessionmaker(bind=engine)()
    revocation_sync_state._malformed.forget()
    try:
        yield session
    finally:
        revocation_sync_state._malformed.forget()
        session.close()
        engine.dispose()


def _store(db, value):
    db.merge(ThemeRevocationSync(id=THEME_REVOCATION_SYNC_ROW_ID, revoked_key_ids=value))
    db.commit()


def _errors(caplog):
    return [record.getMessage() for record in caplog.records
            if record.name == revocation_sync_state.__name__ and record.levelno == logging.ERROR]


def test_no_row_reads_as_no_key(db):
    assert revocation_sync_state.revoked_key_ids(db) == frozenset()
    assert revocation_pull.stored_revoked_keys(db) == []


def test_only_key_ids_count_and_both_callers_read_the_same_set(db, caplog):
    _store(db, [KEY_B, 7, None, KEY_A, {"keyId": KEY_A}])
    with caplog.at_level(logging.ERROR, logger=revocation_sync_state.__name__):
        for _ in range(3):  # the files route asks on every file: one log line, not one per read
            read = revocation_sync_state.revoked_key_ids(db)
            assert isinstance(read, frozenset) and read == {KEY_A, KEY_B}
            assert signature_state.revoked_key_ids(db) == read
            assert revocation_pull.stored_revoked_keys(db) == [KEY_A, KEY_B]
    assert _errors(caplog) == [revocation_sync_state.NOT_KEY_IDS]


def test_a_value_that_is_not_a_list_reads_as_no_key_and_is_logged_once_per_value(db, caplog):
    _store(db, {"revoked": [KEY_A]})
    with caplog.at_level(logging.ERROR, logger=revocation_sync_state.__name__):
        assert revocation_sync_state.revoked_key_ids(db) == frozenset()
        assert revocation_pull.stored_revoked_keys(db) == []
        assert len(_errors(caplog)) == 1 and "not a list (dict)" in _errors(caplog)[0]
        _store(db, [KEY_A])  # well formed in between: the same bad value is logged again
        assert revocation_sync_state.revoked_key_ids(db) == {KEY_A}
        _store(db, {"revoked": [KEY_A]})
        assert revocation_sync_state.revoked_key_ids(db) == frozenset()
    assert len(_errors(caplog)) == 2


def test_the_hosted_edition_never_reads_the_row(db, monkeypatch):
    _store(db, [KEY_A])
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")

    class NoDatabase:
        def get(self, *args, **kwargs):
            raise AssertionError("the hosted edition read theme_revocation_sync")

    assert revocation_sync_state.revoked_key_ids(NoDatabase()) == frozenset()
    assert signature_state.revoked_key_ids(db) == frozenset() and revocation_pull.stored_revoked_keys(db) == []


def test_signature_state_has_no_reader_of_its_own():
    assert signature_state.revoked_key_ids is revocation_sync_state.revoked_key_ids
    source = (THEMES / "signature_state.py").read_text(encoding="utf-8")
    assert "ThemeRevocationSync" not in source and "def revoked_key_ids" not in source


def _module_file(module: str):
    """The source file of a first-party module or package, or None for a name that isn't one."""
    base = _ORCH / module.replace(".", "/")
    return next((path for path in (base.with_suffix(".py"), base / "__init__.py") if path.is_file()), None)


def _first_party_imports(path: Path):
    """The first-party modules a file imports by absolute import (``from x import y`` adds ``x.y`` when y is a module)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            submodules = [f"{node.module}.{alias.name}" for alias in node.names]
            names = [node.module] + [name for name in submodules if _module_file(name) is not None]
        found |= {name for name in names if name.split(".")[0] in FIRST_PARTY_ROOTS}
    return found


def _themes_closure(start: str):
    """Every ``services.themes`` module ``start`` imports, directly or through another of them."""
    seen, todo = set(), [start]
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        todo += [name for name in _first_party_imports(_module_file(module)) if name.startswith(THEMES_PACKAGE + ".")]
    return seen


def test_the_reader_is_a_leaf_so_the_revocation_rule_never_reaches_the_pull_or_the_feed():
    assert _first_party_imports(THEMES / "revocation_sync_state.py") == LEAF_FIRST_PARTY
    reached = _themes_closure("services.themes.revocations")
    assert "services.themes.revocation_sync_state" in reached
    assert not {"services.themes.revocation_pull", "services.themes.revocation_feed"} & reached, sorted(reached)
