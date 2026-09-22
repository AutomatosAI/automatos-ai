"""F066 — creating a database source introspects it.

The create endpoint's docstring always said "Introspects schema"; it never did.
The response read success with 0 tables and every query then failed "No schema
metadata available. Please run introspection first." — an instruction nothing
on screen offered. (Probe 2026-09-22: source 34 created with 0 tables; POST
/34/introspect then found 8 in 888 ms.) Creation now goes through the same
introspect_and_persist the /{id}/introspect endpoint uses.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

from fastapi import HTTPException

import api.database_knowledge as dk
from core.models.database_knowledge import DatabaseKnowledgeSourceCreate

EIGHT_TABLES = {"tables": [{"name": f"t{i}"} for i in range(8)], "relationships": []}


class _Db:
    def __init__(self, source):
        self.source, self.commits, self.rollbacks = source, 0, 0

    def query(self, *_a):
        return self

    def filter(self, *_a):
        return self

    def first(self):
        return self.source

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _wire(monkeypatch, *, introspect):
    source = NS(id=34, name="Harbourline Shop", credential_id=7, dialect="postgresql",
                schema_metadata={}, last_introspected=None)
    service = NS(add_database_source=lambda **_k: _async(source))
    tools = NS(create_database_tools=lambda _s: [])
    monkeypatch.setattr(dk, "get_services", lambda: (service, None, tools))
    monkeypatch.setattr(dk, "CredentialStore",
                        lambda _db: NS(get_decrypted_credential=lambda **_k: {"host": "x"}))

    class _Introspector:
        def __init__(self, **_k):
            pass

        def introspect(self, **_k):
            return introspect()

    monkeypatch.setattr(dk, "DatabaseIntrospectionService", _Introspector)
    return source, _Db(source)


async def _async(value):
    return value


def _create(db):
    body = DatabaseKnowledgeSourceCreate(name="Harbourline Shop", credential_id=7, dialect="postgresql")
    ctx = NS(workspace_id=uuid4(), user=None)
    return asyncio.run(dk.create_database_source(source=body, ctx=ctx, db=db))


def test_creating_a_source_introspects_it_and_reports_the_real_table_count(monkeypatch):
    source, db = _wire(monkeypatch, introspect=lambda: EIGHT_TABLES)
    out = _create(db)
    assert out["success"] is True and out["introspected"] is True
    assert out["schema_tables"] == 8, "was always 0"
    assert source.schema_metadata == EIGHT_TABLES and source.last_introspected is not None
    assert db.commits >= 1


def test_a_failed_introspection_keeps_the_source_and_says_how_to_retry(monkeypatch):
    def _boom():
        raise RuntimeError("connection refused")

    _, db = _wire(monkeypatch, introspect=_boom)
    out = _create(db)
    assert out["success"] is True, "the owner's source is not thrown away"
    assert out["introspected"] is False and out["schema_tables"] == 0
    assert "introspection" in out["message"].lower() and "again" in out["message"].lower()
    assert out["introspection_error"]


def test_the_introspect_endpoint_and_creation_share_one_implementation(monkeypatch):
    source, db = _wire(monkeypatch, introspect=lambda: EIGHT_TABLES)
    metadata = dk.introspect_and_persist(db, source)
    assert metadata == EIGHT_TABLES and source.schema_metadata == EIGHT_TABLES


def test_bad_credentials_are_named_as_the_failed_step(monkeypatch):
    source, db = _wire(monkeypatch, introspect=lambda: EIGHT_TABLES)

    def _no_creds(**_k):
        raise RuntimeError("decrypt failed")

    monkeypatch.setattr(dk, "CredentialStore", lambda _db: NS(get_decrypted_credential=_no_creds))
    try:
        dk.introspect_and_persist(db, source)
    except HTTPException as e:
        assert "credentials" in str(e.detail).lower()
    else:
        raise AssertionError("expected a 400 naming the credentials step")
