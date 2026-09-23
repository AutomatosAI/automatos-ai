"""F077 — platform_query_data answers, by name, by id, or with no id at all.

Night 3, harbourline_shop connected as source 36: Auto 1/7 on NL2SQL questions.
Both paths of the handler were broken:

* ``database_id`` given as a NAME was compared with the integer id in SQL —
  psycopg2 InvalidTextRepresentation — and the aborted transaction then failed
  every later tool in the turn (the F074 pattern);
* no ``database_id`` built ``DatabaseKnowledgeService()`` with none of its five
  dependencies — TypeError.

The handler now runs the one in-process NL2SQL path (``run_nl2sql``): the shared
service, workspace-scoped resolution by id or name (``match_source``), one audit
row. These tests drive the real resolver through the real handler with only the
database reads and the query itself faked.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from modules.nl2sql.service import DatabaseKnowledgeService, match_source
from modules.tools.discovery.handlers_scheduling import query_data

HARBOURLINE = (36, "harbourline_shop")
CRM = (37, "crm")
WS = uuid4()


# ── the resolver ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reference", [36, "36", " 36 ", "harbourline_shop", "Harbourline_Shop", " harbourline_shop "])
def test_a_source_is_found_by_id_or_by_exact_name(reference):
    assert match_source([HARBOURLINE, CRM], reference) == 36


def test_no_reference_uses_the_only_source_and_refuses_to_guess_between_several():
    assert match_source([HARBOURLINE], None) == 36
    assert match_source([HARBOURLINE], "  ") == 36
    assert match_source([HARBOURLINE, CRM], None) is None
    assert match_source([], None) is None


@pytest.mark.parametrize("reference", [999, "999", "harbourlineXshop", "harbourline%", True, "crm2"])
def test_an_unknown_reference_matches_nothing(reference):
    """``_`` and ``%`` are not wildcards; a digit string that is no id here is
    tried as a name and fails."""
    assert match_source([HARBOURLINE, CRM], reference) is None


def test_a_name_made_of_digits_still_resolves_when_no_id_matches():
    assert match_source([(40, "2024")], "2024") == 40


# ── the handler ─────────────────────────────────────────────────────────────

class _Service(DatabaseKnowledgeService):
    """The real resolver (resolve_source_id → match_source); the database reads
    and the NL2SQL query are the only fakes."""

    def __init__(self, sources, fail_reads=0):
        self.sources, self.fail_reads = list(sources), fail_reads
        self.queries, self.audits = [], []

    async def active_sources(self, workspace_id, db_session=None):
        if self.fail_reads:
            self.fail_reads -= 1
            raise RuntimeError("current transaction is aborted")
        return list(self.sources)

    async def query_database(self, **kwargs):
        self.queries.append(kwargs)
        return {"success": True, "sql": "SELECT count(*) AS orders FROM orders",
                "data": [{"orders": 412}], "columns": ["orders"], "row_count": 1}

    async def write_nl_audit(self, **kwargs):
        self.audits.append(kwargs)


class _Db:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


def _ask(monkeypatch, service, db=None, **params):
    monkeypatch.setattr("modules.nl2sql.get_database_knowledge_service", lambda: service)
    db = db or _Db()
    result = asyncio.run(query_data(db, WS, {"question": "how many orders?", "_user_id": "7", **params}))
    return result, db


@pytest.mark.parametrize("reference", ["harbourline_shop", 36, "36"])
def test_a_name_or_an_id_queries_that_source(monkeypatch, reference):
    service = _Service([HARBOURLINE, CRM])
    result, _ = _ask(monkeypatch, service, database_id=reference)
    assert result["success"] is True and "412" in result["answer"]
    [call] = service.queries
    assert call["source_id"] == "36" and call["workspace_id"] == str(WS) and call["user_id"] == "7"
    assert len(service.audits) == 1                      # PRD-160 S4: every NL query is audited


def test_no_id_uses_the_workspaces_only_database(monkeypatch):
    service = _Service([HARBOURLINE])
    result, _ = _ask(monkeypatch, service)
    assert result["success"] is True and service.queries[0]["source_id"] == "36"


def test_several_databases_and_no_id_names_them_instead_of_guessing(monkeypatch):
    service = _Service([HARBOURLINE, CRM])
    result, _ = _ask(monkeypatch, service)
    assert result["success"] is False and service.queries == []
    assert "harbourline_shop (#36)" in result["error"] and "crm (#37)" in result["error"]


def test_a_bad_id_says_what_exists(monkeypatch):
    service = _Service([HARBOURLINE])
    result, _ = _ask(monkeypatch, service, database_id=999)
    assert result["success"] is False and service.queries == []
    assert "'999'" in result["error"] and "harbourline_shop (#36)" in result["error"]


@pytest.mark.parametrize("reference", [True, 3.5, {"id": 36}, [36]])
def test_a_reference_of_the_wrong_type_is_refused_before_any_read(monkeypatch, reference):
    service = _Service([HARBOURLINE])
    result, _ = _ask(monkeypatch, service, database_id=reference)
    assert result == {"success": False, "error": "database_id must be a database source's id or its name"}
    assert service.queries == []


def test_the_session_is_usable_after_a_failed_read(monkeypatch):
    """A failure hands the turn's session back rolled back; the next ask works."""
    service = _Service([HARBOURLINE], fail_reads=1)
    failed, db = _ask(monkeypatch, service, database_id="harbourline_shop")
    assert failed == {"success": False, "error": "Database query failed."} and db.rollbacks == 1
    retried, _ = _ask(monkeypatch, service, db=db, database_id="harbourline_shop")
    assert retried["success"] is True and db.rollbacks == 1


# ── the read itself rolls a borrowed session back ───────────────────────────

class _BorrowedSession:
    """The request session: the first read fails (as an aborted transaction
    does); after a rollback it answers."""

    def __init__(self):
        self.rollbacks, self.reads = 0, 0

    def rollback(self):
        self.rollbacks += 1

    def query(self, *_columns):
        self.reads += 1
        if self.rollbacks == 0:
            raise RuntimeError("InFailedSqlTransaction")
        return self

    def filter(self, *_a):
        return self

    def order_by(self, *_a):
        return self

    def all(self):
        return [(36, "harbourline_shop")]

    def close(self):  # pragma: no cover - a borrowed session is never closed
        raise AssertionError("a borrowed session belongs to the caller")


def test_a_failed_read_on_a_borrowed_session_rolls_it_back_and_it_then_works():
    service = DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)
    session = _BorrowedSession()
    with pytest.raises(RuntimeError):
        asyncio.run(service.active_sources(str(WS), db_session=session))
    assert session.rollbacks == 1
    assert asyncio.run(service.active_sources(str(WS), db_session=session)) == [(36, "harbourline_shop")]
    assert asyncio.run(service.resolve_source_id(str(WS), "HARBOURLINE_SHOP", db_session=session)) == "36"
