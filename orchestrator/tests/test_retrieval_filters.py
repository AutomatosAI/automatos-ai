"""PRD-157 S1 — centralized retrieval filter builder.

Two layers:

* pure unit tests for ``build_retrieval_filters`` / ``RetrievalFilters`` /
  ``scope_where_clause`` / ``allowed_document_ids`` (run anywhere, no DB);
* a DB-backed **tenancy matrix** that seeds ``documents`` with varied
  ``team_access`` and asserts visibility through ``allowed_document_ids`` — the
  matrix the PRD asks to extend to every path through the builder.
"""

from __future__ import annotations

import uuid

import pytest

from modules.rag.retrieval_filters import (
    RetrievalFilters,
    RetrievalScopeError,
    allowed_document_ids,
    build_retrieval_filters,
    scope_where_clause,
    SCOPE_WHERE_TEAM,
    SCOPE_WHERE_WORKSPACE_ONLY,
)


class _Agent:
    """Stand-in for the SQLAlchemy Agent row (has .workspace_id / .team)."""

    def __init__(self, workspace_id=None, team=None):
        self.workspace_id = workspace_id
        self.team = team


# --------------------------------------------------------------------------- #
# build_retrieval_filters — derivation, precedence, normalization, fail-closed
# --------------------------------------------------------------------------- #

class TestBuildRetrievalFilters:
    def test_explicit_workspace_and_team(self):
        f = build_retrieval_filters(workspace_id="ws-1", team="Support")
        assert f.workspace_id == "ws-1"
        assert f.team_terms == ["support"]  # normalized
        assert f.team == "support"
        assert f.has_team_restriction is True

    def test_team_normalization_collapses_case_and_whitespace(self):
        a = build_retrieval_filters(workspace_id="ws", team="Support")
        b = build_retrieval_filters(workspace_id="ws", team="  support ")
        assert a.team_terms == b.team_terms == ["support"]

    def test_no_team_means_no_restriction(self):
        f = build_retrieval_filters(workspace_id="ws")
        assert f.team_terms == []
        assert f.has_team_restriction is False
        assert f.team is None

    def test_derives_from_agent(self):
        agent = _Agent(workspace_id="ws-agent", team="Sales")
        f = build_retrieval_filters(agent=agent)
        assert f.workspace_id == "ws-agent"
        assert f.team_terms == ["sales"]

    def test_explicit_args_win_over_agent(self):
        agent = _Agent(workspace_id="ws-agent", team="sales")
        f = build_retrieval_filters(agent=agent, workspace_id="ws-explicit", team="support")
        assert f.workspace_id == "ws-explicit"
        assert f.team_terms == ["support"]

    def test_derives_from_context_dict(self):
        f = build_retrieval_filters(context={"workspace_id": "ws-ctx", "team": "Ops"})
        assert f.workspace_id == "ws-ctx"
        assert f.team_terms == ["ops"]

    def test_multiple_teams(self):
        f = build_retrieval_filters(workspace_id="ws", teams=["Support", "sales", "", "  "])
        assert f.team_terms == ["support", "sales"]  # blanks dropped, normalized

    def test_fail_closed_when_no_workspace(self):
        with pytest.raises(RetrievalScopeError):
            build_retrieval_filters(team="support")

    def test_fail_closed_when_agent_has_no_workspace(self):
        with pytest.raises(RetrievalScopeError):
            build_retrieval_filters(agent=_Agent(workspace_id=None, team="support"))

    def test_require_workspace_false_allows_unscoped_diag(self):
        f = build_retrieval_filters(require_workspace=False)
        assert f.workspace_id == ""
        assert f.team_terms == []

    def test_uuid_workspace_coerced_to_str(self):
        wsid = uuid.uuid4()
        f = build_retrieval_filters(workspace_id=wsid)
        assert f.workspace_id == str(wsid)

    def test_sql_params_shape(self):
        f = build_retrieval_filters(workspace_id="ws", team="support")
        assert f.sql_params() == {"workspace_id": "ws", "team_terms": ["support"]}


# --------------------------------------------------------------------------- #
# scope_where_clause — predicate selection
# --------------------------------------------------------------------------- #

class TestScopeWhereClause:
    def test_team_restriction_uses_overlap_clause(self):
        f = RetrievalFilters(workspace_id="ws", team_terms=["support"])
        assert scope_where_clause(f) == SCOPE_WHERE_TEAM
        assert "team_access" in scope_where_clause(f)

    def test_no_team_uses_workspace_only_clause(self):
        f = RetrievalFilters(workspace_id="ws", team_terms=[])
        assert scope_where_clause(f) == SCOPE_WHERE_WORKSPACE_ONLY
        assert "team_access" not in scope_where_clause(f)


# --------------------------------------------------------------------------- #
# allowed_document_ids — fail-closed application (fake db, no Postgres)
# --------------------------------------------------------------------------- #

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeDB:
    """Records the executed SQL/params and returns canned rows (or raises)."""

    def __init__(self, rows=None, raise_on_execute=False):
        self.rows = rows or []
        self.raise_on_execute = raise_on_execute
        self.last_sql = None
        self.last_params = None

    def execute(self, sql, params):
        self.last_sql = str(sql)
        self.last_params = params
        if self.raise_on_execute:
            raise RuntimeError("boom")
        return _FakeResult(self.rows)


class TestAllowedDocumentIds:
    def test_returns_visible_subset(self):
        db = _FakeDB(rows=[("1",), ("3",)])
        f = build_retrieval_filters(workspace_id="ws", team="support")
        allowed = allowed_document_ids(db, ["1", "2", "3"], f)
        assert allowed == {"1", "3"}
        # the scope clause and bind params are routed through the builder
        assert "team_access" in db.last_sql
        assert db.last_params["workspace_id"] == "ws"
        assert db.last_params["team_terms"] == ["support"]

    def test_fail_closed_on_db_error(self):
        db = _FakeDB(raise_on_execute=True)
        f = build_retrieval_filters(workspace_id="ws", team="support")
        assert allowed_document_ids(db, ["1", "2"], f) == set()

    def test_empty_doc_ids_returns_empty(self):
        db = _FakeDB(rows=[("1",)])
        f = build_retrieval_filters(workspace_id="ws", team="support")
        assert allowed_document_ids(db, [], f) == set()

    def test_unset_workspace_returns_empty(self):
        db = _FakeDB(rows=[("1",)])
        f = RetrievalFilters(workspace_id="", team_terms=["support"])
        assert allowed_document_ids(db, ["1"], f) == set()

    def test_non_int_ids_ignored(self):
        db = _FakeDB(rows=[])
        f = build_retrieval_filters(workspace_id="ws", team="support")
        # all non-int -> nothing queryable -> empty
        assert allowed_document_ids(db, ["abc", None, ""], f) == set()


# --------------------------------------------------------------------------- #
# DB-backed tenancy matrix (real Postgres via the shared db_session fixture)
# --------------------------------------------------------------------------- #

# (agent_team, doc_team_access, expected_visible)
_MATRIX = [
    (None, [], True),            # no team restriction, public doc
    (None, ["sales"], True),     # no team restriction → workspace-wide
    ("support", [], True),       # team agent sees public docs
    ("support", ["support"], True),   # exact team overlap
    ("Support", ["support"], True),   # case-insensitive normalization
    ("support", ["sales"], False),    # different team → hidden
    ("support", ["sales", "support"], True),  # overlap among several
]


@pytest.mark.integration
@pytest.mark.parametrize("agent_team,doc_team_access,expected_visible", _MATRIX)
def test_tenancy_matrix(db_session, agent_team, doc_team_access, expected_visible):
    from sqlalchemy import text

    workspace_id = str(uuid.uuid4())
    # Seed one document with the matrix's team_access.
    doc_id = db_session.execute(
        text(
            """
            INSERT INTO documents (filename, workspace_id, team_access, status, upload_date)
            VALUES (:fn, :ws::uuid, :team_access, 'processed', NOW())
            RETURNING id
            """
        ),
        {"fn": "matrix.txt", "ws": workspace_id, "team_access": doc_team_access},
    ).scalar()
    db_session.flush()

    filters = build_retrieval_filters(workspace_id=workspace_id, team=agent_team)
    allowed = allowed_document_ids(db_session, [doc_id], filters)

    assert (str(doc_id) in allowed) is expected_visible


@pytest.mark.integration
def test_tenancy_matrix_isolates_other_workspace(db_session):
    """A document in another workspace is never visible, regardless of team."""
    from sqlalchemy import text

    ws_a = str(uuid.uuid4())
    ws_b = str(uuid.uuid4())
    doc_b = db_session.execute(
        text(
            """
            INSERT INTO documents (filename, workspace_id, team_access, status, upload_date)
            VALUES (:fn, :ws::uuid, '{}', 'processed', NOW())
            RETURNING id
            """
        ),
        {"fn": "other-ws.txt", "ws": ws_b},
    ).scalar()
    db_session.flush()

    filters = build_retrieval_filters(workspace_id=ws_a)  # querying as workspace A
    allowed = allowed_document_ids(db_session, [doc_b], filters)
    assert str(doc_b) not in allowed
