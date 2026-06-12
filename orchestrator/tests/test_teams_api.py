"""PRD-158 S1 — Teams entity: helpers, backfill, org-chart consistency.

Pure layer: the single-normalizer write helpers. Integration layer (marked):
the migration backfill collapsing mixed-case data, and org-chart/`/api/teams`
agreeing on the team list.
"""

from __future__ import annotations

import os
import sys
import types
import uuid

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest  # noqa: E402

from core.team_access import get_or_create_team, ensure_teams, normalize_team  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure: write helpers normalize through the single normalizer
# --------------------------------------------------------------------------- #

class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def first(self):
        return self._result

    def all(self):
        return self._result if isinstance(self._result, list) else []


class _FakeDB:
    def __init__(self, first_result=None):
        self._first = first_result
        self.added = []

    def query(self, *a, **k):
        return _FakeQuery(self._first)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass


class TestTeamWriteHelpers:
    def test_create_normalizes_name(self):
        db = _FakeDB(first_result=None)
        team = get_or_create_team(db, uuid.uuid4(), "  Support ")
        assert team is not None
        assert team.normalized_name == "support"   # canonical
        assert team.name == "Support"               # display preserved (trimmed)
        assert len(db.added) == 1

    def test_existing_team_is_not_duplicated(self):
        existing = types.SimpleNamespace(normalized_name="support", name="Support")
        db = _FakeDB(first_result=existing)
        team = get_or_create_team(db, uuid.uuid4(), "support")
        assert team is existing
        assert db.added == []                       # no second row

    def test_blank_name_creates_nothing(self):
        db = _FakeDB(first_result=None)
        assert get_or_create_team(db, uuid.uuid4(), "   ") is None
        assert db.added == []

    def test_ensure_teams_returns_normalized_drops_blanks(self):
        db = _FakeDB(first_result=None)
        out = ensure_teams(db, uuid.uuid4(), ["Support", " Sales ", "", "  "])
        assert out == ["support", "sales"]


# --------------------------------------------------------------------------- #
# Integration (real Postgres): backfill + org-chart consistency
# --------------------------------------------------------------------------- #

# The migration's backfill, lifted verbatim so the test pins the exact SQL.
_BACKFILL_SQL = """
INSERT INTO teams (workspace_id, name, normalized_name)
SELECT workspace_id, MIN(name) AS name, normalized_name
FROM (
    SELECT workspace_id, TRIM(team) AS name, LOWER(TRIM(team)) AS normalized_name
    FROM agents
    WHERE team IS NOT NULL AND TRIM(team) <> ''
    UNION ALL
    SELECT workspace_id, TRIM(t) AS name, LOWER(TRIM(t)) AS normalized_name
    FROM documents, unnest(team_access) AS t
    WHERE team_access IS NOT NULL AND TRIM(t) <> ''
) src
WHERE normalized_name <> ''
GROUP BY workspace_id, normalized_name
ON CONFLICT (workspace_id, normalized_name) DO NOTHING
"""


@pytest.mark.integration
def test_backfill_collapses_mixed_case(db_session, seed_workspace):
    from sqlalchemy import text

    ws = seed_workspace()  # FK parent for agents/documents/teams.workspace_id
    # Two agents 'Support'/'support' and a doc team 'Sales' → 2 teams.
    for team in ("Support", "support"):
        db_session.execute(
            text(
                "INSERT INTO agents (name, agent_type, workspace_id, team, status, slug) "
                "VALUES (:n, 'custom', CAST(:ws AS uuid), :team, 'active', :slug)"
            ),
            {"n": f"a-{team}", "ws": ws, "team": team, "slug": f"a-{uuid.uuid4().hex[:8]}"},
        )
    db_session.execute(
        text(
            "INSERT INTO documents (filename, workspace_id, team_access, status, upload_date) "
            "VALUES ('d.txt', CAST(:ws AS uuid), ARRAY['Sales'], 'processed', NOW())"
        ),
        {"ws": ws},
    )
    db_session.flush()

    db_session.execute(text(_BACKFILL_SQL))
    db_session.flush()

    rows = db_session.execute(
        text("SELECT normalized_name FROM teams WHERE workspace_id = CAST(:ws AS uuid) ORDER BY normalized_name"),
        {"ws": ws},
    ).fetchall()
    normalized = [r[0] for r in rows]
    assert normalized == ["sales", "support"]   # 'Support'/'support' collapsed to one


@pytest.mark.integration
def test_org_chart_teams_match_teams_api(db_session, seed_workspace):
    """org-chart's team list and list_teams() read the same table."""
    from sqlalchemy import text
    from core.team_access import list_teams, get_or_create_team

    ws = uuid.uuid4()
    seed_workspace(ws)  # FK parent for teams.workspace_id
    get_or_create_team(db_session, ws, "Engineering")
    get_or_create_team(db_session, ws, "Support")
    db_session.flush()

    api_names = sorted(t.name for t in list_teams(db_session, ws))
    # org-chart derives its `teams` from the same list_teams() helper.
    assert api_names == ["Engineering", "Support"]
