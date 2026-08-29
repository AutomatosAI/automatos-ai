"""PRD-233 S3 — the local-edition first-run seed (the two-minute demo).

Pure tests (no DB) pin the gate (saas ⇒ the session is never touched), the
fingerprint mechanics (current ⇒ untouched / prior ⇒ refreshed / anything else
⇒ never overwritten), the demo Playbook's validity through the SAME validators
the create route runs, the native-tools-only invariant, content hygiene (no
personal data), and the load_seed_data wiring.

@integration tests run the real seed against the database with a THROWAWAY
workspace id monkeypatched into config — never the live default workspace —
and sweep every row they made: twice ⇒ identical state, rows workspace-scoped
and listable exactly as the API lists them, user edits preserved, prior-version
rows refreshed, deleted rows not resurrected. Skip without Postgres; CI runs them.
"""
from __future__ import annotations

import json
import pathlib
import re
import uuid
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import config
import core.seeds.seed_local_first_run as seed_mod
from core.models.core import WorkflowTemplate
from core.seeds.seed_local_first_run import (
    LEDGER_KEY,
    PLAYBOOK,
    PLAYBOOK_TEMPLATE_ID,
    ROSTER,
    WELCOME_CONTENT,
    WELCOME_SLUG,
    _base_slug,
    _slug_candidates,
    WELCOME_TITLE,
    agent_fingerprint,
    current_fingerprints,
    playbook_fingerprint,
    post_fingerprint,
    seed_local_first_run,
    validate_playbook,
)

_ORCH = pathlib.Path(__file__).resolve().parents[1]
_SEED_SRC = pathlib.Path(seed_mod.__file__).read_text(encoding="utf-8")
ROSTER_SLUGS = [spec["slug"] for spec in ROSTER]


def _fake_ids() -> dict[str, int]:
    return {slug: 100 + index for index, slug in enumerate(ROSTER_SLUGS)}


# --------------------------------------------------------------------------- #
# Gate — SaaS invariant: the seed never runs outside the local edition
# --------------------------------------------------------------------------- #


def test_saas_is_a_no_op_that_never_touches_the_session(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", str(uuid.uuid4()))
    db = MagicMock()
    assert seed_local_first_run(db) == {"skipped": "not-local"}
    db.execute.assert_not_called()
    db.query.assert_not_called()
    db.add.assert_not_called()


def test_local_without_a_default_workspace_is_a_no_op(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", "  ")
    db = MagicMock()
    assert seed_local_first_run(db) == {"skipped": "no-default-workspace"}
    db.execute.assert_not_called()
    db.add.assert_not_called()


# --------------------------------------------------------------------------- #
# Refresh mechanics — the hash pattern (current / prior / customized)
# --------------------------------------------------------------------------- #


def test_agent_row_current_prior_and_customized_outcomes(monkeypatch):
    spec = ROSTER[0]
    columns = seed_mod._agent_columns(spec, uuid.uuid4(), None)
    current = agent_fingerprint(SimpleNamespace(**columns))
    row = SimpleNamespace(**columns)

    assert seed_mod._refresh(row, columns, seed_mod._AGENT_CONTENT_FIELDS, agent_fingerprint, current) == "current"

    row.description = "my own words about this agent"
    assert seed_mod._refresh(row, columns, seed_mod._AGENT_CONTENT_FIELDS, agent_fingerprint, current) == "customized"
    assert row.description == "my own words about this agent"  # never overwritten

    monkeypatch.setattr(seed_mod, "PRIOR_SEED_FINGERPRINTS", frozenset({agent_fingerprint(row)}))
    assert seed_mod._refresh(row, columns, seed_mod._AGENT_CONTENT_FIELDS, agent_fingerprint, current) == "refreshed"
    assert row.description == spec["description"]
    # a second pass over the refreshed row is a no-op
    assert seed_mod._refresh(row, columns, seed_mod._AGENT_CONTENT_FIELDS, agent_fingerprint, current) == "current"


def test_playbook_fingerprint_is_stable_across_re_created_agent_ids():
    """Agent ids are normalised to roster slugs, so a roster agent re-created
    with a new id still hashes as seed content (and can be re-pointed)."""
    ws = uuid.uuid4()
    ids_a = _fake_ids()
    ids_b = {slug: agent_id + 1000 for slug, agent_id in ids_a.items()}
    row_a = SimpleNamespace(**seed_mod._playbook_columns(PLAYBOOK, ws, None, ids_a))
    row_b = SimpleNamespace(**seed_mod._playbook_columns(PLAYBOOK, ws, None, ids_b))
    fp_a = playbook_fingerprint(row_a, {v: k for k, v in ids_a.items()})
    fp_b = playbook_fingerprint(row_b, {v: k for k, v in ids_b.items()})
    assert fp_a == fp_b
    # an edited prompt is a different fingerprint
    row_a.steps[0]["prompt_template"] = "do something else"
    assert playbook_fingerprint(row_a, {v: k for k, v in ids_a.items()}) != fp_b


def test_assign_only_writes_when_different_and_rebuilds_json_values():
    row = SimpleNamespace(tags=["a"], description="x")
    assert seed_mod._assign(row, "description", "x") is False
    original = ["a"]
    assert seed_mod._assign(row, "tags", original) is False  # equal ⇒ untouched
    assert seed_mod._assign(row, "tags", ["a", "b"]) is True
    assert row.tags == ["a", "b"] and row.tags is not original


def test_ledger_is_rebuilt_never_mutated():
    ws = SimpleNamespace(settings={"other": 1})
    before = ws.settings
    ctx = SimpleNamespace(ledger={"agents": {"local-writer"}, "playbooks": set(), "deliverables": set()})
    monkeypatch_present = {"agents": {"local-researcher"}, "playbooks": {PLAYBOOK_TEMPLATE_ID}, "deliverables": set()}
    original = seed_mod._present_identities
    seed_mod._present_identities = lambda _ctx: monkeypatch_present
    try:
        assert seed_mod._record_ledger(ws, ctx) is True
    finally:
        seed_mod._present_identities = original
    assert ws.settings is not before and before == {"other": 1}
    assert ws.settings["other"] == 1
    assert ws.settings[LEDGER_KEY]["seeded"] == {
        "agents": ["local-researcher", "local-writer"],
        "playbooks": [PLAYBOOK_TEMPLATE_ID],
        "deliverables": [],
    }
    assert seed_mod._read_ledger(ws) == {
        "agents": {"local-researcher", "local-writer"},
        "playbooks": {PLAYBOOK_TEMPLATE_ID},
        "deliverables": set(),
    }


def test_current_fingerprints_cover_every_seeded_identity():
    fps = current_fingerprints()
    assert set(fps) == set(ROSTER_SLUGS) | {PLAYBOOK_TEMPLATE_ID, WELCOME_SLUG}
    assert all(re.fullmatch(r"[0-9a-f]{64}", fp) for fp in fps.values())
    assert len(set(fps.values())) == len(fps)


# --------------------------------------------------------------------------- #
# The demo Playbook — valid through the create route's own validators,
# runnable with only an LLM key
# --------------------------------------------------------------------------- #


def test_demo_playbook_passes_the_create_routes_validators():
    ids = _fake_ids()
    columns = seed_mod._playbook_columns(PLAYBOOK, uuid.uuid4(), None, ids)
    recipe = WorkflowTemplate(**columns)  # ORM object, no session needed
    validate_playbook(recipe)  # raises on any validator failure
    for field in ("template_id", "name", "description", "template_definition", "steps"):
        assert columns[field], f"create route requires {field}"
    assert columns["schedule_config"] == {"type": "manual"}
    assert columns["execution_config"]["mode"] == "sequential"


def test_demo_playbook_steps_bind_only_roster_agents_and_the_declared_input():
    ids = _fake_ids()
    columns = seed_mod._playbook_columns(PLAYBOOK, uuid.uuid4(), None, ids)
    assert [s["order"] for s in columns["steps"]] == [1, 2, 3]
    for step in columns["steps"]:
        assert step["agent_id"] in ids.values()
        assert "agent" not in step and "agent_slug" not in step  # GET /{id} enrichment key stays free
        assert "{input.topic}" in step["prompt_template"]
        assert step["error_handling"] == "stop"
    assert set(PLAYBOOK["inputs"]) == {"topic"}
    assert PLAYBOOK["inputs"]["topic"]["default"]
    # LLM-only: nothing to install, no integration named as a requirement
    assert columns["required_tools"] == []
    assert columns["recommended_agents"] == []
    assert columns["is_public"] is True  # the list route defaults to is_public=True


def test_roster_is_native_tools_only_and_runtime_llm_resolution():
    for spec in ROSTER:
        columns = seed_mod._agent_columns(spec, uuid.uuid4(), None)
        assert columns["is_system_agent"] is False  # visible on the Agents page
        assert columns["model_config"] is None  # resolves the workspace's key at runtime
        assert columns["configuration"] == {}  # no Composio app ids / tool bindings
        assert columns["use_custom_persona"] is True and columns["custom_persona_prompt"]
        assert spec["slug"].startswith("local-")


# --------------------------------------------------------------------------- #
# Source guards — hygiene, config discipline, wiring
# --------------------------------------------------------------------------- #


def test_seed_module_reads_config_only_and_carries_no_personal_data():
    assert "os.getenv" not in _SEED_SRC
    content = json.dumps([ROSTER, PLAYBOOK, WELCOME_CONTENT, WELCOME_TITLE])
    assert "@" not in content, "seed content must not carry email addresses"
    assert "gerard" not in _SEED_SRC.lower()
    assert "clerk" not in content.lower()


def test_welcome_post_is_the_db_resident_outputs_member():
    columns = seed_mod._post_columns(uuid.uuid4(), None)
    assert columns["file_path"] is None  # content served from blog_posts.content — no worker fetch
    assert columns["status"] == "published"
    assert columns["content"] == WELCOME_CONTENT
    assert len(columns["excerpt"]) <= 500
    assert "Settings" in WELCOME_CONTENT and "Two-minute brief" in WELCOME_CONTENT


def test_load_seed_data_calls_the_seed_gated_inside_its_own_try_except():
    src = (_ORCH / "core" / "database" / "load_seed_data.py").read_text(encoding="utf-8")
    at = src.index("seed_local_first_run(db)")
    before = src[max(0, at - 500):at]
    after = src[at:at + 400]
    assert "from core.seeds.seed_local_first_run import seed_local_first_run" in before
    assert "try:" in before
    assert "except Exception" in after, "a seed failure must not abort the other seeders"
    # the gate lives in the seed itself (config.AUTH_EDITION), not in the loader
    assert 'config.AUTH_EDITION != "local"' in _SEED_SRC


# --------------------------------------------------------------------------- #
# @integration — the real seed against the database, throwaway workspace only
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PRD-233 S3 seed integration test needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def local_edition(monkeypatch, engine, new_session):
    """One throwaway local workspace per test: config patched to local + a
    fresh DEFAULT_WORKSPACE_ID (NEVER the live default workspace). Teardown
    sweeps every row the seed made for that workspace, and the operator user
    only if this test created it."""
    from sqlalchemy import text

    ws_id = uuid.uuid4()
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", str(ws_id))
    probe = new_session()
    operator_existed = probe.execute(
        text("SELECT 1 FROM users WHERE email = :e"), {"e": config.LOCAL_OPERATOR_EMAIL}
    ).fetchone() is not None
    probe.close()

    yield ws_id

    sweep = new_session.sweep()
    try:
        params = {"w": str(ws_id)}
        sweep.execute(text(
            "DELETE FROM agent_skills WHERE agent_id IN "
            "(SELECT id FROM agents WHERE workspace_id = CAST(:w AS uuid))"
        ), params)
        sweep.execute(text("DELETE FROM agents WHERE workspace_id = CAST(:w AS uuid)"), params)
        # blog_posts + workflow_recipes cascade from the workspace
        sweep.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), params)
        if not operator_existed:
            sweep.execute(text("DELETE FROM users WHERE email = :e"), {"e": config.LOCAL_OPERATOR_EMAIL})
        sweep.commit()
    finally:
        sweep.close()


def _snapshot(session, ws_id: uuid.UUID) -> dict:
    """Every column of every seeded row (updated_at included) — the identical-
    state oracle for the run-twice test."""
    from sqlalchemy import text

    params = {"w": str(ws_id)}

    def rows(sql: str):
        return [tuple(r) for r in session.execute(text(sql), params).fetchall()]

    return {
        "agents": rows("SELECT * FROM agents WHERE workspace_id = CAST(:w AS uuid) ORDER BY slug"),
        "playbooks": rows("SELECT * FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid) ORDER BY template_id"),
        "posts": rows("SELECT * FROM blog_posts WHERE workspace_id = CAST(:w AS uuid) ORDER BY slug"),
        "workspace": rows("SELECT name, slug, settings, is_active, updated_at FROM workspaces WHERE id = CAST(:w AS uuid)"),
    }


def _run(session) -> dict:
    out = seed_local_first_run(session)
    session.commit()
    return out


@pytest.mark.integration
def test_seed_twice_yields_identical_state(local_edition, new_session):
    s = new_session()
    first = _run(s)
    assert first["workspace_id"] == config.DEFAULT_WORKSPACE_ID == str(local_edition)
    assert first["workspace"] == "created"
    assert first["operator_user"] in ("created", "present")
    assert first["agents"] == {"created": len(ROSTER)}
    assert first["playbooks"] == {"created": 1}
    assert first["deliverables"] == {"created": 1}
    assert first["ledger_updated"] is True

    before = _snapshot(s, local_edition)
    second = _run(s)
    assert second["agents"] == {"current": len(ROSTER)}
    assert second["playbooks"] == {"current": 1}
    assert second["deliverables"] == {"current": 1}
    assert second["ledger_updated"] is False
    assert _snapshot(s, local_edition) == before


@pytest.mark.integration
def test_rows_are_workspace_scoped_attributed_and_listable_like_the_api(local_edition, new_session):
    from sqlalchemy import text

    from core.models.core import Agent, BlogPost

    s = new_session()
    outcome = _run(s)
    ws = local_edition

    operator_id = s.execute(
        text("SELECT id FROM users WHERE email = :e"), {"e": config.LOCAL_OPERATOR_EMAIL}
    ).scalar()
    assert operator_id is not None

    roster = s.query(Agent).filter(Agent.workspace_id == ws, Agent.slug.in_([c for b in ROSTER_SLUGS for c in _slug_candidates(b, ws)])).all()
    assert sorted(_base_slug(a.slug, ws) for a in roster) == sorted(ROSTER_SLUGS)
    for agent in roster:
        assert agent.is_system_agent is False and agent.owner_type == "workspace"
        assert agent.created_by_user_id == operator_id
        assert agent.model_config is None
    roster_ids = {a.id for a in roster}
    assert s.execute(
        text("SELECT count(*) FROM agent_app_assignments WHERE agent_id = ANY(:ids)"),
        {"ids": list(roster_ids)},
    ).scalar() == 0  # native tools only — no Composio bindings

    # Auto: the existing per-workspace seeder ran (reused, not duplicated)
    auto = s.query(Agent).filter(Agent.workspace_id == ws, Agent.slug == f"auto-{ws}").all()
    assert len(auto) == 1 and auto[0].id == outcome["auto_agent_id"]

    # The list route's exact filter finds the Playbook; its steps pass the
    # create route's agent check and validators.
    listed = (
        s.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.owner_type == "workspace",
            WorkflowTemplate.workspace_id == ws,
            WorkflowTemplate.is_public.is_(True),
        )
        .all()
    )
    assert [p.template_id.startswith(PLAYBOOK_TEMPLATE_ID) for p in listed] == [True]
    playbook = listed[0]
    validate_playbook(playbook)
    assert {step["agent_id"] for step in playbook.steps} <= roster_ids
    assert playbook.created_by_user_id == operator_id
    assert playbook.created_by == config.LOCAL_OPERATOR_EMAIL

    post = s.query(BlogPost).filter(BlogPost.workspace_id == ws, BlogPost.slug == WELCOME_SLUG).one()
    assert post.title == WELCOME_TITLE and post.status == "published"
    assert post.content == WELCOME_CONTENT and post.file_path is None
    assert post.author_agent_id == outcome["auto_agent_id"]

    # Surfaces in the Deliverables tab (v_workspace_outputs) when the view exists
    if s.execute(text("SELECT to_regclass('v_workspace_outputs')")).scalar():
        rows = s.execute(
            text("SELECT title, artifact_type, status FROM v_workspace_outputs WHERE workspace_id = CAST(:w AS uuid)"),
            {"w": str(ws)},
        ).fetchall()
        assert (WELCOME_TITLE, "blog_post", "published") in [tuple(r) for r in rows]


@pytest.mark.integration
def test_user_edits_are_never_overwritten(local_edition, new_session):
    from sqlalchemy import text

    from core.models.core import Agent, BlogPost

    s = new_session()
    _run(s)
    ws = local_edition

    writer = s.query(Agent).filter(Agent.workspace_id == ws, Agent.slug.in_(_slug_candidates("local-writer", ws))).one()
    writer.description = "Writes only haiku now."
    playbook = s.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == ws).one()
    edited_steps = [dict(step) for step in playbook.steps]
    edited_steps[0] = {**edited_steps[0], "prompt_template": "Research {input.topic} my way."}
    playbook.steps = edited_steps
    post = s.query(BlogPost).filter(BlogPost.workspace_id == ws, BlogPost.slug == WELCOME_SLUG).one()
    post.status = "draft"
    s.commit()

    second = _run(s)
    assert second["agents"] == {"current": len(ROSTER) - 1, "customized": 1}
    assert second["playbooks"] == {"customized": 1}
    assert second["deliverables"] == {"customized": 1}

    s.expire_all()
    assert s.query(Agent).filter(Agent.id == writer.id).one().description == "Writes only haiku now."
    assert s.query(WorkflowTemplate).filter(WorkflowTemplate.id == playbook.id).one().steps[0]["prompt_template"] == "Research {input.topic} my way."
    assert s.execute(text("SELECT status FROM blog_posts WHERE id = :id"), {"id": str(post.id)}).scalar() == "draft"


@pytest.mark.integration
def test_rows_still_on_a_prior_seed_version_are_refreshed(local_edition, new_session, monkeypatch):
    from core.models.core import Agent

    s = new_session()
    _run(s)
    ws = local_edition

    researcher = s.query(Agent).filter(Agent.workspace_id == ws, Agent.slug.in_(_slug_candidates("local-researcher", ws))).one()
    researcher.description = "An older shipped description."
    s.commit()
    older_fingerprint = agent_fingerprint(researcher)
    monkeypatch.setattr(seed_mod, "PRIOR_SEED_FINGERPRINTS", frozenset({older_fingerprint}))

    second = _run(s)
    assert second["agents"] == {"current": len(ROSTER) - 1, "refreshed": 1}
    s.expire_all()
    assert s.query(Agent).filter(Agent.id == researcher.id).one().description == ROSTER[0]["description"]
    assert _run(s)["agents"] == {"current": len(ROSTER)}


@pytest.mark.integration
def test_deleted_seed_rows_are_not_resurrected(local_edition, new_session):
    from sqlalchemy import text

    s = new_session()
    _run(s)
    # Scoped to THIS workspace: template_id is globally unique and another
    # workspace (the live default one) may own the plain id.
    s.execute(text("DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"), {"w": str(local_edition)})
    s.commit()

    second = _run(s)
    assert second["playbooks"] == {"deleted_by_user": 1}
    assert s.execute(
        text("SELECT count(*) FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"), {"w": str(local_edition)}
    ).scalar() == 0


@pytest.mark.integration
def test_saas_gate_writes_nothing_to_the_database(local_edition, new_session, monkeypatch):
    from sqlalchemy import text

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    s = new_session()
    assert _run(s) == {"skipped": "not-local"}
    assert s.execute(
        text("SELECT count(*) FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": str(local_edition)}
    ).scalar() == 0
