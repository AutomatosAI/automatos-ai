"""The deploy itself moves Auto to Haiku 5.5, with no manual step (Gerard, 19 Oct 2026).

"I do not understand why you are giving me these to run": the Alembic data migration
auto_haiku_default_20261019 sets the platform default (system_settings
``orchestrator_llm``) to ``anthropic`` / ``claude-haiku-5-5`` and moves the Auto rows
still on the platform default to the value it stored. On the hosted edition the
setting is always replaced (production held one tenant's ``openai/gpt-5.5``); on the
local edition a self-hoster's own choice is kept, and their seed rows move to it.
Rows a workspace chose and non-Auto agents are never touched, and a second run
changes nothing.

The pure rules are tested on their own; the upgrade runs against the real schema in
``db_session``'s transaction, rolled back after (rows other tests left are not
asserted on).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import event, text

from core.llm import auto_model_sync as sync

MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "auto_haiku_default_20261019.py"

HAIKU = ("anthropic", "claude-haiku-5-5")
GPT55 = ("openrouter", "openai/gpt-5.5")
OLD_SHIPPED = ("openrouter", "google/gemini-2.5-flash")
GEMINI_SEED = {"provider": "openrouter", "model_id": "google/gemini-2.5-flash", "temperature": 0.7, "max_tokens": 4000}
GPT4O_SEED = {"provider": "openrouter", "model_id": "openai/gpt-4o", "temperature": 0.7, "max_tokens": 4000,
              "fallback_model_id": None}
CHOSEN = {"provider": "openrouter", "model_id": "openai/gpt-5.5", "temperature": 0.2, "max_tokens": 4000}
WORKSPACE_GEMINI = {**GEMINI_SEED, "source": "workspace_choice"}


def _migration():
    spec = importlib.util.spec_from_file_location("auto_haiku_default_20261019", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mig = _migration()


# ── the pure rules ──────────────────────────────────────────────────────────

def test_the_migration_chains_onto_the_harness_runs_head():
    assert (mig.revision, mig.down_revision) == ("auto_haiku_default_20261019", "prd263_harness_runs")
    assert mig.TARGET == HAIKU


def test_the_edition_is_configs_own_never_a_second_reading_of_the_environment():
    """AGENTS.md: settings come from config.py only. The migration reads
    config.AUTH_EDITION (unset or unknown is saas there), with no noqa carve-out."""
    source = MIGRATION.read_text(encoding="utf-8")
    assert "config.AUTH_EDITION" in source
    assert "os.environ" not in source and "os.getenv" not in source and "TID251" not in source


@pytest.mark.parametrize("auth_edition, stored, written", [
    ("saas", GPT55, HAIKU),                  # production: one tenant's save is replaced
    ("saas", OLD_SHIPPED, HAIKU),
    ("saas", (None, None), HAIKU),
    ("saas", HAIKU, None),                   # already there: nothing written
    ("local", GPT55, None),                  # a self-hoster's own choice survives
    ("local", OLD_SHIPPED, HAIKU),           # still the old shipped pair
    ("local", (None, None), HAIKU),
    ("local", ("openrouter", None), HAIKU),  # half a pair is no choice: default_route ignores it
    ("local", HAIKU, None),
])
def test_the_platform_default_policy(auth_edition, stored, written):
    assert mig.platform_default_to_write(auth_edition, stored) == written


def _route(model_config):
    return model_config["provider"], model_config["model_id"]


def test_a_seed_row_moves_and_keeps_every_other_key():
    moved = mig.restamped(dict(GPT4O_SEED), HAIKU)
    assert _route(moved) == HAIKU
    assert moved["source"] == "platform_default" and moved["platform_route"] == {"provider": HAIKU[0],
                                                                                 "model_id": HAIKU[1]}
    assert {k: moved[k] for k in ("temperature", "max_tokens", "fallback_model_id")} == {
        "temperature": 0.7, "max_tokens": 4000, "fallback_model_id": None}
    assert sync.is_inheriting(moved)


@pytest.mark.parametrize("model_config", [
    CHOSEN, WORKSPACE_GEMINI, None, "not json", {},
    {**sync.inheriting_model_config(dict(GEMINI_SEED)), "model_id": "openai/gpt-5.5"},  # marked, then re-pointed
])
def test_a_chosen_or_unreadable_row_stays(model_config):
    assert mig.restamped(model_config, HAIKU) is None


def test_a_row_already_on_the_target_stays():
    assert mig.restamped(mig.restamped(dict(GEMINI_SEED), HAIKU), HAIKU) is None


def test_the_migration_reads_rows_as_auto_model_sync_does():
    """The migration carries its own copy of the rules (it imports no application code)."""
    assert mig.KNOWN_SEED_ROUTES == sync.KNOWN_SEED_ROUTES
    samples = [GEMINI_SEED, GPT4O_SEED, CHOSEN, WORKSPACE_GEMINI, {},
               sync.inheriting_model_config({**GEMINI_SEED, "provider": HAIKU[0], "model_id": HAIKU[1]}),
               {**sync.inheriting_model_config(dict(GEMINI_SEED)), "model_id": "x"}]
    assert [mig.is_inheriting(c) for c in samples] == [sync.is_inheriting(c) for c in samples]


# ── the upgrade, against the real schema ────────────────────────────────────

def _set_stored(db, route):
    db.execute(text("DELETE FROM system_settings WHERE category = 'orchestrator_llm' AND key IN "
                    "('provider', 'model', 'llm_provider', 'llm_model')"))
    for key, value in zip(("provider", "model"), route or ()):
        db.execute(text(
            "INSERT INTO system_settings (category, key, value, value_type, is_required, default_value) "
            "VALUES ('orchestrator_llm', :key, :value, 'string', true, :value)"), {"key": key, "value": value})


def _shipped_defaults(db):
    rows = db.execute(text(
        "SELECT key, default_value FROM system_settings WHERE category = 'orchestrator_llm' "
        "AND key IN ('provider', 'model')"
    )).fetchall()
    values = {key: value for key, value in rows}
    return values.get("provider"), values.get("model")


def _stored(db):
    rows = db.execute(text(
        "SELECT key, value FROM system_settings WHERE category = 'orchestrator_llm' AND key IN ('provider', 'model')"
    )).fetchall()
    values = {key: value for key, value in rows}
    return values.get("provider"), values.get("model"), len(rows)


def _agent(db, model_config, *, ws=None, slug=None, system=False):
    from core.models.core import Agent

    agent = Agent(name="Auto" if system else "Barista", agent_type="system" if system else "custom",
                  status="active", workspace_id=UUID(ws) if ws else None, slug=slug, is_system_agent=system,
                  model_config=dict(model_config))
    db.add(agent)
    db.flush()
    return agent.id


@pytest.fixture
def world(db_session, seed_workspace, monkeypatch):
    """Four workspaces' Auto rows (two seed pairs, one chosen before the markers, one
    marked as the workspace's choice), a workspace agent on a seed pair and a global
    ``auto-…`` agent; ``run(edition)`` runs the upgrade on db_session's connection."""
    ws = [seed_workspace(name=f"haiku-{i}") for i in range(5)]
    rows = {
        "gemini": _agent(db_session, GEMINI_SEED, ws=ws[0], slug=f"auto-{ws[0]}", system=True),
        "gpt4o": _agent(db_session, GPT4O_SEED, ws=ws[1], slug=f"auto-{ws[1]}", system=True),
        "chosen": _agent(db_session, CHOSEN, ws=ws[2], slug=f"auto-{ws[2]}", system=True),
        "workspace_choice": _agent(db_session, WORKSPACE_GEMINI, ws=ws[3], slug=f"auto-{ws[3]}", system=True),
        "worker": _agent(db_session, GEMINI_SEED, ws=ws[4], slug=f"barista-{ws[4]}"),
        "global": _agent(db_session, GEMINI_SEED, slug=f"auto-cto-{ws[4]}", system=True),
    }

    def run(edition):
        from config import config

        monkeypatch.setattr(config, "AUTH_EDITION", edition)
        monkeypatch.setattr(mig, "op", SimpleNamespace(get_bind=db_session.connection))
        mig.upgrade()
        db_session.expire_all()

    return db_session, rows, run


def _config(db, agent_id):
    from core.models.core import Agent

    return db.get(Agent, agent_id).model_config


def _untouched(db, rows):
    return {
        "chosen": _config(db, rows["chosen"]) == CHOSEN,
        "workspace_choice": _config(db, rows["workspace_choice"]) == WORKSPACE_GEMINI,
        "worker": _config(db, rows["worker"]) == GEMINI_SEED,
        "global": _config(db, rows["global"]) == GEMINI_SEED,
    }


def test_saas_sets_haiku_and_moves_the_seed_rows(world):
    db, rows, run = world
    _set_stored(db, GPT55)

    run("saas")

    assert _stored(db) == (*HAIKU, 2)
    assert _shipped_defaults(db) == HAIKU   # Reset to defaults restores Haiku, not the old pair
    for key in ("gemini", "gpt4o"):
        moved = _config(db, rows[key])
        assert _route(moved) == HAIKU and moved["max_tokens"] == 4000 and sync.is_inheriting(moved)
    assert all(_untouched(db, rows).values()), _untouched(db, rows)


def test_a_second_run_writes_nothing(world):
    db, rows, run = world
    _set_stored(db, GPT55)
    run("saas")
    before = {key: _config(db, agent_id) for key, agent_id in rows.items()}
    written = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE")):
            written.append(statement)

    connection = db.connection()
    event.listen(connection, "before_cursor_execute", record)
    try:
        run("saas")
    finally:
        event.remove(connection, "before_cursor_execute", record)

    assert written == []
    assert {key: _config(db, agent_id) for key, agent_id in rows.items()} == before
    assert _stored(db) == (*HAIKU, 2)


def test_local_keeps_a_self_hosters_choice_and_moves_the_seed_rows_to_it(world):
    db, rows, run = world
    _set_stored(db, GPT55)

    run("local")

    assert _stored(db) == (*GPT55, 2)
    assert _shipped_defaults(db) == HAIKU   # the choice is kept; the shipped default is Haiku
    assert _route(_config(db, rows["gemini"])) == GPT55 and sync.is_inheriting(_config(db, rows["gemini"]))
    assert _route(_config(db, rows["gpt4o"])) == GPT55
    assert all(_untouched(db, rows).values()), _untouched(db, rows)


def test_local_on_the_old_shipped_pair_moves_to_haiku(world):
    db, rows, run = world
    _set_stored(db, OLD_SHIPPED)

    run("local")

    assert _stored(db) == (*HAIKU, 2)
    assert _route(_config(db, rows["gemini"])) == HAIKU


def test_missing_setting_rows_are_inserted(world):
    db, rows, run = world
    _set_stored(db, None)

    run("saas")

    assert _stored(db) == (*HAIKU, 2)
    assert _route(_config(db, rows["gpt4o"])) == HAIKU
