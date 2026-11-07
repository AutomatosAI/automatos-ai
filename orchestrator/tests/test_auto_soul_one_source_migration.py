"""auto_soul_one_source_20261107: the soul users heard moves onto the Auto row, and the setting goes.

A workspace in ``custom`` mode whose ``settings.orchestrator.custom_soul`` differs from its Auto
row (slug ``auto-<workspace_id>``, a system agent) gets that soul on the row, with
``use_custom_persona`` and ``personality_mode = 'custom'``. ``custom_soul`` then leaves every
workspace that has an Auto row. When the row already speaks with that soul in custom mode, or the
settings mode is not custom, the row is not written; the same text under a preset mode gets its
mode fixed, so Settings shows it. A workspace with no Auto row is left as it is. A second run writes nothing.

The pure rules are tested on their own; the upgrade runs against the real schema in
``db_session``'s transaction, rolled back after (rows other tests left are not asserted on).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import event, text

MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "auto_soul_one_source_20261107.py"

HEARD = "I'm Auto, from Galway: straight answers, no waffle, and I keep my word."
SHOWN = "I'm Auto, your workspace assistant."


def _migration():
    spec = importlib.util.spec_from_file_location("auto_soul_one_source_20261107", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mig = _migration()


# ── the pure rules ──────────────────────────────────────────────────────────

def test_the_migration_chains_onto_the_workspace_move_jobs_head():
    assert (mig.revision, mig.down_revision) == ("auto_soul_one_source_20261107", "prd273_w1_workspace_move_jobs")


@pytest.mark.parametrize("orchestrator, soul", [
    ({"personality_mode": "custom", "custom_soul": HEARD}, HEARD),
    ({"personality_mode": "custom", "custom_soul": "   "}, None),   # the chat fell back to a preset
    ({"personality_mode": "custom"}, None),
    ({"personality_mode": "friendly", "custom_soul": HEARD}, None),  # never what the chat used
    ({"custom_soul": HEARD}, None),
    ("not a mapping", None),
    (None, None),
])
def test_only_a_custom_modes_non_empty_soul_is_copied(orchestrator, soul):
    assert mig.soul_to_copy(orchestrator) == soul


def test_a_row_speaks_with_the_soul_only_with_the_same_text_in_use_in_custom_mode():
    custom = {"personality_mode": "custom"}
    assert mig.row_speaks_with(HEARD, f"  {HEARD}\n", True, custom)
    assert not mig.row_speaks_with(HEARD, SHOWN, True, custom)
    assert not mig.row_speaks_with(HEARD, HEARD, False, custom)
    assert not mig.row_speaks_with(HEARD, HEARD, None, custom)
    assert not mig.row_speaks_with(HEARD, None, True, custom)
    # The same text under a preset mode: Settings would show the preset, so the mode is written.
    assert not mig.row_speaks_with(HEARD, HEARD, True, {"personality_mode": "friendly"})
    assert not mig.row_speaks_with(HEARD, HEARD, True, None)


def test_the_setting_is_removed_into_a_new_dict_and_every_other_key_kept():
    settings = {"brand_kit": {"tone": "warm"}, "orchestrator": {"custom_soul": HEARD, "thinking_level": "high"}}

    cleaned = mig.without_soul(settings)

    assert cleaned == {"brand_kit": {"tone": "warm"}, "orchestrator": {"thinking_level": "high"}}
    assert settings["orchestrator"]["custom_soul"] == HEARD  # the input is untouched
    assert mig.without_soul({"orchestrator": {"thinking_level": "high"}}) is None
    assert mig.without_soul(None) is None


def test_the_mode_becomes_custom_and_the_other_keys_stay():
    configuration = {"personality_mode": "friendly", "thinking_level": "high"}

    assert mig.with_custom_mode(configuration) == {"personality_mode": "custom", "thinking_level": "high"}
    assert configuration["personality_mode"] == "friendly"
    assert mig.with_custom_mode(None) == {"personality_mode": "custom"}


# ── the upgrade, against the real schema ────────────────────────────────────

def _set_settings(db, ws, settings):
    db.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:id AS uuid)"),
               {"s": json.dumps(settings), "id": ws})


def _auto(db, ws, prompt, *, use_custom=True, mode="custom", slug=None, system=True):
    from core.models.core import Agent

    agent = Agent(name="Auto", agent_type="system", status="active", workspace_id=UUID(ws),
                  slug=slug or f"auto-{ws}", is_system_agent=system, custom_persona_prompt=prompt,
                  use_custom_persona=use_custom, configuration={"personality_mode": mode, "thinking_level": "low"},
                  model_config={})
    db.add(agent)
    db.flush()
    return agent.id


@pytest.fixture
def world(db_session, seed_workspace, monkeypatch):
    """Seven workspaces: drifted, equal, equal text under a preset mode, preset mode, no Auto row,
    only a namesake row, no setting. ``run()`` runs the upgrade on db_session's connection."""
    ws = {key: seed_workspace(name=f"soul-{key}") for key in
          ("drifted", "equal", "same_text", "preset", "no_auto", "namesake", "untouched")}
    _set_settings(db_session, ws["drifted"], {"orchestrator": {"personality_mode": "custom", "custom_soul": HEARD,
                                                               "thinking_level": "high"}, "brand_kit": {"a": 1}})
    _set_settings(db_session, ws["equal"], {"orchestrator": {"personality_mode": "custom", "custom_soul": HEARD}})
    _set_settings(db_session, ws["same_text"], {"orchestrator": {"personality_mode": "custom", "custom_soul": HEARD}})
    _set_settings(db_session, ws["preset"], {"orchestrator": {"personality_mode": "friendly", "custom_soul": HEARD}})
    _set_settings(db_session, ws["no_auto"], {"orchestrator": {"personality_mode": "custom", "custom_soul": HEARD}})
    _set_settings(db_session, ws["namesake"], {"orchestrator": {"personality_mode": "custom", "custom_soul": HEARD}})
    _set_settings(db_session, ws["untouched"], {"orchestrator": {"personality_mode": "custom"}})
    rows = {
        "drifted": _auto(db_session, ws["drifted"], SHOWN, mode="friendly"),
        "equal": _auto(db_session, ws["equal"], HEARD),
        "same_text": _auto(db_session, ws["same_text"], HEARD, mode="friendly"),
        "preset": _auto(db_session, ws["preset"], SHOWN, mode="friendly"),
        "namesake": _auto(db_session, ws["namesake"], SHOWN, mode="friendly", system=False),
        "untouched": _auto(db_session, ws["untouched"], SHOWN),
    }

    def run():
        monkeypatch.setattr(mig, "op", SimpleNamespace(get_bind=db_session.connection))
        mig.upgrade()
        db_session.expire_all()

    return db_session, ws, rows, run


def _row(db, agent_id):
    from core.models.core import Agent

    agent = db.get(Agent, agent_id)
    return agent.custom_persona_prompt, agent.use_custom_persona, (agent.configuration or {}).get("personality_mode")


def _settings(db, ws):
    return db.execute(text("SELECT settings FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws}).scalar() or {}


def _orchestrator(db, ws):
    return _settings(db, ws).get("orchestrator")


def _configuration(db, agent_id):
    from core.models.core import Agent

    return db.get(Agent, agent_id).configuration


def test_a_drifted_soul_moves_onto_the_row_and_the_setting_goes(world):
    db, ws, rows, run = world

    run()

    assert _row(db, rows["drifted"]) == (HEARD, True, "custom")
    assert _configuration(db, rows["drifted"]) == {"personality_mode": "custom", "thinking_level": "low"}
    assert _settings(db, ws["drifted"]) == {"orchestrator": {"personality_mode": "custom", "thinking_level": "high"},
                                            "brand_kit": {"a": 1}}


def test_the_same_text_under_a_preset_mode_is_marked_custom_so_settings_shows_it(world):
    db, ws, rows, run = world

    run()

    assert _row(db, rows["same_text"]) == (HEARD, True, "custom")
    assert _orchestrator(db, ws["same_text"]) == {"personality_mode": "custom"}


def test_an_equal_soul_or_a_preset_mode_leaves_the_row_and_drops_the_setting(world):
    db, ws, rows, run = world

    run()

    assert _row(db, rows["equal"]) == (HEARD, True, "custom")
    assert _row(db, rows["preset"]) == (SHOWN, True, "friendly")   # the chat never used that setting
    assert _orchestrator(db, ws["equal"]) == {"personality_mode": "custom"}
    assert _orchestrator(db, ws["preset"]) == {"personality_mode": "friendly"}


def test_a_workspace_without_an_auto_row_is_left_alone(world):
    db, ws, rows, run = world

    run()

    for key in ("no_auto", "namesake"):  # a user agent on the slug is not Auto
        assert _orchestrator(db, ws[key]) == {"personality_mode": "custom", "custom_soul": HEARD}
    assert _row(db, rows["namesake"]) == (SHOWN, True, "friendly")
    assert _row(db, rows["untouched"]) == (SHOWN, True, "custom")
    assert _orchestrator(db, ws["untouched"]) == {"personality_mode": "custom"}


def test_a_second_run_writes_nothing(world):
    db, ws, rows, run = world
    run()
    before = {key: _row(db, agent_id) for key, agent_id in rows.items()}
    written = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE")):
            written.append(statement)

    connection = db.connection()
    event.listen(connection, "before_cursor_execute", record)
    try:
        run()
    finally:
        event.remove(connection, "before_cursor_execute", record)

    assert written == []
    assert {key: _row(db, agent_id) for key, agent_id in rows.items()} == before
