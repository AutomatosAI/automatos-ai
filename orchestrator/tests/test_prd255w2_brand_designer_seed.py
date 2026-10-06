"""PRD-255 Wave 2, US-011: a Brand designer, seeded per workspace.

PURE (no database):
  * the runtime by edition: a Claude Code session (cli, claude, a valid model)
    where the instance runs sessions, the API elsewhere; both pass the runtime
    validator, and neither names a host-specific field;
  * the instructions carry each rule (analyse the logo first, derive from it,
    accent sparingly, look at every rendered page, change the kit only through
    platform_update_brand_kit after the owner approves) and name the real tools;
  * one persona source: the seeded row copies the Socials package's persona;
  * Auto's seed seeds the designer, and a failing designer seed never costs
    the workspace its Auto.

@integration (the orchestrator-tests job's Postgres, one rolled-back transaction
each):
  * seeded once per workspace, idempotent, no host, the skill linked;
  * the Socials package installed afterwards reuses it, and a workspace that
    installed the package first gets no second designer;
  * a designer the owner removed is not seeded again.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402

import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.seeds.seed_auto_agent as auto_seed  # noqa: E402
import core.seeds.seed_brand_designer as designer_seed  # noqa: E402
from core.cli_runtime import (  # noqa: E402
    CONFIG_PROVIDER_KEY,
    CONFIG_RUNTIME_KEY,
    CONFIG_WORKING_DIRECTORY_KEY,
    PROVIDER_CLAUDE,
    RUNTIME_API,
    RUNTIME_CLI,
    validate_runtime_configuration,
)
from core.models.core import Agent  # noqa: E402
from core.seeds.seed_socials_package import BRAND_DESIGNER, SOCIALS_AGENTS  # noqa: E402
from tests.test_prd251w1_socials_package import (  # noqa: E402
    INSTALL_ROUTE,
    SKILL_NAMES,
    _boot,
    _client,
    _rolled_back,
    _skills_manifest,
    _workspace,
)

SOCIALS_DESIGNER = next(spec for spec in SOCIALS_AGENTS if spec["slug"] == BRAND_DESIGNER)
PERSONA = SOCIALS_DESIGNER["custom_persona_prompt"]


# ---------------------------------------------------------------------------
# 1. The runtime, by edition
# ---------------------------------------------------------------------------


def test_where_sessions_run_the_designer_is_a_claude_code_session():
    configuration = designer_seed.designer_configuration(True)
    assert configuration[CONFIG_RUNTIME_KEY] == RUNTIME_CLI
    assert configuration[CONFIG_PROVIDER_KEY] == PROVIDER_CLAUDE
    assert validate_runtime_configuration(configuration, cli_enabled=True) == []


def test_elsewhere_the_designer_is_an_api_agent_the_hosted_edition_accepts():
    configuration = designer_seed.designer_configuration(False)
    assert configuration == {CONFIG_RUNTIME_KEY: RUNTIME_API}
    assert validate_runtime_configuration(configuration, cli_enabled=False) == []


@pytest.mark.parametrize("cli_enabled", [True, False])
def test_the_designer_is_created_with_no_host(cli_enabled):
    configuration = designer_seed.designer_configuration(cli_enabled)
    assert CONFIG_WORKING_DIRECTORY_KEY not in configuration
    assert not [key for key in configuration if "host" in key]


# ---------------------------------------------------------------------------
# 2. The instructions: one persona, every rule
# ---------------------------------------------------------------------------


def test_the_seeded_row_copies_the_socials_packages_persona():
    columns = designer_seed._columns(uuid.uuid4(), 11, designer_seed.designer_configuration(False))
    assert columns["custom_persona_prompt"] == PERSONA and columns["use_custom_persona"] is True
    assert columns["name"] == SOCIALS_DESIGNER["name"] == designer_seed.BRAND_DESIGNER_NAME
    assert columns["cloned_from_id"] == 11 and columns["owner_type"] == "workspace"
    source = (_ORCH / "core" / "seeds" / "seed_brand_designer.py").read_text(encoding="utf-8")
    assert "You are the Brand Designer" not in source  # no second persona


def test_the_designer_analyses_the_logo_before_proposing_and_derives_from_it():
    assert "Before you propose anything" in PERSONA
    assert "Analyse the logo first" in PERSONA
    for quality in ("shape", "colours", "sector", "tone", "sophistication"):
        assert quality in PERSONA
    assert "Derive everything from the logo" in PERSONA


def test_the_designer_uses_the_accent_sparingly_and_looks_at_every_output():
    assert "Use the accent sparingly" in PERSONA and 'accent_use stays "sparing"' in PERSONA
    assert "Look at every output" in PERSONA and "look at its rendered page" in PERSONA


def test_the_kit_changes_only_through_the_update_tool_after_the_owner_approves():
    assert "Change the kit only through platform_update_brand_kit" in PERSONA
    assert "only after the owner approves that proposal" in PERSONA
    assert "Any answer other than Approve" in PERSONA


@pytest.mark.parametrize("tool", [
    "platform_get_brand_kit", "render_preview", "platform_render_preview", "platform_ask_human",
    "platform_update_brand_kit", "create_template", "update_template", "generate_document",
])
def test_the_instructions_name_the_real_tools(tool):
    assert tool in PERSONA


def test_the_designer_keeps_to_document_templates_and_never_changes_a_starter_or_the_logo():
    assert "(pdf, docx, xlsx)" in PERSONA and "Social template layouts are not yours to change" in PERSONA
    assert "A starter is never changed" in PERSONA
    assert "Never generate, redraw or change the logo" in PERSONA


# ---------------------------------------------------------------------------
# 3. Auto's seed seeds the designer
# ---------------------------------------------------------------------------


def _auto_seed_fakes(monkeypatch):
    monkeypatch.setattr(auto_seed, "ensure_builtin_skill", lambda db, name: None)
    monkeypatch.setattr(
        auto_seed, "_get_default_model_config",
        lambda: {"provider": "test", "model_id": "test", "max_tokens": 4000},
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None  # no Auto yet
    return db


def test_seeding_auto_seeds_the_workspaces_designer(monkeypatch):
    db = _auto_seed_fakes(monkeypatch)
    seeded = []
    monkeypatch.setattr(auto_seed, "seed_brand_designer", lambda session, ws: seeded.append((session, ws)))
    ws = uuid.uuid4()
    auto_seed.seed_auto_agent(db, ws)
    assert seeded == [(db, ws)]
    db.begin_nested.assert_called_once_with()


def test_a_failing_designer_seed_never_costs_the_workspace_its_auto(monkeypatch, caplog):
    db = _auto_seed_fakes(monkeypatch)

    def boom(session, ws):
        raise RuntimeError("no marketplace row")

    monkeypatch.setattr(auto_seed, "seed_brand_designer", boom)
    agent = auto_seed.seed_auto_agent(db, uuid.uuid4())
    assert agent.name == "Auto"
    assert "Brand designer: seeding failed" in caplog.text


# ---------------------------------------------------------------------------
# 4. On Postgres
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            for table in ("agents", "skills", "agent_skills", "workflow_recipes", "marketplace_packages",
                          "workspace_enabled_skills", "agent_tool_assignments", "workspaces"):
                conn.execute(sa.text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Brand designer seed tests need the test database: {exc}")
    yield engine
    engine.dispose()


def _designers(session, ws):
    return session.query(Agent).filter(Agent.workspace_id == ws, Agent.name == SOCIALS_DESIGNER["name"]).all()


def _linked_skill_names(agent):
    return {skill.name for skill in agent.skills}


@pytest.mark.integration
@pytest.mark.parametrize("cli_enabled", [True, False])
def test_the_designer_is_seeded_once_per_workspace_with_the_editions_runtime(
    pg_engine, monkeypatch, tmp_path, cli_enabled,
):
    monkeypatch.setattr(designer_seed.config, "CLI_RUNTIME_ENABLED", cli_enabled, raising=False)
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))
        ws = _workspace(conn)
        first = designer_seed.seed_brand_designer(session, ws)
        again = designer_seed.seed_brand_designer(session, ws)
        session.flush()

        assert again.id == first.id and [row.id for row in _designers(session, ws)] == [first.id]
        assert first.slug == designer_seed.designer_slug(ws) and first.owner_type == "workspace"
        assert first.configuration == designer_seed.designer_configuration(cli_enabled)
        assert first.custom_persona_prompt == PERSONA
        assert "brand-kit-builder" in _linked_skill_names(first)
        other = _workspace(conn)
        assert designer_seed.seed_brand_designer(session, other).id != first.id


@pytest.mark.integration
def test_installing_the_socials_package_reuses_the_seeded_designer(pg_engine, monkeypatch, tmp_path):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))
        ws = _workspace(conn)
        seeded = designer_seed.seed_brand_designer(session, ws)
        session.commit()

        installed = _client(session, ws).post(INSTALL_ROUTE)
        assert installed.status_code == 200 and installed.json()["success"] is True, installed.text
        assert [row.id for row in _designers(session, ws)] == [seeded.id]


@pytest.mark.integration
def test_a_workspace_that_installed_the_package_first_gets_no_second_designer(pg_engine, monkeypatch, tmp_path):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))
        ws = _workspace(conn)
        installed = _client(session, ws).post(INSTALL_ROUTE)
        assert installed.status_code == 200, installed.text
        [clone] = _designers(session, ws)

        assert designer_seed.seed_brand_designer(session, ws).id == clone.id
        session.flush()
        assert [row.id for row in _designers(session, ws)] == [clone.id]


@pytest.mark.integration
def test_a_designer_the_owner_removed_is_not_seeded_again(pg_engine, tmp_path):
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))
        ws = _workspace(conn)
        seeded = designer_seed.seed_brand_designer(session, ws)
        session.flush()
        session.delete(seeded)
        session.flush()

        assert designer_seed.seed_brand_designer(session, ws) is None
        session.flush()
        assert _designers(session, ws) == []
