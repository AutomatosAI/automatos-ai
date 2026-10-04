"""PRD-251C Wave 1, US-C101 and US-C102 — research comes with the first plan, and comes back.

On the S0.3b API harness (SQLite), with the real package installer: only its agent cascade
is faked, so a plan's save clones the marketplace **Content bank research** row into the
workspace as the installer does. Pinned:

* the first plan's save installs the playbook once and records it
  (``settings['socials'].research_installed_at``); a second plan installs nothing, and
  another workspace has nothing;
* a workspace with a plan but no flag (made before PRD-251C) installs on its next save;
* a copy the owner deleted is not put back by a save; a copy already there (edited) is
  left as it is and only the flag is set;
* a failing installer never fails the save: nothing is installed and no flag is set;
* the installed copy is what ``installed_playbook`` finds, so research runs it;
* the content bank says why research cannot run (not set up, or its playbook removed);
* an install that meets another one in the workspace (the lock taken) installs nothing;
* Research again puts a missing playbook back and runs it (a deleted one too); it is 409,
  with why, only when it cannot: no marketplace playbook, a plan full of agents, another
  install under way;
* the weekly run never installs: it says why research is off, once a day; a trial
  workspace on the hosted edition gets no weekly run;
* whoever may save a plan may also add the agent and playbook its save installs: the role
  matrix keeps the two together, so a role that splits them fails here first.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import modules.tools.discovery.cascade_installer as ci  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from config import config  # noqa: E402
from modules.policy.roles import WorkspaceRole, workspace_has_permission  # noqa: E402
from core.models.core import RecipeExecution, WorkflowTemplate  # noqa: E402
from core.models.socials import SocialPost, SocialTopic  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.seeds.seed_socials_package import RESEARCH_PLAYBOOK_TEMPLATE_ID  # noqa: E402
from modules.socials import plan_notify  # noqa: E402
from services import package_installer, socials_plan_research  # noqa: E402
from services.agent_quota import AgentLimitReached  # noqa: E402
from services import socials_research_setup as setup  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402

api = api_harness.api
UTC = timezone.utc
NOW = datetime(2026, 10, 14, 7, 30, tzinfo=UTC)  # a Wednesday: the plans' Monday research is due
MARKETPLACE_ID = 1
EDITED_STEPS = [{"step_id": "research", "prompt_template": "The owner's own research prompt"}]


@pytest.fixture
def research(api, monkeypatch):
    """The harness with the content bank's table, the marketplace research row and the
    installer's agent cascade faked (the Director's clone is PRD-230's, tested there)."""
    engine = api.session.get_bind()
    SocialPost.metadata.create_all(engine, tables=[SocialTopic.__table__])
    copies = sa.MetaData()
    api_harness._sqlite_copy(RecipeExecution.__table__, copies)
    copies.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sa.insert(WorkflowTemplate.__table__).values(
            id=MARKETPLACE_ID, template_id=RESEARCH_PLAYBOOK_TEMPLATE_ID, name="Content bank research",
            description="Fills a plan's content bank", owner_type="marketplace", owner_id="marketplace",
            steps=[{"step_id": "research", "agent_name": "Social Media Director"}], created_by="seed",
        ))
    cascades = []

    async def cascade(**kwargs):
        cascades.append(kwargs)
        return ci.CascadeResult()

    monkeypatch.setattr(ci, "cascade_recipe_dependencies", cascade)
    launched, told = [], []
    engine_fake = SimpleNamespace(launch=lambda **kwargs: launched.append(kwargs))
    monkeypatch.setattr("services.playbook_engine.get_playbook_engine", lambda: engine_fake)
    monkeypatch.setattr(plan_notify, "notify_plan", lambda ws, plan_id, event, title: told.append((event[0], title)))
    monkeypatch.setattr("core.database.database.SessionLocal", sessionmaker(bind=engine))
    api.cascades, api.launched, api.told = cascades, launched, told
    return api


def _copies(api, workspace_id):
    api.session.expire_all()
    return (
        api.session.query(WorkflowTemplate)
        .filter(WorkflowTemplate.workspace_id == workspace_id, WorkflowTemplate.cloned_from_id == MARKETPLACE_ID)
        .all()
    )


def _flag(api, workspace_id):
    api.session.expire_all()
    return setup.research_installed_at(api.session.get(Workspace, workspace_id).settings)


def _note(api, plan):
    resp = api.client.get(f"/api/socials/plans/{plan['id']}/topics")
    assert resp.status_code == 200, resp.text
    return resp.json()["research_note"]


def test_the_first_plan_installs_research_once_and_a_second_plan_installs_nothing(research):
    plan = _create_plan(research)
    (copy,) = _copies(research, WS_A)
    assert (copy.owner_type, copy.name, copy.steps) == ("workspace", "Content bank research", [{"step_id": "research", "agent_name": "Social Media Director"}])
    assert copy.template_id != RESEARCH_PLAYBOOK_TEMPLATE_ID  # the workspace's own row
    assert _flag(research, WS_A) is not None
    assert len(research.cascades) == 1 and research.cascades[0]["remap_steps"] is True  # its step gets the Director's clone
    _create_plan(research, name="A second plan")
    assert len(_copies(research, WS_A)) == 1 and len(research.cascades) == 1
    assert _copies(research, WS_B) == [] and _flag(research, WS_B) is None  # another workspace has nothing
    assert _note(research, plan) is None


def test_the_installed_copy_is_what_research_runs(research):
    _create_plan(research)
    (copy,) = _copies(research, WS_A)
    found = socials_plan_research.installed_playbook(research.session, WS_A)
    assert found is not None and found.id == copy.id


def test_a_workspace_with_a_plan_but_no_flag_installs_on_its_next_save(research, monkeypatch):
    with monkeypatch.context() as before_this_prd:
        before_this_prd.setattr(setup, "after_plan_save", lambda *args: None)
        plan = _create_plan(research)
    assert _copies(research, WS_A) == [] and _flag(research, WS_A) is None
    assert _note(research, plan) == setup.NOT_SET_UP
    resp = research.client.put(f"/api/socials/plans/{plan['id']}", json={"goal": "Book demos at the stand"})
    assert resp.status_code == 200, resp.text
    assert len(_copies(research, WS_A)) == 1 and _flag(research, WS_A) is not None


def test_a_copy_the_owner_deleted_is_not_put_back_by_a_save(research):
    plan = _create_plan(research)
    (copy,) = _copies(research, WS_A)
    table = WorkflowTemplate.__table__
    with research.session.get_bind().begin() as conn:  # as Playbooks deletes it: the row goes
        conn.execute(sa.delete(table).where(table.c.id == copy.id))
    _create_plan(research, name="A second plan")
    research.client.put(f"/api/socials/plans/{plan['id']}", json={"goal": "Book demos at the stand"})
    assert _copies(research, WS_A) == [] and len(research.cascades) == 1
    assert _note(research, plan) == setup.REMOVED


def test_an_edited_copy_is_left_as_it_is_and_only_the_flag_is_set(research):
    with research.session.get_bind().begin() as conn:
        conn.execute(sa.insert(WorkflowTemplate.__table__).values(
            id=50, template_id="socials-content-research", name="Content bank research", description="Edited",
            owner_type="workspace", owner_id=str(WS_A), workspace_id=WS_A, cloned_from_id=MARKETPLACE_ID,
            steps=EDITED_STEPS, created_by="owner",
        ))
    _create_plan(research)
    (copy,) = _copies(research, WS_A)
    assert (copy.id, copy.steps) == (50, EDITED_STEPS)
    assert research.cascades == []  # the installer never ran
    assert _flag(research, WS_A) is not None


@pytest.mark.parametrize("failure, level", [
    (RuntimeError("the database went away"), logging.ERROR),
    (package_installer.PackageInstallError("Marketplace playbook not found"), logging.WARNING),
])
def test_a_failing_install_never_fails_the_plans_save(research, monkeypatch, caplog, failure, level):
    async def failing(*args, **kwargs):
        raise failure

    monkeypatch.setattr(package_installer, "install_playbook", failing)
    with caplog.at_level(logging.WARNING, logger=setup.__name__):
        plan = _create_plan(research)
    assert research.client.get(f"/api/socials/plans/{plan['id']}").status_code == 200
    assert _copies(research, WS_A) == [] and _flag(research, WS_A) is None  # tried again on the next save
    (record,) = [r for r in caplog.records if r.name == setup.__name__]
    assert record.levelno == level and "research" in record.getMessage()
    assert _note(research, plan) == setup.NOT_SET_UP


class _LockHeldElsewhere:
    """A Postgres session whose try-lock answers false: another install holds it."""

    def __init__(self):
        self.ended = []

    def get_bind(self):
        return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

    def execute(self, statement, params):
        assert params == {"namespace": setup.RESEARCH_LOCK_NAMESPACE, "key": str(WS_A)}
        return SimpleNamespace(scalar=lambda: False)

    def query(self, *entities):
        pytest.fail("an install without the lock read nothing")

    def commit(self):
        self.ended.append("commit")

    def rollback(self):
        self.ended.append("rollback")


def test_an_install_that_meets_another_installs_nothing(monkeypatch):
    monkeypatch.setattr(package_installer, "install_playbook", lambda *a, **k: pytest.fail("installed without the lock"))
    db = _LockHeldElsewhere()
    outcome = asyncio.run(setup.install(db, WS_A, now=datetime.now(timezone.utc), first_only=False))
    assert outcome == setup.BUSY and db.ended == ["commit"]


def test_the_flag_is_written_into_a_new_settings_object():
    settings = {"socials": {"enabled": True}, "other": {"kept": 1}}
    now = datetime(2026, 10, 4, 9, tzinfo=timezone.utc)
    written = setup.with_research_installed_at(settings, now)
    assert written == {"socials": {"enabled": True, "research_installed_at": "2026-10-04T09:00:00+00:00"}, "other": {"kept": 1}}
    assert settings == {"socials": {"enabled": True}, "other": {"kept": 1}}  # unchanged
    assert setup.research_installed_at(written) == "2026-10-04T09:00:00+00:00"
    assert setup.research_installed_at({"socials": "not an object"}) is None and setup.research_installed_at(None) is None


# ── US-C102: Research again puts it back; the weekly run never does ─────────


def _research_again(api, plan):
    return api.client.post(f"/api/socials/plans/{plan['id']}/research")


def _delete_copies(api, workspace_id):
    table = WorkflowTemplate.__table__
    with api.session.get_bind().begin() as conn:
        conn.execute(sa.delete(table).where(table.c.workspace_id == workspace_id, table.c.cloned_from_id == MARKETPLACE_ID))


def _weekly(now=NOW):
    return anyio.run(functools.partial(anyio.to_thread.run_sync, socials_plan_research.launch_due, now))


@pytest.mark.parametrize("had_it", [False, True], ids=["never-installed", "deleted-by-the-owner"])
def test_research_again_puts_a_missing_playbook_back_and_runs_it(research, monkeypatch, had_it):
    with monkeypatch.context() as saves:
        if not had_it:
            saves.setattr(setup, "after_plan_save", lambda *args: None)
        plan = _create_plan(research)
    _delete_copies(research, WS_A)
    started = _research_again(research, plan)
    assert started.status_code == 202, started.text
    (copy,) = _copies(research, WS_A)
    (launch,) = research.launched
    assert launch["recipe_id"] == copy.id and launch["recipe_execution_id"] == started.json()["execution_id"]
    assert _flag(research, WS_A) is not None and _note(research, plan) is None
    again = _research_again(research, plan)
    assert again.status_code == 202 and len(_copies(research, WS_A)) == 1  # nothing to put back now


def test_research_again_is_409_only_when_the_marketplace_has_no_playbook(research):
    plan = _create_plan(research)
    _delete_copies(research, WS_A)
    table = WorkflowTemplate.__table__
    with research.session.get_bind().begin() as conn:  # the seed never ran on this platform
        conn.execute(sa.delete(table).where(table.c.id == MARKETPLACE_ID))
    refused = _research_again(research, plan)
    assert (refused.status_code, refused.json()["detail"]) == (409, setup.NO_MARKETPLACE_PLAYBOOK)
    assert research.launched == []


def test_research_again_says_why_it_cannot_set_research_up(research, monkeypatch):
    plan = _create_plan(research)
    _delete_copies(research, WS_A)

    async def full(*args, **kwargs):
        raise AgentLimitReached({"message": "Your basic plan includes 5 agents and this workspace has 5."})

    with monkeypatch.context() as plan_full:
        plan_full.setattr(package_installer, "install_playbook", full)
        refused = _research_again(research, plan)
    assert refused.status_code == 409
    assert refused.json()["detail"] == "Research could not be set up: Your basic plan includes 5 agents and this workspace has 5."
    monkeypatch.setattr(setup, "_lock_is_ours", lambda db, workspace_id: False)
    busy = _research_again(research, plan)
    assert (busy.status_code, busy.json()["detail"]) == (409, setup.SETTING_UP)
    assert research.launched == [] and _copies(research, WS_A) == []


def test_the_weekly_run_never_puts_the_playbook_back_and_says_why_once_a_day(research, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    plan = _create_plan(research, starts_on="2026-10-12")
    _delete_copies(research, WS_A)
    assert _weekly() == 0 and _weekly() == 0
    assert _copies(research, WS_A) == [] and research.launched == [] and len(research.cascades) == 1
    assert research.told == [("social_plan_research_unavailable", f"{plan['name']}: {setup.REMOVED}")]


def test_a_hosted_trial_gets_no_weekly_run(research, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr("services.trial_ledger.is_trial_active_workspace", lambda workspace: True)
    _create_plan(research, starts_on="2026-10-12")
    assert _weekly() == 0 and research.launched == [] and research.told == []
    monkeypatch.setattr("services.trial_ledger.is_trial_active_workspace", lambda workspace: False)
    assert _weekly() == 1 and len(research.launched) == 1  # a paid workspace's research comes round


def test_every_role_that_saves_a_plan_may_add_what_its_save_installs():
    """A plan's save installs a marketplace playbook and its agent (US-C101) behind the plan
    routes' documents:create and documents:update. That is safe only while every role holding
    those also holds agents:create and playbooks:create."""
    for role in WorkspaceRole:
        if workspace_has_permission(role.value, "documents:create") or workspace_has_permission(role.value, "documents:update"):
            assert workspace_has_permission(role.value, "agents:create"), role
            assert workspace_has_permission(role.value, "playbooks:create"), role

