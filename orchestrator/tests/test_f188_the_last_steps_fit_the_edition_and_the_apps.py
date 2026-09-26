"""F188 (night 6) — onboarding's last steps fit the install and what is connected.

ca73421f6 made the setup checklist say "Connect an app" while none is connected,
but the powerup prompt still told Auto to present "connect a second app". And
229ae4569 moves a local workspace from boom to completed, since powerup has no
local UI, but Auto's own platform_update_onboarding(advance_to="powerup") still
landed on powerup. Now the prompt words the item from the live count, and on
local Auto's advance to powerup completes onboarding and says why.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from modules.context.sections.base import SectionContext
from modules.context.sections.onboarding import OnboardingSection


@pytest.fixture
def connected(monkeypatch):
    import core.composio.entity_manager as em

    def set_apps(apps):
        monkeypatch.setattr(em.EntityManager, "get_connected_apps", lambda self, ws: list(apps))
    return set_apps


def _powerup():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = NS(
        onboarding={"stage": "powerup", "stages": {}, "segment": {}})
    ctx = SectionContext(agent=None, workspace_id="ws-1", db_session=db, messages=[])
    return " ".join(asyncio.run(OnboardingSection().render(ctx)).split())


def test_with_no_app_connected_the_first_one_is_the_ask(connected):
    connected([])
    assert "run-and-learn checklist (connect an app · invite a teammate" in _powerup()


def test_with_one_connected_it_is_a_second(connected):
    connected(["GMAIL"])
    assert "run-and-learn checklist (connect a second app · invite a teammate" in _powerup()


class _Db:
    def __init__(self, workspace):
        self.workspace = workspace

    def query(self, *a):
        return self

    def filter(self, *a):
        return self

    def first(self):
        return self.workspace

    def add(self, obj):
        pass

    def flush(self):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.mark.parametrize("local, stage", [(True, "completed"), (False, "powerup")], ids=["local", "saas"])
def test_autos_advance_to_powerup_fits_the_install(monkeypatch, local, stage):
    from config import config
    from modules.tools.discovery.handlers_onboarding import update_onboarding

    monkeypatch.setattr(type(config), "IS_LOCAL_EDITION", local)
    workspace = NS(id="ws-1", onboarding={"stage": "boom", "stages": {"boom": "2026-09-26T02:02:53+00:00"},
                                          "segment": {}})
    result = asyncio.run(update_onboarding(_Db(workspace), "ws-1", {"advance_to": "powerup"}))

    assert result["success"] is True and workspace.onboarding["stage"] == stage
    assert ("no powerup step" in result.get("message", "")) is local
