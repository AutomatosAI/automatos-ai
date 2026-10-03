"""F231 (night 6, B119): asked to run the café playbook for a new café, Auto copied it
and ran the copy.

14:56:20, 2 Oct: the owner asked for the new-café playbook (#102, "New Cafe
Onboarding") for Quayside Pantry. Auto made playbook #109 "New Cafe Onboarding -
Quayside Pantry", a third café playbook beside #102 and #103, and ran that; it failed
at once. Each new café would leave another playbook behind. A name that is one of
the workspace's playbooks with a qualifier added is refused now, and the result says
to run the existing one with the details as input_data.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text


@pytest.fixture
def shop(db_session, seed_workspace):
    return NS(db=db_session, ws=UUID(seed_workspace()))


def _create(shop, name):
    from modules.tools.discovery.handlers_playbooks import create_playbook

    return asyncio.run(create_playbook(shop.db, shop.ws, {"name": name, "description": "Welcomes a new café"}))


def _count(shop):
    return shop.db.execute(text("SELECT count(*) FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"),
                           {"w": str(shop.ws)}).scalar()


@pytest.fixture
def cafe(shop):
    """The workspace's own café playbook (unique per run)."""
    name = f"New Cafe Onboarding {uuid.uuid4().hex[:6]}"
    made = _create(shop, name)
    assert made["success"] is True
    return NS(name=name, id=made["playbook"]["id"])


@pytest.mark.parametrize("qualified, detail", [
    ("{name} - Quayside Pantry", "Quayside Pantry"),              # night 6's #109
    ("{name} (Quayside Pantry)", "Quayside Pantry"),
    ("{name}: Quayside Pantry", "Quayside Pantry"),
    ("{name} for Quayside Pantry", "Quayside Pantry"),
])
def test_a_copy_of_a_playbook_for_one_run_is_refused(shop, cafe, qualified, detail):
    before = _count(shop)
    reply = _create(shop, qualified.format(name=cafe.name))

    assert reply["success"] is False and _count(shop) == before          # night 6: a third café playbook
    assert "platform_execute_playbook" in reply["error"] and "input_data" in reply["error"]
    assert f"To run it for {detail}," in reply["error"] and reply["existing_playbook_id"] == cafe.id


def test_a_playbook_whose_own_name_has_a_qualifier_still_matches(shop):
    printable = f"Tom's Checklist {uuid.uuid4().hex[:6]} (Printable)"
    assert _create(shop, printable)["success"] is True
    reply = _create(shop, f"{printable} - Week 41")
    assert reply["success"] is False and "To run it for Week 41," in reply["error"]


def test_a_name_of_its_own_is_made(shop, cafe):
    assert _create(shop, f"Quayside Pantry welcome {uuid.uuid4().hex[:6]}")["success"] is True
    assert _create(shop, f"Coffee - the tasting notes {uuid.uuid4().hex[:6]}")["success"] is True   # no playbook 'Coffee'
