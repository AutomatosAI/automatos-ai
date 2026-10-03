"""F222 (night 6, 2 Oct): asked to add the marketplace's "Weekly social posts", Auto
made an empty playbook of that name and said it was done.

The owner: "In the marketplace there's a ready-made one called 'Weekly social
posts' ... Can you add it for me". No action installs a single marketplace
playbook. Auto called one that does not exist, then platform_create_playbook with
the same name, and told the owner it was there. Now that name is refused, with a
result saying the marketplace holds it and the owner installs it there. Auto
says it can't, instead of pretending.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

# Unique per run: the test database may hold the real catalogue (the Socials package's playbooks).
LISTED = f"Weekly social posts {uuid.uuid4().hex[:6]}"


@pytest.fixture
def shop(db_session, seed_workspace):
    return NS(db=db_session, ws=UUID(seed_workspace()))


def _listed(shop, name=LISTED, approved=True):
    """A ready-made playbook on the marketplace."""
    from core.models.core import WorkflowTemplate

    playbook = WorkflowTemplate(name=name, template_id=f"mkt-{uuid.uuid4().hex[:8]}", description="Drafts posts",
                                owner_type="marketplace", owner_id="marketplace", is_approved=approved,
                                created_by="marketplace",
                                tags=[], template_definition={"steps": []})
    shop.db.add(playbook)
    shop.db.flush()
    return playbook.id


def _create(shop, name):
    from modules.tools.discovery.handlers_playbooks import create_playbook

    return asyncio.run(create_playbook(shop.db, shop.ws, {"name": name, "description": "Reads the week's work"}))


def _mine(shop):
    return shop.db.execute(text("SELECT count(*) FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"),
                           {"w": str(shop.ws)}).scalar()


def test_a_marketplace_playbooks_name_is_not_copied_into_an_empty_one(shop):
    listed = _listed(shop)
    reply = _create(shop, f"  {LISTED.upper()} ")              # any case, any spacing: night 6 said "Weekly Social Posts"

    assert reply["success"] is False and reply["marketplace_playbook_id"] == listed
    assert "ready-made playbook in the Marketplace" in reply["error"]
    assert "can't install" in reply["error"]                   # so Auto says it can't
    assert _mine(shop) == 0                                    # night 6: an empty "Weekly Social Posts"


def test_a_playbook_the_marketplace_does_not_show_is_no_bar(shop):
    _listed(shop, approved=False)                              # not approved: the owner never sees it
    assert _create(shop, LISTED)["success"] is True


def test_another_name_is_still_made(shop):
    _listed(shop)
    assert _create(shop, f"Quayside Pantry welcome {uuid.uuid4().hex[:6]}")["success"] is True


def test_a_namesake_in_the_workspace_is_still_refused_first(shop):
    """F185's rule is kept: the workspace's own playbook is named before the marketplace's."""
    _listed(shop)
    mine = f"New Cafe Onboarding {uuid.uuid4().hex[:6]}"
    first = _create(shop, mine)
    again = _create(shop, mine)
    assert first["success"] is True and again["success"] is False
    assert again["existing_playbook_id"] == first["playbook"]["id"]
