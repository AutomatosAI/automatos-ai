"""F379 (night 11, 7 Oct): social media work is the Social Media Director's ticket, never NEWSROOM's.

Night 11: "my social media person" (ticket 2155) and "my social media director" (2157) went to
NEWSROOM, which made a markdown caption and a storyboard and no post; the Social Media Director
was never picked (friction 17). Now:

PURE: the lane reads the ask (the owner's social media person, a channel's post, a carousel, a
reel; not a question, not the posts already made, not an ask Auto keeps); it pins the ASSIGN lane
for the Director, ahead of the tiers and never mid-onboarding, and marks the turn; the turn's note
says the ticket is a Socials post that exists and has rendered; and the classifier's "my social
media person" resolves to the Director, never to NEWSROOM.

@integration (the orchestrator-tests job's Postgres, one rolled-back transaction): the Director is
the workspace's clone of the Socials package's agent, then the active agent of that name, and a
paused one is no Director.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from consumers.chatbot import socials_assign_lane as lane  # noqa: E402
from consumers.chatbot.auto import Action, AutoBrain, apply_assign_bias  # noqa: E402

WS = "00000000-0000-0000-0000-0000000379d1"
DIRECTOR = NS(id=348, name="Social Media Director")
NEWSROOM = NS(id=274, name="NEWSROOM")


@pytest.mark.parametrize("said", [
    "Auto: my social media person should draft the October Harvest Club post.",
    "Can you get my social media director to do a 15-second video of our top three cafés?",
    "Draft an Instagram post for the October Harvest Club box.",
    "Make a carousel for the Harvest Club: two coffees, £32, on sale Monday.",
    "Have our social media team put together a LinkedIn post about the wholesale offer.",
])
def test_social_media_work_is_read_as_the_directors(said):
    assert lane.asks_for_social_work(said) is True


@pytest.mark.parametrize("said", [
    "What's my social media person working on?",
    "Which Instagram posts are waiting for approval?",
    "Can you list my Instagram posts from this week?",
    "Make the carousel yourself, don't bother the team.",
    "Write a blog post about the Harvest Club.",
    "How do I give my agent two photos?",
])
def test_a_question_the_posts_already_made_and_an_ask_auto_keeps_are_not(said):
    assert lane.asks_for_social_work(said) is False


@pytest.fixture
def director(monkeypatch):
    monkeypatch.setattr(lane, "find_social_media_director", lambda db, workspace_id: DIRECTOR)


def test_the_ask_is_the_assign_lane_for_the_director_and_the_ticket_is_a_socials_post(director):
    assessment = lane.director_assignment(object(), WS, "Draft an Instagram post for the October Harvest Club box.")

    assert assessment.action == Action.ASSIGN and assessment.confidence == 1.0
    assert (assessment.target_agent_id, assessment.target_agent_name) == (348, "Social Media Director")
    apply_assign_bias(assessment, "Draft an Instagram post for the October Harvest Club box.")
    directive = assessment.context_directive
    assert "file this as a board ticket" in directive and 'assigned_agent_name="Social Media Director"' in directive


def test_the_directors_turn_gets_what_its_ticket_must_say_never_how_to_make_the_post():
    from consumers.chatbot.socials_turn_note import socials_note

    async def turn():
        lane._director.set("Social Media Director")
        return socials_note(["Draft an Instagram post for the October Harvest Club box."], [])

    note = asyncio.run(turn())
    assert note.startswith("The ticket is a Socials post. This is social media work, the Social Media Director's job.")
    assert "done means the post exists in the Socials tab and has rendered, with its post id" in note
    assert "Don't make the post yourself in this reply." in note


def test_no_director_leaves_the_turn_to_the_tiers(monkeypatch):
    monkeypatch.setattr(lane, "find_social_media_director", lambda db, workspace_id: None)
    assert lane.director_assignment(object(), WS, "Draft an Instagram post for Friday.") is None


def test_a_director_that_cannot_be_read_leaves_the_turn_to_the_tiers(monkeypatch):
    def broken(db, workspace_id):
        raise RuntimeError("database gone")

    monkeypatch.setattr(lane, "find_social_media_director", broken)
    assert lane.director_assignment(object(), WS, "Draft an Instagram post for Friday.") is None


class _Brain:
    def __init__(self, onboarding=False):
        self._db, self._workspace_id, self.onboarding = object(), WS, onboarding

    def _onboarding_active(self):
        return self.onboarding


def _assessed(brain, message):
    """The wrapped assessment and the turn's mark, read inside the turn's own context."""
    async def tiers(brain, message, conversation_length=0):
        return "the tiers"

    async def turn():
        result = await lane.social_work_goes_to_the_director(tiers)(brain, message)
        return result, lane.director_turn()

    return asyncio.run(turn())


def test_the_lane_comes_before_the_tiers_and_marks_the_turn(director):
    pinned, mark = _assessed(_Brain(), "Auto: my social media person should draft the Harvest Club post.")
    other, unmarked = _assessed(_Brain(), "Send Rosa the invoice.")
    onboarding, still = _assessed(_Brain(onboarding=True), "Draft an Instagram post for Friday.")

    assert pinned.target_agent_id == 348 and mark == "Social Media Director"
    assert (other, unmarked) == ("the tiers", None)
    assert (onboarding, still) == ("the tiers", None)


def test_autobrain_gives_my_social_media_person_to_the_director(director, monkeypatch):
    brain = AutoBrain(object(), WS)
    monkeypatch.setattr(brain, "_onboarding_active", lambda: False)

    assessment = asyncio.run(brain.assess("Auto: my social media person should draft the Harvest Club post."))

    assert assessment.action == Action.ASSIGN and assessment.target_agent_name == "Social Media Director"


def test_the_classifiers_social_media_person_is_the_director_never_newsroom():
    brain = AutoBrain(object(), WS)
    message = "Get my social media person on the Harvest Club box."

    assert brain._match_roster_agent("my social media person", [NEWSROOM, DIRECTOR], message=message) == (
        348, "Social Media Director")
    assert brain._match_roster_agent("my social media person", [NEWSROOM], message=message) == (None, None)
    assert lane.social_media_role("the newsroom", [NEWSROOM, DIRECTOR]) == (None, None)


# ---------------------------------------------------------------------------
# On Postgres: the Director is the Socials package's agent in the workspace
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    import sqlalchemy as sa

    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            for table in ("agents", "skills", "agent_skills", "workflow_recipes", "marketplace_packages", "workspaces"):
                conn.execute(sa.text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Social Media Director lookup needs the test database: {exc}")
    yield engine
    engine.dispose()


@pytest.mark.integration
def test_the_director_is_the_packages_clone_then_the_agent_of_that_name(pg_engine, monkeypatch, tmp_path):
    import core.auth.workspace_permission as permission_mod
    from tests.test_prd251w1_socials_package import (
        INSTALL_ROUTE, SKILL_NAMES, _boot, _client, _rolled_back, _skills_manifest, _workspace,
    )

    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))
        ws = _workspace(conn)
        assert lane.find_social_media_director(session, ws) is None
        installed = _client(session, ws).post(INSTALL_ROUTE)
        assert installed.status_code == 200, installed.text

        clone = lane.find_social_media_director(session, ws)
        assert clone is not None and clone.workspace_id == ws and clone.cloned_from_id is not None

        clone.cloned_from_id = None   # an agent of that name, made by hand
        session.flush()
        assert lane.find_social_media_director(session, ws).id == clone.id
        clone.status = "paused"
        session.flush()
        assert lane.find_social_media_director(session, ws) is None
