"""PRD-251B Wave 2, US-B203 and US-B204 — the content bank, by hand and by research.

Pinned:

* the routes: a topic is added, listed (unused first, pinned by their day), edited,
  pinned and deleted; a fact without a source, a title the bank holds (case and spacing
  aside), and a "never say" phrase of the plan are each refused with 422 and the reason;
  another workspace's plan is a 404;
* the research tools: ``platform_get_social_plan`` reads the plan and its bank;
  ``platform_add_social_topics`` adds what passes and lists each refusal with why,
  ``origin`` research, the agent as its author; both refuse while Socials is off;
* the seeded **Content bank research** playbook reads the plan, adds topics and never
  makes, approves or publishes a post;
* the make tick's pick: a topic pinned to the slot's day first, then the oldest unused one
  that suits the format; a used one never again.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialCampaign, SocialPost, SocialTopic  # noqa: E402
from core.seeds.seed_socials_package import RESEARCH_PLAYBOOK_TEMPLATE_ID, SOCIALS_PLAYBOOKS  # noqa: E402
from modules.socials import topics  # noqa: E402
from modules.tools.discovery import handlers_socials  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B, WS_OFF, _ctx  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402

api = api_harness.api
SOURCE = {"kind": "web", "ref": "https://example.com/launch", "label": "Launch page"}
AGENT = {"_agent_id": 7, "_agent_name": "Social Media Director"}


@pytest.fixture
def bank(api):
    SocialPost.metadata.create_all(api.session.get_bind(), tables=[SocialTopic.__table__])
    return api


def _topic(**overrides):
    body = {"title": "Three weeks to Lisbon", "angle": "Why visit the stand", "formats": ["image"],
            "facts": [{"text": "We demo on stand B12.", "source": SOURCE}]}
    body.update(overrides)
    return body


def _add(api, plan, **overrides):
    return api.client.post(f"/api/socials/plans/{plan['id']}/topics", json=_topic(**overrides))


# ── the routes ─────────────────────────────────────────────────────────────


def test_a_topic_is_added_listed_edited_pinned_and_deleted(bank):
    plan = _create_plan(bank)
    added = _add(bank, plan)
    assert added.status_code == 201, added.text
    topic = added.json()
    assert (topic["origin"], topic["plan_id"], topic["used_at"]) == ("person", plan["id"], None)
    second = _add(bank, plan, title="The roadmap", facts=[]).json()
    pinned = bank.client.put(f"/api/socials/plans/{plan['id']}/topics/{second['id']}/pin", json={"pinned_on": "2026-10-14"})
    assert pinned.json()["pinned_on"] == "2026-10-14"
    listed = bank.client.get(f"/api/socials/plans/{plan['id']}/topics").json()
    assert [t["title"] for t in listed["topics"]] == ["The roadmap", "Three weeks to Lisbon"]  # pinned first
    assert (listed["total"], listed["unused"]) == (2, 2)
    edited = bank.client.put(f"/api/socials/plans/{plan['id']}/topics/{topic['id']}", json={"angle": "Book a demo"})
    assert edited.json()["angle"] == "Book a demo" and edited.json()["facts"] == topic["facts"]
    assert bank.client.delete(f"/api/socials/plans/{plan['id']}/topics/{topic['id']}").status_code == 204
    assert bank.client.get(f"/api/socials/plans/{plan['id']}").json()["bank"] == {"topics": 1, "unused": 1}


@pytest.mark.parametrize("overrides, reason", [
    ({"facts": [{"text": "Unsourced figure."}]}, "has no source"),
    ({"facts": [{"text": "Odd source.", "source": {"kind": "rumour", "ref": "x", "label": "x"}}]}, "has no source"),
    ({"title": "Our CHEAPEST plan yet"}, "never-say"),
    ({"facts": [{"text": "It is the cheapest.", "source": SOURCE}]}, "never-say"),
    ({"formats": ["poster"]}, "formats"),
])
def test_the_bank_refuses_with_the_reason(bank, overrides, reason):
    plan = _create_plan(bank, sources={"never_say": ["cheapest"]})
    resp = _add(bank, plan, **overrides)
    assert resp.status_code == 422 and reason in resp.text


def test_a_title_the_bank_holds_is_refused_case_and_spacing_aside(bank):
    plan = _create_plan(bank)
    assert _add(bank, plan).status_code == 201
    again = _add(bank, plan, title="  three WEEKS   to lisbon ")
    assert again.status_code == 422 and "already has" in again.text


def test_another_workspaces_bank_is_not_found(bank):
    plan = _create_plan(bank)
    bank.ctx = _ctx(WS_B)
    assert bank.client.get(f"/api/socials/plans/{plan['id']}/topics").status_code == 404
    assert _add(bank, plan).status_code == 404


# ── the research tools ─────────────────────────────────────────────────────


def _tool(handler, bank, workspace_id=WS_A, **params):
    return asyncio.run(handler(bank.session, workspace_id, {**AGENT, **params}))


def test_research_reads_the_plan_and_adds_what_passes(bank):
    plan = _create_plan(bank, sources={"never_say": ["cheapest"]})
    _add(bank, plan)
    read = _tool(handlers_socials.get_social_plan, bank, plan_id=plan["id"])
    assert read["success"] is True and read["plan"]["sources"]["never_say"] == ["cheapest"]
    assert read["bank"] == [{"title": "Three weeks to Lisbon", "formats": ["image"], "used": False}]
    answer = _tool(handlers_socials.add_social_topics, bank, plan_id=plan["id"], topics=[
        _topic(title="A new release", facts=[{"text": "v2 shipped on Monday.", "source": {"kind": "github", "ref": "https://github.com/org/repo/releases/v2", "label": "Release v2"}}]),
        _topic(),  # the bank holds it
        _topic(title="Cheapest ever"),  # never say
        _topic(title="No source", facts=[{"text": "A figure."}]),
    ])
    assert answer["success"] is True and [t["title"] for t in answer["added"]] == ["A new release"]
    assert [r["index"] for r in answer["refused"]] == ["1", "2", "3"]
    row = bank.session.query(SocialTopic).filter(SocialTopic.title == "A new release").one()
    assert (row.origin, row.created_by) == ("research", "agent:7")


def test_the_research_tools_refuse_while_socials_is_off_and_for_an_unknown_plan(bank):
    plan = _create_plan(bank)
    off = _tool(handlers_socials.add_social_topics, bank, workspace_id=WS_OFF, plan_id=plan["id"], topics=[_topic()])
    assert off["success"] is False and off.get("socials_off") is True
    assert _tool(handlers_socials.get_social_plan, bank, plan_id=str(uuid.uuid4()))["success"] is False
    assert _tool(handlers_socials.get_social_plan, bank, workspace_id=WS_B, plan_id=plan["id"])["success"] is False
    assert bank.session.query(SocialTopic).count() == 0


def test_the_seeded_research_playbook_only_fills_the_bank():
    (spec,) = [s for s in SOCIALS_PLAYBOOKS if s["template_id"] == RESEARCH_PLAYBOOK_TEMPLATE_ID]
    (step,) = spec["steps"]
    prompt = step["prompt_template"]
    assert spec["name"] == "Content bank research" and step["agent_slug"] == "social-media-director"
    assert "{input.plan_id}" in prompt and "platform_get_social_plan" in prompt and "platform_add_social_topics" in prompt
    assert "platform_create_social_post" not in prompt  # it never makes a post


# ── the pick ───────────────────────────────────────────────────────────────


def test_the_next_topic_pinned_first_then_the_oldest_that_suits(bank):
    plan_row = SocialCampaign(id=uuid.uuid4(), workspace_id=WS_A, name="P", created_by="u", kind="plan", approved_hash_set=[])
    bank.session.add(plan_row)
    bank.session.commit()
    day = date(2026, 10, 14)
    video = topics.add_topic(bank.session, plan_row, {"title": "Video only", "formats": ["video"]}, created_by="u")
    oldest = topics.add_topic(bank.session, plan_row, {"title": "Any format"}, created_by="u")
    other_day = topics.add_topic(bank.session, plan_row, {"title": "Pinned elsewhere"}, created_by="u")
    other_day.pinned_on = date(2026, 10, 20)
    pinned = topics.add_topic(bank.session, plan_row, {"title": "Pinned today", "formats": ["image"]}, created_by="u")
    pinned.pinned_on = day
    bank.session.commit()
    assert topics.next_topic(bank.session, plan_row, "image", day).id == pinned.id
    assert topics.next_topic(bank.session, plan_row, "video", day).id == video.id
    topics.mark_used(pinned, SimpleNamespaceLike(uuid.uuid4()), datetime(2026, 10, 14, 7, tzinfo=timezone.utc))
    bank.session.commit()
    assert topics.next_topic(bank.session, plan_row, "image", day).id == oldest.id
    assert topics.next_topic(bank.session, plan_row, "image", date(2026, 10, 20)).id == other_day.id


class SimpleNamespaceLike:
    """A post as mark_used reads it: its id."""

    def __init__(self, post_id):
        self.id = post_id
