"""PRD-251C Wave 4, US-C405 (research weighs results) and C9's dated topics from research.

On the S0.3b API harness (SQLite). Pinned:

* **History carries the numbers:** each post's numbers and engagement once read, none before;
* **the bank's order with results:** the next topic is the unused one most like the plan's best
  performers, the oldest without results; a topic pinned to the day still comes first;
* **research's prompt** reads the numbers and adds dated topics; **a dated topic** from research
  is pinned to its day, within the plan's dates, and refused outside them.
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialCampaign, SocialPost, SocialPostStat, SocialPostTarget, SocialTopic  # noqa: E402
from core.seeds.seed_socials_package import _RESEARCH_PROMPT  # noqa: E402
from modules.socials import history, topics  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc
FACT = {"text": "Stand B12.", "source": {"kind": "note", "ref": "owner", "label": "The owner"}}


def _plan_row(api, plan):
    row = api.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    api.session.refresh(row)
    return row


def _went_out(api, plan, title, engagement=None, topic_title=None):
    """A post of the plan that went out yesterday, made from a topic, read when ``engagement`` is given."""
    post = SocialPost(id=uuid.uuid4(), workspace_id=WS_A, created_by="u", title=title, content_hash="0" * 64, status="published",
                      format="image", campaign_id=uuid.UUID(plan["id"]), created_at=datetime.now(UTC) - timedelta(days=2))
    target = SocialPostTarget(id=uuid.uuid4(), post_id=post.id, toolkit="linkedin", post_kind="image", action_plan={},
                              idempotency_key=f"sp:{post.id}:linkedin", status="published", remote_id="urn:li:share:1",
                              published_at=datetime.now(UTC) - timedelta(days=1))
    rows = [post, target]
    if topic_title:
        rows.append(SocialTopic(workspace_id=WS_A, campaign_id=post.campaign_id, title=topic_title, facts=[], formats=[], created_by="u",
                                used_post_id=post.id, used_at=datetime.now(UTC) - timedelta(days=1)))
    if engagement is not None:
        rows.append(SocialPostStat(workspace_id=WS_A, post_id=post.id, target_id=target.id, reading=1, read_at=datetime.now(UTC),
                                   numbers={"reactions": engagement}, source_action="LINKEDIN_LIST_REACTIONS"))
    api.session.add_all(rows)
    api.session.commit()
    return post


def _bank_topic(api, plan, title, angle=None, pinned_on=None):
    row = _plan_row(api, plan)
    topic = topics.add_topic(api.session, row, {"title": title, "angle": angle, "facts": [FACT], "pinned_on": pinned_on}, created_by="u")
    api.session.commit()
    return topic


def test_history_carries_each_posts_numbers_once_read(bank):  # noqa: F811
    plan = _create_plan(bank)
    _went_out(bank, plan, "Read one", engagement=42)
    _went_out(bank, plan, "Not read yet")
    items = {item["title"]: item for item in history.history(bank.session, WS_A)}
    assert (items["Read one"]["numbers"], items["Read one"]["engagement"]) == ({"reactions": 42}, 42)
    assert (items["Not read yet"]["numbers"], items["Not read yet"]["engagement"]) == (None, None)


def test_the_next_topic_is_the_one_most_like_the_best_performers(bank):  # noqa: F811
    plan = _create_plan(bank)
    older = _bank_topic(bank, plan, "Pricing explained")
    like_best = _bank_topic(bank, plan, "The stand tour route", angle="Walk the stand with us")
    day = date(2026, 10, 14)
    assert topics.next_topic(bank.session, _plan_row(bank, plan), "image", day).id == older.id  # no results: the oldest
    _went_out(bank, plan, "Come see the stand", engagement=40, topic_title="Stand tour")
    _went_out(bank, plan, "Our pricing tiers", engagement=2, topic_title="Pricing tiers")
    assert topics.next_topic(bank.session, _plan_row(bank, plan), "image", day).id == like_best.id
    pinned = _bank_topic(bank, plan, "Doors open", pinned_on=day)
    assert topics.next_topic(bank.session, _plan_row(bank, plan), "image", day).id == pinned.id  # its day comes first


def test_research_adds_dated_topics_within_the_plans_dates(bank):  # noqa: F811
    plan = _create_plan(bank)  # 12 Oct to 8 Nov 2026
    row = _plan_row(bank, plan)
    added, refused = topics.add_topics(bank.session, row, [
        {"title": "Web Summit in 3 days", "facts": [FACT], "pinned_on": "2026-11-05"},
        {"title": "After the summit", "facts": [FACT], "pinned_on": "2026-11-20"},
        {"title": "Some day", "facts": [FACT], "pinned_on": "soon"},
    ], created_by="research")
    assert [(topic.title, topic.pinned_on) for topic in added] == [("Web Summit in 3 days", date(2026, 11, 5))]
    assert [item["reason"] for item in refused] == ["pinned_on must fall within the plan's dates", "pinned_on must be a day, such as 2026-11-05"]


def test_researchs_prompt_reads_the_numbers_and_adds_dated_topics():
    assert "numbers and engagement" in _RESEARCH_PROMPT and "most engagement" in _RESEARCH_PROMPT
    assert "pinned_on" in _RESEARCH_PROMPT and "countdown" in _RESEARCH_PROMPT
