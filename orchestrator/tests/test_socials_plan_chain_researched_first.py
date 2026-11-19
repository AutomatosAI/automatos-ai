"""19 Nov 2026: a plan's week is made from researched topics, and Auto writes every template word.

On 18 Nov the plan's Content bank research run was offered none of the actions its prompt names
(the dispatcher's ranking cut them), added no topics, ended "completed" and told nobody. Two
minutes later the tick made the week from ten titles with no facts, and the composer, told to
answer null for anything the brief did not spell out, left a Fact card's figure, two carousels'
points and two videos' 55 to 60 fields empty: five posts waited on the owner. Pinned:

* a prompt's named ``platform_*`` actions head the narrowed enum, gate-filtered and never cut by
  the cap (``modules/tools/named_actions.py``, ``tool_router._apply_named_prior``);
* research that sends a title the bank holds with no facts gives that topic its facts; without
  facts it is refused; the plan read says how many facts each topic has;
* no post of a plan is made while its research run is unfinished (for at most RESEARCH_WAIT);
  a finished run that gave the bank nothing is told once, in the run's words;
* the composer writes every variable not marked claim; only a claim waits for a fact;
* a render the template refused is tried once with another template Auto picks;
* the research prompt researches the owner's topics first, and an unedited copy that holds the
  prompt seeded until today takes the new one.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import anyio
import sqlalchemy as sa

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.core import RecipeExecution  # noqa: E402
from core.models.socials import SocialCampaign, SocialTopic  # noqa: E402
from core.seeds import seed_socials_package as seed  # noqa: E402
from core.seeds import socials_research_prompts as prompts  # noqa: E402
from modules.socials import compose, render  # noqa: E402
from modules.tools import named_actions, tool_router  # noqa: E402
from modules.tools.discovery import handlers_socials  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from services import socials_research_outcome as outcome  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_maker import NOW, _install, _text_plan, maker, playbooks  # noqa: E402,F401  (fixtures)
from tests.test_prd251bw2_topics import AGENT, SOURCE, bank  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc
FACT = {"text": "Automatos Go runs Auto on your phone.", "source": SOURCE}


# ── the actions a prompt names ─────────────────────────────────────────────


def test_named_in_reads_each_platform_action_once_in_order():
    text = "Read with platform_get_social_plan, then platform_add_social_topics; platform_get_social_plan again."
    assert named_actions.named_in(text) == ["platform_get_social_plan", "platform_add_social_topics"]
    assert named_actions.named_in(None) == [] and named_actions.named_in("no actions here") == []


def test_the_named_actions_head_a_narrowed_enum_and_the_cap_never_cuts_them():
    ranked = [f"platform_ranked_{n}" for n in range(5)]
    query = "Call platform_get_social_plan and platform_add_social_topics, never platform_admin_reset."
    allowed, reason, from_pins = named_actions.with_named_first(
        (ranked, None, False), query, lambda name: name != "platform_admin_reset", cap=3,
    )
    assert allowed == ["platform_get_social_plan", "platform_add_social_topics", "platform_ranked_0"]
    assert (reason, from_pins) == (named_actions.NAMED_REASON, False)


def test_a_full_enum_or_a_query_naming_none_is_left_as_it_is():
    assert named_actions.with_named_first((None, "flag off", False), "platform_x", lambda n: True, 40) == (None, "flag off", False)
    narrowed = (["platform_a"], "ranked", False)
    assert named_actions.with_named_first(narrowed, "nothing named", lambda n: True, 40) is narrowed


def test_the_router_gates_named_actions_as_ranking_does(monkeypatch):
    monkeypatch.setattr(tool_router, "_page_action_passes_gate", lambda name, admin, su: name == "platform_get_social_plan")
    allowed, _reason, _pins = tool_router._apply_named_prior(
        (["platform_list_deliverables"], None, False), "platform_get_social_plan then platform_gated_one", False, False,
    )
    assert allowed == ["platform_get_social_plan", "platform_list_deliverables"]


# ── research gives the owner's topics their facts ──────────────────────────


def _tool(handler, env, **params):
    import asyncio

    return asyncio.run(handler(env.session, WS_A, {**AGENT, **params}))


def _plan_with_an_idea(env, title="Automatos Go: work from anywhere"):
    from tests.test_prd251bw2_plans import _create as create_plan

    plan = create_plan(env)
    resp = env.client.post(f"/api/socials/plans/{plan['id']}/topics", json={"title": title, "facts": []})
    assert resp.status_code == 201, resp.text
    return plan


def test_research_gives_a_factless_topic_its_facts_and_the_read_says_how_many(bank):  # noqa: F811
    plan = _plan_with_an_idea(bank)
    assert _tool(handlers_socials.get_social_plan, bank, plan_id=plan["id"])["bank"][0]["facts"] == 0

    answer = _tool(handlers_socials.add_social_topics, bank, plan_id=plan["id"], topics=[
        {"title": "automatos go:  work from anywhere", "angle": "Run the business from the train", "facts": [FACT], "formats": ["image"]},
    ])

    assert answer["success"] is True and [t["title"] for t in answer["added"]] == ["Automatos Go: work from anywhere"]
    bank.session.expire_all()
    (row,) = bank.session.query(SocialTopic).filter(SocialTopic.campaign_id == uuid.UUID(plan["id"])).all()
    assert row.origin == "person" and row.facts[0]["text"] == FACT["text"] and row.angle == "Run the business from the train"
    assert _tool(handlers_socials.get_social_plan, bank, plan_id=plan["id"])["bank"][0]["facts"] == 1


def test_research_sending_a_factless_topic_without_facts_is_refused(bank):  # noqa: F811
    plan = _plan_with_an_idea(bank)
    answer = _tool(handlers_socials.add_social_topics, bank, plan_id=plan["id"], topics=[
        {"title": "Automatos Go: work from anywhere", "facts": []},
    ])
    assert answer["success"] is False and "send it with its facts" in answer["refused"][0]["reason"]


# ── the batch waits for research; an empty run is told ─────────────────────


def _run(env, plan, *, status, started_at, final_output=""):
    _install(env, WS_A)
    run_id = f"research-{uuid.uuid4().hex[:8]}"
    with env.session.get_bind().begin() as conn:
        conn.execute(sa.insert(RecipeExecution.__table__).values(
            execution_id=run_id, recipe_id=2, workspace_id=WS_A, status=status,
            started_at=started_at.replace(tzinfo=None), output_data={"final_output": final_output},
        ))
    row = env.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    row.research = {**(row.research or {}), "last_run_id": run_id, "last_run_at": started_at.isoformat()}
    env.session.commit()
    return run_id


def _set_status(env, run_id, status):
    with env.session.get_bind().begin() as conn:
        conn.execute(sa.update(RecipeExecution.__table__).where(RecipeExecution.__table__.c.execution_id == run_id).values(status=status))


def test_no_post_is_made_while_the_plans_research_runs(playbooks):  # noqa: F811
    from tests.test_prd251bw2_maker import _topic

    plan = _text_plan(playbooks)
    _topic(playbooks, plan, "Three weeks to Lisbon")
    run_id = _run(playbooks, plan, status="running", started_at=NOW - timedelta(minutes=5))

    assert maker_mod.collect_due(NOW) == []
    assert maker_mod.collect_due(NOW + outcome.RESEARCH_WAIT) != []  # a run that never ends holds nothing for long
    _set_status(playbooks, run_id, "completed")
    assert maker_mod.collect_due(NOW) == [(uuid.UUID(plan["id"]), "r1|2026-10-14|09:00")]


def test_a_finished_run_that_gave_the_bank_nothing_is_told_once_in_its_words(playbooks):  # noqa: F811
    plan = _text_plan(playbooks)
    later = datetime.now(UTC) + timedelta(hours=1)  # after every topic's clock
    _run(playbooks, plan, status="completed", started_at=later,
         final_output="No topics were added. This session doesn't have the two tools the ticket depends on.\nMore.")

    assert outcome.check_finished(playbooks.factory) == 1
    assert outcome.check_finished(playbooks.factory) == 0  # the plan remembers the run
    (event, title) = playbooks.told.plan[-1]
    assert event == outcome.EMPTY_EVENT[0]
    assert title == "Countdown to Lisbon: No topics were added. This session doesn't have the two tools the ticket depends on."


def test_a_finished_run_that_gave_facts_is_not_told(playbooks):  # noqa: F811
    from tests.test_prd251bw2_maker import _topic

    plan = _text_plan(playbooks)
    _run(playbooks, plan, status="completed", started_at=datetime.now(UTC) - timedelta(hours=1))
    _topic(playbooks, plan, "Three weeks to Lisbon")  # research's topic, with its fact

    assert outcome.check_finished(playbooks.factory) == 0 and playbooks.told.plan == []


def test_an_unfinished_run_is_not_checked_yet():
    plan = NS(research={"last_run_id": "r"}, workspace_id=WS_A, id="p", name="P")
    db = NS(query=lambda *a: NS(filter=lambda *c: NS(first=lambda: NS(status="running", execution_id="r",
                                                                       started_at=NOW, output_data={}))))
    assert outcome.tell_if_empty(db, plan) is False


# ── the composer writes the template's words ───────────────────────────────


def test_the_composer_writes_every_word_and_only_a_claim_waits_for_a_fact():
    for note in (compose.FACTS_NOTE, compose.FILL_NOTE):
        assert compose.TEMPLATE_WORDS_NOTE in note and "never leave it empty" in note
    assert "answer null" in compose.FILL_NOTE and "variable marked claim" in compose.FILL_NOTE
    assert "states no figure, result or customer" in compose.TEMPLATE_WORDS_NOTE


# ── a refused template is tried once with another ──────────────────────────


def _in_a_worker(fn):
    async def main():
        return await anyio.to_thread.run_sync(fn)
    return anyio.run(main)


def _refused_rig(monkeypatch, picked, refusal=None):
    calls = NS(proposed=[], edited=[], rendered=0)

    def propose(db, plan, slot, topic, now, visual_slots=(), **options):
        calls.proposed.append(options)
        return {"title": "Go", "copy": {"base": "Go.", "channels": {}}, "variables": {"headline": {"value": "GO"}},
                "sources": {}, "template_id": picked}

    async def edit_post(db, post, actor, changes, agent=None):
        calls.edited.append(changes)

    async def render_post(db, workspace, post, actor):
        calls.rendered += 1
        if refusal is not None:
            raise refusal

    monkeypatch.setattr(maker_mod, "_propose", propose)
    monkeypatch.setattr(maker_mod, "_posts_api", lambda: NS(edit_post=edit_post, render_post=render_post))
    return calls


def _another(post):
    plan, slot = NS(created_by="owner", id="plan"), NS(template_id="fact-card")
    return _in_a_worker(lambda: maker_mod._another_template(None, None, plan, slot, NS(), post, NOW))


def test_a_refused_template_is_tried_once_with_another_that_auto_picks(monkeypatch):
    calls = _refused_rig(monkeypatch, picked="image-card")
    proposal, refused = _another(NS(template_id="fact-card", id="post"))
    assert refused is None and calls.rendered == 1 and proposal["template_id"] == "image-card"
    assert calls.proposed == [{"instead_of": "fact-card"}] and calls.edited[0]["template_id"] == "image-card"


def test_no_other_template_picked_leaves_the_post_to_wait(monkeypatch):
    calls = _refused_rig(monkeypatch, picked="fact-card")
    assert _another(NS(template_id="fact-card", id="post")) is None and calls.edited == []
    assert _another(NS(template_id=None, id="post")) is None


def test_a_second_refusal_is_handed_back_for_the_owner(monkeypatch):
    again = render.NotRenderable("Fill in Headline before rendering.", missing=("headline",))
    calls = _refused_rig(monkeypatch, picked="image-card", refusal=again)
    _proposal, refused = _another(NS(template_id="fact-card", id="post"))
    assert refused is again and calls.rendered == 1


# ── the research prompt ────────────────────────────────────────────────────


def test_the_research_prompt_researches_the_owners_topics_first_and_names_its_actions():
    (step,) = next(p for p in seed.SOCIALS_PLAYBOOKS if p["template_id"] == seed.RESEARCH_PLAYBOOK_TEMPLATE_ID)["steps"]
    prompt = step["prompt_template"]
    assert prompt == prompts.RESEARCH_PROMPT and '"facts": 0' in prompt and "the bank gives that topic its facts" in prompt
    named = named_actions.named_in(prompt)
    assert {"platform_get_social_plan", "platform_add_social_topics", "platform_list_deliverables"} <= set(named)


def test_a_copy_holding_the_prompt_seeded_until_today_takes_the_new_one():
    assert prompts.RESEARCH_PROMPT_F383 in seed.SEEDED_BEFORE[seed.RESEARCH_PLAYBOOK_TEMPLATE_ID]["research"]
    assert prompts.RESEARCH_PROMPT_F383 != prompts.RESEARCH_PROMPT
    assert seed._NEVER_PUBLISH == prompts.NEVER_PUBLISH

