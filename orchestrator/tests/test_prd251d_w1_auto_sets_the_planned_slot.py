"""PRD-251D Wave 1, US-D106 (E6): "post it Friday at nine" puts the post on Friday at nine, waiting for approval.

The create and update tools took no day or time, and the weekly Playbook wrote "Proposed for: <day and
time>" into the brief as text. Now both tools take ``planned_for`` (an ISO date and time) and
``timezone`` (an IANA zone): the post's planned slot, where it waits in the Socials calendar until a
person approves it. Pinned, on the US-116 harness (SQLite, the real router and executor):

* the rule for the time (``social_post_checks.planned_slot``): no offset is the wall-clock time in
  timezone, else in the post's own zone; an offset is that moment; no zone at all, a past time, an
  unknown zone and a day with no time are refused with the reason, and nothing is saved;
* a created post carries the slot and an update moves it on a draft and on a post waiting for
  approval, written with ``service.set_planned_for`` alone: the content hash, the status and the
  history are untouched, and the post is not scheduled;
* an approved or scheduled post's slot is refused in plain words and left as it was, the rest of the
  call with it;
* only the owner's approval schedules the post into its slot: no tool call reaches ``schedule``,
  ``reslot``, ``apply_slot``, ``approve`` or the publisher;
* every answer, ``platform_get_social_post`` and ``platform_list_social_posts`` say the slot in the
  post's timezone, in words;
* the weekly Playbook sets planned_for, and an installed copy nobody edited takes the new prompts.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from tests.test_prd251w1_social_post_tools import (  # its environment first
    _created,
    _draft_fields,
    _names_in,
    _posts,
    _row,
    _run,
    _template,
    _tool,
)
from tests.test_prd251w1_social_post_tools import env as _env

import api.socials as socials_api  # noqa: E402
import api.socials_slots as socials_slots  # noqa: E402
import modules.socials.publish_lifecycle as publish_lifecycle  # noqa: E402
import modules.socials.publishing as publishing  # noqa: E402
from core.seeds import seed_socials_package as seed  # noqa: E402
from modules.socials import service  # noqa: E402
from modules.tools.discovery import (  # noqa: E402
    handlers_socials,
    social_post_checks,
    social_post_list,
    social_post_results,
)
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402
from modules.tools.discovery.social_post_checks import planned_slot, slot_asked, slot_held  # noqa: E402
from modules.tools.discovery.social_post_list import slot_words  # noqa: E402

env = _env  # the US-116 fixture: SQLite, the real router and executor

NEW_YORK, TOKYO = "America/New_York", "Asia/Tokyo"  # zones whose offset is never 0, so a zone read wrong shows
NOW = datetime(2026, 11, 16, 14, 5, tzinfo=timezone.utc)  # a Monday
NEXT_YEAR = datetime.now(timezone.utc).year + 1
AHEAD = f"{NEXT_YEAR}-11-20T09:00"  # always ahead: 09:00 in New York is 14:00 UTC (November is past the DST change)
AHEAD_UTC = datetime(NEXT_YEAR, 11, 20, 14, 0, tzinfo=timezone.utc)
LATER = f"{NEXT_YEAR}-11-21T10:30"
LATER_UTC = datetime(NEXT_YEAR, 11, 21, 15, 30, tzinfo=timezone.utc)
PAST = f"{NEXT_YEAR - 2}-11-20T09:00"  # an agent's wrong year
AHEAD_WORDS = f"{datetime(NEXT_YEAR, 11, 20):%A} 20 November {NEXT_YEAR}, 09:00 in America/New_York"

# What only a person may reach (D6, D14): the review, the schedule, the slot route's reschedule, the way out.
NOT_FOR_TOOLS = {"approve", "schedule", "unschedule", "reslot", "apply_slot", "start_publish", "run_publish",
                 "publish_target", "approve_social_post", "schedule_social_post", "publish_social_post_now"}


def _utc(value):
    """A stored slot as UTC: SQLite hands DateTime(timezone=True) back without its offset."""
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


def _text_post(env, **fields):
    """A text post a person could approve as it is (no template, no claims)."""
    return _created(_tool(env, "platform_create_social_post", title="Friday's post", format="text",
                          copy={"base": "See you on Friday."}, render=False, **fields))


def _approve(env, post_id):
    row = _row(env, post_id)
    answer = env.client.post(f"/api/socials/posts/{post_id}/approve", json={"content_hash": row.content_hash})
    assert answer.status_code == 200, answer.text
    return answer.json()


# ---------------------------------------------------------------------------
# The rule for the time (no database)
# ---------------------------------------------------------------------------


def test_a_time_without_an_offset_is_read_in_its_timezone():
    assert planned_slot("2030-11-22T09:00", NEW_YORK, now=NOW) == (
        (datetime(2030, 11, 22, 14, 0, tzinfo=timezone.utc), NEW_YORK), None)
    assert planned_slot("2030-11-22 09:00", TOKYO, now=NOW) == (
        (datetime(2030, 11, 22, 0, 0, tzinfo=timezone.utc), TOKYO), None)


def test_a_time_with_an_offset_is_that_moment_shown_in_the_zone():
    (at, zone), problem = planned_slot("2030-11-22T09:00:00Z", NEW_YORK, now=NOW)
    assert problem is None and (at, zone) == (datetime(2030, 11, 22, 9, 0, tzinfo=timezone.utc), NEW_YORK)
    assert slot_words(at, zone) == "Friday 22 November 2030, 04:00 in America/New_York"


def test_left_out_the_timezone_is_the_posts_own_and_with_none_the_slot_is_refused():
    assert planned_slot("2030-11-22T09:00", None, TOKYO, now=NOW) == (
        (datetime(2030, 11, 22, 0, 0, tzinfo=timezone.utc), TOKYO), None)
    assert planned_slot("2030-11-22T09:00", "", TOKYO, now=NOW)[0][1] == TOKYO
    # The call's own timezone wins over the post's.
    assert planned_slot("2030-11-22T09:00", NEW_YORK, TOKYO, now=NOW)[0] == (
        datetime(2030, 11, 22, 14, 0, tzinfo=timezone.utc), NEW_YORK)
    for sent in ("2030-11-22T09:00", "2030-11-22T09:00:00+00:00"):  # nothing guesses a zone, even beside an offset
        assert planned_slot(sent, None, None, now=NOW) == (None, social_post_checks.NO_SLOT_ZONE)
    assert planned_slot("2030-11-22T09:00", None, "Not/AZone", now=NOW) == (None, social_post_checks.NO_SLOT_ZONE)


def test_a_past_time_an_unknown_zone_and_a_day_without_a_time_are_refused_with_why():
    slot, problem = planned_slot("2025-11-21T09:00", NEW_YORK, now=NOW)
    assert slot is None and problem.startswith("planned_for is in the past: Friday 21 November 2025, 09:00 in "
                                               "America/New_York, and it is now Monday 16 November 2026, 09:05 in "
                                               "America/New_York.")
    assert "check the year" in problem and problem.endswith("Nothing was saved.")
    assert planned_slot("2026-11-16T09:05", NEW_YORK, now=NOW)[0] is None  # this very minute is not ahead
    assert planned_slot("2030-11-22T09:00", "Mars/Olympus", now=NOW) == (
        None, "timezone 'Mars/Olympus' is not an IANA timezone, such as Europe/Dublin or America/New_York. "
              "Nothing was saved.")
    assert planned_slot("2030-11-22", NEW_YORK, now=NOW) == (
        None, "planned_for '2030-11-22' is a day with no time: send the time too, such as 2030-11-22T09:00. "
              "Nothing was saved.")
    for sent in ("Friday at nine", 20301122, ["2030-11-22T09:00"]):
        slot, problem = planned_slot(sent, NEW_YORK, now=NOW)
        assert slot is None and "is not an ISO date and time, written YYYY-MM-DDTHH:MM" in problem, sent


def test_no_planned_for_is_no_slot_and_a_timezone_alone_changes_nothing():
    for sent in (None, ""):
        assert planned_slot(sent, NEW_YORK, now=NOW) == (None, None)
        assert slot_asked(sent, NEW_YORK, NS(id="p", status="approved", timezone=None), now=NOW) == (None, None)


def test_the_slot_is_said_in_words_in_the_posts_own_timezone():
    # A stored slot without an offset is UTC (the database's convention), never the machine's own zone.
    assert slot_words("2030-11-22T14:00:00", NEW_YORK) == "Friday 22 November 2030, 09:00 in America/New_York"
    assert slot_words("2030-11-22T14:00:00+00:00", NEW_YORK) == "Friday 22 November 2030, 09:00 in America/New_York"
    assert slot_words("2030-11-22T14:00:00+00:00", None) == "Friday 22 November 2030, 14:00 in UTC"
    assert slot_words(None, NEW_YORK) is None and slot_words("", NEW_YORK) is None


@pytest.mark.parametrize("status", sorted(socials_slots.SLOT_EDIT_STATUSES))
def test_a_draft_a_post_waiting_for_approval_and_one_sent_back_take_a_slot(status):
    assert slot_held(NS(status=status)) is None


@pytest.mark.parametrize(("status", "why"), [
    ("approved", "is approved: the owner moves it in Socials"),
    ("scheduled", "is scheduled: the owner moves it in Socials"),
    ("missed", "missed its slot: the owner gives it a new one in Socials"),
    ("rendering", "is rendering: set its slot once the render has finished"),
    ("failed", "failed its render: render it again (render true), then set its slot"),
    ("publishing", "is publishing: its slot can no longer move"),
    ("published", "is published: its slot can no longer move"),
    ("partially_published", "is partially published: its slot can no longer move"),
    ("archived", "is archived: its slot can no longer move"),
])
def test_any_other_post_takes_no_slot_from_a_tool_said_in_plain_words(status, why):
    assert slot_held(NS(status=status)) == why
    assert slot_asked("2030-11-22T09:00", NEW_YORK, NS(id="p1", status=status, timezone=None), now=NOW) == (
        None, f"planned_for was refused, and nothing was saved: post p1 {why}.")


# ---------------------------------------------------------------------------
# The tools' parameters
# ---------------------------------------------------------------------------


def test_create_and_update_take_the_slot_and_say_what_it_is():
    registry = get_action_registry()
    for name in ("platform_create_social_post", "platform_update_social_post"):
        props = registry.get(name).parameters["properties"]
        assert props["planned_for"]["type"] == props["timezone"]["type"] == "string", name
        said = props["planned_for"]["description"]
        for words in ("ISO date and time", "slot in the Socials calendar", "the owner still approves the post",
                      "approving it schedules it into that slot", "A time already past is refused"):
            assert words in said, (name, words)
        assert "IANA timezone" in props["timezone"]["description"], name


# ---------------------------------------------------------------------------
# The tools set the slot alone
# ---------------------------------------------------------------------------


def test_a_created_post_carries_its_slot_and_is_not_scheduled(env):
    made = _tool(env, "platform_create_social_post", title="Friday's post", format="text",
                 copy={"base": "See you on Friday."}, render=False, planned_for=AHEAD, timezone=NEW_YORK)

    post = _created(made)
    row = _row(env, post["id"])
    assert _utc(row.planned_for) == AHEAD_UTC and row.timezone == NEW_YORK
    assert (row.status, row.scheduled_for, row.approved_hash) == ("draft", None, None)
    assert [entry["action"] for entry in row.review_log] == ["draft"]  # the slot is no edit of the content
    assert post["planned"] == AHEAD_WORDS
    assert made["slot"] == social_post_results.PLANNED_NOTE.format(when=AHEAD_WORDS)
    assert "it is not scheduled until then" in made["slot"]
    # The slot is not content: the same post made without one has the same hash.
    unplanned = _row(env, _text_post(env)["id"])
    assert row.content_hash == unplanned.content_hash == service.compute_content_hash(row)
    assert unplanned.planned_for is None


def test_an_update_moves_the_slot_of_a_draft_and_of_a_post_waiting_for_approval(env):
    post = _text_post(env)
    before = _row(env, post["id"])
    status, content_hash, log = before.status, before.content_hash, list(before.review_log)

    moved = _tool(env, "platform_update_social_post", post_id=post["id"], planned_for=AHEAD, timezone=NEW_YORK)

    assert moved["success"] is True, moved
    row = _row(env, post["id"])
    assert _utc(row.planned_for) == AHEAD_UTC and row.timezone == NEW_YORK
    assert (row.status, row.content_hash, row.review_log) == (status, content_hash, log)
    assert moved["post"]["planned"] == AHEAD_WORDS and "not scheduled until then" in moved["slot"]

    assert _tool(env, "platform_submit_social_post", post_id=post["id"])["success"] is True
    waiting = _row(env, post["id"])
    content_hash, log = waiting.content_hash, list(waiting.review_log)  # values: _row hands back the same object
    again = _tool(env, "platform_update_social_post", post_id=post["id"], planned_for=LATER)  # the post's own zone

    assert again["success"] is True, again
    row = _row(env, post["id"])
    assert _utc(row.planned_for) == LATER_UTC and row.timezone == NEW_YORK
    assert (row.status, row.content_hash, row.approved_hash) == ("needs_approval", content_hash, None)
    assert row.review_log == log and row.scheduled_for is None

    # A timezone alone moves nothing, and says so.
    alone = _tool(env, "platform_update_social_post", post_id=post["id"], timezone=TOKYO)
    assert alone == {"success": False, "error": handlers_socials.NOTHING_TO_CHANGE}
    assert _row(env, post["id"]).timezone == NEW_YORK


def test_a_change_and_a_slot_in_one_update_save_both_and_the_slot_is_no_edit(env):
    post = _text_post(env)

    both = _tool(env, "platform_update_social_post", post_id=post["id"], copy={"base": "See you on Saturday."},
                 planned_for=AHEAD, timezone=NEW_YORK)

    assert both["success"] is True, both
    row = _row(env, post["id"])
    assert row.copy == {"base": "See you on Saturday."} and _utc(row.planned_for) == AHEAD_UTC
    assert [entry["action"] for entry in row.review_log] == ["draft", "edit"] and row.review_log[-1]["fields"] == ["copy"]
    assert row.status == "draft" and row.content_hash == service.compute_content_hash(row)
    assert both["post"]["planned"] == AHEAD_WORDS


def test_a_post_made_on_a_template_renders_and_waits_for_approval_in_its_slot(env):
    template_id = _template(env)

    made = _tool(env, "platform_create_social_post", **_draft_fields(), template=str(template_id),
                 planned_for=AHEAD, timezone=NEW_YORK)  # render is the default

    assert made["success"] is True and made["render"] == {"started": True}, made
    assert made["post"]["planned"] == AHEAD_WORDS and "not scheduled until then" in made["slot"]
    (job,) = env.launched
    done, _renderer, _store = _run(job, env)
    assert done is True
    row = _row(env, made["post"]["id"])
    assert (row.status, _utc(row.planned_for), row.scheduled_for) == ("needs_approval", AHEAD_UTC, None)


def test_a_slot_the_commit_refuses_after_the_save_fails_the_call_and_says_the_post_was_saved(env, monkeypatch):
    def _raced(db, post, *, status, content_hash):
        raise service.StaleContent("0" * 64)  # another writer committed between the save and the slot

    monkeypatch.setattr(socials_api, "_commit_unchanged", _raced)  # create_post commits on its own
    made = _tool(env, "platform_create_social_post", title="Raced", format="text", copy={"base": "Hello."},
                 render=False, planned_for=AHEAD, timezone=NEW_YORK)

    assert made["success"] is False and made["error"].startswith(
        f'Saved post {made["post"]["id"]} ("Raced"), but its planned slot was NOT set: the post changed since you '
        "opened it"), made
    row = _row(env, made["post"]["id"])
    assert row is not None and (row.status, row.planned_for) == ("draft", None) and made["post"]["planned"] is None


def test_an_approved_or_scheduled_posts_slot_is_refused_and_left_as_it_was(env):
    planned = _text_post(env, planned_for=AHEAD, timezone=NEW_YORK)
    assert _tool(env, "platform_submit_social_post", post_id=planned["id"])["success"] is True
    scheduled = _approve(env, planned["id"])  # the owner's approval schedules it into its slot
    assert scheduled["status"] == "scheduled" and scheduled["scheduled_for"] == scheduled["planned_for"]

    refused = _tool(env, "platform_update_social_post", post_id=planned["id"], planned_for=LATER, timezone=NEW_YORK)

    assert refused == {"success": False, "error": (
        f"planned_for was refused, and nothing was saved: post {planned['id']} is scheduled: the owner moves it "
        "in Socials.")}
    row = _row(env, planned["id"])
    assert _utc(row.planned_for) == _utc(row.scheduled_for) == AHEAD_UTC and row.status == "scheduled"
    assert row.approved_hash == row.content_hash

    approved = _text_post(env)
    assert _tool(env, "platform_submit_social_post", post_id=approved["id"])["success"] is True
    assert _approve(env, approved["id"])["status"] == "approved"
    both = _tool(env, "platform_update_social_post", post_id=approved["id"], copy={"base": "See you Saturday."},
                 planned_for=AHEAD, timezone=NEW_YORK)

    assert both["success"] is False and "is approved: the owner moves it in Socials" in both["error"]
    row = _row(env, approved["id"])  # the copy was not saved either, so the approval stands
    assert (row.status, row.planned_for, row.copy) == ("approved", None, {"base": "See you on Friday."})
    assert row.approved_hash == row.content_hash


def test_a_bad_timezone_and_a_past_time_are_refused_and_nothing_is_saved(env):
    before = _posts(env)
    for sent, words in (({"planned_for": AHEAD, "timezone": "Mars/Olympus"}, "is not an IANA timezone"),
                        ({"planned_for": PAST, "timezone": NEW_YORK}, "planned_for is in the past"),
                        ({"planned_for": AHEAD}, "planned_for needs its timezone")):
        made = _tool(env, "platform_create_social_post", title="Refused", format="text", render=False, **sent)
        assert made["success"] is False and words in made["error"], made
    assert _posts(env) == before

    post = _text_post(env)
    for sent in ({"planned_for": PAST, "timezone": NEW_YORK}, {"planned_for": AHEAD, "timezone": "Mars/Olympus"}):
        changed = _tool(env, "platform_update_social_post", post_id=post["id"], copy={"base": "Changed."}, **sent)
        assert changed["success"] is False and changed["error"].endswith("Nothing was saved."), changed
    row = _row(env, post["id"])
    assert row.planned_for is None and row.copy == {"base": "See you on Friday."}


def test_no_tool_call_schedules_approves_or_publishes_only_the_owners_approval_does(env, monkeypatch):
    written = []
    real = service.set_planned_for
    monkeypatch.setattr(service, "set_planned_for", lambda post, *slot: written.append(slot) or real(post, *slot))
    with pytest.MonkeyPatch.context() as only_people:
        def _never(*args, **kwargs):
            raise AssertionError("a tool reached a step only a person may take")

        for target, name in ((service, "schedule"), (service, "reslot"), (service, "approve"),
                             (socials_slots, "apply_slot"), (publish_lifecycle, "start_publish"),
                             (publishing, "run_publish")):
            only_people.setattr(target, name, _never)
        post = _text_post(env, planned_for=AHEAD, timezone=NEW_YORK)
        assert _tool(env, "platform_update_social_post", post_id=post["id"], planned_for=LATER)["success"] is True
        assert _tool(env, "platform_submit_social_post", post_id=post["id"])["success"] is True
    assert [_utc(at) for at, _zone in written] == [AHEAD_UTC, LATER_UTC]
    assert _row(env, post["id"]).scheduled_for is None
    assert _approve(env, post["id"])["status"] == "scheduled"  # the owner's approval, outside the tools

    # Nor does anything the tools' own modules name: the slot is written with set_planned_for alone.
    names = set()
    for module in (handlers_socials, social_post_checks, social_post_results, social_post_list):
        names |= _names_in(inspect.getsource(module))
    assert names & NOT_FOR_TOOLS == set()
    assert "set_planned_for" in _names_in(inspect.getsource(handlers_socials))


def test_get_and_list_say_the_slot_in_the_posts_timezone(env):
    post = _text_post(env, planned_for=AHEAD, timezone=NEW_YORK)
    plain = _text_post(env)

    read = _tool(env, "platform_get_social_post", post_id=post["id"])["post"]
    assert read["planned"] == AHEAD_WORDS and read["timezone"] == NEW_YORK
    rows = {row["id"]: row for row in _tool(env, "platform_list_social_posts")["posts"]}
    assert rows[post["id"]]["planned"] == AHEAD_WORDS and rows[plain["id"]]["planned"] is None
    assert _tool(env, "platform_get_social_post", post_id=plain["id"])["post"]["planned"] is None


# ---------------------------------------------------------------------------
# The weekly Playbook and Auto's turn note
# ---------------------------------------------------------------------------


(WEEKLY,) = [spec for spec in seed.SOCIALS_PLAYBOOKS if spec["template_id"] == seed.WEEKLY_PLAYBOOK_TEMPLATE_ID]


class _Rows:
    """db.query(...).filter(...): the marketplace row (first) and its workspace copies (all)."""

    def __init__(self, row, copies):
        self.row, self.copies = row, copies

    def query(self, *entities):
        return self

    def filter(self, *clauses):
        return self

    def first(self):
        return self.row

    def all(self):
        return self.copies


def _steps(plan_prompt, draft_prompt):
    plan, draft = WEEKLY["steps"]
    return [{"step_id": plan["step_id"], "order": 1, "agent_id": 412, "prompt_template": plan_prompt},
            {"step_id": draft["step_id"], "order": 2, "agent_id": 412, "prompt_template": draft_prompt}]


def test_the_weekly_playbook_sets_the_slot_instead_of_writing_it_in_the_brief():
    plan, draft = (step["prompt_template"] for step in WEEKLY["steps"])
    assert "Proposed for:" not in draft and "Proposed for: <day and time>" in seed._WEEKLY_DRAFT_PROMPT_251
    assert "planned_for" in draft and "timezone" in draft and "approving it schedules it there" in draft
    assert '"proposed_for" (YYYY-MM-DDTHH:MM), "timezone" (its IANA zone)' in plan
    assert seed._NEVER_PUBLISH in draft  # still never publishes, schedules or approves


def test_an_unedited_copy_takes_the_new_weekly_prompts_and_an_edited_one_keeps_its_own():
    plan, draft = (step["prompt_template"] for step in WEEKLY["steps"])
    unedited = NS(id=1, workspace_id="ws-1", owner_type="workspace",
                  steps=_steps(seed._WEEKLY_PLAN_PROMPT_251, seed._WEEKLY_DRAFT_PROMPT_251))
    edited = NS(id=2, workspace_id="ws-2", owner_type="workspace",
                steps=_steps("Our own plan: the roastery's week.", seed._WEEKLY_DRAFT_PROMPT_251))
    row = NS(id=99, owner_type="marketplace", steps=_steps(seed._WEEKLY_PLAN_PROMPT_251, seed._WEEKLY_DRAFT_PROMPT_251))

    assert seed._ensure_playbook(_Rows(row, [unedited, edited]), WEEKLY, {}) == seed.UPDATED

    assert [step["prompt_template"] for step in row.steps] == [plan, draft]
    assert [step["prompt_template"] for step in unedited.steps] == [plan, draft]
    assert [step["agent_id"] for step in unedited.steps] == [412, 412]  # the copy's own agent stays
    assert [step["prompt_template"] for step in edited.steps] == ["Our own plan: the roastery's week.", draft]
    assert seed.refreshed_steps(WEEKLY, row.steps) is None  # up to date now


def test_autos_turn_note_says_how_a_day_and_time_is_set_and_that_it_is_not_scheduled():
    from consumers.chatbot.socials_turn_note import socials_note

    note = socials_note(["Make a LinkedIn post about the Harvest Club and post it Friday at nine."], [])
    assert "send planned_for" in note and "timezone, the IANA zone of today's date line" in note
    assert "never say it is scheduled before they approve" in note
