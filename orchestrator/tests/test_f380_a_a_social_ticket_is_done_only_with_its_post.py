"""F380 (night 11, 7 Oct): a social ticket is done only when its post exists and has rendered.

Night 11's Social Media Director tickets closed done with nothing in Socials: #2158's
first run answered with a one-line caption, and #2160 made a generate_document PNG and
said the draft was "waiting for your approval in the Socials tab". The posts it did save
had no render; the owner rendered each one by hand. The owner: "'done' must mean the
post exists and has rendered."

Now a ticket whose brief asks for a post, or whose answer says one was made, goes to
review when the run saved no post (saying a generate_document image is not one), or
when its post's render failed or was refused. A post saved without its render gets its
render started, and the ticket says so. A question, a ticket with Socials off and any
other ticket close as before.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from core.services import ticket_reasons as tr
from services import social_ticket_post as stp
from services.social_post_asks import asks_for_a_social_post, claims_a_social_post, made_a_document_image

WS = "febae41b-374b-4580-a5ef-f698bdd382e4"
AGENT = 348
ACTOR = f"agent:{AGENT}"
STARTED = datetime(2026, 10, 7, 1, 8, 0, tzinfo=timezone.utc)

TITLE_2158 = "September numbers card for Instagram"
BRIEF_2158 = "A stats card: 1,240 bags roasted, 412 subscribers, 31 wholesale cafés. Draft only."
ANSWER_2158 = "Here's a snapshot of our September activity. Thanks for your continued support! #SeptemberStats"
TITLE_2160 = "15-second video: September top three cafés"
ANSWER_2160 = ("I have drafted an Instagram image post using the 'Stats card' template. This draft is now waiting "
               "for your approval in the Socials tab: http://localhost:9000/automatos/generated-documents/c.png")


# ── which tickets ask for a post, and which answers say one was made ────────

@pytest.mark.parametrize("title, description", [
    (TITLE_2158, BRIEF_2158),
    ("Quote card: Rosa at Lantern Kitchen", "Her words: Harbour Blend by name now."),
    (TITLE_2160, "Lantern Kitchen 62 kg, Gull & Anchor 48 kg, The Copper Kettle 40 kg."),
    ("Wholesale offer post for LinkedIn", "First 5 kg of Harbour Blend at £16/kg."),
    ("Before and after: the Guji re-roast", "An Instagram before/after post. Ask me for the photos."),
])
def test_night_11s_director_briefs_ask_for_a_post(title, description):
    assert asks_for_a_social_post(title, description)


@pytest.mark.parametrize("title, description", [
    ("Summarize last week's Instagram posts", "Which did best?"),
    ("Plan next week's social posts", "Five ideas for the café posts."),
    ("Upload Guji Re-Roast Photos", "The photos for the Instagram post."),
    ("Align Socials Visuals with Documents", "Make the social cards look like the invoice."),
    ("Caption for the Guji post", "Caption only, for Instagram."),
    ("Thank-you email to Rosa", "Thank her for the quote she gave us."),
    ("Generate Brand Board document", "Add sample social cards on the brand board."),
])
def test_briefs_about_posts_that_ask_for_none_are_left_alone(title, description):
    assert not asks_for_a_social_post(title, description)


def test_2160s_answer_claims_a_post_and_links_a_document_image():
    assert claims_a_social_post(ANSWER_2160)
    assert made_a_document_image(ANSWER_2160, [])
    assert made_a_document_image("Done.", ["generate_document"])
    assert not claims_a_social_post("The invoice is in Deliverables, waiting for you.")
    assert not made_a_document_image("The post is in Socials.", ["platform_create_social_post"])


# ── a post's render, and which posts are the run's ──────────────────────────

def _post(**fields):
    base = dict(id=uuid4(), title="September at a Glance", status="draft", media={}, template_id=uuid4(),
                format="image", review_log=[], created_by=ACTOR, created_at=STARTED + timedelta(seconds=20))
    return NS(**{**base, **fields})


def test_where_a_post_stands_on_its_render():
    failed = [{"action": "render", "by": ACTOR}, {"action": "render_failed", "comment": "media-render timed out"}]

    assert stp.render_state(_post(media={"4:5": [{"deliverable_id": "d1"}]})) == (stp.RENDERED, "")
    assert stp.render_state(_post(status="rendering")) == (stp.UNDER_WAY, "")
    assert stp.render_state(_post(template_id=None, format="text")) == (stp.NO_RENDER_NEEDED, "")
    assert stp.render_state(_post(status="failed", review_log=failed)) == (stp.FAILED_RENDER, "media-render timed out.")
    assert stp.render_state(_post(status="needs_approval")) == (stp.UNRENDERED, "")   # submitted with no picture
    assert stp.render_state(_post(status="approved")) == (stp.LEFT_AS_IT_IS, "")


def test_a_post_is_the_runs_when_its_agent_made_or_changed_it_during_the_run():
    since = STARTED
    edit = {"at": (STARTED + timedelta(minutes=1)).isoformat(), "by": ACTOR, "action": "edit"}

    assert stp.made_or_changed_by(_post(), ACTOR, since)
    assert stp.made_or_changed_by(_post(created_by="user:1", created_at=STARTED - timedelta(days=1),
                                        review_log=[edit]), ACTOR, since)
    assert not stp.made_or_changed_by(_post(created_by="agent:347"), ACTOR, since)
    assert not stp.made_or_changed_by(_post(created_at=STARTED - timedelta(hours=1)), ACTOR, since)


# ── the completion writer ───────────────────────────────────────────────────

@pytest.fixture
def finish(monkeypatch):
    """A Social Media Director ticket, review mode auto, ending through finalize_board_task_run."""
    from api import board_tasks
    import services.result_files as result_files
    import services.ticket_owner_ask as ticket_owner_ask

    async def _no(*a, **k):
        return False

    async def _none(*a, **k):
        return None

    monkeypatch.setattr(ticket_owner_ask, "park_if_the_result_asks", _no)
    monkeypatch.setattr(result_files, "check_named_files", _none)
    monkeypatch.setattr(board_tasks, "_dispatch_task_complete", _none)
    monkeypatch.setattr(board_tasks, "_auto_create_task_report", _none)
    monkeypatch.setattr(stp, "socials_on", lambda db, ws: NS(id=UUID(WS)))
    renders = []

    def run(answer, *, title=TITLE_2158, description=BRIEF_2158, posts=(), render=(True, ""), actions=None):
        async def _render(db, workspace, post, actor):
            renders.append((post.title, actor))
            return render

        monkeypatch.setattr(stp, "runs_posts", lambda db, ws, actor, since: list(posts))
        monkeypatch.setattr(stp, "start_render", _render)
        task = NS(id=2158, status="in_progress", result=None, error_message=None, completed_at=None,
                  runtime_ref=None, review_mode="auto", source_type="user", source_id=None, title=title,
                  description=description, started_at=STARTED, created_at=STARTED, assigned_agent_id=AGENT)
        session = NS(get=lambda *a, **k: task, commit=lambda: None, rollback=lambda: None)
        execution = {"execution": {"actions": actions}} if actions is not None else {}
        asyncio.run(board_tasks.finalize_board_task_run(
            session, task_id=2158, workspace_id=WS, agent_id=AGENT,
            exec_result={"status": "success", "result": answer, **execution}, review_mode="auto"))
        return task

    run.renders = renders
    return run


def test_2158s_caption_with_no_post_goes_to_review(finish):
    task = finish(ANSWER_2158)

    assert task.status == "review"                                       # night 11: done
    assert task.result.startswith(ANSWER_2158)                           # nothing the agent wrote is lost
    assert task.result.endswith(stp.NO_POST_NOTE.format(image=""))
    assert tr.review_reason(task) == tr.NOTHING_DONE


def test_2160s_document_image_is_not_a_post(finish):
    task = finish(ANSWER_2160, title=TITLE_2160, description="Top three cafés by kilos.",
                  actions=["generate_document", "platform_list_templates"])

    assert task.status == "review"
    assert stp.IMAGE_IS_NOT_A_POST in task.result


def test_an_answer_that_claims_a_post_needs_one_whatever_the_brief(finish):
    answer = "Your Instagram post is drafted and waiting for your approval in the Socials tab."
    task = finish(answer, title="Something for Rosa", description="She was kind to us this week.")

    assert task.status == "review"
    assert "no Socials post was saved in this run" in task.result


def test_a_post_saved_without_its_render_gets_it_started(finish):
    task = finish("Your Instagram post draft has been created.", posts=[_post()])

    assert finish.renders == [("September at a Glance", ACTOR)]
    assert task.status == "done"
    assert task.result.endswith(stp.RENDER_STARTED_NOTE.format(title="September at a Glance"))


def test_a_render_that_is_refused_sends_it_back_with_the_reason(finish):
    refused = (False, "fill in headline, point_1_title before rendering.")
    task = finish("Your Instagram post draft has been created.", posts=[_post()], render=refused)

    assert task.status == "review"
    assert 'the post "September at a Glance" was saved but did not render: fill in headline' in task.result
    assert "platform_update_social_post with render true" in task.result


def test_a_failed_render_sends_it_back_with_the_renderers_reason(finish):
    failed = [{"action": "render_failed", "comment": "media-render timed out", "by": ACTOR}]
    task = finish("The post is in Socials.", posts=[_post(status="failed", review_log=failed)])

    assert task.status == "review"
    assert "did not render: media-render timed out." in task.result
    assert finish.renders == []                                          # a failed render is not retried blind


@pytest.mark.parametrize("post", [
    _post(media={"4:5": [{"deliverable_id": "d1"}]}),
    _post(status="rendering"),
])
def test_a_rendered_post_or_one_rendering_closes_done_as_it_is(finish, post):
    answer = "The Stats card post is rendering and waits for your approval in Socials."
    task = finish(answer, posts=[post])

    assert (task.status, task.result) == ("done", answer)


def test_socials_off_and_other_tickets_close_as_before(finish, monkeypatch):
    task = finish("Thanks Rosa, the 12 kg goes Monday.", title="Reply to Rosa", description="About her order.")
    assert (task.status, task.result) == ("done", "Thanks Rosa, the 12 kg goes Monday.")

    monkeypatch.setattr(stp, "socials_on", lambda db, ws: None)
    task = finish(ANSWER_2158)
    assert (task.status, task.result) == ("done", ANSWER_2158)


def test_a_result_that_only_asks_is_left_to_its_question(monkeypatch):
    def _never(*a, **k):
        raise AssertionError("an asking answer is F183's: no post check")

    monkeypatch.setattr(stp, "socials_on", _never)
    task = NS(id=2162, status="in_progress", runtime_ref=None, source_type="user", source_id=None,
              title="Before and after: the Guji re-roast", description="An Instagram before/after post.",
              started_at=STARTED, assigned_agent_id=AGENT)
    session = NS(get=lambda *a, **k: task)
    answer = "Please provide the two phone photos. Once I have them, I can create the Instagram draft."

    assert asyncio.run(stp.social_post_check(session, task_id=2162, workspace_id=WS, agent_id=AGENT,
                                             exec_result={"status": "success", "result": answer})) is None
