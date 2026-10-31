"""P256-FIX-SOC-5 (F437, socials day 14 Oct): a render says since when, and when it gives up.

Renders took 4 to 18 minutes and said nothing: a post showed "Rendering" with no start time or
deadline, ``SocialPost.to_dict`` had no render field (the run's "rendered_at null" read a key that
does not exist), and Auto could not confirm a render it started. Now the render route records when
the render gives up on its ``render`` entry, from the same clock as the start
(``service.start_render(gives_up_after=)``), and ``to_dict`` carries a ``render`` block derived from
the history (``core/social_render_state.py``), right after ``status`` so a cut tool answer keeps
it. On the render harness (``tests/test_prd251w1_render_lifecycle.py``: the real route, the
renderer mocked over HTTP), pinned:

* after the route starts a render: rendering, its start the ``render`` entry's time, giving up
  ``SOCIALS_RENDER_MAX_WAIT_SECONDS`` later; ``render`` sits right after ``status``, in the post
  and in Auto's read of it;
* after the render finishes: done, with when; after it fails: failed, with why;
* a draft the platform refused reads failed with no start, and one that waits for you reads
  waiting; the block's action names are the lifecycle's.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core import social_render_state  # noqa: E402
from modules.socials import service  # noqa: E402
from modules.tools.discovery.social_post_results import read_view  # noqa: E402
from tests.test_prd251w1_render_lifecycle import FakeStore, Renderer, _post, _renderable, _run, _start, env  # noqa: E402,F401

WAIT_SECONDS = 60  # the harness's SOCIALS_RENDER_MAX_WAIT_SECONDS


def _started(env):  # noqa: F811
    post = _renderable(env)
    _, job = _start(env, post)
    return post, job


def test_a_started_render_says_when_it_started_and_when_it_gives_up(env):  # noqa: F811
    post, _job = _started(env)

    row = _post(env, post["id"])
    entry = row.review_log[-1]
    shown = row.to_dict()

    assert entry["action"] == service.ACTION_RENDER
    gives_up = (datetime.fromisoformat(entry["at"]) + timedelta(seconds=WAIT_SECONDS)).isoformat()
    assert shown["render"] == {"state": "rendering", "started_at": entry["at"], "gives_up_at": gives_up,
                               "finished_at": None, "why": None}
    keys = list(shown)
    assert keys[keys.index("status") + 1] == "render"
    assert list(read_view(shown))[:3] == ["id", "status", "render"]  # Auto's read keeps it near the top


def test_a_finished_render_is_done_and_a_failed_one_says_why(env):  # noqa: F811
    done_post, done_job = _started(env)
    assert _run(done_job, Renderer(), FakeStore(), env.factory) is True
    done = _post(env, done_post["id"])
    started = next(entry for entry in done.review_log if entry["action"] == service.ACTION_RENDER)
    finished = done.review_log[-1]
    assert finished["action"] == service.ACTION_RENDER_DONE
    assert {key: done.to_dict()["render"][key] for key in ("state", "started_at", "finished_at", "why")} == {
        "state": "done", "started_at": started["at"], "finished_at": finished["at"], "why": None,
    }

    failed_post, failed_job = _started(env)
    renderer = Renderer(status="failed", error={"code": "render_timed_out", "message": "the render ran past 900 s"})
    assert _run(failed_job, renderer, FakeStore(), env.factory) is False
    failed = _post(env, failed_post["id"]).to_dict()["render"]
    assert (failed["state"], failed["why"]) == ("failed", "The render failed: the render ran past 900 s.")
    assert failed["started_at"] is not None and failed["finished_at"] is not None


def test_a_refused_draft_reads_failed_without_a_start_and_a_waiting_one_reads_waiting():
    refused = [{"at": "t1", "action": "draft"}, {"at": "t2", "action": "render"}, {"at": "t3", "action": "render_done"},
               {"at": "t4", "action": "edit"}, {"at": "t5", "action": "render_failed", "comment": "No render minutes."}]
    assert social_render_state.render_block("failed", refused) == {
        "state": "failed", "started_at": None, "gives_up_at": None, "finished_at": "t5", "why": "No render minutes.",
    }
    waiting = [{"at": "t1", "action": "draft"}, {"at": "t2", "action": "needs_input", "comment": "Fill in Big number before rendering."}]
    assert social_render_state.render_block("draft", waiting)["state"] == "waiting"
    assert social_render_state.render_block("draft", waiting)["why"] == "Fill in Big number before rendering."
    assert social_render_state.render_block("draft", [{"at": "t1", "action": "draft"}]) is None


def test_the_blocks_action_names_are_the_lifecycles():
    assert (social_render_state.RENDER, social_render_state.RENDER_DONE, social_render_state.RENDER_FAILED,
            social_render_state.NEEDS_INPUT) == (service.ACTION_RENDER, service.ACTION_RENDER_DONE,
                                                 service.ACTION_RENDER_FAILED, service.ACTION_NEEDS_INPUT)
    assert (social_render_state.RENDERING_STATUS, social_render_state.DRAFT_STATUS) == (service.RENDERING, service.DRAFT)
