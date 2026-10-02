"""P251W1-RVW-5 — a footage shot's spend is booked before the provider can charge for it.

Wave 1 booked a render's footage in the toolkit run's ``finally``, after polling
for up to ``SOCIALS_FOOTAGE_MAX_WAIT_SECONDS``: a process that died in between (a
redeploy's SIGKILL, an OOM, a crash) left the provider's charge with no booking,
and the post cap, the monthly media cap and the budget gate all under-counted
it. Each shot is now booked before its submit (``modules/socials/media_ledger``)
and settled in place when its job ends. On the Wave 1 footage harness
(``test_prd251w1_footage.py``: the real Socials router over in-memory SQLite, the
real ``llm_usage``, Composio mocked where the recipe calls it). Pins:

* (a) the fal status poll kills the process right after FAL_AI_SUBMIT_ASYNC_JOB
  was accepted: llm_usage holds the shot's estimate against the post;
* (b) the same after a Higgsfield submit: its ceiling;
* the boot reaper fails that post and leaves the shot's booking in place;
* the estimate is in llm_usage before fal is asked to make the shot, and the
  finished job settles that same row: one amount, counted once by the caps and
  by the budget gate's spend-to-date;
* a submit fal refuses reverses its booking; a shot it takes without a request
  id keeps it (fal may still make it, and bill it); a booking that cannot be
  written submits nothing;
* a credit-billed toolkit's balance difference is shared across its shots'
  bookings by their prices: one row per shot, together the difference;
* the ledger settles a booking once, and a settle that fails keeps the
  committed amount.

The process's death is ``tests/helpers_socials_process_death.py``'s stand-in:
nothing after it reaches the database, as with a real SIGKILL.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w1_footage as footage_harness  # noqa: E402
from tests.test_prd251w1_footage import (  # noqa: E402  (the footage harness's pieces)
    CLIP, FAL_ESTIMATE, FAL_MODEL, FAL_STATUS, FAL_SUBMIT, HF_BALANCE, HF_VIDEO, HF_WAIT, PROMPT, STILL, TOKEN, WS,
    FakeStore, Renderer, _cache_fal, _cache_higgsfield, _connect, _create, _failed, _last_log, _media_rows, _ok,
    _post, _render, _script_fal, _script_higgsfield,
)
from tests.helpers_socials_process_death import mortal, paid_media_now, render_until_it_dies  # noqa: E402
import sqlalchemy as sa  # noqa: E402

import core.boot.reaper as reaper  # noqa: E402
import core.media_render_quota as render_quota  # noqa: E402
import modules.socials.media_caps as media_caps  # noqa: E402
import modules.socials.media_ledger as media_ledger  # noqa: E402
from core.llm.providers import MEDIA_RENDER_PROVIDER  # noqa: E402
from core.models.core import LLMUsage  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.policy.budget import spend_to_date  # noqa: E402

# The footage harness's fixture, bound by assignment so pytest finds it here and
# ruff does not read it as an unused import.
env = footage_harness.env

HF_IMAGE = "HIGGSFIELD_MCP_GENERATE_IMAGE"
ESTIMATE = 0.46
# Higgsfield prices nothing: a shot is booked at its ceiling (the harness's
# SOCIALS_FOOTAGE_CEILING_*_USD), in credit at SOCIALS_HIGGSFIELD_USD_PER_CREDIT.
VIDEO_CEILING, IMAGE_CEILING, HIGGSFIELD_USD_PER_CREDIT = 2.5, 0.3, 0.0625


def _fal_dies_at_its_first_poll(env, monkeypatch):
    """fal takes the shot; the process dies as it asks for the job's status."""
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env, estimate=ESTIMATE)
    process = mortal(env, monkeypatch)
    env.composio.answers[FAL_STATUS] = process.die
    post = _create(env)
    render_until_it_dies(env, post["id"], process, renderer=Renderer(FakeStore()), store=FakeStore(), token=TOKEN)
    return post


def _counted(env, post_id):
    """What the post's cap, the workspace's monthly media cap and the budget gate count."""
    env.session.expire_all()
    spend = media_caps.media_spend(env.session, env.session.get(Workspace, WS), post_id)
    return spend.post_usd, spend.month_usd, spend_to_date(env.session, WS, "month")["cost_usd"]


# ---------------------------------------------------------------------------
# (a), (b) — a process that dies after the toolkit took the shot
# ---------------------------------------------------------------------------


def test_a_fal_shot_is_booked_at_its_estimate_before_the_process_can_die(env, monkeypatch):
    """(a) The fal status poll kills the process right after FAL_AI_SUBMIT_ASYNC_JOB
    was accepted: the shot's estimate is in llm_usage against the post."""
    post = _fal_dies_at_its_first_poll(env, monkeypatch)

    assert env.composio.slugs() == [FAL_ESTIMATE, FAL_SUBMIT, FAL_STATUS]
    (row,) = _media_rows(env, post["id"])
    assert (row.provider, row.model_id, row.input_tokens, row.status) == ("fal_ai", FAL_MODEL, 5, "pending")
    assert row.total_cost == pytest.approx(ESTIMATE) and bool(row.is_byok) is True
    assert _post(env, post["id"]).status == "rendering", "nothing ended the render: the boot reaper will"
    assert _counted(env, post["id"]) == (pytest.approx(ESTIMATE),) * 3


def test_a_higgsfield_shot_is_booked_at_its_ceiling_before_the_process_can_die(env, monkeypatch):
    """(b) The same after a Higgsfield submit: the shot's ceiling, in Higgsfield's credit."""
    _cache_higgsfield(env)
    _connect(env, "HIGGSFIELD_MCP")
    _script_higgsfield(env, before=100, after=92)
    process = mortal(env, monkeypatch)
    env.composio.answers[HF_WAIT] = process.die
    post = _create(env)

    render_until_it_dies(env, post["id"], process, renderer=Renderer(FakeStore()), store=FakeStore(), token=TOKEN)

    assert env.composio.slugs()[:3] == [HF_BALANCE, HF_VIDEO, HF_WAIT]
    (row,) = _media_rows(env, post["id"])
    assert (row.provider, row.model_id, row.status) == ("higgsfield_mcp", "kling3_0", "pending")
    assert row.total_cost == pytest.approx(VIDEO_CEILING)
    assert row.input_tokens == VIDEO_CEILING / HIGGSFIELD_USD_PER_CREDIT == 40
    assert _counted(env, post["id"]) == (pytest.approx(VIDEO_CEILING),) * 3


def test_the_boot_reaper_fails_the_post_and_keeps_its_shots_booking(env, monkeypatch):
    """The next boot fails the post the dead process left rendering, and deletes
    the render's hold on the quota; the shot's booking stays: fal may bill it."""
    post = _fal_dies_at_its_first_poll(env, monkeypatch)
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    now = datetime.now(timezone.utc)
    posts = SocialPost.__table__
    env.session.execute(sa.update(posts).where(posts.c.id == uuid.UUID(post["id"])).values(updated_at=now - timedelta(hours=2)))
    env.session.commit()
    later = now + render_quota.reservation_lifetime() + timedelta(seconds=1)
    cutoff = later - timedelta(minutes=30)

    assert reaper._reap_social_renders(env.session, cutoff, later) == 1
    assert reaper._reap_render_reservations(env.session, cutoff, later) == 1
    env.session.commit()

    saved = _post(env, post["id"])
    assert saved.status == "failed" and saved.review_log[-1]["by"] == "orphaned_on_restart"
    assert env.session.query(LLMUsage).filter(LLMUsage.provider == MEDIA_RENDER_PROVIDER).count() == 0
    (row,) = _media_rows(env, post["id"])
    assert (row.provider, row.status) == ("fal_ai", "pending") and row.total_cost == pytest.approx(ESTIMATE)


# ---------------------------------------------------------------------------
# Booked before the submit, settled in place
# ---------------------------------------------------------------------------


def test_the_estimate_is_booked_before_fal_is_asked_and_the_job_settles_that_row(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env, estimate=ESTIMATE)
    post = _create(env)
    submit, at_submit = env.composio.answers[FAL_SUBMIT], []

    def submitting(params):
        at_submit.extend(paid_media_now(env, post["id"]))
        return submit

    env.composio.answers[FAL_SUBMIT] = submitting

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is True
    assert at_submit == [{"provider": "fal_ai", "usd": pytest.approx(ESTIMATE), "units": 5, "status": "pending"}]
    (row,) = _media_rows(env, post["id"])
    assert (row.status, row.input_tokens, row.error_message) == ("success", 5, None)
    assert row.total_cost == pytest.approx(ESTIMATE) and row.latency_ms is not None
    # One amount, counted once, by the post's cap, the monthly cap and the budget gate.
    assert _counted(env, post["id"]) == (pytest.approx(ESTIMATE),) * 3


def test_a_submit_fal_refuses_reverses_its_booking(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)
    at_submit = []

    def refusing(params):
        at_submit.extend(paid_media_now(env, post["id"]))
        return _failed("insufficient balance")

    env.composio.answers[FAL_SUBMIT] = refusing

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is False and renderer.bundles == [] and FAL_STATUS not in env.composio.slugs()
    assert [entry["status"] for entry in at_submit] == ["pending"], "booked while fal was being asked"
    assert "fal.ai's FAL_AI_SUBMIT_ASYNC_JOB failed: insufficient balance" in _last_log(env, post["id"])["comment"]
    assert _media_rows(env, post["id"]) == [], "fal never took it: nothing stays booked"


def test_a_shot_fal_took_without_a_request_id_keeps_its_booking(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    env.composio.answers[FAL_SUBMIT] = _ok({"status": "IN_QUEUE"})
    post = _create(env)

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is False and FAL_STATUS not in env.composio.slugs()
    assert "fal.ai took the hook footage but gave no request id" in _last_log(env, post["id"])["comment"]
    (row,) = _media_rows(env, post["id"])
    assert row.status == "success" and row.total_cost == pytest.approx(ESTIMATE), "fal may still make it, and bill it"


def test_a_booking_that_cannot_be_written_submits_nothing(env, monkeypatch):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)

    def cannot_book(*args, **kwargs):
        raise media_ledger.BookingFailed("fal_ai's spend could not be booked")

    monkeypatch.setattr(media_ledger, "commit", cannot_book)

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is False and env.composio.slugs() == [FAL_ESTIMATE], "priced, never submitted"
    assert "what fal.ai would spend could not be booked, so it was not submitted" in _last_log(env, post["id"])["comment"]
    assert _media_rows(env, post["id"]) == []


# ---------------------------------------------------------------------------
# A credit-billed toolkit: its balance difference, shared across its shots
# ---------------------------------------------------------------------------


def test_a_credit_billed_toolkits_difference_is_shared_across_its_shots_bookings(env):
    """Two Higgsfield shots, booked at their ceilings ($2.50 of footage, a $0.30
    still): the 8 credits Higgsfield used ($0.50) settle as each shot's share by
    price, one row per shot, together the difference and nothing more."""
    _cache_higgsfield(env)
    _connect(env, "HIGGSFIELD_MCP")
    links = {"job-7": "https://cdn.higgsfield.ai/jobs/job-7/result.mp4",
             "job-8": "https://cdn.higgsfield.ai/jobs/job-8/result.png"}
    env.files.update({links["job-7"]: CLIP, links["job-8"]: STILL})
    env.composio.answers[HF_BALANCE] = [_ok({"credits": 100}), _ok({"credits": 92})]
    env.composio.answers[HF_VIDEO] = _ok({"results": [{"id": "job-7", "status": "queued"}]})
    env.composio.answers[HF_IMAGE] = _ok({"results": [{"id": "job-8", "status": "queued"}]})

    def wait(params):
        (job,) = params["jobs"]
        return _ok({"jobs": [{"id": job, "status": "completed", "results": [{"url": links[job]}]}]})

    env.composio.answers[HF_WAIT] = wait
    post = _create(env, footage={"hook": {"prompt": PROMPT}, "still": {"prompt": "an open logbook"}})

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is True
    budget, used_usd = VIDEO_CEILING + IMAGE_CEILING, 8 * HIGGSFIELD_USD_PER_CREDIT
    rows = _media_rows(env, post["id"])
    assert [(row.model_id, row.input_tokens, row.status) for row in rows] == [
        ("kling3_0", 7, "success"), ("gpt_image_2", 1, "success"),  # the 8 credits used, split 7.14 / 0.86 whole
    ]
    assert [row.total_cost for row in rows] == [
        pytest.approx(used_usd * VIDEO_CEILING / budget), pytest.approx(used_usd * IMAGE_CEILING / budget),
    ]
    assert _counted(env, post["id"]) == (pytest.approx(used_usd),) * 3
    footage_of = _post(env, post["id"]).footage
    assert footage_of["hook"]["cost_usd"] == pytest.approx(rows[0].total_cost)
    assert footage_of["still"]["cost_usd"] == pytest.approx(rows[1].total_cost)


# ---------------------------------------------------------------------------
# The ledger: a booking is settled once, and a failed settle keeps it
# ---------------------------------------------------------------------------


class _CommitFails:
    """A session whose commit fails, as a dropped connection's would."""

    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        return getattr(self.session, name)

    def commit(self):
        raise sa.exc.OperationalError("COMMIT", {}, Exception("the database went away"))


def test_a_booking_is_settled_once_and_a_failed_settle_keeps_the_committed_amount(env):
    post_id = uuid.uuid4()
    booking = media_ledger.commit(env.factory, WS, post_id, provider="fal_ai", model_id=FAL_MODEL, units=5, usd=ESTIMATE)
    settled = media_ledger.Settlement(booking, units=5, usd=0.40)

    assert media_ledger.settle(lambda: _CommitFails(env.factory()), [settled], 7) is False
    assert paid_media_now(env, post_id) == [{"provider": "fal_ai", "usd": pytest.approx(ESTIMATE), "units": 5, "status": "pending"}]

    assert media_ledger.settle(env.factory, [media_ledger.Settlement(booking, units=5, usd=float("nan"))], 7) is True
    assert paid_media_now(env, post_id)[0]["status"] == "pending", "garbage never settles a booking"

    assert media_ledger.settle(env.factory, [settled], 7) is True
    assert media_ledger.settle(env.factory, [media_ledger.Settlement(booking, reverse=True)]) is True
    assert paid_media_now(env, post_id) == [{"provider": "fal_ai", "usd": pytest.approx(0.40), "units": 5, "status": "success"}]


@pytest.mark.parametrize("usd", [float("nan"), float("inf"), -0.46, None, "0.46", True])
def test_a_price_that_is_not_an_amount_of_dollars_is_never_booked(env, usd):
    """Garbage is refused, never booked as $0: then the shot is not submitted."""
    post_id = uuid.uuid4()
    with pytest.raises(media_ledger.BookingFailed, match="is not an amount of dollars to book"):
        media_ledger.commit(env.factory, WS, post_id, provider="fal_ai", model_id=FAL_MODEL, units=5, usd=usd)
    assert paid_media_now(env, post_id) == []
