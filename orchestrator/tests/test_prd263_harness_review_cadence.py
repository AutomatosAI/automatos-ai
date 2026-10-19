"""PRD-263 US-023 — Auto's weekly HARNESS review text describes the report-only loop.

The cadence block said "HARNESS runs Sunday 02:00 UTC" and told Auto to list queued
prescriptions with ``platform_list_tasks tag=harness status=todo``. HARNESS now runs
on the installation's weekday cron (US-012), queues nothing (US-016) and reports
coverage and verdict counts (US-014a). The review names no day or time, names the
ISO week this week's run judges, and tells Auto to wait when that run has not
happened yet.
"""
from __future__ import annotations

from datetime import datetime, timezone

from core.services import auto_cadence

MONDAY = datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc)
JUDGED_WEEK = "2026-W42"  # the last complete ISO week on MONDAY


def _block(**cadence) -> str:
    return auto_cadence.build_cadence_block({"cadence": cadence}, now=MONDAY)


def test_the_review_names_no_day_time_or_queued_cards():
    block = _block(harness_review={"enabled": True, "day": "mon"})

    assert "## Weekly HARNESS Review" in block
    for gone in ("Sunday", "tag=harness", "02:00", "queued-for-review", "Monday HARNESS"):
        assert gone not in block
    for said in ("coverage", "INSUFFICIENT_DATA", "WATCH", "REGRESSION"):
        assert said in block


def test_the_review_waits_for_this_weeks_run():
    block = _block(harness_review={"enabled": True, "day": "mon"})

    assert JUDGED_WEEK in block
    assert "scheduled_not_run_yet" in block and f"cur_window is not {JUDGED_WEEK}" in block
    assert "the review waits for it" in block


def test_the_template_itself_names_no_day():
    for gone in ("Sunday", "tag=harness"):
        assert gone not in auto_cadence._HARNESS_REVIEW


def test_the_daily_brief_no_longer_promises_auto_applied_changes():
    block = _block(daily_brief={"enabled": True})

    assert "## Auto Daily Brief" in block
    assert "auto-applied" not in block and "HARNESS prescriptions" not in block


def test_only_the_harness_review_is_formatted(monkeypatch):
    # Guards _prompt: another cadence's text is kept verbatim, braces and all (formatting it raises KeyError).
    monkeypatch.setitem(auto_cadence._PROMPT_MAP, "team_review", "## Team Review {literal}")

    block = _block(team_review={"enabled": True, "day": "mon"}, harness_review={"enabled": True, "day": "mon"})

    assert "## Team Review {literal}" in block and JUDGED_WEEK in block
