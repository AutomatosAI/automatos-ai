"""PRD-268 US-C11c (§6.3, K9, K18) — when a follow-up may be, and the first cap one more breaks.

Pure: no database. ``modules.proactive.follow_up_caps`` reads a follow-up's time from
``in_minutes`` or an ISO ``at`` with its offset (a time with none is refused) and names the
first cap a request breaks: sooner than 10 minutes, later than 7 days, a sixth pending
follow-up for the person, a 21st in their last 24 hours, the workspace's 101st. The seven
settings are config's (and the config surface's), and every sentence the tools say is in
English and Portuguese (Brazil).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from modules.proactive.follow_up_caps import (
    BOTH_TIMES,
    NAIVE_TIME,
    NO_TIME,
    NOT_A_TIME,
    CapBroken,
    FollowUpCaps,
    FollowUpCounts,
    FollowUpTimeRefused,
    count_broken,
    first_broken,
    follow_up_time,
    time_broken,
)

ORCH = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 30, 14, 0, tzinfo=timezone.utc)
CAPS = FollowUpCaps()
NONE_COUNTED = FollowUpCounts()
SETTINGS = (
    "PROACTIVE_FOLLOW_UP_MIN_MINUTES", "PROACTIVE_FOLLOW_UP_MAX_DAYS", "PROACTIVE_FOLLOW_UP_MAX_PENDING",
    "PROACTIVE_FOLLOW_UP_MAX_PER_DAY", "PROACTIVE_FOLLOW_UP_MAX_PER_WORKSPACE_PER_DAY",
    "PROACTIVE_FOLLOW_UP_CONTEXT_MESSAGES", "PROACTIVE_FOLLOW_UP_REASON_CHARS",
)


def _in(minutes):
    return follow_up_time(minutes, None, NOW)


# ── the time ────────────────────────────────────────────────────────────────

def test_nine_minutes_is_refused_and_ten_allowed():
    assert time_broken(CAPS, _in(9), NOW) == CapBroken("min_minutes", 10)
    assert time_broken(CAPS, _in(10), NOW) is None


def test_seven_days_is_allowed_and_seven_days_and_a_minute_refused():
    week = 7 * 24 * 60
    assert time_broken(CAPS, _in(week), NOW) is None
    assert time_broken(CAPS, _in(week + 1), NOW) == CapBroken("max_days", 7)


def test_a_time_with_no_offset_is_refused():
    with pytest.raises(FollowUpTimeRefused) as refused:
        follow_up_time(None, "2026-10-30T15:10:00", NOW)
    assert refused.value.code == NAIVE_TIME


@pytest.mark.parametrize("at", ["2026-10-30T16:10:00+01:00", "2026-10-30T15:10:00Z", "2026-10-30T12:10:00-03:00"])
def test_a_time_with_its_offset_is_read_in_utc(at):
    assert follow_up_time(None, at, NOW) == datetime(2026, 10, 30, 15, 10, tzinfo=timezone.utc)


@pytest.mark.parametrize("minutes", [30, "30", " 30 ", 30.0])
def test_in_minutes_counts_from_now(minutes):
    assert follow_up_time(minutes, None, NOW) == NOW + timedelta(minutes=30)


def test_an_empty_in_minutes_beside_an_at_is_no_time_given():
    assert follow_up_time("", "2026-10-30T15:10:00Z", NOW) == datetime(2026, 10, 30, 15, 10, tzinfo=timezone.utc)


@pytest.mark.parametrize("minutes, at, code", [
    (None, None, NO_TIME), (None, "", NO_TIME), ("", "", NO_TIME), (30, "2026-10-30T15:10:00Z", BOTH_TIMES),
    (True, None, NOT_A_TIME), ("half an hour", None, NOT_A_TIME), (10.5, None, NOT_A_TIME),
    (None, "tomorrow at ten", NOT_A_TIME), (None, 1730300000, NOT_A_TIME),
    # Past the calendar's ends, or digits int() cannot read: refused, never raised.
    (10 ** 13, None, NOT_A_TIME), (1e300, None, NOT_A_TIME), (float("inf"), None, NOT_A_TIME),
    ("9" * 10, None, NOT_A_TIME), ("\u00b2", None, NOT_A_TIME), (None, "0001-01-01T00:00:00+05:00", NOT_A_TIME),
])
def test_a_time_that_cannot_be_read_says_why(minutes, at, code):
    with pytest.raises(FollowUpTimeRefused) as refused:
        follow_up_time(minutes, at, NOW)
    assert refused.value.code == code


def test_a_time_in_the_past_breaks_the_earliest():
    assert time_broken(CAPS, _in(-5), NOW) == CapBroken("min_minutes", 10)


# ── the counts ──────────────────────────────────────────────────────────────

def test_the_fifth_pending_is_allowed_and_the_sixth_refused():
    assert count_broken(CAPS, FollowUpCounts(pending=4)) is None
    assert count_broken(CAPS, FollowUpCounts(pending=5)) == CapBroken("max_pending", 5)


def test_the_twentieth_of_the_day_is_allowed_and_the_twenty_first_refused():
    assert count_broken(CAPS, FollowUpCounts(today=19)) is None
    assert count_broken(CAPS, FollowUpCounts(today=20)) == CapBroken("max_per_day", 20)


def test_the_workspaces_hundredth_is_allowed_and_its_hundred_and_first_refused():
    assert count_broken(CAPS, FollowUpCounts(workspace_today=99)) is None
    assert count_broken(CAPS, FollowUpCounts(workspace_today=100)) == CapBroken("max_per_workspace_per_day", 100)


def test_the_first_cap_broken_is_named_in_order():
    everything = FollowUpCounts(pending=5, today=20, workspace_today=100)
    assert first_broken(CAPS, _in(5), NOW, everything) == CapBroken("min_minutes", 10)
    assert first_broken(CAPS, _in(60), NOW, everything) == CapBroken("max_pending", 5)
    assert first_broken(CAPS, _in(60), NOW, FollowUpCounts(today=20, workspace_today=100)) == CapBroken(
        "max_per_day", 20)
    assert first_broken(CAPS, _in(60), NOW, NONE_COUNTED) is None


def test_a_cap_of_zero_allows_nothing():
    assert count_broken(FollowUpCaps(max_pending=0), NONE_COUNTED) == CapBroken("max_pending", 0)


@pytest.mark.parametrize("wrong", [{"max_pending": -1}, {"max_days": "7"}, {"min_minutes": True},
                                   {"min_minutes": 10_081}])
def test_caps_set_wrong_are_refused(wrong):
    with pytest.raises(ValueError):
        FollowUpCaps(**wrong)


# ── config ──────────────────────────────────────────────────────────────────

def test_the_defaults_are_k9s_and_config_sets_them():
    from services.proactive.follow_ups import platform_caps

    assert FollowUpCaps() == FollowUpCaps(min_minutes=10, max_days=7, max_pending=5, max_per_day=20,
                                          max_per_workspace_per_day=100, reason_chars=500)
    assert platform_caps() == FollowUpCaps()


def test_the_seven_settings_are_in_config_and_the_config_surface():
    from config import config

    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text())["settings"]
    assert {name: getattr(config, name) for name in SETTINGS} == {
        "PROACTIVE_FOLLOW_UP_MIN_MINUTES": 10, "PROACTIVE_FOLLOW_UP_MAX_DAYS": 7,
        "PROACTIVE_FOLLOW_UP_MAX_PENDING": 5, "PROACTIVE_FOLLOW_UP_MAX_PER_DAY": 20,
        "PROACTIVE_FOLLOW_UP_MAX_PER_WORKSPACE_PER_DAY": 100, "PROACTIVE_FOLLOW_UP_CONTEXT_MESSAGES": 5,
        "PROACTIVE_FOLLOW_UP_REASON_CHARS": 500,
    }
    assert set(SETTINGS) <= set(surface) and surface == sorted(surface)


# ── what the tools say ──────────────────────────────────────────────────────

def test_every_sentence_is_in_english_and_portuguese():
    from services.proactive.follow_up_strings import CAP_KEYS, STRINGS

    assert set(STRINGS) == {"en", "pt-BR"}
    assert set(STRINGS["en"]) == set(STRINGS["pt-BR"])
    assert set(CAP_KEYS) <= set(STRINGS["en"])
    assert all(value.strip() for table in STRINGS.values() for value in table.values())


@pytest.mark.parametrize("locale", ["en", "pt-BR"])
def test_a_cap_sentence_says_the_caps_name_and_value(locale):
    from services.proactive.follow_up_strings import cap_key, say

    for cap, limit in (("min_minutes", 10), ("max_days", 7), ("max_pending", 5), ("max_per_day", 20),
                       ("max_per_workspace_per_day", 100)):
        sentence = say(locale, cap_key(cap), limit=limit)
        assert cap in sentence and str(limit) in sentence


def test_the_use_a_watch_sentence_is_the_prds():
    from services.proactive.follow_up_strings import say

    assert say("en", "use_a_watch") == "Use a watch for that: it supervises the work to a verdict and posts back here."


def test_the_language_is_the_contexts_else_english():
    from services.proactive.follow_up_strings import locale_of, say

    assert locale_of({"locale": "pt-BR"}) == "pt-BR"
    assert locale_of({"locale": "pt"}) == "pt-BR"
    assert locale_of({"locale": None}) == locale_of(None) == locale_of({}) == "en"
    assert say("fr", "not_auto") == say("en", "not_auto")
