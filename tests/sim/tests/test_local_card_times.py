"""F091 (night 3) — the persona reads card times in its own local time.

Card times came through as UTC ISO strings and the persona took four live cards
for expired. ``questions`` and ``inventory`` now print local time and how far
off it is.
"""
from datetime import datetime, timedelta, timezone

from tests.sim import customer_ops as ops

LISBON = timezone(timedelta(hours=1), "WEST")      # the persona's summer zone, no tzdata needed
NOW = datetime(2026, 9, 23, 0, 30, tzinfo=timezone.utc)


def test_a_utc_expiry_reads_as_local_time_and_how_long_is_left():
    assert ops.local_time("2026-09-23T01:00:00+00:00", now=NOW, tz=LISBON) == "2026-09-23 02:00 WEST (in 30m)"
    assert ops.local_time("2026-09-24T03:15:00Z", now=NOW, tz=LISBON) == "2026-09-24 04:15 WEST (in 1d 2h)"


def test_a_past_time_says_how_long_ago_and_naive_values_are_utc():
    assert ops.local_time("2026-09-22T23:30:00", now=NOW, tz=LISBON) == "2026-09-23 00:30 WEST (1h 0m ago)"


def test_nothing_or_garbage_is_shown_as_is():
    assert ops.local_time(None) == "—"
    assert ops.local_time("soon") == "soon"


def test_the_inventory_shows_each_cards_expiry_locally():
    inv = {"questions": [{"id": 600, "kind": "approval", "reason": "Delete a document",
                          "expires_at": "2099-01-01T00:00:00+00:00"}]}
    line = [ln for ln in ops.render_inventory(inv).splitlines() if ln.startswith("**Pending")][0]
    assert "#600 approval: Delete a document (expires 2099-01-01" in line and "(in " in line
