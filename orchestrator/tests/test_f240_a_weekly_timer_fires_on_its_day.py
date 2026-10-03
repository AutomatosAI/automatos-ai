"""F240 — a weekly timer fires on the day it names, the day the calendar shows.

Night 7: APScheduler's ``CronTrigger.from_crontab`` read cron's weekday as
0 = Monday, so '0 7 * * 5' (Friday's playbook) fired on Saturday, Monday's
weekly heartbeat on Tuesday and '30 6 * * 6' on Sunday, while croniter's next
run (what the screen showed) named the right day. Every cron string now becomes
a trigger through ``schedule_util.scheduler_cron_trigger``, which hands the
weekday to APScheduler as day names.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

# Thursday 1 Oct 2026, half a minute past midnight: no expression below fires on it.
THURSDAY = datetime(2026, 10, 1, 0, 0, 30, tzinfo=timezone.utc)
FIRES_TO_COMPARE = 12

SAVED_SCHEDULES = (
    "0 7 * * 5",        # night 7: Friday's playbook ran on Saturday
    "0 9 * * 1",        # the weekly heartbeat (interval_to_cron of a week): Monday, not Tuesday
    "30 6 * * 6",       # Saturday, not Sunday
    "0 8 * * 0",        # Sunday as 0
    "0 8 * * 7",        # Sunday as 7
    "0 9 * * 1-5",      # weekdays
    "0 10 * * 0,6",     # the weekend
    "0 9 * * mon-fri",  # names already meant the right days
    "0 12 * * */2",     # Sun, Tue, Thu, Sat
    "*/15 * * * *",
    "0 9 * * *",
)


def _fires(trigger, count):
    fired, now = [], THURSDAY
    for _ in range(count):
        at = trigger.get_next_fire_time(fired[-1] if fired else None, now)
        fired.append(at)
        now = at + timedelta(seconds=1)
    return fired


def _calendar(expression, count):
    from services.schedule_util import next_run

    shown, now = [], THURSDAY
    for _ in range(count):
        at = next_run(expression, now=now)
        shown.append(at)
        now = at
    return shown


def test_fridays_timer_fires_on_friday():
    from services.playbook_scheduler import cron_trigger

    fired = _fires(cron_trigger("0 7 * * 5", "UTC"), 2)

    assert fired[0] == datetime(2026, 10, 2, 7, 0, tzinfo=timezone.utc)   # night 7: Sat 3 Oct
    assert [at.strftime("%A") for at in fired] == ["Friday", "Friday"]


@pytest.mark.parametrize("expression", SAVED_SCHEDULES)
def test_every_saved_schedule_fires_when_the_calendar_says(expression):
    from services.schedule_util import scheduler_cron_trigger

    fired = _fires(scheduler_cron_trigger(expression, timezone="UTC"), FIRES_TO_COMPARE)

    assert fired == _calendar(expression, FIRES_TO_COMPARE)


def test_a_playbook_in_a_zone_fires_on_its_day_in_that_zone():
    from services.playbook_scheduler import cron_trigger

    fired = _fires(cron_trigger("30 6 * * 6", "Europe/London"), 1)[0]

    assert (fired.strftime("%A"), fired.hour, fired.minute) == ("Saturday", 6, 30)


def test_the_weekly_heartbeat_fires_on_monday():
    from services.heartbeat_service import HeartbeatService

    fired = _fires(HeartbeatService._interval_to_cron_trigger(7 * 24 * 60), 2)

    assert [at.astimezone(timezone.utc).strftime("%A") for at in fired] == ["Monday", "Monday"]


@pytest.mark.parametrize("expression", ("0 7 * * 5#3", "0 7 * * L5", "0 7 * * 8", "0 7 * *", "0 7 * * xyz"))
def test_a_weekday_the_scheduler_cannot_honour_is_refused_by_name(expression):
    from services.playbook_scheduler import cron_trigger

    with pytest.raises(ValueError, match=re.escape(f"invalid cron '{expression}'")):
        cron_trigger(expression, "UTC")
