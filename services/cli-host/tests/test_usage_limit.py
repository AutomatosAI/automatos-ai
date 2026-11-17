"""F083 — a CLI's usage limit is a pause, not a failure.

Gerard runs sprints until his Claude window closes and resumes after it, so the
limit WILL hit the runtime agents. Before this, a session that exited on the
limit became "<cli> exited (code N) before finishing the turn": every in-flight
ticket errored and every later claim errored on spawn until the window reopened.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from automatos_cli_host import usage_limit
from automatos_cli_host.config import HostConfig
from automatos_cli_host.hook_server import HookServer
from automatos_cli_host.host import Host
from automatos_cli_host.session import Session

from conftest import FAKE_CLAUDE
from fake_claude import LIMIT_COMMAND, LIMIT_MENU, LIMIT_NOTICE

NOW = datetime(2026, 9, 23, 13, 5, tzinfo=timezone.utc)
NOV17 = datetime(2026, 11, 17, 9, 30, tzinfo=timezone.utc)
# A weekly limit's screen as a stalled session's terminal recorded it (fake_claude.py):
# the notice, then the usage limit menu Claude Code waits on for a choice.
WEEKLY_LIMIT_SCREEN = (LIMIT_NOTICE.format(notice="You've hit your weekly limit · resets Nov 20, 9am (Europe/Berlin)")
                       + LIMIT_COMMAND + LIMIT_MENU)
_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _weekly_notice(days_ahead: int = 3):
    """Claude Code's weekly limit notice for a reset ``days_ahead`` from today, and that moment."""
    from zoneinfo import ZoneInfo

    reset = (datetime.now(ZoneInfo("Europe/Berlin")) + timedelta(days=days_ahead)).replace(
        hour=9, minute=0, second=0, microsecond=0)
    notice = f"You've hit your weekly limit · resets {_MONTH_NAMES[reset.month - 1]} {reset.day}, 9am (Europe/Berlin)"
    return notice, reset


# ── what the CLI says ───────────────────────────────────────────────────────

def test_the_persona_runners_limit_wordings_are_recognised():
    for said in ("You've hit your limit · resets 3pm (Europe/Dublin)",
                 "Claude usage limit reached. Your limit will reset at 15:00.",
                 "API Error: 429 {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\"}}",
                 "Error: 529 overloaded",
                 '{"type":"result","subtype":"error_usage_limit","is_error":true}'):
        assert usage_limit.is_usage_limit(said), said
    for said in ("Traceback (most recent call last): KeyError: 'x'", "zsh: command not found: claude", ""):
        assert not usage_limit.is_usage_limit(said), said


def test_the_reset_time_is_read_when_the_cli_gives_one():
    assert usage_limit.resets_at("try again in 2 hours", NOW) == NOW + timedelta(hours=2)
    assert usage_limit.resets_at("Please try again in 30 minutes.", NOW) == NOW + timedelta(minutes=30)
    dublin = usage_limit.resets_at("You've hit your limit · resets 3pm (Europe/Dublin)", NOW)
    assert (dublin.hour, dublin.minute, str(dublin.tzinfo)) == (15, 0, "Europe/Dublin")
    assert dublin > NOW and dublin - NOW < timedelta(hours=24)
    earlier_today = usage_limit.resets_at("limit will reset at 09:00", NOW)
    assert earlier_today.astimezone(timezone.utc).date() == (NOW + timedelta(days=1)).date()   # already past: tomorrow
    assert usage_limit.resets_at("Usage limit reached.", NOW) is None


def test_without_a_reset_time_the_host_checks_again_in_fifteen_minutes():
    until, known = usage_limit.pause("Usage limit reached.", NOW)
    assert (until, known) == (NOW + timedelta(minutes=15), False)
    assert "checking again at" in usage_limit.describe("claude", until, known)
    until, known = usage_limit.pause("try again in 1 hour", NOW)
    line = usage_limit.describe("claude", until, known)
    assert line.startswith("paused: claude usage limit, resumes ~") and "key" not in line


def test_claude_codes_usage_limit_menu_is_a_parked_turn():
    """The weekly limit opened Claude Code's usage limit menu. The host saw the limit
    only at the 4 h session timeout, and three tickets looped on it for about 20 h.
    The menu draws some words with cursor moves; they read as spaces."""
    assert usage_limit.is_parked(LIMIT_MENU)
    assert usage_limit.is_parked(WEEKLY_LIMIT_SCREEN) and usage_limit.is_usage_limit(WEEKLY_LIMIT_SCREEN)
    notice = "You've hit your weekly limit · resets Nov 20, 9am (Europe/Berlin)"
    assert usage_limit.is_usage_limit(notice)                    # the notice's own words are a limit
    # Only the open menu holds the turn: not the notice, a warning that names the menu's
    # command, or its first option's sentence quoted on its own, numbered or not.
    sentence = "Stop and wait for limit to reset"
    for said in (notice, "Approaching weekly limit · resets Nov 20, 9am · /rate-limit-options to see options",
                 sentence, f"The menu's first option is 1. {sentence}, and the session waits on it."):
        assert not usage_limit.is_parked(said), said


def test_a_file_that_describes_the_menu_does_not_park_the_session_showing_it():
    """A session that shows this module or the README (reading or editing them) must not
    be ended as parked and requeued: neither file reads as the open menu."""
    module = Path(usage_limit.__file__)
    for path in (module, module.parent.parent / "README.md"):
        assert not usage_limit.is_parked(path.read_text(encoding="utf-8")), path


def test_a_weekly_limits_reset_date_is_read_not_a_short_recheck():
    """The screen said "resets Nov 20, 9am (Europe/Berlin)"; the host read no time,
    re-checked after 15 minutes, called the window reopened and claimed again."""
    berlin = usage_limit.resets_at(WEEKLY_LIMIT_SCREEN, NOV17)
    assert (berlin.month, berlin.day, berlin.hour, berlin.minute, str(berlin.tzinfo)) == (11, 20, 9, 0, "Europe/Berlin")
    until, known = usage_limit.pause(WEEKLY_LIMIT_SCREEN, NOV17)
    assert known and until - NOV17 > timedelta(days=2)
    at = usage_limit.resets_at("limit · resets Nov 20 at 9:30pm", NOV17)
    assert (at.day, at.hour, at.minute) == (20, 21, 30)
    new_year = usage_limit.resets_at("resets Jan 2, 9am (Europe/Berlin)", datetime(2026, 12, 30, 12, tzinfo=timezone.utc))
    assert (new_year.year, new_year.month, new_year.day) == (2027, 1, 2)
    week = usage_limit.resets_at("resets Nov 24, 9am", NOV17)   # a weekly limit's furthest: 7 days ahead
    assert (week.month, week.day) == (11, 24)


def test_a_reset_that_cannot_be_read_falls_back_to_the_short_recheck():
    """Tolerant: no crash, and never a pause for a moment the screen did not state."""
    for said in (LIMIT_MENU,                                    # "reset", then the next option's "2.": no hour
                 "Stop and wait for limit to reset\r\n  2. Wait here, then continue",   # the same, no escapes
                 "resets Feb 30, 9am",                          # no such day
                 "resets Nov 20, 27pm",                         # no such hour
                 "resets Nov 16, 9am (Europe/Berlin)",          # already past: not a pause until next year
                 "resets Nov 26, 9am"):                         # 9 days ahead: no plan window is that long
        assert usage_limit.resets_at(said, NOV17) is None, said
        assert usage_limit.pause(said, NOV17) == (NOV17 + timedelta(seconds=usage_limit.RECHECK_SECONDS), False)


# ── the session: limit output mid-turn is a pause ───────────────────────────

def _cfg(short_tmp, **over) -> HostConfig:
    base = dict(state_dir=short_tmp / "state", cli_binaries={"claude": str(FAKE_CLAUDE)}, use_worktrees=False,
                session_timeout_seconds=120)
    base.update(over)
    return HostConfig(**base)


def _ticket(workdir: Path, **over) -> dict:
    t = {"task_id": 42, "attempt": 1, "session_id": str(uuid.uuid4()), "agent_name": "Dwight",
         "title": "Say hi", "prompt": "OBJECTIVE: write hello.txt\nOUTPUT: the file\nTOOLS: Write\nBOUNDARIES: this dir",
         "cwd": str(workdir), "model": "sonnet", "allowed_tools": [], "provider": "claude"}
    t.update(over)
    return t


def _run(short_tmp, ticket):
    cfg = _cfg(short_tmp)
    hooks = HookServer(cfg.socket_path)
    hooks.start()
    allow = [str(short_tmp / "ws")]
    s = Session(ticket, cfg, allow, cfg.socket_path, default_root=allow[0])
    hooks.register(str(ticket["task_id"]), s.handle_hook)
    try:
        return s, s.run()
    finally:
        hooks.unregister(str(ticket["task_id"]))
        hooks.stop()


def test_limit_output_mid_turn_is_a_usage_limit_not_an_error(short_tmp, fake_home, env_clean, monkeypatch):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "usage-limit")
    _, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "usage_limit", out.error
    assert out.error.startswith("paused: claude usage limit, resumes ~")
    assert "exited (code" not in out.error and "key" not in out.error
    assert datetime.fromisoformat(out.resets_at).hour == 15
    payload = out.as_result_payload(1)
    assert payload["status"] == "usage_limit" and payload["resets_at"] == out.resets_at


def test_claude_codes_parked_line_is_a_limit_and_its_warning_is_not():
    for said in ("\x1b[38;2;153;153;153m●\x1b[3GUsage limit reached · continuing automatically at 9am · esc to cancel",
                 "⚠ Usage limit reached · limit resets 9am\r\x1b[1B\x1b[39m    Continuing automatically at 9am · Esc",
                 "5-hour limit reached · continuing automatically at 3pm"):
        assert usage_limit.is_parked(said) and usage_limit.is_usage_limit(said), said
    for said in ("Approaching usage limit · resets at 9am", "We are continuing automatically at 10 tomorrow.", ""):
        assert not usage_limit.is_parked(said), said


def test_a_session_parked_at_its_limit_ends_at_once_as_a_pause(short_tmp, fake_home, env_clean, monkeypatch):
    """5 Nov 2026: Claude Code parked two turns on "continuing automatically at 9am" and
    the host waited out its 4 h session timeout, holding both tickets and both slots."""
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "usage-limit-parked")
    started = time.monotonic()
    _, out = _run(short_tmp, _ticket(workdir))
    assert time.monotonic() - started < 60                      # not the 120 s session timeout
    assert out.status == "usage_limit", out.error
    assert out.exit_reason == "parked_at_limit"
    assert out.error.startswith("paused: claude usage limit, resumes ~")
    assert datetime.fromisoformat(out.resets_at).hour == 9      # Claude Code's "resets 9am (Europe/Berlin)"


def test_a_session_at_the_usage_limit_menu_ends_at_once_and_pauses_until_the_stated_reset(
        short_tmp, fake_home, env_clean, monkeypatch):
    """The menu sat on screen until the 4 h session timeout, and the pause that followed
    was 15 minutes although the screen said the limit reset days later."""
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    notice, reset = _weekly_notice()
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "usage-limit-menu")
    monkeypatch.setenv("FAKE_CLAUDE_LIMIT_TEXT", notice)
    started = time.monotonic()
    _, out = _run(short_tmp, _ticket(workdir))
    assert time.monotonic() - started < 60                      # not the 120 s session timeout
    assert (out.status, out.exit_reason) == ("usage_limit", "parked_at_limit"), out.error
    assert datetime.fromisoformat(out.resets_at) == reset       # the stated day and hour, not a re-check
    assert out.reset_stated and out.error.startswith("paused: claude usage limit, resumes ~")


def test_the_menu_is_seen_when_one_redraw_puts_more_than_the_scan_window_after_it(short_tmp, fake_home, env_clean):
    """A 200-column frame draws wide borders under the menu in the same read: the
    menu's line must not fall out of the scanned window before it is read."""
    workdir = short_tmp / "ws" / "repo"
    s = Session(_ticket(workdir), _cfg(short_tmp), [str(short_tmp / "ws")], short_tmp / "h.sock",
                default_root=str(short_tmp / "ws"))
    border = ("─" * 200 + "\r\n").encode("utf-8") * 8         # about 4.8 KB after the menu
    s._watch_for_parking(WEEKLY_LIMIT_SCREEN.encode("utf-8") + border)
    assert s.parked.is_set()


def test_a_parked_turn_is_a_pause_even_when_its_notice_scrolled_out_of_the_tail(
        short_tmp, fake_home, env_clean, monkeypatch):
    """More than the 4000 characters the limit check reads came after the menu: the
    turn the host ended at its limit is a pause all the same, never a failed ticket.
    With no reset time left to read, it degrades to the short re-check."""
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "usage-limit-menu")
    monkeypatch.setenv("FAKE_CLAUDE_LIMIT_PADDING_LINES", "21")       # 21 x 202 characters after the menu
    _, out = _run(short_tmp, _ticket(workdir))
    assert (out.status, out.exit_reason) == ("usage_limit", "parked_at_limit"), out.error
    assert not out.reset_stated and out.error.startswith("paused: claude usage limit")


def test_a_turn_ended_on_a_rate_limit_api_error_is_a_pause_not_a_failure(short_tmp, fake_home, env_clean, monkeypatch):
    """Claude Code's StopFailure of type rate_limit is the plan's window closing: the
    ticket goes back to the queue until the reset its words state."""
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    notice, reset = _weekly_notice()
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "api-error")
    monkeypatch.setenv("FAKE_CLAUDE_API_ERROR", "rate_limit")
    monkeypatch.setenv("FAKE_CLAUDE_API_ERROR_TEXT", notice)
    started = time.monotonic()
    _, out = _run(short_tmp, _ticket(workdir))
    assert time.monotonic() - started < 60
    assert (out.status, out.exit_reason) == ("usage_limit", "api_error"), out.error
    assert datetime.fromisoformat(out.resets_at) == reset and out.reset_stated


def test_an_api_error_is_judged_by_its_own_words_not_the_output_before_it(
        short_tmp, fake_home, env_clean, monkeypatch):
    """An API that cannot be reached is a failed ticket, even when the session's own
    earlier output talked about a rate limit (a tool's answer, say): only the error's
    words decide, and they name no limit."""
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "api-error")
    monkeypatch.setenv("FAKE_CLAUDE_API_ERROR", "server_error")
    monkeypatch.setenv("FAKE_CLAUDE_API_ERROR_TEXT",
                       "API Error: Connection refused — a firewall or proxy may be blocking it (ECONNREFUSED)")
    monkeypatch.setenv("FAKE_CLAUDE_OUTPUT_BEFORE", "GitHub answered 403: API rate limit exceeded · resets 3pm")
    _, out = _run(short_tmp, _ticket(workdir))
    assert (out.status, out.exit_reason) == ("error", "api_error"), out.error
    assert "ECONNREFUSED" in out.error and "server_error" in out.error and out.resets_at is None


def test_a_normal_non_zero_exit_still_errors(short_tmp, fake_home, env_clean, monkeypatch):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "exit-early")
    _, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "error" and "code 3" in out.error and out.resets_at is None


# ── the host: no claims for a paused CLI until its window reopens ───────────

class _Api:
    def __init__(self, tasks):
        self.tasks = tasks

    def claim(self, host_id, limit):
        return {"tasks": list(self.tasks), "parked": []}


def _host(short_tmp, tasks=()):
    host = Host(_cfg(short_tmp, url="http://127.0.0.1:9", allow_dirs=[short_tmp / "ws"]))
    host.api = _Api(tasks)
    return host


def test_a_paused_cli_is_not_served_and_says_why(short_tmp, fake_home, env_clean):
    host = _host(short_tmp)
    assert "claude" in host.capabilities()["providers"]
    until = (datetime.now().astimezone() + timedelta(hours=1)).isoformat()
    host._last_heartbeat = time.time()
    host._pause_cli("claude", until, "paused: claude usage limit, resumes ~15:00")
    caps = host.capabilities()
    assert "claude" not in caps["providers"]                    # the backend's claim filter
    assert caps["clis"]["claude"]["served"] is False
    assert caps["clis"]["claude"]["reason"] == "paused: claude usage limit, resumes ~15:00"
    assert caps["paused"] == {"claude": "paused: claude usage limit, resumes ~15:00"}
    assert host._last_heartbeat == 0.0                          # the backend hears on the next tick


def test_the_window_reopening_restores_the_cli(short_tmp, fake_home, env_clean):
    host = _host(short_tmp)
    past = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat()
    host._pause_cli("claude", past, "paused: claude usage limit, resumes ~now")
    caps = host.capabilities()
    assert "claude" in caps["providers"] and "paused" not in caps and host.limited == {}


def test_a_stated_reset_stands_against_a_recheck_and_yields_to_the_next_stated_one(
        short_tmp, fake_home, env_clean):
    """Both slots stop on the same limit. When the second reads no reset time, its
    15-minute re-check must not replace the stated one: the host would call the
    window reopened and claim again days before it reset. A newly stated reset
    always replaces the held one, so a wrong long pause does not stick."""
    host = _host(short_tmp)
    now = datetime.now().astimezone()
    weekly, known = usage_limit.pause(_weekly_notice()[0], now)
    assert known
    host._pause_cli("claude", weekly.isoformat(), usage_limit.describe("claude", weekly, known), stated=True)
    recheck, stated = usage_limit.pause("Usage limit reached.", now)
    assert not stated
    host._pause_cli("claude", recheck.isoformat(), usage_limit.describe("claude", recheck, stated), stated=stated)
    assert host.limited["claude"][0] == weekly.timestamp()
    assert "claude" not in host.capabilities()["providers"]
    sooner, known = usage_limit.pause("You've hit your limit · resets 3pm (Europe/Dublin)", now)
    host._pause_cli("claude", sooner.isoformat(), usage_limit.describe("claude", sooner, known), stated=known)
    assert host.limited["claude"][0] == sooner.timestamp()


def test_a_ticket_claimed_for_a_paused_cli_goes_straight_back(short_tmp, fake_home, env_clean):
    ticket = {"task_id": 77, "attempt": 2, "provider": "claude", "session_id": str(uuid.uuid4())}
    host = _host(short_tmp, tasks=[ticket])
    until = (datetime.now().astimezone() + timedelta(hours=1)).isoformat()
    host._pause_cli("claude", until, "paused: claude usage limit, resumes ~15:00")
    host._claim_and_start("h1")
    assert host.sessions == {}                                  # nothing spawned
    released = host.pending_results["77"]
    assert released["status"] == "usage_limit" and released["attempt"] == 2
    assert released["resets_at"] == until and released["error"].startswith("paused: claude usage limit")


# ── PRD-253 S1.6: GitHub Copilot's own words for a plan that ran out ─────────


@pytest.mark.parametrize("said", [
    "Error: You've run out of your AI credits for this month.",
    "You have used your included AI credits for the month.",
    "You've been rate limited on the current model. Would you like to switch to auto mode to continue?",
    "You've hit your rate limit.",
    "You've reached your weekly rate limit.",
    "Your session rate limit was reached.",
])
def test_copilots_limit_sentences_pause(said):
    assert usage_limit.is_usage_limit(said)


def test_copilots_model_policy_error_is_not_a_pause():
    """The organisation's model policy is an error the operator fixes, not a window that reopens."""
    assert not usage_limit.is_usage_limit("Error: No supported model available")
    assert not usage_limit.is_usage_limit("No model available. Check policy enablement under GitHub Settings > Copilot")
