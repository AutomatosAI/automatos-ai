"""PRD-268 US-C03a (§6.2.1, §6.2.2, §6.2.4, §6.9) — the check-in rules, as pure functions.

``modules/proactive/`` decides; the services read the database and pass in what they read.
Proven here, function by function:

* the window starts where the last successful run read up to, else a day ago; a failed
  or skipped run after the last success leaves it there;
* titles normalise (Unicode form, case, spacing, trailing punctuation) and the three
  subject keys are built the same way every time, a ticket's from its board number; a
  title that says nothing is refused;
* a suggestion repeats an open one with the same key or title, or a hidden one hidden
  within 14 days (13 days ago: a repeat; 15: not);
* the §6.9 caps are the defaults, frozen and validated, and each check refuses with
  ``skipped_cap`` at its limit; the oldest open suggestions expire to make room;
* a streak counts one error class in a row; the interval doubles from the third failure,
  up to 24 hours, and a success resets it;
* evidence must name one to five reads the run made, a ticket by its board number;
* the package imports nothing but the standard library and itself, and mutates nothing.
"""
from __future__ import annotations

import ast
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.proactive import caps as caps_module
from modules.proactive import dedupe, provenance, streaks, window
from modules.proactive.caps import CapRefusal, Caps, clamp_interval, oldest_to_expire, run_refusal, suggestion_refusal
from modules.proactive.dedupe import (
    Candidate,
    agent_error_key,
    board_number,
    find_duplicate,
    is_duplicate,
    normalise_title,
    ticket_key,
    title_key,
)
from modules.proactive.provenance import provenance_ok
from modules.proactive.streaks import consecutive_failures, is_due, next_interval
from modules.proactive.window import check_in_window, last_success_at

PACKAGE = Path(__file__).resolve().parents[1] / "modules" / "proactive"
NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)


def _run(outcome: str, hours_ago: float, error_class=None, *, read_until_hours_ago=None) -> SimpleNamespace:
    window_until = NOW - timedelta(hours=read_until_hours_ago) if read_until_hours_ago is not None else None
    return SimpleNamespace(outcome=outcome, created_at=NOW - timedelta(hours=hours_ago), window_until=window_until,
                           error_class=error_class)


def _row(title: str, subject_key: str, *, decided_days_ago=None, created_days_ago: float = 20) -> SimpleNamespace:
    decided_at = NOW - timedelta(days=decided_days_ago) if decided_days_ago is not None else None
    created_at = NOW - timedelta(days=created_days_ago)
    return SimpleNamespace(title=title, subject_key=subject_key, decided_at=decided_at,
                           last_seen_at=created_at, created_at=created_at)


# ── the package is pure ─────────────────────────────────────────────────────

ALLOWED_IMPORTS = {"__future__", "collections", "dataclasses", "datetime", "re", "typing", "unicodedata", "modules"}


def test_the_package_imports_nothing_but_the_standard_library_and_itself():
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in ALLOWED_IMPORTS, f"{path.name} imports {name}"
                assert name.split(".")[0] != "modules" or name.startswith("modules.proactive"), (
                    f"{path.name} imports {name}")


def test_the_names_it_shares_are_the_models_and_the_boards():
    from core.models import proactive as models
    from services.ticket_numbers import NUMBER_DIGITS, format_number

    assert window.SUCCESSFUL_OUTCOMES == {models.RUN_SILENT, models.RUN_SUGGESTED}
    assert streaks.FAILED == models.RUN_ERROR
    assert caps_module.CAPPED == models.RUN_SKIPPED_CAP
    assert dedupe.SUBJECT_KEY_LENGTH == models.SUBJECT_KEY_LENGTH
    assert dedupe.TICKET_DIGITS == NUMBER_DIGITS
    assert ticket_key(format_number(846)) == "ticket:0846"


# ── the window (§6.2.1) ─────────────────────────────────────────────────────

def test_with_no_successful_run_the_window_is_the_last_day():
    assert check_in_window(None, NOW) == NOW - DAY


def test_the_window_starts_at_the_last_success():
    assert check_in_window(NOW - timedelta(hours=3), NOW) == NOW - timedelta(hours=3)


def test_a_window_after_a_failed_run_starts_at_the_last_success():
    runs = [_run("error", 1, "401"), _run("suggested", 6), _run("skipped_cap", 0.5), _run("silent", 10)]

    since = check_in_window(last_success_at(runs), NOW)

    assert since == NOW - timedelta(hours=6)


def test_the_next_window_starts_where_the_last_success_read_up_to_not_when_it_was_recorded():
    runs = [_run("suggested", 6, read_until_hours_ago=6.25), _run("silent", 9, read_until_hours_ago=9.5)]

    assert last_success_at(runs) == NOW - timedelta(hours=6.25)


def test_runs_that_never_succeeded_leave_the_last_day():
    runs = [_run("error", 1, "timeout"), _run("skipped_quiet", 2), _run("skipped_off", 3)]

    assert last_success_at(runs) is None
    assert check_in_window(last_success_at(runs), NOW) == NOW - DAY
    assert last_success_at([]) is None


def test_a_success_stamped_after_now_gives_an_empty_window_not_a_reversed_one():
    assert check_in_window(NOW + timedelta(minutes=5), NOW) == NOW


# ── titles and subject keys (§6.2.4) ────────────────────────────────────────

@pytest.mark.parametrize("title,normalised", [
    ("Fix THE Board", "fix the board"),
    ("  fix   the\tboard \n", "fix the board"),
    ("Fix the board!!!", "fix the board"),
    ("Fix the board ?", "fix the board"),
    ("Fix the board…", "fix the board"),
    ("Fix the board. ", "fix the board"),
    ("VECTOR's check-ins FAIL: since 09:00.", "vector's check-ins fail: since 09:00"),
    ("?!", ""),
    (None, ""),
    ("\uff26ix the board", "fix the board"),
    ("Cafe\u0301 is closed", "caf\u00e9 is closed"),
])
def test_a_title_normalises_case_spacing_and_trailing_punctuation(title, normalised):
    assert normalise_title(title) == normalised


@pytest.mark.parametrize("agent_id,error_class,key", [
    (578, "401", "agent:578:error:401"),
    ("578", " Timeout ", "agent:578:error:timeout"),
    (12, "Provider  refused", "agent:12:error:provider_refused"),
])
def test_an_agent_error_streak_key(agent_id, error_class, key):
    assert agent_error_key(agent_id, error_class) == key


@pytest.mark.parametrize("error_class", ["", "  ", None])
def test_a_streak_key_needs_its_error_class(error_class):
    with pytest.raises(ValueError):
        agent_error_key(578, error_class)


@pytest.mark.parametrize("number,key", [
    ("#0846", "ticket:0846"),
    ("846", "ticket:0846"),
    (846, "ticket:0846"),
    (" 0846 ", "ticket:0846"),
    ("#0051.3", "ticket:0051.3"),
    ("#12345", "ticket:12345"),
])
def test_a_ticket_key_is_its_board_number(number, key):
    assert ticket_key(number) == key


@pytest.mark.parametrize("number", ["abc", "#", "#08a6", "", "1.2.3", 846.0, 188.3, True])
def test_a_ticket_key_refuses_what_is_not_a_board_number(number):
    assert board_number(number) is None
    with pytest.raises(ValueError):
        ticket_key(number)


def test_a_title_key_is_its_normalised_title_within_the_column():
    assert title_key("Fix the Board!") == "title:fix the board"
    assert len(title_key("word " * 100)) == dedupe.SUBJECT_KEY_LENGTH


@pytest.mark.parametrize("title", ["", "  ", "?!", None])
def test_a_title_that_says_nothing_has_no_key(title):
    with pytest.raises(ValueError):
        title_key(title)
    with pytest.raises(ValueError):
        _ = Candidate(title).key


def test_a_candidate_is_keyed_by_its_own_key_else_its_title():
    assert Candidate("Ticket #0846 is stuck", "ticket:0846").key == "ticket:0846"
    assert Candidate("Fix the board!").key == "title:fix the board"


# ── dedupe: open, and hidden within 14 days (§6.2.4, FR-6) ──────────────────

def test_an_open_suggestion_with_the_same_key_is_a_duplicate():
    open_row = _row("Ticket #0846 has been stuck since Friday", "ticket:0846")

    candidate = Candidate("Look at #0846", "ticket:0846")

    assert find_duplicate(candidate, [open_row], [], NOW) is open_row
    assert is_duplicate(candidate, [open_row], [], NOW)


def test_an_open_suggestion_with_the_same_title_is_a_duplicate_whatever_its_key():
    open_row = _row("Fix the board", "ticket:0846")

    assert is_duplicate(Candidate("fix  the BOARD."), [open_row], [], NOW)


@pytest.mark.parametrize("days_ago,duplicate", [(13, True), (14, True), (15, False)])
def test_a_hidden_suggestion_blocks_its_subject_for_14_days(days_ago, duplicate):
    hidden = _row("Fix the board", "title:fix the board", decided_days_ago=days_ago)

    assert is_duplicate(Candidate("Fix the board"), [], [hidden], NOW) is duplicate
    assert is_duplicate(Candidate("Something else", "title:fix the board"), [], [hidden], NOW) is duplicate


def test_a_hidden_row_without_a_decision_time_is_dated_by_when_it_was_last_seen():
    hidden = _row("Fix the board", "title:fix the board", created_days_ago=13)

    assert is_duplicate(Candidate("Fix the board"), [], [hidden], NOW)


def test_titles_that_say_nothing_do_not_match_each_other():
    open_row = _row("!!", "ticket:0846")

    assert not is_duplicate(Candidate("?", "ticket:0847"), [open_row], [], NOW)


def test_another_subject_is_not_a_duplicate():
    open_row = _row("Fix the board", "ticket:0846")
    hidden = _row("Rotate the key", "agent:578:error:401", decided_days_ago=2)

    candidate = Candidate("Ticket #0847 failed", "ticket:0847")

    assert find_duplicate(candidate, [open_row], [hidden], NOW) is None
    assert not is_duplicate(candidate, (), (), NOW)


# ── caps (§6.9) ─────────────────────────────────────────────────────────────

def test_the_caps_are_the_prd_defaults_and_frozen():
    caps = Caps()

    assert (caps.min_interval_minutes, caps.max_interval_minutes) == (30, 24 * 60)
    assert caps.max_model_runs_per_day == 48
    assert caps.max_suggestions_per_run == 3
    assert caps.max_new_per_person_per_day == 10
    assert caps.max_open_per_person == 20
    assert caps.max_open_per_agent == 3
    assert caps.max_pushes_per_person_per_day == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.max_suggestions_per_run = 5
    own = dataclasses.replace(caps, max_suggestions_per_run=2)
    assert own.max_suggestions_per_run == 2 and caps.max_suggestions_per_run == 3


@pytest.mark.parametrize("overrides", [
    {"max_suggestions_per_run": -1},
    {"max_open_per_agent": 2.5},
    {"max_model_runs_per_day": "48"},
    {"max_new_per_person_per_day": True},
    {"min_interval_minutes": 0},
    {"min_interval_minutes": 120, "max_interval_minutes": 60},
])
def test_caps_a_workspace_got_wrong_are_refused(overrides):
    with pytest.raises(ValueError):
        dataclasses.replace(Caps(), **overrides)


def test_a_cap_of_zero_allows_nothing():
    none_per_run = dataclasses.replace(Caps(), max_suggestions_per_run=0)

    assert suggestion_refusal(none_per_run, made_this_run=0, new_today_for_person=0, open_for_agent=0).limit == 0


def test_the_model_run_cap_refuses_at_its_limit_with_the_skip_outcome():
    assert run_refusal(Caps(), 47) is None
    assert run_refusal(Caps(), 48) == CapRefusal(cap="max_model_runs_per_day", limit=48, count=48)
    assert run_refusal(Caps(), 48).outcome == "skipped_cap"


def test_a_run_makes_at_most_three_suggestions():
    assert suggestion_refusal(Caps(), made_this_run=2, new_today_for_person=0, open_for_agent=0) is None

    refusal = suggestion_refusal(Caps(), made_this_run=3, new_today_for_person=0, open_for_agent=0)

    assert refusal == CapRefusal(cap="max_suggestions_per_run", limit=3, count=3, outcome="skipped_cap")


@pytest.mark.parametrize("new_today,open_for_agent,cap", [
    (10, 0, "max_new_per_person_per_day"),
    (0, 3, "max_open_per_agent"),
    (10, 3, "max_new_per_person_per_day"),
])
def test_each_suggestion_cap_refuses_at_its_limit(new_today, open_for_agent, cap):
    refusal = suggestion_refusal(Caps(), made_this_run=0, new_today_for_person=new_today,
                                 open_for_agent=open_for_agent)

    assert refusal.cap == cap and refusal.outcome == "skipped_cap"


def test_a_workspaces_own_caps_are_the_ones_checked():
    own = dataclasses.replace(Caps(), max_suggestions_per_run=1, max_model_runs_per_day=5)

    assert suggestion_refusal(own, made_this_run=1, new_today_for_person=0, open_for_agent=0).limit == 1
    assert run_refusal(own, 5).limit == 5


def test_the_oldest_open_suggestions_expire_to_make_room():
    rows = [SimpleNamespace(name=f"s{age}", created_at=NOW - timedelta(hours=age)) for age in (5, 30, 1, 22, 9)]
    before = list(rows)
    small = dataclasses.replace(Caps(), max_open_per_person=3)

    expired = oldest_to_expire(small, rows)

    assert [row.name for row in expired] == ["s30", "s22", "s9"]
    assert rows == before
    assert oldest_to_expire(Caps(), rows) == ()
    assert len(oldest_to_expire(Caps(), [SimpleNamespace(created_at=NOW)] * 20)) == 1


@pytest.mark.parametrize("minutes,clamped", [(10, 30), (30, 30), (240, 240), (1440, 1440), (3000, 1440)])
def test_an_interval_is_held_within_the_caps(minutes, clamped):
    assert clamp_interval(Caps(), minutes) == clamped


# ── failing streaks (§6.2.2) ────────────────────────────────────────────────

def test_a_streak_counts_one_error_class_in_a_row_newest_first():
    runs = [_run("error", 1, "401"), _run("skipped_cap", 2), _run("error", 3, "401"),
            _run("error", 4, "401"), _run("silent", 5), _run("error", 6, "401")]

    assert consecutive_failures(runs, "401") == 3


def test_a_success_or_another_error_class_ends_the_streak():
    assert consecutive_failures([_run("suggested", 1), _run("error", 2, "401")], "401") == 0
    assert consecutive_failures([_run("error", 1, "401"), _run("error", 2, "timeout"),
                                 _run("error", 3, "401")], "401") == 1
    assert consecutive_failures([_run("error", 1, "timeout")], "401") == 0
    assert consecutive_failures([], "401") == 0


def test_a_streaks_error_class_compares_as_its_key_writes_it():
    runs = [_run("error", 1, "Timeout"), _run("error", 2, " timeout "), _run("error", 3, "TIMEOUT")]

    assert consecutive_failures(runs, "timeout") == 3
    assert consecutive_failures([_run("error", 1, "Provider refused")], "provider_refused") == 1


def test_the_interval_doubles_from_the_third_failure_capped_at_24_hours():
    schedule = [next_interval(60, failures) for failures in range(13)]

    assert schedule == [60, 60, 60, 120, 240, 480, 960, 1440, 1440, 1440, 1440, 1440, 1440]
    assert next_interval(30, 3) == 60
    assert next_interval(60, 10_000) == 24 * 60


def test_a_success_resets_the_interval():
    runs = [_run("silent", 0.1)] + [_run("error", hours, "401") for hours in range(1, 12)]

    assert next_interval(60, consecutive_failures(runs, "401")) == 60


def test_a_day_long_interval_stays_a_day_and_a_longer_one_is_never_shortened():
    assert next_interval(1440, 5) == 1440
    assert next_interval(2000, 0) == 2000
    assert next_interval(2000, 5) == 2000


@pytest.mark.parametrize("base,failures", [(0, 3), (-60, 3), (60, -1)])
def test_a_nonsense_interval_or_count_is_refused(base, failures):
    with pytest.raises(ValueError):
        next_interval(base, failures)


def test_a_streaks_later_ticks_wait_for_the_doubled_interval():
    two_hours_ago = NOW - timedelta(hours=2)

    assert is_due(two_hours_ago, 60, 3, NOW)  # 120 minutes have passed
    assert not is_due(two_hours_ago, 60, 4, NOW)  # 240 have not
    assert is_due(NOW - timedelta(hours=1), 60, 0, NOW)


# ── provenance (§6.2.3) ─────────────────────────────────────────────────────

READS = [{"type": "ticket", "id": 846}, {"type": "report", "id": "r-9"}, {"type": "run", "id": "b2c"}]


def test_evidence_that_names_reads_the_run_made_is_provenance():
    assert provenance_ok([{"type": "Ticket", "id": "846"}, {"type": "report", "id": "r-9"}], READS)


@pytest.mark.parametrize("cited", ["#0846", "0846", "846", 846])
def test_a_ticket_is_cited_by_its_board_number_however_it_is_written(cited):
    assert provenance_ok([{"type": "ticket", "id": cited}], READS)
    assert provenance_ok([{"type": "ticket", "id": 846}], [{"type": "ticket", "id": "#0846"}])


def test_a_ticket_cited_as_a_float_is_not_its_board_number():
    assert not provenance_ok([{"type": "ticket", "id": 846.0}], READS)


def test_evidence_the_run_did_not_read_is_refused():
    assert not provenance_ok([{"type": "ticket", "id": "846"}, {"type": "document", "id": "d-1"}], READS)
    assert not provenance_ok([{"type": "report", "id": "846"}], READS)


@pytest.mark.parametrize("evidence", [
    [],
    None,
    [{"type": "ticket"}],
    [{"id": "846"}],
    ["ticket:846"],
    [{"type": "ticket", "id": "  "}],
    [{"type": "ticket", "id": 846}] * 6,
])
def test_no_evidence_malformed_evidence_or_more_than_five_refs_is_refused(evidence):
    assert not provenance_ok(evidence, READS)


def test_a_run_with_no_reads_proves_nothing_and_malformed_reads_are_ignored():
    assert not provenance_ok([{"type": "ticket", "id": 846}], None)
    assert not provenance_ok([{"type": "ticket", "id": 846}], [])
    assert provenance_ok([{"type": "ticket", "id": 846}], ["junk", {"type": "ticket"}, {"type": "ticket", "id": 846}])
    assert provenance.MAX_EVIDENCE == 5
