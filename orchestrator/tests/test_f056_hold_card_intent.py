"""F056 — the hold card names the program the command runs, not a path fragment.

Night 2, grant 355: the card read "The agent wants to run **359**". The command
began ``WORK=/…/sessions/359``, and the first-word/basename rule turned an
ASSIGNMENT's path into a "program". The program, three lines down past a blank
line and a comment, was ``grep``.
"""
from __future__ import annotations

from services.cli_host_service import _command_verbs, _plain_intent, session_hold_question

GRANT_355 = (
    "WORK=/Users/gkavanagh/.automatos/cli-host/sessions/359\n"
    "\n"
    "# Exclude files I've already read in depth\n"
    "grep -v -e 'sessions/315/' -e 'sessions/324/' x"
)


def test_the_night_2_card_names_grep_not_the_ticket_number():
    assert _command_verbs(GRANT_355) == ["grep"]
    line = _plain_intent(GRANT_355)
    assert "grep" in line and "359" not in line
    assert "search files for a pattern" in line


def test_assignments_and_comments_are_not_programs():
    assert _command_verbs("A=1 B=/x/y sort data.txt") == ["sort"]
    assert _command_verbs("# just a note\nls") == ["ls"]
    assert _command_verbs("/usr/bin/grep x f") == ["grep"]


def test_a_command_that_only_sets_a_variable_says_so():
    assert "set a shell variable" in _plain_intent("WORK=/tmp/x")


def test_several_programs_are_all_named_once_in_order():
    assert _command_verbs("cat a | sort | uniq && cat b") == ["cat", "sort", "uniq"]
    line = _plain_intent("cat a | sort | uniq")
    assert line.startswith("The agent wants to **read a file** (`cat`), then `sort`, `uniq`")


def test_the_full_card_carries_the_real_intent_and_the_exact_command():
    card = session_hold_question(359, {"subject": GRANT_355, "reason": "could not be parsed"})
    assert "The agent wants to run **359**" not in card
    assert "grep" in card.split("<details>")[0]           # the plain-English lead names it
    assert GRANT_355.splitlines()[0] in card              # the exact command is still there, folded
