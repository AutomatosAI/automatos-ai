"""The Composio hint steers the model to composio_execute; it never forces it.

Its last line said "You MUST call `composio_execute`", and the OpenAI-compatible
clients turned that phrase into tool_choice="required": a 400 on Claude Opus 5.5
and Fable 5.1, and a forced call on turns where the top-N fallback had matched
actions the user never asked about.
"""
from modules.tools.services.composio_hint_service import _header_lines, _match_lines

MATCHES = [
    ("SLACK", ["SLACK_SEND_MESSAGE"]),
    ("GMAIL", ["GMAIL_SEND_EMAIL", "GMAIL_LIST_EMAILS"]),
    ("JIRA", ["JIRA_CREATE_ISSUE"]),
]


def test_apps_with_more_matches_come_first_and_their_actions_are_named():
    lines, named = _match_lines(MATCHES, {})
    assert lines[0] == "- GMAIL available actions: GMAIL_SEND_EMAIL, GMAIL_LIST_EMAILS"
    assert lines[1].startswith("- JIRA") and lines[2].startswith("- SLACK")
    assert named == ["GMAIL_SEND_EMAIL", "GMAIL_LIST_EMAILS", "JIRA_CREATE_ISSUE", "SLACK_SEND_MESSAGE"]
    assert "calling `composio_execute`" in lines[-1]


def test_at_most_six_apps_are_listed():
    lines, named = _match_lines([(f"APP{i}", [f"APP{i}_ACT"]) for i in range(9)], {})
    assert sum(line.startswith("- APP") for line in lines) == 6 and len(named) == 6


def test_no_match_no_call_line():
    lines, named = _match_lines([], {})
    assert lines == [] and named == []


def test_no_hint_text_can_force_a_tool_call():
    lines, _ = _match_lines(MATCHES, {"SLACK_SEND_MESSAGE": "channel, text"})
    text = "\n".join(_header_lines(["SLACK", "GMAIL", "SLACK"]) + lines)
    assert "You MUST call" not in text
    assert "(via Composio): GMAIL, SLACK." in text
