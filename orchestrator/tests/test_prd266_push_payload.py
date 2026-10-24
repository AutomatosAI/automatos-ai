"""PRD-266 US-011a (§4.4, D17, D20, FR-7, FR-8, FR-9; K8, K9) — what a push says and its signature.

``services/push/payload.py`` builds each push type's signed v2 payload and its Expo message:

* Only the five push events push: ``PUSH_EVENT_TYPES`` is exactly them, and the three push
  types are ``core.models.mobile_push.PUSH_TYPES``.
* A question carries its text as plain text (markdown dropped), at most 500 characters with an
  ellipsis, and its options only when there are 1 to 3 of at most 80 characters each (never
  cut: ``/answer`` takes the exact string). Every other push carries its generic line; so does
  a question when the workspace redacts questions.
* The title is "<agent or Auto> · <badge>" with the PRD-264 badge for the phone's locale.
* The ONE shared vector file (``tests/fixtures/prd266_push_signature_vectors.json``, which
  automatos-go copies byte for byte): every case's message and sig recompute, the builder
  reproduces every case's payload from its inputs, and no tampered case verifies.
* A change to any signed field, a ``v`` other than 2, and an ``iat`` one second outside the
  window (7 days old, 300 s ahead) fail ``verify_signature``; every id is a string.

DB-free: no database, no network.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from core.models.mobile_push import PUSH_TYPES
from services.push import payload as push_payload
from services.push.payload import (
    APPROVAL_LINE,
    AUTO_MESSAGE_LINE,
    ELLIPSIS,
    MISSION_PLAN_LINE,
    OPEN_CATEGORY,
    PUSH_EVENT_TYPES,
    PUSH_OPTION_MAX_CHARS,
    PUSH_QUESTION_MAX_CHARS,
    PUSH_SIG_MAX_AGE_SECONDS,
    PUSH_SIG_MAX_FUTURE_SECONDS,
    PUSH_TYPE_EVENTS,
    QUESTION_LINE,
    REVIEW_LINE,
    build_payload,
    canonical_message,
    push_type_for,
    redact_questions_enabled,
    sign,
    to_message,
    verify_signature,
)
from services.push.sender import PRIORITY_DEFAULT, PRIORITY_HIGH, PushMessage

VECTORS = Path(__file__).resolve().parent / "fixtures" / "prd266_push_signature_vectors.json"
FIVE = {"question_pending", "approval_pending", "mission_plan_ready", "review_pending", "auto_message"}
PUSH_KEY = "pk-test1"  # under 10 characters: gitleaks' generic-api-key never matches it
TOKEN = "ExpoTok1"
WS = uuid.UUID("5b0c4c8e-2f6a-4d3e-9a7b-1c2d3e4f5a6b")
NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)
QUESTION = "Ship the Lisbon deck?"
SIGNED_FIELDS = ("type", "workspace_id", "notification_id", "grant_id", "link_type", "link_id",
                 "ticket_number", "question")


def _vectors() -> dict:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def _at(iat: int) -> datetime:
    return datetime.fromtimestamp(iat, tz=timezone.utc)


def _grant(question_md: Any = QUESTION, options: Any = None, grant_id: int = 42) -> SimpleNamespace:
    return SimpleNamespace(id=grant_id, question_md=question_md, options=options)


def _device(locale: Optional[str] = "en") -> SimpleNamespace:
    return SimpleNamespace(expo_push_token=TOKEN, locale=locale)


def _question(options: Any = None, *, question_md: Any = QUESTION, redact: bool = False) -> dict:
    return build_payload(
        PUSH_KEY, event_type="question_pending", workspace_id=WS, notification_id=1001,
        link_type="question", link_id=42, grant=_grant(question_md, options), redact=redact, now=NOW,
    )


def _message(payload: dict, event_type: str = "question_pending", **kwargs: Any) -> PushMessage:
    return to_message(_device(kwargs.pop("locale", "en")), payload, event_type=event_type, **kwargs)


# --------------------------------------------------------------------------- which events push


def test_push_event_types_are_exactly_the_five():
    assert PUSH_EVENT_TYPES == FIVE
    assert set(PUSH_TYPE_EVENTS) == set(PUSH_TYPES)
    assert set(push_payload.GENERIC_LINES) == FIVE
    assert set(push_payload.CHANNEL_BY_TYPE) == set(PUSH_TYPES)


@pytest.mark.parametrize("event_type, push_type", [
    ("question_pending", "needs_answer"), ("approval_pending", "needs_answer"),
    ("mission_plan_ready", "needs_answer"), ("review_pending", "review"), ("auto_message", "auto_message"),
])
def test_each_push_event_has_its_push_type(event_type, push_type):
    assert push_type_for(event_type) == push_type


@pytest.mark.parametrize("event_type", ["task_completed", "agent_error", "", None, 42])
def test_any_other_event_never_pushes(event_type):
    assert push_type_for(event_type) is None
    with pytest.raises(ValueError):
        build_payload(PUSH_KEY, event_type=event_type, workspace_id=WS, now=NOW)


# --------------------------------------------------------------------------- each type's payload and line


@pytest.mark.parametrize("event_type, link_type, line, channel, priority", [
    ("approval_pending", "approval_grant", APPROVAL_LINE, "questions", PRIORITY_HIGH),
    ("mission_plan_ready", "mission", MISSION_PLAN_LINE, "questions", PRIORITY_HIGH),
    ("review_pending", "task", REVIEW_LINE, "review", PRIORITY_DEFAULT),
    ("auto_message", "chat", AUTO_MESSAGE_LINE, "auto", PRIORITY_DEFAULT),
])
def test_every_other_push_carries_its_generic_line(event_type, link_type, line, channel, priority):
    built = build_payload(PUSH_KEY, event_type=event_type, workspace_id=WS, notification_id=7,
                          link_type=link_type, link_id=9, now=NOW)
    assert built["type"] == push_type_for(event_type)
    assert "question" not in built and "options" not in built
    message = _message(built, event_type)
    assert (message.body, message.channel_id, message.priority) == (line, channel, priority)
    assert message.category_id == OPEN_CATEGORY
    assert message.to == TOKEN and dict(message.data) == built


def test_the_generic_lines_read_as_written():
    assert (QUESTION_LINE, APPROVAL_LINE, MISSION_PLAN_LINE, AUTO_MESSAGE_LINE) == (
        "A question needs your answer", "An approval is waiting for you",
        "A Mission plan is ready for you", "Auto messaged you")


def test_a_review_names_its_ticket_number():
    built = build_payload(PUSH_KEY, event_type="review_pending", workspace_id=WS, notification_id=1003,
                          link_type="task", link_id=8812, ticket_number="#0419", now=NOW)
    assert built["ticket_number"] == "#0419"
    assert _message(built, "review_pending").body == "A ticket is ready for review · #0419"


def test_a_question_without_its_grant_is_the_generic_line():
    built = build_payload(PUSH_KEY, event_type="question_pending", workspace_id=WS, now=NOW)
    message = _message(built)
    assert "question" not in built and "grant_id" not in built
    assert (message.body, message.category_id) == (QUESTION_LINE, OPEN_CATEGORY)


@pytest.mark.parametrize("question_md", ["", "   ", None, 7])
def test_a_question_with_no_text_is_the_generic_line(question_md):
    built = _question(["Yes"], question_md=question_md)
    assert "question" not in built and "options" not in built
    assert (_message(built).body, _message(built).category_id) == (QUESTION_LINE, OPEN_CATEGORY)


# --------------------------------------------------------------------------- a question's text and options


def test_a_question_carries_its_text_and_three_options():
    built = _question(["Yes", "No", "Later"])
    message = _message(built, agent_name="Ledger")
    assert built["question"] == QUESTION and built["options"] == ["Yes", "No", "Later"]
    assert built["grant_id"] == "42"
    assert message.body == QUESTION
    assert message.category_id == "automatos.question.3"
    assert (message.channel_id, message.priority) == ("questions", PRIORITY_HIGH)


def test_one_option_is_carried():
    built = _question(["Yes"])
    assert built["options"] == ["Yes"]
    assert _message(built).category_id == "automatos.question.1"


@pytest.mark.parametrize("options", [
    ["A", "B", "C", "D"],
    ["Yes", "x" * (PUSH_OPTION_MAX_CHARS + 1)],
    [],
    None,
    ["Yes", 2],
    ["Yes", "  "],
    "Yes",
])
def test_options_that_do_not_fit_are_never_carried_nor_cut(options):
    built = _question(options)
    assert built["question"] == QUESTION and "options" not in built
    assert _message(built).category_id == "automatos.question.0"


def test_an_option_of_exactly_80_characters_is_carried_whole():
    option = "y" * PUSH_OPTION_MAX_CHARS
    assert _question([option, "No"])["options"] == [option, "No"]


def test_a_question_of_500_characters_is_kept_whole():
    text = "q" * PUSH_QUESTION_MAX_CHARS
    assert _question(question_md=text)["question"] == text


def test_a_question_of_501_characters_ends_in_an_ellipsis():
    question = _question(question_md="q" * (PUSH_QUESTION_MAX_CHARS + 1))["question"]
    assert len(question) == PUSH_QUESTION_MAX_CHARS
    assert question.endswith(ELLIPSIS) and question[:-1] == "q" * (PUSH_QUESTION_MAX_CHARS - 1)


def test_a_question_is_sent_as_plain_text():
    markdown = "# Launch\n**Ship** the [Lisbon deck](https://example.com/deck)?\n\n\n- `now` or _later_"
    assert _question(question_md=markdown)["question"] == "Launch\nShip the Lisbon deck?\n\nnow or later"


@pytest.mark.parametrize("markdown, plain", [
    ("Rename user_id_map?", "Rename user_id_map?"),
    ("Edit __init__.py or _private?", "Edit __init__.py or _private?"),
    ("See [the page](https://en.wikipedia.org/wiki/Lisbon_(city)) first?", "See the page first?"),
    ("**Ship *both* decks** ~~today~~?", "Ship both decks today?"),
    ("Is 2 * 3 * 4 right?", "Is 2 * 3 * 4 right?"),
])
def test_marks_go_and_names_and_arithmetic_stay(markdown, plain):
    assert _question(question_md=markdown)["question"] == plain


def test_a_question_longer_than_the_scan_still_ends_in_an_ellipsis():
    markdown = "Ship it?" + " " * push_payload.QUESTION_SCAN_MAX_CHARS + "tail"
    assert _question(question_md=markdown)["question"] == "Ship it?" + ELLIPSIS
    assert _question(question_md="Ship it?" + " " * push_payload.QUESTION_SCAN_MAX_CHARS)["question"] == "Ship it?"


# --------------------------------------------------------------------------- redaction


def test_redaction_reads_the_workspace_push_setting_and_is_off_by_default():
    assert redact_questions_enabled({"push": {"redact_questions": True}}) is True
    for settings in ({}, None, {"push": {}}, {"push": {"redact_questions": False}}, {"push": "on"}, []):
        assert redact_questions_enabled(settings) is False


def test_a_redacted_question_is_generic_and_open_only():
    built = _question(["Yes", "No"], redact=True)
    message = _message(built)
    assert "question" not in built and "options" not in built
    assert built["grant_id"] == "42" and built["type"] == "needs_answer"
    assert (message.body, message.category_id) == (QUESTION_LINE, OPEN_CATEGORY)


# --------------------------------------------------------------------------- the AI badge (FR-9)


@pytest.mark.parametrize("locale, badge", [("en", "AI"), ("pt", "IA"), ("pt-BR", "IA"), ("de", "KI"),
                                           ("xx", "AI"), (None, "AI")])
def test_every_title_carries_the_badge_for_the_phones_locale(locale, badge):
    built = _question(["Yes"])
    assert _message(built, agent_name="Ledger", locale=locale).title == f"Ledger · {badge}"


@pytest.mark.parametrize("agent_name", [None, "", "   "])
def test_a_push_without_an_agent_is_from_auto(agent_name):
    built = build_payload(PUSH_KEY, event_type="auto_message", workspace_id=WS, now=NOW)
    assert _message(built, "auto_message", agent_name=agent_name).title == "Auto · AI"


# --------------------------------------------------------------------------- the payload's shape


def test_every_id_in_a_built_payload_is_a_string_and_nothing_is_null():
    built = build_payload(PUSH_KEY, event_type="question_pending", workspace_id=WS, notification_id=1001,
                          link_type="question", link_id=42, grant=_grant(options=["Yes"]),
                          ticket_number="#0419", now=NOW)
    for field in ("workspace_id", "notification_id", "grant_id", "link_id", "ticket_number"):
        assert isinstance(built[field], str), field
    assert built["workspace_id"] == str(WS)
    assert (built["v"], built["iat"]) == (2, int(NOW.timestamp()))
    assert None not in built.values()


def test_a_field_with_no_value_is_left_out():
    built = build_payload(PUSH_KEY, event_type="auto_message", workspace_id=WS, now=NOW)
    assert set(built) == {"v", "type", "workspace_id", "iat", "sig"}
    assert verify_signature(PUSH_KEY, built, NOW)


def test_signing_needs_a_push_key():
    with pytest.raises(ValueError):
        build_payload(None, event_type="auto_message", workspace_id=WS, now=NOW)
    with pytest.raises(ValueError):
        sign("", {"type": "auto_message"})
    assert verify_signature("", _question(), NOW) is False


# --------------------------------------------------------------------------- the shared vector file (K9)


def test_the_vector_file_is_k9s():
    vectors = _vectors()
    assert vectors["schema"] == "automatos-push-signature-vectors/v2"
    assert vectors["window"] == {"max_age_seconds": PUSH_SIG_MAX_AGE_SECONDS,
                                 "max_future_seconds": PUSH_SIG_MAX_FUTURE_SECONDS}
    assert (PUSH_SIG_MAX_AGE_SECONDS, PUSH_SIG_MAX_FUTURE_SECONDS) == (604800, 300)
    assert [case["name"] for case in vectors["cases"]] == [
        "question-with-options", "redacted-question", "review", "auto-message", "unicode-question"]
    assert len(vectors["tampered"]) == 4


@pytest.mark.parametrize("case", _vectors()["cases"], ids=lambda case: case["name"])
def test_every_case_recomputes_its_message_and_sig_and_verifies(case):
    key, payload = _vectors()["key"], case["payload"]
    assert canonical_message(payload).decode("utf-8") == case["message"]
    assert sign(key, payload) == case["sig"] == payload["sig"]
    assert verify_signature(key, payload, _at(payload["iat"])) is True


def _case_inputs(name: str) -> dict:
    question_grant = _grant("Ship the Lisbon deck?", ["Yes", "No", "Later"], 42)
    return {
        "question-with-options": dict(event_type="question_pending", notification_id=1001, link_type="question",
                                      link_id=42, grant=question_grant, ticket_number="0419"),
        "redacted-question": dict(event_type="question_pending", notification_id=1002, link_type="question",
                                  link_id=42, grant=question_grant, ticket_number="0419", redact=True),
        "review": dict(event_type="review_pending", notification_id=1003, link_type="task", link_id=8812,
                       ticket_number="0419"),
        "auto-message": dict(event_type="auto_message", notification_id=1004, link_type="chat",
                             link_id=uuid.UUID("0d9f2c51-7a43-4b8e-9c61-2e5f7a1b3c4d")),
        "unicode-question": dict(event_type="question_pending", notification_id=1005, link_type="question",
                                 link_id=7, grant=_grant('Publicar às 09:00 — café ☕?\nLinha "dois"',
                                                         ["Sim", "Não"], 7)),
    }[name]


@pytest.mark.parametrize("case", _vectors()["cases"], ids=lambda case: case["name"])
def test_the_builder_reproduces_every_case(case):
    vectors = _vectors()
    payload = case["payload"]
    built = build_payload(vectors["key"], workspace_id=WS, now=_at(payload["iat"]), **_case_inputs(case["name"]))
    assert built == payload


@pytest.mark.parametrize("case", _vectors()["tampered"], ids=lambda case: case["name"])
def test_no_tampered_case_verifies(case):
    payload = case["payload"]
    assert case["valid"] is False
    assert verify_signature(_vectors()["key"], payload, _at(payload["iat"])) is False


# --------------------------------------------------------------------------- what the signature covers


def _signed_question() -> dict:
    return build_payload(PUSH_KEY, event_type="question_pending", workspace_id=WS, notification_id=1001,
                         link_type="question", link_id=42, grant=_grant(options=["Yes", "No", "Later"]),
                         ticket_number="#0419", now=NOW)


@pytest.mark.parametrize("field", SIGNED_FIELDS)
def test_a_change_to_any_signed_field_fails(field):
    signed = _signed_question()
    assert verify_signature(PUSH_KEY, signed, NOW) is True
    assert verify_signature(PUSH_KEY, {**signed, field: signed[field] + "x"}, NOW) is False


@pytest.mark.parametrize("options", [["Yes", "No", "Never"], ["Later", "No", "Yes"], ["Yes", "No"], []])
def test_a_change_to_an_option_or_their_order_fails(options):
    assert verify_signature(PUSH_KEY, {**_signed_question(), "options": options}, NOW) is False


def test_a_change_to_iat_fails():
    signed = _signed_question()
    assert verify_signature(PUSH_KEY, {**signed, "iat": signed["iat"] + 1}, NOW) is False


def test_a_removed_field_or_another_key_fails():
    signed = _signed_question()
    without = {key: value for key, value in signed.items() if key != "ticket_number"}
    assert verify_signature(PUSH_KEY, without, NOW) is False
    assert verify_signature("pk-test2", signed, NOW) is False


@pytest.mark.parametrize("version", [1, None, "2", True])
def test_any_version_but_2_fails(version):
    signed = {**_signed_question(), "v": version}
    assert verify_signature(PUSH_KEY, signed, NOW) is False


@pytest.mark.parametrize("sig", [None, 42, "", "zz", "é" * 64])
def test_a_missing_or_odd_sig_fails_without_raising(sig):
    assert verify_signature(PUSH_KEY, {**_signed_question(), "sig": sig}, NOW) is False


@pytest.mark.parametrize("field, value", [
    ("options", 5), ("options", "Yes"), ("options", {"Yes"}), ("question", "\ud800"), ("link_id", object()),
])
def test_a_payload_that_cannot_be_canonical_fails_without_raising(field, value):
    assert verify_signature(PUSH_KEY, {**_signed_question(), field: value}, NOW) is False


@pytest.mark.parametrize("seconds_later, valid", [
    (PUSH_SIG_MAX_AGE_SECONDS, True),
    (PUSH_SIG_MAX_AGE_SECONDS + 1, False),
    (-PUSH_SIG_MAX_FUTURE_SECONDS, True),
    (-PUSH_SIG_MAX_FUTURE_SECONDS - 1, False),
])
def test_iat_is_fresh_for_seven_days_and_300_seconds_ahead(seconds_later, valid):
    signed = _signed_question()
    assert verify_signature(PUSH_KEY, signed, NOW + timedelta(seconds=seconds_later)) is valid


def test_an_iat_that_is_not_an_integer_fails():
    signed = _signed_question()
    for iat in (str(signed["iat"]), float(signed["iat"]), None, True):
        assert verify_signature(PUSH_KEY, {**signed, "iat": iat}, NOW) is False
