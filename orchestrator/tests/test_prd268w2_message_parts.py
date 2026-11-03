"""PRD-268 W2 US-C19b (§6.11.2, K4): a background post carries structured parts beside its text.

``post_background_message`` (``services/chat_messenger.py``) saves ``[{type: 'text', text},
*parts]``; ``services/chat_parts.py`` accepts only ``brief@1`` and ``suggestion@1``, each at
most ``CHAT_PART_MAX_BYTES`` of JSON, and refuses anything else before a read or a write.
Proven here:

* the parts module: its two types, a copy of each part, and a refusal naming the type for an
  unknown type, a part that is not a mapping, not JSON or too big, and a bare part;
* the messenger keeps its name and keyword arguments and gains ``parts``; the fail-soft
  ``deliver_background_message`` passes them through unchanged;
* on Postgres: a brief part is saved after the text, whose words are the ones given; a post
  without parts is today's single text part; an unknown or oversized part saves nothing (no
  message, no Auto thread); the owner check and the Auto-thread fallback are unchanged
  (another member's chat_id falls back to the caller's Auto thread).
"""
from __future__ import annotations

import inspect
import uuid

import pytest
from sqlalchemy import create_engine, text

from services.chat_parts import CHAT_PART_MAX_BYTES

PROSE = "Good morning. Two things need you today, and the Lisbon flights are ready to book."
BRIEF = {
    "type": "brief@1",
    "brief": {
        "local_date": "2026-10-29",
        "greeting": "Good morning, Gerard",
        "sections": [{"key": "needs_you", "rows": [{"title": "Approve the Lisbon flights"}]}],
        "counts": {"needs_you": 2},
    },
}
SUGGESTION = {"type": "suggestion@1", "suggestion_id": str(uuid.UUID(int=7))}


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the message-parts tests need Postgres with the PRD-205 schema: {exc}")
    yield eng
    eng.dispose()


def _person(s, tag):
    clerk = f"user_prd268w2mp_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return clerk, int(pk)


def _chat(s, ws, user_int_id):
    chat_id = str(uuid.uuid4())
    s.execute(text(
        "INSERT INTO chats (id, user_id, workspace_id, title, visibility) "
        "VALUES (CAST(:c AS uuid), :u, CAST(:w AS uuid), 'ordinary', 'private')"),
        {"c": chat_id, "u": user_int_id, "w": ws})
    return chat_id


@pytest.fixture
def people(engine, new_session):
    """A workspace, the caller and another member, each with an ordinary chat of their own:
    ``(workspace id, caller, other)``, each person ``(clerk id, users.id, chat id)``."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd268w2-message-parts')"),
              {"id": ws})
    caller, other = _person(s, "caller"), _person(s, "other")
    caller, other = (*caller, _chat(s, ws, caller[1])), (*other, _chat(s, ws, other[1]))
    s.commit()
    yield ws, caller, other
    s = new_session.sweep()
    for stmt in ("DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)",
                 "DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)",
                 "DELETE FROM users WHERE clerk_user_id IN (:a, :b)",
                 "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"):
        s.execute(text(stmt), {"w": ws, "a": caller[0], "b": other[0]})
    s.commit()


def _post(new_session, ws, person, **kwargs):
    """One post through ``post_background_message`` on a committing session."""
    from services.chat_messenger import post_background_message

    s = new_session()
    message = post_background_message(
        s, workspace_id=ws, text=kwargs.pop("words", PROSE), source={"origin": "daily_digest"},
        clerk_user_id=person[0], **kwargs)
    s.commit()
    return message


def _saved(new_session, ws):
    """Every message in the workspace as ``(chat id, chat kind, chat owner, parts)``, oldest first."""
    return [tuple(row) for row in new_session().execute(text(
        "SELECT CAST(m.chat_id AS text), c.kind, c.user_id, m.parts FROM messages m "
        "JOIN chats c ON c.id = m.chat_id WHERE m.workspace_id = CAST(:w AS uuid) "
        "ORDER BY m.created_at"), {"w": ws}).all()]


def _auto_threads(new_session, ws):
    return new_session().execute(text(
        "SELECT count(*) FROM chats WHERE workspace_id = CAST(:w AS uuid) AND kind = 'auto'"),
        {"w": ws}).scalar()


# --------------------------------------------------------------------------- the parts module


def test_the_structured_part_types_are_the_brief_and_the_suggestion():
    from services.chat_parts import STRUCTURED_PART_TYPES

    assert STRUCTURED_PART_TYPES == ("brief@1", "suggestion@1", "calibration@1")  # 2026-10-31: W3's US-C22b
    assert isinstance(CHAT_PART_MAX_BYTES, int) and CHAT_PART_MAX_BYTES > 0


def test_checked_parts_gives_back_a_copy_of_each_part_and_none_for_none():
    from services.chat_parts import checked_parts

    checked = checked_parts([BRIEF, SUGGESTION])

    assert checked == [BRIEF, SUGGESTION] and checked_parts(None) == [] and checked_parts(()) == []
    assert checked[0] is not BRIEF and checked[0]["brief"] is not BRIEF["brief"]


@pytest.mark.parametrize("part, named", [
    ({"type": "text", "text": "a second text part"}, "'text'"),
    ({"type": "card@9", "card": {}}, "'card@9'"),
    ({"brief": BRIEF["brief"]}, "'None'"),
    ("brief@1", "'str'"),
    (["brief@1"], "'list'"),
])
def test_checked_parts_refuses_a_part_of_any_other_type_naming_it(part, named):
    from services.chat_parts import checked_parts

    with pytest.raises(ValueError, match=named):
        checked_parts([BRIEF, part])


def test_checked_parts_refuses_a_part_over_the_byte_limit_and_takes_one_at_it():
    import json

    from services.chat_parts import checked_parts

    empty = len(json.dumps({"type": "brief@1", "brief": ""}, separators=(",", ":")))
    at_limit = {"type": "brief@1", "brief": "a" * (CHAT_PART_MAX_BYTES - empty)}

    assert checked_parts([at_limit]) == [at_limit]
    with pytest.raises(ValueError, match="'brief@1'.*over"):
        checked_parts([{**at_limit, "brief": at_limit["brief"] + "a"}])
    with pytest.raises(ValueError, match="'suggestion@1'.*over"):
        checked_parts([{"type": "suggestion@1", "why": "é" * CHAT_PART_MAX_BYTES}])


@pytest.mark.parametrize("part", [
    {"type": "brief@1", "brief": float("nan")},
    {"type": "suggestion@1", "at": object()},
    {"type": "suggestion@1", "why": "half a pair \ud800"},
])
def test_checked_parts_refuses_a_part_that_is_not_json(part):
    from services.chat_parts import checked_parts

    with pytest.raises(ValueError, match=f"'{part['type']}' is not JSON"):
        checked_parts([part])


def test_checked_parts_refuses_a_part_too_deep_to_encode_as_a_value_error(monkeypatch):
    import services.chat_parts as chat_parts

    def too_deep(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded while encoding a JSON object")

    monkeypatch.setattr(chat_parts.json, "dumps", too_deep)

    with pytest.raises(ValueError, match="'brief@1' is not JSON"):
        chat_parts.checked_parts([BRIEF])


@pytest.mark.parametrize("part", [
    {"type": "suggestion@1", "why": "a\x00b"},
    {"type": "brief@1", "brief": {"rows\x00": []}},
])
def test_checked_parts_refuses_a_nul_character_postgres_would_refuse(part):
    from services.chat_parts import checked_parts

    with pytest.raises(ValueError, match=f"'{part['type']}' holds a NUL"):
        checked_parts([part])


def test_checked_parts_keeps_the_words_backslash_u0000_which_are_not_a_nul():
    from services.chat_parts import checked_parts

    words = {"type": "suggestion@1", "why": r"the path C:\u0000 and C:\\u0000"}

    assert checked_parts([words]) == [words]


def test_a_refused_type_is_quoted_at_most_type_in_error_max_chars():
    from services.chat_parts import TYPE_IN_ERROR_MAX_CHARS, checked_parts

    with pytest.raises(ValueError) as refused:
        checked_parts([{"type": "x" * (TYPE_IN_ERROR_MAX_CHARS * 10)}])

    assert "x" * TYPE_IN_ERROR_MAX_CHARS in str(refused.value)
    assert "x" * (TYPE_IN_ERROR_MAX_CHARS + 1) not in str(refused.value)


@pytest.mark.parametrize("parts", [BRIEF, "brief@1", b"brief@1"])
def test_checked_parts_refuses_a_bare_part_or_a_string_for_the_list(parts):
    from services.chat_parts import checked_parts

    with pytest.raises(ValueError, match="list of parts"):
        checked_parts(parts)


# --------------------------------------------------------------------------- the messenger's seam


def test_post_background_message_keeps_its_keywords_and_gains_parts():
    from services.chat_messenger import post_background_message

    params = inspect.signature(post_background_message).parameters

    assert list(params) == ["db", "workspace_id", "text", "source", "chat_id", "clerk_user_id",
                            "link_type", "link_id", "parts"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for name, p in params.items() if name != "db")
    assert params["parts"].default is None


def test_deliver_background_message_passes_parts_through_unchanged(monkeypatch):
    from services import chat_messenger

    seen = []
    monkeypatch.setattr(chat_messenger, "post_background_message", lambda db, **kw: seen.append(kw))
    parts = [SUGGESTION]

    chat_messenger.deliver_background_message(
        object(), workspace_id=str(uuid.uuid4()), text=PROSE, source={"origin": "suggestion"},
        clerk_user_id="user_x", parts=parts)

    assert seen[0]["parts"] is parts


def test_a_refused_part_raises_before_the_messenger_reads_anything(monkeypatch):
    from services import chat_messenger

    def no_reads(*args, **kwargs):
        raise AssertionError("the messenger read before checking its parts")

    monkeypatch.setattr(chat_messenger, "_resolve_user_int_id", no_reads)
    monkeypatch.setattr(chat_messenger, "find_or_create_auto_chat", no_reads)

    with pytest.raises(ValueError, match="'card@9'"):
        chat_messenger.post_background_message(
            object(), workspace_id=str(uuid.uuid4()), text=PROSE, source={"origin": "suggestion"},
            clerk_user_id="user_x", parts=[{"type": "card@9"}])


# --------------------------------------------------------------------------- on Postgres


def test_a_post_with_a_brief_part_saves_the_text_then_the_brief(people, new_session):
    ws, caller, _ = people

    message = _post(new_session, ws, caller, parts=[BRIEF])

    [(chat_id, kind, owner, parts)] = _saved(new_session, ws)
    assert parts == [{"type": "text", "text": PROSE}, BRIEF] == message.parts
    assert parts[0]["text"] == PROSE
    assert (chat_id, kind, owner) == (str(message.chat_id), "auto", caller[1])


def test_deliver_saves_the_parts_it_is_given(people, new_session):
    from services.chat_messenger import deliver_background_message

    ws, caller, _ = people
    s = new_session()
    message = deliver_background_message(
        s, workspace_id=ws, text="Auto has a suggestion: book the Lisbon flights.",
        source={"origin": "suggestion", "suggestion_id": SUGGESTION["suggestion_id"]},
        clerk_user_id=caller[0], parts=[SUGGESTION])
    s.commit()

    assert message is not None
    assert [row[3] for row in _saved(new_session, ws)] == [
        [{"type": "text", "text": "Auto has a suggestion: book the Lisbon flights."}, SUGGESTION]]


def test_a_post_without_parts_is_todays_single_text_part(people, new_session):
    ws, caller, _ = people

    _post(new_session, ws, caller)

    assert [row[3] for row in _saved(new_session, ws)] == [[{"type": "text", "text": PROSE}]]


def test_an_empty_text_posts_nothing_even_with_a_brief(people, new_session):
    ws, caller, _ = people

    assert _post(new_session, ws, caller, words="   ", parts=[BRIEF]) is None
    assert _saved(new_session, ws) == [] and _auto_threads(new_session, ws) == 0


@pytest.mark.parametrize("part, named", [
    ({"type": "card@9", "card": {}}, "'card@9'"),
    ({"type": "brief@1", "brief": "a" * (CHAT_PART_MAX_BYTES + 1)}, "'brief@1'.*over"),
])
def test_a_refused_part_saves_nothing(people, new_session, part, named):
    ws, caller, _ = people

    with pytest.raises(ValueError, match=named):
        _post(new_session, ws, caller, parts=[BRIEF, part])

    assert _saved(new_session, ws) == [] and _auto_threads(new_session, ws) == 0


def test_deliver_refuses_a_bad_part_fail_soft_and_saves_nothing(people, new_session):
    from services.chat_messenger import deliver_background_message

    ws, caller, _ = people

    assert deliver_background_message(
        new_session(), workspace_id=ws, text=PROSE, source={"origin": "suggestion"},
        clerk_user_id=caller[0], parts=[{"type": "card@9"}]) is None
    assert _saved(new_session, ws) == []


def test_another_members_chat_falls_back_to_the_callers_auto_thread(people, new_session):
    ws, caller, other = people

    message = _post(new_session, ws, caller, chat_id=other[2], parts=[BRIEF])

    [(chat_id, kind, owner, parts)] = _saved(new_session, ws)
    assert chat_id != other[2] and chat_id == str(message.chat_id)
    assert (kind, owner) == ("auto", caller[1]) and parts[1] == BRIEF


def test_the_callers_own_chat_is_honoured(people, new_session):
    ws, caller, _ = people

    _post(new_session, ws, caller, chat_id=caller[2], parts=[BRIEF])

    [(chat_id, kind, owner, _parts)] = _saved(new_session, ws)
    assert (chat_id, owner) == (caller[2], caller[1]) and kind != "auto"
    assert _auto_threads(new_session, ws) == 0


def test_a_chat_id_outside_the_workspace_falls_back_to_the_auto_thread(people, new_session):
    ws, caller, _ = people

    _post(new_session, ws, caller, chat_id=str(uuid.uuid4()), parts=[BRIEF])

    [(_chat_id, kind, owner, _parts)] = _saved(new_session, ws)
    assert (kind, owner) == ("auto", caller[1])


def test_no_valid_chat_and_no_resolvable_person_drops_the_post(people, new_session):
    ws, _, _ = people

    assert _post(new_session, ws, ("user_does_not_exist",), parts=[BRIEF]) is None
    assert _saved(new_session, ws) == [] and _auto_threads(new_session, ws) == 0
