"""PRD-220: persistent & multi-threaded widget chat — backend seams.

Covers the three backend changes:

1. Route order — ``GET /api/chat/search`` must be registered BEFORE
   ``GET /api/chat/{chat_id}``. Routes match in declaration order, so the old
   placement resolved ``/api/chat/search`` as ``chat_id="search"`` and 404'd.
2. page-context injection — the widget's page hint is added prompt-side only,
   AFTER the clean save, by rebuilding (never mutating) the history entries.
   (PRD-221 S2 moved this to ``services.page_context``; invariants preserved.)
3. ``_last_message_previews`` — thread-list previews for ``GET /api/chat/history``.

Pure unit tests — no DB / network (db is mocked), matching
``test_p2w0_chat_identity.py``.
"""
from unittest.mock import MagicMock

import pytest


def _api_chat():
    try:
        import api.chat as chat_module
    except Exception as e:  # env without the heavy router deps
        pytest.skip(f"api.chat not importable in this env: {e}")
    return chat_module


# ---------------------------------------------------------------------------
# 1. Route order — /search before /{chat_id}
# ---------------------------------------------------------------------------

def test_search_route_registered_before_chat_id_param_route():
    chat_module = _api_chat()
    get_paths = [
        r.path
        for r in chat_module.router.routes
        if "GET" in (getattr(r, "methods", None) or set())
    ]
    assert "/api/chat/search" in get_paths, get_paths
    assert "/api/chat/{chat_id}" in get_paths, get_paths
    assert get_paths.index("/api/chat/search") < get_paths.index("/api/chat/{chat_id}"), (
        "GET /search must be declared before GET /{chat_id} or it resolves as "
        f"chat_id='search'. Current GET order: {get_paths}"
    )


# ---------------------------------------------------------------------------
# 2. page-context injection — PRD-221 S2 replaced the old ``_inject_page_context``
#    helper with the ``services.page_context`` pipeline (sanitize → render →
#    inject). The invariants below (last-user-only, rebuild-never-mutate, noop
#    on empty, legacy-label truncation) are preserved through a legacy
#    ``{"page": <label>}`` context, which renders via the same one renderer.
#    Structured-context behaviour is covered in test_prd221_page_context.py.
# ---------------------------------------------------------------------------

def _inject(history, page_label):
    from services.page_context import inject_page_preamble
    return inject_page_preamble(history, {"page": page_label} if page_label else {})


def test_page_context_appended_to_last_user_message_only():
    history = [
        {"role": "user", "parts": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "reply"}]},
        {"role": "user", "parts": [{"type": "text", "text": "second"}]},
    ]
    out = _inject(history, "Dashboard")

    assert len(out) == 3
    # Earlier messages untouched (same objects — no rebuild needed for them).
    assert out[0] is history[0]
    assert out[1] is history[1]
    # Last user message got exactly one extra part carrying the hint.
    assert len(out[2]["parts"]) == 2
    assert out[2]["parts"][0] == {"type": "text", "text": "second"}
    assert "Dashboard" in out[2]["parts"][1]["text"]
    assert out[2]["parts"][1]["type"] == "text"


def test_page_context_never_mutates_inputs():
    """The parts list is the ORM row's JSONB — in-place edits could get flushed."""
    original_parts = [{"type": "text", "text": "hello"}]
    history = [{"role": "user", "parts": original_parts}]

    out = _inject(history, "Agent Management")

    assert original_parts == [{"type": "text", "text": "hello"}]
    assert history[0]["parts"] is original_parts
    assert out[0] is not history[0]
    assert len(out[0]["parts"]) == 2


def test_page_context_noop_when_absent_blank_or_no_user_message():
    history = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]

    assert _inject(history, None) is history
    assert _inject(history, "") is history
    assert _inject(history, "   ") is history

    assistant_only = [{"role": "assistant", "parts": [{"type": "text", "text": "yo"}]}]
    assert _inject(assistant_only, "Dashboard") is assistant_only


def test_page_context_label_is_truncated():
    from services.page_context import _LEGACY_LABEL_MAX_LEN
    history = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
    out = _inject(history, "x" * 500)
    hint_text = out[0]["parts"][1]["text"]
    assert "x" * _LEGACY_LABEL_MAX_LEN in hint_text
    assert "x" * (_LEGACY_LABEL_MAX_LEN + 1) not in hint_text


# ---------------------------------------------------------------------------
# 3. _last_message_previews
# ---------------------------------------------------------------------------

def test_previews_empty_ids_short_circuits_without_query():
    chat_module = _api_chat()
    db = MagicMock()
    assert chat_module._last_message_previews(db, []) == {}
    db.execute.assert_not_called()


def test_previews_maps_and_truncates():
    chat_module = _api_chat()
    long_text = "a" * 200
    row_long = MagicMock()
    row_long.chat_id = "11111111-1111-1111-1111-111111111111"
    row_long.parts = [{"type": "text", "text": long_text}]
    row_multi = MagicMock()
    row_multi.chat_id = "22222222-2222-2222-2222-222222222222"
    row_multi.parts = [
        {"type": "text", "text": "I've created"},
        {"type": "tool", "text": None},
        {"type": "text", "text": "the playbook"},
    ]
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [row_long, row_multi]

    previews = chat_module._last_message_previews(db, [str(row_long.chat_id), str(row_multi.chat_id)])

    assert previews[str(row_multi.chat_id)] == "I've created the playbook"
    truncated = previews[str(row_long.chat_id)]
    assert len(truncated) == chat_module._PREVIEW_MAX_CHARS
    assert truncated.endswith("…")


def test_previews_tolerates_non_list_parts():
    chat_module = _api_chat()
    row = MagicMock()
    row.chat_id = "33333333-3333-3333-3333-333333333333"
    row.parts = None
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [row]

    previews = chat_module._last_message_previews(db, [str(row.chat_id)])
    assert previews[str(row.chat_id)] == ""
