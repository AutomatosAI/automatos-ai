"""PRD-206 S3 — the resume payload ("where did we leave off?").

Shape, recency ordering, the Q7 private-scope visibility rule, next-step
lifting from S2 checkpoints, and the LLM-facing rendering. Pure/mocked —
no DB, no vector store.
"""
import os
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

from modules.memory.injection_filter import visible_to_viewer  # noqa: E402
from modules.memory.resume_context import (  # noqa: E402
    assemble_resume_payload,
    format_resume_for_llm,
)


def _mem(mid, text, mtype, created_at, scope=None, owner=None, chat_id=None):
    meta = {"type": mtype, "category": mtype, "importance": 0.6}
    if scope:
        meta["scope"] = scope
    if owner:
        meta["owner"] = owner
    if chat_id:
        meta["chat_id"] = chat_id
    return {"id": mid, "memory": text, "created_at": created_at, "metadata": meta}


THREADS = [
    {
        "chat_id": "c-2",
        "title": "Academy tutor",
        "updated_at": "2026-07-17T10:00:00+00:00",
        "summary": {
            "topic": "Academy tutor flow",
            "last_summary": "Fixed CORS, tutor next.",
            "next_step": "Re-test the tutor end to end.",
        },
    },
    {
        "chat_id": "c-1",
        "title": "Memory PRD",
        "updated_at": "2026-07-16T22:00:00+00:00",
        "summary": {"topic": "PRD-206", "last_summary": "Phase 1 scoped.", "next_step": None},
    },
    {"chat_id": "c-0", "title": "No checkpoint yet", "updated_at": None, "summary": None},
]

MEMORIES = [
    _mem("m1", "Decided to use per-key CORS allowlists.", "decision",
         "2026-07-17T09:00:00+00:00", scope="workspace", chat_id="c-2"),
    _mem("m2", "Old decision about pilots.", "decision", "2026-07-10T09:00:00+00:00"),
    _mem("m3", "Refresh the tutor corpus.", "open_loop",
         "2026-07-17T09:30:00+00:00", scope="workspace"),
    _mem("m4", "Gerard prefers dark themes.", "preference",
         "2026-07-17T08:00:00+00:00", scope="private", owner="user:7"),
    _mem("m5", "Private note of another user.", "decision",
         "2026-07-17T09:45:00+00:00", scope="private", owner="user:99"),
]


def test_resume_payload_shape():
    payload = assemble_resume_payload(THREADS, MEMORIES, viewer="user:7")
    assert set(payload) == {
        "threads", "recent_decisions", "open_loops",
        "suggested_next_steps", "projects",
    }
    assert payload["threads"] == THREADS
    assert payload["projects"] == []          # S4 (Phase 2) fills this
    assert [l["text"] for l in payload["open_loops"]] == ["Refresh the tutor corpus."]
    # Next steps lifted from checkpoints, nulls skipped, thread order kept.
    assert payload["suggested_next_steps"] == ["Re-test the tutor end to end."]


def test_resume_orders_decisions_by_recency():
    payload = assemble_resume_payload(THREADS, MEMORIES, viewer="user:7")
    texts = [d["text"] for d in payload["recent_decisions"]]
    # m5 is another user's private decision — invisible; newest-first order.
    assert texts == [
        "Decided to use per-key CORS allowlists.",
        "Old decision about pilots.",
    ]
    # Items carry the thread link for deep-linking.
    assert payload["recent_decisions"][0]["chat_id"] == "c-2"


def test_resume_respects_private_scope():
    as_owner = assemble_resume_payload([], MEMORIES, viewer="user:99")
    assert "Private note of another user." in [d["text"] for d in as_owner["recent_decisions"]]

    as_other = assemble_resume_payload([], MEMORIES, viewer="user:7")
    assert "Private note of another user." not in [d["text"] for d in as_other["recent_decisions"]]

    headless = assemble_resume_payload([], MEMORIES, viewer=None)
    assert "Private note of another user." not in [d["text"] for d in headless["recent_decisions"]]
    # Workspace/legacy rows still resume for a headless caller.
    assert len(headless["recent_decisions"]) == 2


def test_resume_caps_items():
    many = [
        _mem(f"d{i}", f"decision {i}", "decision", f"2026-07-{10 + (i % 7):02d}T00:00:00")
        for i in range(20)
    ]
    payload = assemble_resume_payload([], many, viewer=None, limit_items=3)
    assert len(payload["recent_decisions"]) == 3


def test_format_resume_for_llm_sections_and_empty():
    payload = assemble_resume_payload(THREADS, MEMORIES, viewer="user:7")
    text = format_resume_for_llm(payload)
    assert "Recent threads:" in text
    assert "Recent decisions:" in text
    assert "Open loops:" in text
    assert "Suggested next steps:" in text
    assert "Academy tutor" in text

    empty = format_resume_for_llm(assemble_resume_payload([], [], viewer=None))
    assert "No resume context yet" in empty


# ---------------------------------------------------------------------------
# visible_to_viewer — the one read-side scope rule (S3 + S7 share it)
# ---------------------------------------------------------------------------

def test_visible_to_viewer_rules():
    legacy = {"memory": "x", "metadata": {"type": "user_fact"}}
    workspace = {"memory": "x", "metadata": {"scope": "workspace"}}
    private_mine = {"memory": "x", "metadata": {"scope": "private", "owner": "user:7"}}
    private_theirs = {"memory": "x", "metadata": {"scope": "private", "owner": "user:9"}}
    private_ownerless = {"memory": "x", "metadata": {"scope": "private"}}

    assert visible_to_viewer(legacy, "user:7")
    assert visible_to_viewer(legacy, None)
    assert visible_to_viewer(workspace, None)
    assert visible_to_viewer(private_mine, "user:7")
    assert not visible_to_viewer(private_theirs, "user:7")
    assert not visible_to_viewer(private_mine, None)          # unknown viewer fails closed
    assert not visible_to_viewer(private_ownerless, "user:7") # ownerless private: no one


# ---------------------------------------------------------------------------
# The tool handler end-to-end (service mocked)
# ---------------------------------------------------------------------------

class _FakeService:
    is_durable_configured = True

    async def get_all_memories(self, workspace_id, limit=200):
        return MEMORIES


@pytest.mark.asyncio
async def test_resume_tool_handler_formats_payload(monkeypatch):
    import types as _types

    sys.modules.setdefault("camelot", _types.ModuleType("camelot"))
    import modules.memory.unified_memory_service as ums
    from modules.tools.discovery.handlers_workspace import resume_context

    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: _FakeService())

    # Headless caller: no _user_id injected → no personal threads, workspace
    # decisions/open loops still present, formatted string ready for the LLM.
    result = await resume_context(None, "00000000-0000-0000-0000-000000000001", {})
    assert result["success"] is True
    assert result["threads"] == []
    assert len(result["recent_decisions"]) == 2
    assert "Recent decisions:" in result["formatted"]
