"""PRD-206 S2 — thread checkpoints.

The checkpoint distill writes ``chats.summary`` and stores NEW decisions /
open loops / a changed thread summary as typed L3 memories through the S1
write contract. Re-checkpointing an unchanged thread updates the summary
watermark but plans no duplicate memories. The Q3 exclusion validator
applies to every planned item.

Pure/mocked — fake LLM, fake unified service, namespace fakes for the ORM
rows. No DB, no network.
"""
import os
import sys
import types
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

import core.llm as core_llm  # noqa: E402

from modules.memory import thread_checkpoint as tc  # noqa: E402
from modules.memory.write_contract import MEMORY_SCOPE_WORKSPACE  # noqa: E402


CHECKPOINT_JSON = (
    '{"topic": "Academy tutor flow", '
    '"last_summary": "We tightened the tutor CORS flow and picked the env allowlist fix.", '
    '"next_step": "Merge the preflight PR and re-test the tutor.", '
    '"decisions": ["Use per-key allowed_domains for widget CORS preflights."], '
    '"open_questions": ["Refresh the tutor corpus after the CORS fix lands."]}'
)


# ---------------------------------------------------------------------------
# parse_checkpoint
# ---------------------------------------------------------------------------

def test_parse_clean_json():
    parsed = tc.parse_checkpoint(CHECKPOINT_JSON)
    assert parsed["topic"] == "Academy tutor flow"
    assert parsed["decisions"] == ["Use per-key allowed_domains for widget CORS preflights."]
    assert parsed["open_questions"] == ["Refresh the tutor corpus after the CORS fix lands."]
    assert parsed["next_step"].startswith("Merge the preflight PR")


def test_parse_tolerates_fences_and_prose():
    fenced = f"Here is the checkpoint:\n```json\n{CHECKPOINT_JSON}\n```\nDone."
    parsed = tc.parse_checkpoint(fenced)
    assert parsed is not None and parsed["topic"] == "Academy tutor flow"


def test_parse_coerces_item_shapes_and_null_next_step():
    parsed = tc.parse_checkpoint(
        '{"topic": "t", "last_summary": "s", "next_step": null, '
        '"decisions": [{"text": "decided X"}, 42, ""], '
        '"open_questions": "not-a-list"}'
    )
    assert parsed["decisions"] == ["decided X"]
    assert parsed["open_questions"] == []
    assert parsed["next_step"] is None


@pytest.mark.parametrize("junk", ["", "no json here", "[1,2,3]", None])
def test_parse_rejects_garbage(junk):
    assert tc.parse_checkpoint(junk) is None


# ---------------------------------------------------------------------------
# prompt + transcript
# ---------------------------------------------------------------------------

def test_prompt_carries_prior_for_idempotence():
    prior = {"topic": "t", "last_summary": "old", "next_step": None,
             "decisions": ["d1"], "open_questions": []}
    prompt = tc.build_checkpoint_prompt("User: hi", prior)
    assert "previous checkpoint" in prompt
    assert "d1" in prompt
    assert "NEVER include secrets" in prompt
    # No prior → no carry-forward block.
    assert "previous checkpoint" not in tc.build_checkpoint_prompt("User: hi", None)


def test_render_transcript_extracts_aisdk_parts():
    messages = [
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "hi there"},
                                        {"type": "tool-call", "name": "x"}]},
        {"role": "assistant", "parts": []},  # no text → skipped
    ]
    out = tc.render_transcript(messages)
    assert "User: hello" in out and "Auto: hi there" in out
    assert "tool-call" not in out


# ---------------------------------------------------------------------------
# compose + plan (idempotence + exclusions)
# ---------------------------------------------------------------------------

def _parsed():
    return tc.parse_checkpoint(CHECKPOINT_JSON)


def test_compose_summary_shape():
    summary = tc.compose_summary(_parsed(), trigger="on_demand")
    assert {"topic", "last_summary", "next_step", "decisions", "open_questions",
            "updated_at", "checkpointed_at", "trigger"} <= set(summary)
    assert isinstance(summary["checkpointed_at"], float)
    assert summary["trigger"] == "on_demand"


def test_plan_writes_checkpoint_memories_with_contract_metadata():
    planned = tc.plan_typed_memories(
        _parsed(), None,
        owner="user:7", chat_id="chat-1", workspace_id="ws-1", trigger="idle_sweep",
    )
    by_type = {p["metadata"]["type"]: p for p in planned}
    assert set(by_type) == {"decision", "open_loop", "thread_summary"}
    for p in planned:
        meta = p["metadata"]
        assert meta["scope"] == MEMORY_SCOPE_WORKSPACE      # Q7: workspace objects
        assert meta["owner"] == "user:7"
        assert meta["chat_id"] == "chat-1"
        assert meta["source"] == "thread_checkpoint"
        assert p["subject_id"] == "user:7"
    assert "where we left off" in by_type["thread_summary"]["content"]


def test_plan_is_idempotent_against_prior():
    parsed = _parsed()
    prior = tc.compose_summary(parsed, trigger="idle_sweep")
    planned = tc.plan_typed_memories(
        parsed, prior,
        owner="user:7", chat_id="chat-1", workspace_id="ws-1", trigger="idle_sweep",
    )
    assert planned == []          # nothing new → nothing stored


def test_plan_diffs_only_new_items():
    parsed = _parsed()
    prior = tc.compose_summary(parsed, trigger="idle_sweep")
    changed = {
        **parsed,
        "decisions": parsed["decisions"] + ["Ship phase 1 without the panel."],
        "last_summary": parsed["last_summary"],   # unchanged → no new thread_summary
    }
    planned = tc.plan_typed_memories(
        changed, prior,
        owner="user:7", chat_id="chat-1", workspace_id="ws-1", trigger="idle_sweep",
    )
    assert [p["metadata"]["type"] for p in planned] == ["decision"]
    assert planned[0]["content"] == "Ship phase 1 without the panel."


def test_plan_applies_exclusion_validator():
    parsed = {
        "topic": "t", "last_summary": "", "next_step": None,
        "decisions": ["the db password is hunter2"],
        "open_questions": [],
    }
    planned = tc.plan_typed_memories(
        parsed, None, owner=None, chat_id="c", workspace_id="w", trigger="on_demand",
    )
    assert planned == []


# ---------------------------------------------------------------------------
# run_thread_checkpoint (I/O seams monkeypatched)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    async def generate_response(self, messages, tools=None):
        return _FakeResp(self._content)


class _FakeService:
    def __init__(self):
        self.calls = []

    async def store_two_tier(self, **kwargs):
        self.calls.append(kwargs)
        return [("global", {"success": True})]


class _FakeDb:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _patch_io(monkeypatch, chat, messages, llm_content):
    monkeypatch.setattr(tc, "_load_chat", lambda db, ws, cid: chat)
    monkeypatch.setattr(tc, "_load_messages", lambda db, cid: messages)
    monkeypatch.setattr(core_llm, "create_llm_manager", lambda **kw: _FakeLLM(llm_content))
    fake_service = _FakeService()
    import modules.memory.unified_memory_service as ums
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: fake_service)
    return fake_service


_MESSAGES = [
    {"role": "user", "parts": [{"type": "text", "text": f"message {i}"}]}
    for i in range(6)
]


@pytest.mark.asyncio
async def test_checkpoint_writes_chat_summary_and_typed_memories(monkeypatch):
    chat = types.SimpleNamespace(id="chat-1", user_id=7, summary=None)
    db = _FakeDb()
    service = _patch_io(monkeypatch, chat, _MESSAGES, CHECKPOINT_JSON)

    result = await tc.run_thread_checkpoint(
        db, workspace_id="ws-1", chat_id="chat-1", trigger="idle_sweep",
    )

    assert result["success"] is True
    assert db.commits == 1
    assert chat.summary["topic"] == "Academy tutor flow"
    assert chat.summary["trigger"] == "idle_sweep"
    stored_types = {c["metadata"]["type"] for c in service.calls}
    assert stored_types == {"decision", "open_loop", "thread_summary"}
    for call in service.calls:
        assert call["subject_id"] == "user:7"
        assert call["metadata"]["chat_id"] == "chat-1"
    assert result["stored_memories"] == 3


@pytest.mark.asyncio
async def test_checkpoint_idempotent_updates(monkeypatch):
    """Re-checkpointing with an unchanged distill updates the summary
    watermark but stores NO duplicate memories."""
    chat = types.SimpleNamespace(id="chat-1", user_id=7, summary=None)
    db = _FakeDb()
    service = _patch_io(monkeypatch, chat, _MESSAGES, CHECKPOINT_JSON)

    first = await tc.run_thread_checkpoint(
        db, workspace_id="ws-1", chat_id="chat-1", trigger="idle_sweep",
    )
    first_mark = chat.summary["checkpointed_at"]
    calls_after_first = len(service.calls)

    second = await tc.run_thread_checkpoint(
        db, workspace_id="ws-1", chat_id="chat-1", trigger="on_demand",
    )

    assert first["success"] and second["success"]
    assert len(service.calls) == calls_after_first          # no duplicates
    assert second["planned_memories"] == 0
    assert chat.summary["checkpointed_at"] >= first_mark    # watermark moved
    assert chat.summary["trigger"] == "on_demand"
    assert db.commits == 2


@pytest.mark.asyncio
async def test_checkpoint_skips_short_threads(monkeypatch):
    chat = types.SimpleNamespace(id="chat-1", user_id=7, summary=None)
    service = _patch_io(monkeypatch, chat, _MESSAGES[:2], CHECKPOINT_JSON)

    result = await tc.run_thread_checkpoint(
        _FakeDb(), workspace_id="ws-1", chat_id="chat-1", trigger="idle_sweep",
    )
    assert result["success"] is False and result.get("skipped") is True
    assert chat.summary is None
    assert service.calls == []


@pytest.mark.asyncio
async def test_checkpoint_stores_nothing_on_parse_failure(monkeypatch):
    chat = types.SimpleNamespace(id="chat-1", user_id=7, summary=None)
    db = _FakeDb()
    service = _patch_io(monkeypatch, chat, _MESSAGES, "no json in this reply")

    result = await tc.run_thread_checkpoint(
        db, workspace_id="ws-1", chat_id="chat-1", trigger="idle_sweep",
    )
    assert result["success"] is False
    assert chat.summary is None
    assert db.commits == 0
    assert service.calls == []
