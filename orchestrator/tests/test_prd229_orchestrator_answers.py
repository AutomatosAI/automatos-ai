"""PRD-229 US-001 — the answering service (pure tests).

No DB, no LLM, no network:
  * the upstream-digest retrieval is exercised against a tiny fake session that
    returns dependency + task rows (proving answers cite the digest),
  * the external sources are patched to empty so a fixture isolates one source,
  * the composition LLM is an injected stub whose call count is asserted (the
    empty-retrieval short-circuit makes ZERO LLM calls),
  * ``emit_event`` is spied to prove every Q&A lands on the run event trail.

Covers the US-001 acceptance criteria: answerable → cited answer; unretrievable
→ cannot_answer with no LLM call; budget → cannot_answer(budget); governance →
escalate_directly without answering; Q&A recorded; CLARIFICATION_BUDGET in config.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import services.orchestrator_answers as oa  # noqa: E402
from config import Config  # noqa: E402
from core.models.orchestration import (  # noqa: E402
    OrchestrationEvent,
    OrchestrationTask,
    OrchestrationTaskDependency,
)
from core.models.orchestration_enums import EventType  # noqa: E402
from services.orchestrator_answers import (  # noqa: E402
    ClarificationSubject,
    answer_clarification,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)

    def count(self):
        return len(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Dispatches ``query(Model)`` to preset rows — no SQL, no DB."""

    def __init__(self, by_model=None):
        self._by_model = by_model or {}

    def query(self, model):
        return _FakeQuery(self._by_model.get(model, []))


class _LLMStub:
    """A composition LLM whose call count is observable."""

    def __init__(self, content="Use s3://acme-staging (see [1])."):
        self.calls = 0
        self._content = content

    async def generate_response(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self._content)


def _factory(stub):
    def make(**kwargs):
        return stub
    return make


def _upstream_task(task_id=100, output="The staging bucket is s3://acme-staging.", title="Provision storage"):
    return SimpleNamespace(id=task_id, output=output, title=title, sequence_number=1)


def _dep(depends_on_task_id=100):
    return SimpleNamespace(depends_on_task_id=depends_on_task_id, task_id=200)


def _subject(**over):
    base = dict(run_id=uuid4(), workspace_id=uuid4(), task_id=200, task=None)
    base.update(over)
    return ClarificationSubject(**base)


@pytest.fixture
def spy_events(monkeypatch):
    """Record every emit_event the service fires (the run event trail)."""
    calls = []

    def _rec(db, run_id, event_type, actor_type, actor_id=None, task_id=None, payload=None):
        calls.append(SimpleNamespace(event_type=event_type, payload=payload or {}))
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(oa, "emit_event", _rec)
    return calls


@pytest.fixture
def no_external(monkeypatch):
    """Isolate the upstream-digest source: the external sources return nothing."""
    async def _empty(db, subject, question):
        return []

    monkeypatch.setattr(oa, "_external_blocks", _empty)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_clarification_budget_default_is_three():
    assert Config.CLARIFICATION_BUDGET == 3


# ---------------------------------------------------------------------------
# answerable → cited answer, LLM called once, refs point at the digest
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_answerable_returns_answer_with_digest_refs(spy_events, no_external):
    stub = _LLMStub()
    db = _FakeSession({
        OrchestrationTaskDependency: [_dep(depends_on_task_id=100)],
        OrchestrationTask: [_upstream_task(task_id=100)],
        OrchestrationEvent: [],  # no answers spent yet
    })

    result = await answer_clarification(
        db, _subject(), "Which bucket do I write to?", llm_factory=_factory(stub),
    )

    assert "answer" in result and result["answer"]
    assert stub.calls == 1
    # The source ref points at the upstream digest task — not a fabrication.
    refs = result["sources"]
    assert any(r["type"] == "upstream_task" and r["task_id"] == "100" for r in refs)
    # ...and the Q&A is on the run trail as ANSWERED.
    assert [c.event_type for c in spy_events] == [EventType.CLARIFICATION_ANSWERED]


# ---------------------------------------------------------------------------
# unretrievable → cannot_answer, ZERO LLM calls (short-circuit before compose)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unretrievable_makes_no_llm_call(spy_events, no_external):
    stub = _LLMStub()
    db = _FakeSession({
        OrchestrationTaskDependency: [],  # no upstream → nothing retrievable
        OrchestrationTask: [],
        OrchestrationEvent: [],
    })

    result = await answer_clarification(
        db, _subject(), "What did the auditor conclude?", llm_factory=_factory(stub),
    )

    assert result == {"cannot_answer": True, "reason": "unretrievable"}
    assert stub.calls == 0  # empty retrieval short-circuited BEFORE the LLM
    assert [c.event_type for c in spy_events] == [EventType.CLARIFICATION_ESCALATED]


@pytest.mark.asyncio
async def test_no_answer_sentinel_is_cannot_answer(spy_events, no_external):
    """Retrieval hits but the composer says the answer is not in them."""
    stub = _LLMStub(content="NO_ANSWER")
    db = _FakeSession({
        OrchestrationTaskDependency: [_dep()],
        OrchestrationTask: [_upstream_task()],
        OrchestrationEvent: [],
    })

    result = await answer_clarification(
        db, _subject(), "Unrelated question?", llm_factory=_factory(stub),
    )

    assert result == {"cannot_answer": True, "reason": "unretrievable"}
    assert stub.calls == 1  # composed, then declined — no fabrication


# ---------------------------------------------------------------------------
# budget → cannot_answer(budget), no retrieval, no LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_exhausted_returns_budget(spy_events, no_external):
    stub = _LLMStub()
    db = _FakeSession({
        OrchestrationTaskDependency: [_dep()],  # would be answerable...
        OrchestrationTask: [_upstream_task()],
        OrchestrationEvent: [SimpleNamespace()] * Config.CLARIFICATION_BUDGET,  # ...but spent
    })

    result = await answer_clarification(
        db, _subject(), "Which bucket?", llm_factory=_factory(stub),
    )

    assert result == {"cannot_answer": True, "reason": "budget"}
    assert stub.calls == 0
    assert [c.event_type for c in spy_events] == [EventType.CLARIFICATION_ESCALATED]


@pytest.mark.asyncio
async def test_budget_below_limit_still_answers(spy_events, no_external):
    stub = _LLMStub()
    db = _FakeSession({
        OrchestrationTaskDependency: [_dep()],
        OrchestrationTask: [_upstream_task()],
        OrchestrationEvent: [SimpleNamespace()] * (Config.CLARIFICATION_BUDGET - 1),
    })

    result = await answer_clarification(
        db, _subject(), "Which bucket?", llm_factory=_factory(stub),
    )

    assert "answer" in result
    assert stub.calls == 1


# ---------------------------------------------------------------------------
# governance → escalate_directly, no answering attempt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("declared", ["spend", "destructive", "scope"])
@pytest.mark.asyncio
async def test_declared_governance_escalates_directly(declared, spy_events, monkeypatch):
    stub = _LLMStub()
    external_called = {"n": 0}

    async def _spy_external(db, subject, question):
        external_called["n"] += 1
        return [{"text": "x", "source": {"type": "upstream_task", "task_id": "1"}}]

    monkeypatch.setattr(oa, "_external_blocks", _spy_external)

    db = _FakeSession({OrchestrationEvent: []})
    result = await answer_clarification(
        db, _subject(), "A perfectly routine question.",
        category=declared, llm_factory=_factory(stub),
    )

    assert result == {"escalate_directly": True, "reason": "governance", "category": declared}
    assert stub.calls == 0
    assert external_called["n"] == 0  # governance skips retrieval entirely
    assert [c.event_type for c in spy_events] == [EventType.CLARIFICATION_ESCALATED]
    assert spy_events[0].payload["category"] == declared


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Should I delete all the production records?", "destructive"),
        ("Do I have budget to purchase the enterprise plan?", "spend"),
        ("This is a scope change — add a new feature?", "scope"),
        ("Which staging bucket should I write to?", None),
    ],
)
def test_governance_keyword_detection(question, expected):
    assert oa._governance_category(question, None) == expected


def test_declared_category_wins_over_benign_text():
    assert oa._governance_category("totally benign", "spend") == "spend"


def test_non_governance_declared_category_is_ignored():
    # a non-governance declared category does not force escalation
    assert oa._governance_category("benign text", "logistics") is None


# ---------------------------------------------------------------------------
# budget counts ONLY answered clarifications (escalations don't consume it)
# ---------------------------------------------------------------------------

def test_answers_used_counts_only_answered_events():
    db = _FakeSession({OrchestrationEvent: [SimpleNamespace(), SimpleNamespace()]})
    assert oa._answers_used(db, _subject()) == 2
