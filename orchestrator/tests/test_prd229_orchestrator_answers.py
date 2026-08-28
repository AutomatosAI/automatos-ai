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
from unittest.mock import MagicMock
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


@pytest.mark.parametrize("reply", [
    "NO_ANSWER",
    "NO_ANSWER.",
    "**NO_ANSWER**",
    "NO_ANSWER - not in context",
    "NO_ANSWER — the context does not cover this",
    "> NO_ANSWER",
    "no_answer",
])
@pytest.mark.asyncio
async def test_no_answer_near_miss_is_cannot_answer(reply, spy_events, no_external):
    """P229-RVW-3 — a non-bare declination (punctuation / markdown / caveat) is
    cannot_answer, NOT a cited answer. Exact equality only caught the bare form,
    letting the refusal text leak WITH sources — a fabrication-adjacent bug."""
    stub = _LLMStub(content=reply)
    db = _FakeSession({
        OrchestrationTaskDependency: [_dep()],
        OrchestrationTask: [_upstream_task()],
        OrchestrationEvent: [],
    })

    result = await answer_clarification(
        db, _subject(), "Unrelated question?", llm_factory=_factory(stub),
    )

    assert result == {"cannot_answer": True, "reason": "unretrievable"}
    assert stub.calls == 1  # composed, then declined — no leaked sources


@pytest.mark.asyncio
async def test_genuine_answer_mentioning_sentinel_is_returned(spy_events, no_external):
    # A real grounded answer that merely MENTIONS the token mid-sentence is not a
    # declination — it is returned normally with its sources.
    stub = _LLMStub(
        content="Write to s3://acme-staging; the pipeline emits NO_ANSWER only on empty input (see [1]).",
    )
    db = _FakeSession({
        OrchestrationTaskDependency: [_dep()],
        OrchestrationTask: [_upstream_task()],
        OrchestrationEvent: [],
    })

    result = await answer_clarification(
        db, _subject(), "Where do I write?", llm_factory=_factory(stub),
    )

    assert "answer" in result and result["answer"]
    assert result["sources"]
    assert [c.event_type for c in spy_events] == [EventType.CLARIFICATION_ANSWERED]


def test_is_no_answer_unit():
    # bare + non-bare forms decline; genuine answers (incl. mid-sentence mention) do not
    assert oa._is_no_answer("NO_ANSWER")
    assert oa._is_no_answer("**NO_ANSWER**")
    assert oa._is_no_answer("NO_ANSWER.")
    assert oa._is_no_answer("  no_answer  ")
    assert oa._is_no_answer("")
    assert not oa._is_no_answer("Use output B (see [1]).")
    assert not oa._is_no_answer("The pipeline emits NO_ANSWER on empty input.")


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


@pytest.mark.asyncio
async def test_budget_soft_cap_is_bounded_by_committed_trail(no_external):
    """P229-RVW-4 — the budget is a SOFT cap BY DESIGN, counted off the run's
    COMMITTED event trail. The check and the ANSWERED record are intentionally
    not atomic: concurrent asks run their agent I/O on SEPARATE DB sessions
    (coordinator Phase 2), and emit_event flushes without committing, so a
    sibling's in-flight answer is invisible under READ COMMITTED. Two asks that
    read the SAME committed baseline (budget-1 spent) therefore BOTH answer — an
    overspend bounded by max_concurrent that we accept (escalations are never
    budget-limited; hardening would mean committing a borrowed session mid-run).

    This pins the contract the check-site doc relies on: the gate is off the
    committed count, and once an answer IS committed/visible the next declines.
    """
    stub = _LLMStub()
    # committed trail shows budget-1 answers; a concurrent sibling's answer is
    # flushed-not-committed → invisible here, so both asks see the same baseline.
    baseline = _FakeSession({
        OrchestrationTaskDependency: [_dep()],
        OrchestrationTask: [_upstream_task()],
        OrchestrationEvent: [SimpleNamespace()] * (Config.CLARIFICATION_BUDGET - 1),
    })
    r1 = await answer_clarification(baseline, _subject(), "Which bucket?", llm_factory=_factory(stub))
    r2 = await answer_clarification(baseline, _subject(), "Which bucket?", llm_factory=_factory(stub))
    assert "answer" in r1 and "answer" in r2  # accepted, bounded soft-cap overspend

    # once the trail COMMITS to the budget, the next ask declines without composing
    committed = _FakeSession({
        OrchestrationTaskDependency: [_dep()],
        OrchestrationTask: [_upstream_task()],
        OrchestrationEvent: [SimpleNamespace()] * Config.CLARIFICATION_BUDGET,
    })
    calls_before = stub.calls
    r3 = await answer_clarification(committed, _subject(), "Which bucket?", llm_factory=_factory(stub))
    assert r3 == {"cannot_answer": True, "reason": "budget"}
    assert stub.calls == calls_before  # no compose once the committed cap is reached


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
# P229-RVW-7 — natural phrasing escalates (the keyword backstop had gaps)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        # spend — word-order-independent budget + money stems the list missed
        ("Should I increase the compute budget for this task?", "spend"),
        ("What will this cost the client?", "spend"),
        ("Do I expense the new tool?", "spend"),
        ("Is there a fee to process this?", "spend"),
        # destructive — verbs the original list missed
        ("Can I overwrite the existing records?", "destructive"),
        ("Should I revoke the old API key?", "destructive"),
        ("Do I deactivate the stale account?", "destructive"),
        # scope — implicit "also"/additional-work phrasing
        ("Should I also fix the login page?", "scope"),
        ("While I'm here, should I refactor the parser?", "scope"),
    ],
)
def test_natural_phrasing_governance_keyword_detection(question, expected):
    # the unit-level backstop now recognises natural phrasings, not only the
    # verbatim keywords the original list required.
    assert oa._governance_category(question, None) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Should I increase the compute budget for this task?",
        "Can I overwrite the existing records?",
        "Should I also fix the login page while I'm here?",
    ],
)
@pytest.mark.asyncio
async def test_natural_phrasing_governance_escalates_without_llm(question, spy_events):
    # end-to-end: a natural-phrasing governance question escalates directly with
    # ZERO retrieval and ZERO LLM composition — Auto never auto-answers a
    # spend/destructive/scope decision (G3), it hands it to a human.
    stub = _LLMStub()
    result = await answer_clarification(
        MagicMock(), _subject(), question, llm_factory=_factory(stub),
    )
    assert result["escalate_directly"] is True
    assert result["reason"] == "governance"
    assert stub.calls == 0  # no composition — the backstop caught it first


# ---------------------------------------------------------------------------
# budget counts ONLY answered clarifications (escalations don't consume it)
# ---------------------------------------------------------------------------

def test_answers_used_counts_only_answered_events():
    db = _FakeSession({OrchestrationEvent: [SimpleNamespace(), SimpleNamespace()]})
    assert oa._answers_used(db, _subject()) == 2


# ---------------------------------------------------------------------------
# P229-RVW-2 — upstream retrieval is fenced to the subject's OWN run. A foreign
# dependency task (different run) is never surfaced as a "grounded" answer, even
# if a same-shaped task_id reaches _upstream_blocks. Uses a filter-AWARE fake
# (the _FakeSession above ignores filters, so it cannot exercise scoping).
# ---------------------------------------------------------------------------

import operator as _operator  # noqa: E402
from sqlalchemy.sql.operators import in_op  # noqa: E402


def _row_matches(row, expr) -> bool:
    key = getattr(getattr(expr, "left", None), "key", None)
    if key is None:
        return True
    actual = getattr(row, key, None)
    op = getattr(expr, "operator", None)
    if op is _operator.eq:
        return actual == expr.right.value
    if op is in_op:
        return actual in (expr.right.value or [])
    return True


class _ScopedQuery:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *exprs):
        rows = self._rows
        for e in exprs:
            rows = [r for r in rows if _row_matches(r, e)]
        return _ScopedQuery(rows)

    def order_by(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def count(self):
        return len(self._rows)


class _ScopedSession:
    def __init__(self, by_model):
        self._by_model = by_model

    def query(self, model):
        return _ScopedQuery(self._by_model.get(model, []))


def test_upstream_blocks_scopes_out_foreign_run():
    # dep task id=100 lives in run-B; the subject is run-A → no blocks.
    subject = _subject(run_id="run-A", task_id=200)
    foreign = SimpleNamespace(id=100, run_id="run-B", output="secret", title="x", sequence_number=1)
    db = _ScopedSession({
        OrchestrationTaskDependency: [SimpleNamespace(task_id=200, depends_on_task_id=100)],
        OrchestrationTask: [foreign],
    })
    assert oa._upstream_blocks(db, subject) == []


def test_upstream_blocks_returns_same_run_deps():
    subject = _subject(run_id="run-A", task_id=200)
    same = SimpleNamespace(id=100, run_id="run-A", output="ours", title="x", sequence_number=1)
    db = _ScopedSession({
        OrchestrationTaskDependency: [SimpleNamespace(task_id=200, depends_on_task_id=100)],
        OrchestrationTask: [same],
    })
    blocks = oa._upstream_blocks(db, subject)
    assert len(blocks) == 1
    assert blocks[0]["source"]["task_id"] == "100"


def test_upstream_blocks_guards_missing_run():
    subject = _subject(run_id=None, task_id=200)
    db = _ScopedSession({
        OrchestrationTaskDependency: [SimpleNamespace(task_id=200, depends_on_task_id=100)],
        OrchestrationTask: [SimpleNamespace(id=100, run_id="run-A", output="x", title="t", sequence_number=1)],
    })
    assert oa._upstream_blocks(db, subject) == []


# ---------------------------------------------------------------------------
# P229-RVW-9 — sources are the CITED blocks, not the whole retrieval set
# ---------------------------------------------------------------------------

def test_cited_sources_unit():
    blocks = [
        {"source": {"type": "upstream_task", "task_id": "1"}},
        {"source": {"type": "memory", "id": "m2"}},
        {"source": {"type": "field", "id": "f3"}},
    ]
    # cites [3] then [1], repeats [3], and a bogus [9] → valid indices only, in
    # first-cited order, de-duped; [9] ignored (never invents a source).
    assert oa._cited_sources("see [3] and [1] and [3] and [9]", blocks) == [
        {"type": "field", "id": "f3"},
        {"type": "upstream_task", "task_id": "1"},
    ]
    # no valid citation → fall back to ALL retrieved sources
    assert oa._cited_sources("no markers here", blocks) == [b["source"] for b in blocks]
    assert oa._cited_sources("only bogus [42]", blocks) == [b["source"] for b in blocks]


@pytest.mark.asyncio
async def test_sources_reflect_cited_block_not_full_retrieval(spy_events, monkeypatch):
    # several blocks retrieved, the answer cites ONLY [2] → sources is exactly that
    # one block's ref, not all three (the grounded/cited guarantee, not over-claim).
    blocks = [
        {"text": "alpha", "source": {"type": "upstream_task", "task_id": "1"}},
        {"text": "beta", "source": {"type": "memory", "id": "m2"}},
        {"text": "gamma", "source": {"type": "field", "id": "f3"}},
    ]

    async def _blocks(db, subject, question):
        return blocks

    monkeypatch.setattr(oa, "_retrieve", _blocks)
    stub = _LLMStub(content="Use the memory note (see [2]).")
    db = _FakeSession({OrchestrationEvent: []})  # 0 answers spent → under budget

    result = await answer_clarification(db, _subject(), "Q?", llm_factory=_factory(stub))

    assert result["answer"]
    assert result["sources"] == [{"type": "memory", "id": "m2"}]  # ONLY [2]


@pytest.mark.asyncio
async def test_sources_fall_back_to_all_when_answer_cites_nothing(spy_events, monkeypatch):
    blocks = [
        {"text": "alpha", "source": {"type": "upstream_task", "task_id": "1"}},
        {"text": "beta", "source": {"type": "memory", "id": "m2"}},
    ]

    async def _blocks(db, subject, question):
        return blocks

    monkeypatch.setattr(oa, "_retrieve", _blocks)
    stub = _LLMStub(content="Write to the staging bucket.")  # no [n] markers
    db = _FakeSession({OrchestrationEvent: []})

    result = await answer_clarification(db, _subject(), "Q?", llm_factory=_factory(stub))

    assert result["sources"] == [b["source"] for b in blocks]  # fallback: all retrieved
