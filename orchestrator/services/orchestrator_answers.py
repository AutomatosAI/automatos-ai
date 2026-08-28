"""PRD-229 US-001 — the answering service: grounded, budgeted, cited.

Given a run/task subject and a mid-run clarification, Auto answers ONLY from
retrievable context (upstream task results, the mission field, durable memory,
the intake/business corpus, and — for questions about the floor — fleet state),
citing the refs it used. Nothing retrievable → ``cannot_answer``: the LLM call
is composition over retrieved context, and an EMPTY retrieval short-circuits
BEFORE any LLM call. The reviewer hunts invented answers — there are none: every
answer carries the source refs it was grounded on, and the composition prompt is
told to reply ``NO_ANSWER`` rather than guess.

Budget: ``Config.CLARIFICATION_BUDGET`` ANSWERS per run (counted off the run's
own event trail). Escalations are never budget-limited — that ladder is US-003.
Governance-category questions (destructive / spend / scope, per PRD-223) skip
answering entirely and return ``escalate_directly``. Every Q&A — answered,
cannot-answer, or escalate — is appended to the run event trail (``emit_event``).

Pure by construction: the one LLM call is an injectable seam (``llm_factory``,
mirroring ``digest_service``) and every external retrieval source is fail-soft,
so the unit suite runs with no LLM and no external infra.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import Config
from core.models.orchestration_enums import ActorType, EventType
from services.orchestration_state import emit_event

logger = logging.getLogger(__name__)

# ``cannot_answer`` reasons.
REASON_BUDGET = "budget"
REASON_UNRETRIEVABLE = "unretrievable"

# Governance categories (PRD-223 framing) — never answered by Auto; escalated.
GOVERNANCE_CATEGORIES = ("destructive", "spend", "scope")

# The exact sentinel the composition prompt returns when the retrieved context
# does not actually answer the question — treated as cannot_answer (grounding
# guard: retrieval hits do not guarantee an answer lives in them).
_NO_ANSWER = "NO_ANSWER"

# Conservative keyword probes for governance detection when the caller does not
# declare a category. A false positive only OVER-escalates (safe + visible); it
# never fabricates. Substring match, lower-cased.
_GOVERNANCE_KEYWORDS: Dict[str, tuple] = {
    "destructive": (
        "delete", "drop table", "truncate", "wipe", "destroy", "purge",
        "remove all", "rm -rf", "erase", "tear down",
    ),
    "spend": (
        "spend", "purchase", "buy ", "pay ", "payment", "invoice", "charge",
        "budget increase", "upgrade the plan", "upgrade plan", "$",
    ),
    "scope": (
        "out of scope", "scope change", "change the scope", "expand the scope",
        "new requirement", "add a feature", "change the goal",
    ),
}

# Retrieval caps — keep the composed context bounded (the answer round runs
# inside the task's execution envelope; US-002 pins the time-box).
_UPSTREAM_PER_OUTPUT_LIMIT = 4000
_MAX_BLOCKS = 12


@dataclass(frozen=True)
class ClarificationSubject:
    """The run/task the clarification is raised against (server-resolved).

    ``task`` is the OrchestrationTask ORM row when available (the upstream-digest
    source reads its dependency outputs); ``task``/``task_id`` may be absent for
    a run-level question. Immutable — never mutated by the answering path.
    """

    run_id: Any
    workspace_id: Any
    task_id: Any = None
    task: Any = None
    field_id: Optional[str] = None
    agent_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def answer_clarification(
    db: Any,
    subject: ClarificationSubject,
    question: str,
    *,
    category: Optional[str] = None,
    llm_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Answer a mid-run clarification, or decline (budget / governance / empty).

    Returns exactly one of:
      * ``{"answer": str, "sources": [ref, ...]}``            — grounded answer
      * ``{"cannot_answer": True, "reason": str}``            — budget|unretrievable
      * ``{"escalate_directly": True, "reason": "governance", "category": str}``
    """
    question = (question or "").strip()

    # 1. Governance short-circuit — never answered, always escalated. No
    #    retrieval, no LLM.
    gov = _governance_category(question, category)
    if gov:
        _record(db, subject, question, outcome="escalate_directly", extra={"category": gov})
        return {"escalate_directly": True, "reason": "governance", "category": gov}

    # 2. Budget — count answers already spent on THIS run. Spent → cannot_answer.
    if _answers_used(db, subject) >= _budget():
        _record(db, subject, question, outcome="cannot_answer", extra={"reason": REASON_BUDGET})
        return {"cannot_answer": True, "reason": REASON_BUDGET}

    # 3. Retrieve grounded context. Empty → cannot_answer BEFORE any LLM call.
    blocks = await _retrieve(db, subject, question)
    if not blocks:
        _record(db, subject, question, outcome="cannot_answer", extra={"reason": REASON_UNRETRIEVABLE})
        return {"cannot_answer": True, "reason": REASON_UNRETRIEVABLE}

    # 4. Compose over the retrieved context (the LLM's only job). A blank or
    #    NO_ANSWER reply means the hits did not actually answer → cannot_answer.
    answer = await _compose_answer(
        question, blocks, workspace_id=subject.workspace_id, llm_factory=llm_factory
    )
    if not answer or answer.strip() == _NO_ANSWER:
        _record(db, subject, question, outcome="cannot_answer", extra={"reason": REASON_UNRETRIEVABLE})
        return {"cannot_answer": True, "reason": REASON_UNRETRIEVABLE}

    sources = [b["source"] for b in blocks]
    _record(db, subject, question, outcome="answered", extra={"sources": sources})
    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Governance detection
# ---------------------------------------------------------------------------

def _governance_category(question: str, declared: Optional[str]) -> Optional[str]:
    """Return the governance category if this question must be escalated, else
    None. The caller's DECLARED category wins; otherwise a conservative keyword
    scan of the question text."""
    if declared:
        normalized = declared.strip().lower()
        if normalized in GOVERNANCE_CATEGORIES:
            return normalized
    text = (question or "").lower()
    for category, needles in _GOVERNANCE_KEYWORDS.items():
        if any(n in text for n in needles):
            return category
    return None


# ---------------------------------------------------------------------------
# Budget (counted off the run's own event trail — no new counter, no schema)
# ---------------------------------------------------------------------------

def _budget() -> int:
    try:
        return int(Config.CLARIFICATION_BUDGET)
    except (TypeError, ValueError):
        return 3


def _answers_used(db: Any, subject: ClarificationSubject) -> int:
    """How many clarifications Auto has ANSWERED on this run so far."""
    try:
        from core.models.orchestration import OrchestrationEvent

        return (
            db.query(OrchestrationEvent)
            .filter(
                OrchestrationEvent.run_id == subject.run_id,
                OrchestrationEvent.event_type == EventType.CLARIFICATION_ANSWERED.value,
            )
            .count()
        )
    except Exception:  # noqa: BLE001 — degrade to "no answers spent" (fail-open)
        logger.warning("[clarify] budget count failed; treating as 0", exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# Retrieval — upstream digest (real) + external sources (fail-soft)
# ---------------------------------------------------------------------------

async def _retrieve(db: Any, subject: ClarificationSubject, question: str) -> List[Dict[str, Any]]:
    """Grounded context blocks: ``[{"text": str, "source": {ref}}, ...]``.

    The upstream-digest source is the same data ``_prepare_task`` dispatches on
    (upstream-dependency task outputs); the rest are best-effort so a missing
    Qdrant / RAG / fleet read degrades the answer set, never breaks the round.
    """
    blocks = list(_upstream_blocks(db, subject))
    blocks += await _external_blocks(db, subject, question)
    return blocks[:_MAX_BLOCKS]


def _upstream_blocks(db: Any, subject: ClarificationSubject) -> List[Dict[str, Any]]:
    """Upstream-dependency task outputs — the run's own produced context.

    Mirrors ``CoordinatorService._collect_upstream_outputs`` (same digest
    source) with a local query, so the answering path never imports the hot
    coordinator module (circular-import + heavy-load hazard)."""
    task_id = subject.task_id if subject.task_id is not None else getattr(subject.task, "id", None)
    if task_id is None:
        return []
    try:
        from core.models.orchestration import (
            OrchestrationTask,
            OrchestrationTaskDependency,
        )

        deps = (
            db.query(OrchestrationTaskDependency)
            .filter(OrchestrationTaskDependency.task_id == task_id)
            .all()
        )
        if not deps:
            return []
        dep_ids = [d.depends_on_task_id for d in deps]
        rows = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.id.in_(dep_ids))
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )
        blocks: List[Dict[str, Any]] = []
        for row in rows:
            text = (getattr(row, "output", None) or "").strip()
            if not text:
                continue
            blocks.append({
                "text": text[:_UPSTREAM_PER_OUTPUT_LIMIT],
                "source": {
                    "type": "upstream_task",
                    "task_id": _idstr(getattr(row, "id", None)),
                    "title": getattr(row, "title", None),
                },
            })
        return blocks
    except Exception:  # noqa: BLE001
        logger.warning("[clarify] upstream retrieval failed", exc_info=True)
        return []


async def _external_blocks(db: Any, subject: ClarificationSubject, question: str) -> List[Dict[str, Any]]:
    """Field + memory + intake corpus + fleet, each fail-soft. One seam the
    unit suite patches to isolate the upstream-digest behaviour."""
    out: List[Dict[str, Any]] = []
    out += await _field_blocks(subject, question)
    out += await _memory_blocks(subject, question)
    out += await _corpus_blocks(subject, question)
    out += _fleet_blocks(db, subject, question)
    return out


async def _field_blocks(subject: ClarificationSubject, question: str) -> List[Dict[str, Any]]:
    """Semantic hits from the mission's shared field (PRD-108)."""
    if not subject.field_id:
        return []
    try:
        from modules.context.factory import get_shared_context

        field = get_shared_context()
        if not field:
            return []
        results = await field.query(
            context_id=subject.field_id,
            query=question,
            agent_id=subject.agent_id,
            top_k=Config.FIELD_QUERY_TOP_K,
        )
        return [
            {"text": str(r.get("value", "")), "source": {"type": "mission_field", "key": r.get("key")}}
            for r in (results or [])
            if r.get("value")
        ]
    except Exception:  # noqa: BLE001
        logger.warning("[clarify] field retrieval failed", exc_info=True)
        return []


async def _memory_blocks(subject: ClarificationSubject, question: str) -> List[Dict[str, Any]]:
    """Durable workspace memory (PRD-206). Naturally empty when unconfigured."""
    try:
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()
        if not getattr(service, "is_durable_configured", False):
            return []
        results = await service.search_long_term(
            workspace_id=str(subject.workspace_id), query=question, limit=5,
        )
        blocks: List[Dict[str, Any]] = []
        for r in (results or []):
            text = str(r.get("memory") or r.get("text") or r.get("content") or "").strip()
            if not text:
                continue
            blocks.append({"text": text, "source": {"type": "memory", "id": _idstr(r.get("id"))}})
        return blocks
    except Exception:  # noqa: BLE001
        logger.warning("[clarify] memory retrieval failed", exc_info=True)
        return []


async def _corpus_blocks(subject: ClarificationSubject, question: str) -> List[Dict[str, Any]]:
    """Intake / business corpus via the existing RAG stack (S3 Vectors only)."""
    try:
        from modules.rag.service import RAGService

        rag = RAGService()
        result = await rag.retrieve_context(
            query=question, workspace_id=str(subject.workspace_id), max_chunks=5,
        )
        blocks: List[Dict[str, Any]] = []
        for chunk in (getattr(result, "chunks", None) or []):
            text = str(chunk.get("expanded_content") or chunk.get("content") or "").strip()
            if not text:
                continue
            blocks.append({
                "text": text,
                "source": {
                    "type": "corpus",
                    "source_file": chunk.get("source_file") or chunk.get("filename"),
                    "document_id": _idstr(chunk.get("document_id")),
                },
            })
        return blocks
    except Exception:  # noqa: BLE001
        logger.warning("[clarify] corpus retrieval failed", exc_info=True)
        return []


# Floor-question keywords — the fleet read is noise for most questions, so it is
# gated to questions plausibly about who/what is running.
_FLEET_KEYWORDS = (
    "fleet", "agent", "who is working", "who's working", "workload", "busy",
    "queue", "idle", "the floor", "capacity", "assigned to",
)


def _fleet_blocks(db: Any, subject: ClarificationSubject, question: str) -> List[Dict[str, Any]]:
    """228's fleet read-model — only when the question is about the floor."""
    text = (question or "").lower()
    if not any(k in text for k in _FLEET_KEYWORDS):
        return []
    try:
        from services.fleet_state import get_fleet_state

        state = get_fleet_state(db, subject.workspace_id)
        agents = (state or {}).get("agents") or []
        if not agents:
            return []
        summary = "; ".join(
            f"{a.get('name')}: {(a.get('current') or {}).get('title') or 'idle'}"
            for a in agents[:20]
        )
        return [{
            "text": f"Fleet snapshot ({len(agents)} agents): {summary}",
            "source": {"type": "fleet_state", "generated_at": (state or {}).get("generated_at")},
        }]
    except Exception:  # noqa: BLE001
        logger.warning("[clarify] fleet retrieval failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Composition — the one LLM call (injectable seam)
# ---------------------------------------------------------------------------

_COMPOSE_SYSTEM = (
    "You are Auto, answering an executing worker agent's clarification DURING a "
    "run. Answer ONLY from the numbered context below — it is the sole source of "
    "truth. If the context does not contain the answer, reply with exactly "
    f"{_NO_ANSWER} and nothing else. Never invent facts, sources, ids, or values. "
    "Cite the [n] blocks you used. Be concise and decisive."
)


def _default_llm_factory(**kwargs: Any) -> Any:
    from core.llm.manager import create_llm_manager

    return create_llm_manager(**kwargs)


async def _compose_answer(
    question: str,
    blocks: List[Dict[str, Any]],
    *,
    workspace_id: Any,
    llm_factory: Optional[Any] = None,
) -> str:
    """Compose a grounded answer from retrieved blocks. Fail-soft → '' (which the
    caller treats as cannot_answer)."""
    factory = llm_factory or _default_llm_factory
    try:
        llm = factory(
            service_name="orchestrator",
            workspace_id=workspace_id,
            request_type="clarification",
        )
        response = await llm.generate_response(_build_messages(question, blocks))
        return (getattr(response, "content", None) or "").strip()
    except Exception:  # noqa: BLE001
        logger.warning("[clarify] composition failed", exc_info=True)
        return ""


def _build_messages(question: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context = "\n\n".join(
        f"[{i + 1}] ({_ref_label(b['source'])})\n{b['text']}"
        for i, b in enumerate(blocks)
    )
    return [
        {"role": "system", "content": _COMPOSE_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]


# ---------------------------------------------------------------------------
# Event trail
# ---------------------------------------------------------------------------

def _record(
    db: Any,
    subject: ClarificationSubject,
    question: str,
    *,
    outcome: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append the Q&A to the run's event trail. Best-effort — a trail fault
    never fails the clarification."""
    event_type = (
        EventType.CLARIFICATION_ANSWERED
        if outcome == "answered"
        else EventType.CLARIFICATION_ESCALATED
    )
    payload = {"outcome": outcome, "question": (question or "")[:2000]}
    if extra:
        payload.update(extra)
    try:
        emit_event(
            db,
            run_id=subject.run_id,
            event_type=event_type,
            actor_type=ActorType.COORDINATOR,
            actor_id="auto",
            task_id=subject.task_id,
            payload=payload,
        )
    except Exception:  # noqa: BLE001
        logger.warning("[clarify] failed to record Q&A on run trail", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_label(source: Dict[str, Any]) -> str:
    kind = source.get("type", "source")
    for key in ("title", "source_file", "key", "id", "generated_at"):
        if source.get(key):
            return f"{kind}:{source[key]}"
    return kind


def _idstr(value: Any) -> Optional[str]:
    return None if value is None else str(value)
