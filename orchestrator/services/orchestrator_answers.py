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
import re
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

# The sentinel the composition prompt returns when the retrieved context does
# not actually answer the question — treated as cannot_answer (grounding guard:
# retrieval hits do not guarantee an answer lives in them).
_NO_ANSWER = "NO_ANSWER"

# Characters a model tends to wrap the bare sentinel in — markdown emphasis,
# blockquote/list markers, quotes, whitespace — stripped from the LEFT before
# the leading-sentinel test so '**NO_ANSWER**' / '> NO_ANSWER' still decline.
_SENTINEL_LEADING = "*_`~'\"#>- \t"

# Conservative keyword probes for governance detection when the caller does not
# declare a category. A false positive only OVER-escalates (safe + visible); it
# never fabricates. Substring match, lower-cased.
_GOVERNANCE_KEYWORDS: Dict[str, tuple] = {
    "destructive": (
        "delete", "drop table", "truncate", "wipe", "destroy", "purge",
        "remove all", "rm -rf", "erase", "tear down",
        # P229-RVW-7: common destructive verbs the original list missed.
        "overwrite", "revoke", "deactivate", "reformat", "uninstall",
    ),
    "spend": (
        "spend", "purchase", "buy ", "pay ", "payment", "invoice", "charge",
        "upgrade the plan", "upgrade plan", "$",
        # P229-RVW-7: word-order-independent "budget" (so "increase the budget"
        # matches, not only "budget increase") + the money stems the original
        # list missed. Governance detection biases toward OVER-escalation
        # (safe + visible; it never fabricates), so broad money words are by
        # design — a false hit escalates a routine question, never invents one.
        "budget", "cost", "expense", "fee", "billing",
    ),
    "scope": (
        "out of scope", "scope change", "change the scope", "expand the scope",
        "new requirement", "add a feature", "change the goal",
        # P229-RVW-7: implicit "also"/additional-work phrasing ("should I also
        # fix X while I'm here?") — targeted action bigrams, NOT bare "also"
        # (which would over-escalate every incidental mention).
        "also fix", "also add", "also implement", "also build", "also handle",
        "also update", "also create", "also refactor", "while i'm here",
        "while i am here", "additional feature", "additional work",
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
    #
    # SOFT CAP, by design (P229-RVW-4). This check and the ANSWERED emit_event at
    # the end are two unlocked operations, so concurrent asks in one run can both
    # read the same pre-increment count and both answer — the run's answered count
    # can exceed CLARIFICATION_BUDGET by up to (max_concurrent - 1). We accept the
    # soft cap deliberately rather than hardening it, because a correct atomic
    # spend is unsafe here and the overspend is cheap:
    #   * The concurrent tasks in one mission-run tick run on a SHARED DB session
    #     (P229-RVW-5 corrected the earlier "separate sessions" claim):
    #     coordinator_service opens ONE SessionLocal per tick (:1521), builds
    #     AgentFactory(db_session=db) per task (:2242), and runs their agent I/O
    #     concurrently via asyncio.gather (:1750). The ANSWERED marker here is
    #     emit_event-FLUSHED but NOT committed, and this answer path deliberately
    #     never commits: a commit would durably persist every SIBLING task's
    #     in-flight, uncommitted work on that shared session (a worse bug than the
    #     overspend). So no app-level or advisory lock can close the check→record
    #     window without committing the shared session mid-tool-execution.
    #   * A schema-encoded slot / separate counter is ruled out: PRD-229 mandates
    #     zero migrations and budget "counted off the run's own event trail".
    #   * The blast radius is bounded by max_concurrent (default 3) and is pure
    #     cost-control softness — escalations are never budget-limited, so the
    #     human-ask safety valve is unaffected. (The ESCALATION path DOES commit —
    #     unlike this ephemeral answer path — because it must durably park a
    #     human-visible ask; see clarification_ladder.escalate_clarification.)
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
    if not answer or _is_no_answer(answer):
        _record(db, subject, question, outcome="cannot_answer", extra={"reason": REASON_UNRETRIEVABLE})
        return {"cannot_answer": True, "reason": REASON_UNRETRIEVABLE}

    sources = _cited_sources(answer, blocks)
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


def task_clarification_rounds(db: Any, subject: ClarificationSubject) -> int:
    """How many clarification ANSWER rounds THIS task has already spent (per-task,
    counted off the run's own event trail — no new counter, no schema). Bounds the
    cumulative answer-round time a single task can burn inside its execution
    envelope (P229-RVW-8): only ANSWERED rounds let the agent continue (a
    cannot_answer / escalate round parks the task), so this is the number of prior
    calls that entered the (up-to-CLARIFICATION_ANSWER_TIMEOUT) answer round."""
    if subject.task_id is None:
        return 0
    try:
        from core.models.orchestration import OrchestrationEvent

        count = (
            db.query(OrchestrationEvent)
            .filter(
                OrchestrationEvent.run_id == subject.run_id,
                OrchestrationEvent.task_id == subject.task_id,
                OrchestrationEvent.event_type == EventType.CLARIFICATION_ANSWERED.value,
            )
            .count()
        )
        return count if isinstance(count, int) else 0
    except Exception:  # noqa: BLE001 — degrade to "no rounds spent" (fail-open)
        logger.warning("[clarify] per-task round count failed; treating as 0", exc_info=True)
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
    if task_id is None or subject.run_id is None:
        return []
    try:
        from core.models.orchestration import (
            OrchestrationRun,
            OrchestrationTask,
            OrchestrationTaskDependency,
        )

        # P229-RVW-10: make the run→workspace defence-in-depth REAL for THIS read.
        # The run_id fence below is intra-run only; _load_task's workspace scope
        # gates the PARK write, not this read (which uses subject.run_id directly,
        # never subject.task). So confirm the run belongs to the subject's
        # workspace here: if the run is loadable and owned by a DIFFERENT
        # workspace, refuse — a run_id that diverged from workspace_id (the known
        # cached-agent-runtime workspace-staleness pattern) must not surface
        # another tenant's outputs as "grounding" while _external_blocks reads the
        # (different) subject.workspace_id. Fail-open only when the run is
        # unloadable — a bogus run_id has no dep outputs to leak, and the run_id
        # fence still applies.
        if subject.workspace_id is not None:
            run = (
                db.query(OrchestrationRun)
                .filter(OrchestrationRun.id == subject.run_id)
                .first()
            )
            if run is not None and getattr(run, "workspace_id", None) != subject.workspace_id:
                logger.warning(
                    "[clarify] upstream read refused — run %s is not in the subject's workspace",
                    subject.run_id,
                )
                return []

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
            .filter(
                OrchestrationTask.id.in_(dep_ids),
                # P229-RVW-2: fence upstream reads to the subject's OWN run. A
                # task's upstream deps are intra-run by DAG construction, so this
                # loses no legitimate context; it stops a foreign task_id from
                # surfacing another tenant's outputs as a "grounded" answer. The
                # tenant boundary for this read is the run→workspace confirmation
                # above + the executor strip (which binds run_id/task_id
                # self-consistently from server field_context) — NOT _load_task.
                OrchestrationTask.run_id == subject.run_id,
            )
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


# Citation markers the composed answer uses to reference numbered blocks ([1],
# [2], … — the 1-based index _build_messages assigns).
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _cited_sources(answer: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The sources the composed answer actually CITED — not every retrieved block
    (P229-RVW-9). Parse the answer's [n] markers (1-based, as _build_messages
    numbers them) back to blocks, in first-cited order, de-duped by block index.

    Falls back to ALL retrieved sources ONLY when the answer cited none (or cited
    no VALID index) — so the returned ``sources`` mean "what this answer was
    grounded on", the guarantee the module docstring makes, instead of over-
    claiming the whole retrieval set as its grounding. Every returned ref is still
    a real retrieved source; this only ever NARROWS, never invents."""
    seen: set = set()
    cited: List[Dict[str, Any]] = []
    for marker in _CITATION_RE.findall(answer or ""):
        idx = int(marker) - 1
        if 0 <= idx < len(blocks) and idx not in seen:
            seen.add(idx)
            cited.append(blocks[idx]["source"])
    return cited if cited else [b["source"] for b in blocks]


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

def _is_no_answer(reply: str) -> bool:
    """True when the composed reply is a declination, robust to non-bare forms
    (P229-RVW-3). The prompt asks for exactly ``NO_ANSWER``; models often add
    trailing punctuation, a caveat, or markdown emphasis. Any reply that LEADS
    with the sentinel — after uppercasing and stripping surrounding markup —
    declines; a mid-sentence mention inside a genuine answer does NOT (so a real
    answer that references the token is still returned). Exact-string equality
    let '**NO_ANSWER**' / 'NO_ANSWER — not in context' leak as a cited answer."""
    if not reply:
        return True
    normalized = reply.strip().upper().lstrip(_SENTINEL_LEADING)
    return normalized.startswith(_NO_ANSWER)


def _ref_label(source: Dict[str, Any]) -> str:
    kind = source.get("type", "source")
    for key in ("title", "source_file", "key", "id", "generated_at"):
        if source.get(key):
            return f"{kind}:{source[key]}"
    return kind


def _idstr(value: Any) -> Optional[str]:
    return None if value is None else str(value)
