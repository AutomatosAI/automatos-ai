"""F085-A (night 3): retrieval first — a question in a workspace that has
documents is searched before the model's first call.

Night 3's RAG test put 40 product questions to Auto in a workspace holding the
product's own 54-page manual: one search_knowledge call in 40, twenty
find_tools calls, 14/40 right — it answered from its own idea of the product.
Now, when the workspace has documents and the owner's message is a question,
the turn runs search_knowledge (the very call the model would make: the agent's
team scope, the documents' real names, the F088 citations) before the first
model call, and the passages that clear a relevance floor go into the prompt,
cited by file. The model can still search again; a near-identical query is
skipped as already done.

Cost: one embedding and one search per question turn in a workspace with
documents — booked to the turn in llm_usage like any search, logged here, and
shown in the reply's activity trail. Dial: chatbot.knowledge_prefetch (on
unless set false); off is the path as it was.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

PREFETCH_TOOL = "search_knowledge"
MAX_QUERY_CHARS = 1000
PREFETCH_HEADER = (
    "Passages from this workspace's documents, found for the owner's question before you answered "
    "(search_knowledge ran automatically). Answer from them where they apply and name the file; call "
    "search_knowledge again if they do not cover the question."
)
# F077/F078 (refresh-3 retest): with a database connected, a summary document's
# figure answered the number questions (415 for 400, 18 for 19); the database was
# never asked. The passages stay; for a number they say where the number lives.
PREFETCH_DATABASE_NOTE = (
    "This workspace also has a connected database: for current counts, totals or rankings, call "
    "smart_query_database rather than answering from these passages; a document's figure may be out of date."
)

# An instruction, even one phrased as a question ("Can you create an agent?").
_INSTRUCTION = re.compile(
    r"^\s*(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:create|make|build|write|draft|send|email|message|delete|remove|add|schedule|run|launch|start|stop|"
    r"cancel|assign|update|set|change|connect|install|generate|post|publish|book|upload|move|rename|invite|"
    r"fix|deploy)\b",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(
    r"^\s*(?:(?:and|so|ok|okay|hey|hi|also|quick ones?)\b[\s,:—-]*)?(?:please\s+)?"
    r"(?:what|which|who|whom|whose|when|where|why|how|is|are|was|were|does|do|did|can|could|should|would|"
    r"will|may|has|have|had|tell me|explain|describe|remind me|look (?:in|up|through|at)|"
    r"check (?:my|the|our)|search (?:my|the|our)|find (?:out|the|my|our))\b",
    re.IGNORECASE,
)

# Asking, anywhere in the message ("Quick ones — … please answer each …").
_ASKING = re.compile(
    r"\b(?:answer (?:each|these|this|them|the following)|tell me (?:where|what|which|how|when|who|why|if|whether)|"
    r"do you know|i want to (?:know|check))\b",
    re.IGNORECASE,
)


def is_question(message: Optional[str]) -> bool:
    """A message asking something, not one telling Auto to do something."""
    t = (message or "").strip()
    if len(t) < 4 or _INSTRUCTION.match(t):
        return False
    return "?" in t or bool(_QUESTION_START.match(t)) or bool(_ASKING.search(t))


def documents_in(db: Any, workspace_id: Any) -> int:
    """Searchable documents in the workspace."""
    row = db.execute(
        text("SELECT count(*) FROM documents WHERE workspace_id = CAST(:ws AS uuid) "
             "AND status IN ('completed', 'processed')"),
        {"ws": str(workspace_id)},
    ).first()
    return int(row[0]) if row else 0


@dataclass(frozen=True)
class Prefetch:
    args: Dict[str, Any]
    message: Optional[Dict[str, str]]     # what the model reads; None when nothing cleared the floor
    passages: int
    files: List[str]
    found: int
    elapsed_ms: int
    frontend_data: Any = None

    @property
    def summary(self) -> str:
        """The activity-trail line."""
        if not self.passages:
            return f"searched automatically — nothing above the relevance floor ({self.found} found)"
        names = ", ".join(self.files[:3]) + (" …" if len(self.files) > 3 else "")
        return f"{self.passages} passage{'s' if self.passages != 1 else ''} from {names} — searched automatically"


def _has_database(db: Any, workspace_id: Any) -> bool:
    """Whether the workspace has an active database source (the documents
    section's own reading). Cannot tell: the passages go as they did."""
    from modules.context.sections.documents_inventory import connected_databases

    try:
        return bool(connected_databases(db, workspace_id))
    except Exception:  # noqa: BLE001 — a note must never be why a turn fails
        logger.warning("[F085] retrieval first: could not read the database sources", exc_info=True)
        return False


async def prefetch(
    db: Any,
    workspace_id: Any,
    message: Optional[str],
    *,
    search: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    enabled: bool,
    limit: int,
    min_score: float,
    question_only: bool = True,
    header: Optional[str] = None,
) -> Optional[Prefetch]:
    """Search the documents for a question before the model answers, or None
    when this turn does not qualify (dial off, not a question, no documents)
    or the search failed. ``search`` runs search_knowledge with the given args
    and returns the tool router's result (``raw_result`` / ``frontend_data``)."""
    if not enabled or (question_only and not is_question(message)):
        return None
    try:
        if documents_in(db, workspace_id) < 1:
            return None
    except Exception:  # noqa: BLE001 — cannot tell: the turn runs as it did
        logger.warning("[F085] retrieval first skipped: could not count documents", exc_info=True)
        return None
    args = {"query": (message or "").strip()[:MAX_QUERY_CHARS], "limit": limit}
    started = time.monotonic()
    try:
        result = await search(args) or {}
    except Exception:  # noqa: BLE001 — the model can still search for itself
        logger.warning("[F085] retrieval first failed", exc_info=True)
        return None
    elapsed = int((time.monotonic() - started) * 1000)
    found = [r for r in ((result.get("raw_result") or {}).get("results") or []) if isinstance(r, dict)]
    kept = [r for r in found if float(r.get("similarity") or 0.0) >= min_score]
    files = list(dict.fromkeys(str(r.get("filename") or r.get("source") or "document") for r in kept))
    logger.info(
        "[F085] retrieval first fired (ws=%s): %d of %d passages at or above %.2f from %d file(s) in %d ms — "
        "one embedding and one search, booked to this turn",
        workspace_id, len(kept), len(found), min_score, len(files), elapsed,
    )
    message_for_model = None
    if kept:
        from modules.tools.formatting.result_formatter import ToolResultFormatter

        body = ToolResultFormatter.format_for_llm({"success": True, "results": kept}, PREFETCH_TOOL)
        if header is None:
            header = f"{PREFETCH_HEADER} {PREFETCH_DATABASE_NOTE}" if _has_database(db, workspace_id) else PREFETCH_HEADER
        message_for_model = {"role": "system", "content": f"{header}\n\n{body}"}
    return Prefetch(args=args, message=message_for_model, passages=len(kept), files=files, found=len(found),
                    elapsed_ms=elapsed, frontend_data=result.get("frontend_data"))
