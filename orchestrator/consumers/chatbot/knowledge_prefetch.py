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

F227/F085 (2 Oct, night 6): a message asking several questions got ONE search
for all of them, and under load the model answered every question from those
five passages without searching (F088KB 3/8, was 8/8). A numbered or bulleted
list of questions, or several sentences ending in "?", is now searched once per
question (up to MAX_QUESTIONS, in order), each with its own passages; the header
tells the model to search for itself any question its passages do not answer.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy import text

from core.database.read_release import release_if_read_only

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
# F227: several questions in one message, each searched on its own.
MAX_QUESTIONS = 8
MIN_QUESTION_CHARS = 8
PER_QUESTION_MIN = 2
MULTI_HEADER = (
    "Passages from this workspace's documents, found for each of the owner's questions before you answered "
    "(search_knowledge ran automatically, once per question). Answer each question from its own passages and "
    "name the file. Where a question's passages are missing or do not answer it, call search_knowledge for that "
    "question before you answer it; never answer it from another question's passages."
)
QUESTION_HEADING = "Question {number}: {question}"
NOTHING_FOR_QUESTION = "No passage cleared the relevance floor: search for it yourself before you answer it."
# Linear on any input (no lazy run before a trailing \s*, no unanchored scan for "?").
_ITEM = re.compile(r"^\s*(?:\(?\d{1,2}[.)]|\(?[a-h][.)]|[-*\u2022\u2013])\s+(?P<text>\S.*)", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[?.!])\s+|\n")

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


def asks_the_documents(message: Optional[str]) -> bool:
    """A question the documents may answer. F263 (night 7b): a question about the
    board or a card is answered from the board, whose tools give its live state;
    five passages from old reports had named done cards as waiting in Review."""
    from consumers.chatbot.board_questions import about_the_board

    return is_question(message) and not about_the_board(message)


def split_questions(message: Optional[str]) -> List[str]:
    """The separate questions of a message that asks several: a numbered or
    bulleted list, else the sentences ending in "?". [] for a single question."""
    lines = [line for line in (message or "").splitlines() if line.strip()]
    items = [m.group("text").strip() for m in (_ITEM.match(line) for line in lines) if m]
    if len(items) < 2:
        sentences = (sentence.strip() for sentence in _SENTENCE_BREAK.split(message or ""))
        items = [sentence for sentence in sentences if sentence.endswith("?")]
    items = [item for item in items if len(item) >= MIN_QUESTION_CHARS]
    return items[:MAX_QUESTIONS] if len(items) >= 2 else []


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
    # Every search that ran (one per question for several, F227): the tool loop
    # counts each, so the model's repeat of one is skipped as done.
    searches: List[Dict[str, Any]] = field(default_factory=list)

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


@dataclass(frozen=True)
class _Found:
    args: Dict[str, Any]
    kept: List[Dict[str, Any]]
    found: int
    frontend_data: Any = None


async def _search_one(search: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], query: str, limit: int,
                      min_score: float) -> _Found:
    args = {"query": query.strip()[:MAX_QUERY_CHARS], "limit": limit}
    result = await search(args) or {}
    found = [r for r in ((result.get("raw_result") or {}).get("results") or []) if isinstance(r, dict)]
    kept = [r for r in found if float(r.get("similarity") or 0.0) >= min_score]
    return _Found(args=args, kept=kept, found=len(found), frontend_data=result.get("frontend_data"))


def _passages(kept: List[Dict[str, Any]]) -> str:
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    return ToolResultFormatter.format_for_llm({"success": True, "results": kept}, PREFETCH_TOOL)


def _content_for(questions: List[str], searches: List[_Found], header: str) -> Optional[str]:
    """What the model reads: the passages of one search, or of each question under its heading."""
    if not any(found.kept for found in searches):
        return None
    if not questions:
        return f"{header}\n\n{_passages(searches[0].kept)}"
    blocks = [QUESTION_HEADING.format(number=n, question=q) + "\n" + (_passages(f.kept) if f.kept else NOTHING_FOR_QUESTION)
              for n, (q, f) in enumerate(zip(questions, searches), start=1)]
    return f"{header}\n\n" + "\n\n".join(blocks)


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
    and returns the tool router's result (``raw_result`` / ``frontend_data``).
    A message asking several questions is searched once per question (F227)."""
    if not enabled or (question_only and not asks_the_documents(message)):
        return None
    try:
        if documents_in(db, workspace_id) < 1:
            return None
    except Exception:  # noqa: BLE001 — cannot tell: the turn runs as it did
        logger.warning("[F085] retrieval first skipped: could not count documents", exc_info=True)
        return None
    # F330 (night 9c): the count opened the turn's transaction; the searches
    # (an embedding call, then a search on sessions of their own) must not keep
    # its connection "idle in transaction". Kept as is if the turn wrote.
    release_if_read_only(db)
    # The owner's own questions, each searched; a brief (F201's draft guides) is
    # searched whole: its questions are a customer's, and its query asks for the rules.
    questions = split_questions(message) if question_only else []
    queries = questions or [message or ""]
    each = max(PER_QUESTION_MIN, limit // len(queries)) if questions else limit
    started = time.monotonic()
    try:
        searches = [await _search_one(search, query, each, min_score) for query in queries]
    except Exception:  # noqa: BLE001 — the model can still search for itself
        logger.warning("[F085] retrieval first failed", exc_info=True)
        return None
    elapsed = int((time.monotonic() - started) * 1000)
    kept = [passage for found in searches for passage in found.kept]
    files = list(dict.fromkeys(str(r.get("filename") or r.get("source") or "document") for r in kept))
    logger.info(
        "[F085] retrieval first fired (ws=%s): %d search(es), %d of %d passages at or above %.2f from %d file(s) "
        "in %d ms — booked to this turn",
        workspace_id, len(searches), len(kept), sum(f.found for f in searches), min_score, len(files), elapsed,
    )
    if header is None:
        base = MULTI_HEADER if questions else PREFETCH_HEADER
        header = f"{base} {PREFETCH_DATABASE_NOTE}" if kept and _has_database(db, workspace_id) else base
    content = _content_for(questions, searches, header)
    return Prefetch(args=searches[0].args if not questions else {"query": (message or "").strip()[:MAX_QUERY_CHARS],
                                                                "limit": limit},
                    message={"role": "system", "content": content} if content else None,
                    passages=len(kept), files=files, found=sum(f.found for f in searches), elapsed_ms=elapsed,
                    frontend_data=searches[0].frontend_data, searches=[f.args for f in searches])
