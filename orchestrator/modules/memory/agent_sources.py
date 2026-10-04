"""F325 (night 9b): a memory that names an agent's report as where a fact came from is
never handed to Auto or an agent as the owner's facts.

Night 9b, chat 99a25490 (MORNING-REPORT.md L13, Friction 6): asked "Where did that come
from?" about Quay's November terms, Auto searched its memory (platform_search_memory)
and answered "I found that information in my memory, which was stored from a document
titled '2026-10-04_125617_e5c675_task-payment-terms-for-cafés.md'": ticket #1851's
answer, filed on night 9 as document #1547 (source_type agent_output). F305 took an
agent's documents out of every document search, not out of memory. On night 9 (chat
9a128fe3, before F305) Auto had cited that file in answer to the same question, and a
chat turn's durable facts are distilled into memory
(consumers/chatbot/smart_memory.store_conversation; a promoted transcript keeps the
words too), so the citation outlived the search fix: every memory search and every
memory put into a prompt passed it on.

Now a memory that names a document an agent wrote is left out of what Auto and the
agents read: by its filed name (a report's ``YYYY-MM-DD_HHMMSS_xxxxxx_<title>.md``, a
mission's ``mission-output-<goal>.md``) everywhere, and, where a database session is
at hand, by the name of any agent_output document in the workspace:
- platform_search_memory and platform_browse_memories (handlers_search) leave it out
  and say how many they left out;
- the memories put into a prompt (``injection_filter.filter_injectable_memories``:
  Auto's chat recall and an agent's context) leave it out.
Nothing is deleted: the owner still sees such a memory on the Memory page.
"""
from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Iterable, List, Set
from uuid import UUID

# services/report_service.py files a report as "<date>_<time>_<6 hex>_<title>.md";
# services/coordinator_service.py files a mission's output as "mission-output-<goal>.md".
_FILED_BY_AN_AGENT = re.compile(r"\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{6}_\S+|mission-output-[\w-]+\.md", re.IGNORECASE)
_REPORT_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{6}_")
# A document's name without its filing prefix and extension still names it when it is
# this long ("task-payment-terms-for-cafés"); a shorter one could be anybody's word.
MIN_NAME_CHARS = 12
NAMES_READ = 2000
LEFT_OUT_NOTE = ("{count} memories were left out: each names an agent's report or output as where a fact came "
                 "from, and an agent's writing is never the owner's facts. The owner sees them on the Memory page.")


def memory_text(memory: Any) -> str:
    """A memory row's words, whichever shape it takes (Mem0's ``memory``, L2's ``content``)."""
    if not isinstance(memory, dict):
        return ""
    return str(memory.get("memory") or memory.get("content") or "")


def names_an_agents_document(text: Any, names: Iterable[str] = ()) -> bool:
    """Whether ``text`` names a document an agent wrote: by its filed name, or by one of
    ``names`` (lower case, from ``agents_document_names``)."""
    said = str(text or "").lower()
    if not said:
        return False
    return bool(_FILED_BY_AN_AGENT.search(said)) or any(name in said for name in names)


def owners_memories(memories: Iterable[Any], names: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """The memories (dicts) that name no document an agent wrote, in their order."""
    known = frozenset(names)
    return [m for m in memories if isinstance(m, dict) and not names_an_agents_document(memory_text(m), known)]


def left_out_note(count: int) -> Dict[str, Any]:
    """What a memory tool's answer says about the memories it left out; nothing when none."""
    return {"left_out": count, "left_out_note": LEFT_OUT_NOTE.format(count=count)} if count else {}


def _forms(name: Any) -> Set[str]:
    """A document's name as a memory may give it: whole, and without its filing prefix
    and extension when that is still long enough to be its own."""
    whole = str(name or "").strip().lower()
    if not whole:
        return set()
    bare = _REPORT_PREFIX.sub("", whole).rsplit(".", 1)[0]
    return {whole} | ({bare} if len(bare) >= MIN_NAME_CHARS else set())


def agents_document_names(db: Any, workspace_id: Any) -> FrozenSet[str]:
    """The names of the workspace's documents an agent wrote (newest first, up to
    ``NAMES_READ``), lower case; none without a database session or a workspace."""
    from sqlalchemy.orm import Session

    if not isinstance(db, Session) or workspace_id is None:
        return frozenset()
    from core.models.core import Document
    from services.knowledge_flywheel import AGENT_OUTPUT_SOURCE_TYPE

    rows = (db.query(Document.filename, Document.original_filename)
            .filter(Document.workspace_id == UUID(str(workspace_id)),
                    Document.source_type == AGENT_OUTPUT_SOURCE_TYPE)
            .order_by(Document.id.desc()).limit(NAMES_READ).all())
    names: Set[str] = set()
    for filename, original in rows:
        names |= _forms(filename) | _forms(original)
    return frozenset(names)


__all__ = ["LEFT_OUT_NOTE", "agents_document_names", "left_out_note", "memory_text", "names_an_agents_document",
           "owners_memories"]
