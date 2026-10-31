"""PRD-268 §6.11.5, K16 — what the person's answer to their calibration remembers.

Through the existing memory service only (``modules.memory.unified_memory_service``
``store_long_term``), its metadata built by the one write contract
(``modules.memory.write_contract.build_memory_metadata``):

* **Sounds right** stores each line not rejected, as the person confirmed it: the line is the
  content, fact_type ``user_fact``, owner ``user:<id>`` (so the scope defaults to private: only
  they recall it), source_type ``platform_verified`` (the person confirmed it), extra
  {origin ``calibration``, calibration_id, line_id}.
* **Reject** stores one correction: content ``Not so: <the line>``, the same type, owner, scope
  and source, extra {origin, calibration_id, line_id, correction: true}.

Every content first passes ``violates_exclusions`` (the memory system's one consent gate): one
that names a secret is never stored, and the caller says so; only the rule's name is logged,
never the content. ``subject_id`` is ``user:<id>`` (PRD-196: the person's memories are erased
together). A write the service refuses (its answer carries an error, or success false: no
durable store set up, a timeout) or that raises is ``MemoryNotStored``: the caller gives its
state change back and answers 503. The durable store keeps one copy of the same content per
owner (it dedups by content hash), so writing a line again after a failure adds none.

``remember`` runs ``store`` on the app's event loop (``anyio.from_thread.run``) from a plain
``def`` route's worker thread: the memory service's clients live on that loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import anyio

from modules.memory.forget import subject_of
from modules.memory.write_contract import build_memory_metadata, violates_exclusions

__all__ = [
    "CORRECTION_PREFIX", "MemoryNotStored", "MemoryWrite", "correction_write", "excluded", "line_write",
    "memory_metadata", "memory_service", "remember", "store",
]

logger = logging.getLogger(__name__)

ORIGIN = "calibration"
# The person's own facts: private to them once an owner is set (write_contract.PRIVATE_DEFAULT_TYPES).
FACT_TYPE = "user_fact"
# The person confirmed it, or said it was wrong: it is not a model's guess.
SOURCE_TYPE = "platform_verified"
CORRECTION_PREFIX = "Not so: "


class MemoryNotStored(Exception):
    """The memory service did not store a calibration answer."""


@dataclass(frozen=True)
class MemoryWrite:
    """One memory an answer writes: a line the person confirmed, or the correction of one."""

    workspace_id: str
    user_id: int
    calibration_id: str
    line_id: str
    content: str
    correction: bool = False

    @property
    def subject(self) -> Optional[str]:
        """The person as the memory's owner and data subject: ``user:<users.id>``."""
        return subject_of(self.user_id)


def line_write(workspace_id: Any, user_id: int, calibration_id: Any, line_id: str, text: str) -> MemoryWrite:
    """The memory of a line the person confirmed (Sounds right)."""
    return MemoryWrite(str(workspace_id), int(user_id), str(calibration_id), str(line_id), str(text))


def correction_write(workspace_id: Any, user_id: int, calibration_id: Any, line_id: str, text: str) -> MemoryWrite:
    """The correction of a line the person rejected: ``Not so: <the line>``."""
    return MemoryWrite(str(workspace_id), int(user_id), str(calibration_id), str(line_id),
                       CORRECTION_PREFIX + str(text), correction=True)


def excluded(write: MemoryWrite) -> Optional[str]:
    """The exclusion rule the write's content trips (it is never stored), else None."""
    return violates_exclusions(write.content)


def memory_metadata(write: MemoryWrite) -> Dict[str, Any]:
    """The write's metadata, through the one write contract (K16)."""
    extra: Dict[str, Any] = {"origin": ORIGIN, "calibration_id": write.calibration_id, "line_id": write.line_id}
    if write.correction:
        extra = {**extra, "correction": True}
    return build_memory_metadata(fact_type=FACT_TYPE, source_type=SOURCE_TYPE, owner=write.subject, extra=extra)


def memory_service() -> Any:
    """The memory service (its singleton)."""
    from modules.memory.unified_memory_service import get_unified_memory_service

    return get_unified_memory_service()


def _refused(result: Any) -> bool:
    return not isinstance(result, Mapping) or bool(result.get("error")) or result.get("success") is False


async def store(write: MemoryWrite) -> None:
    """Store ``write`` in the person's long-term memory. Raises ``ValueError`` for content an
    exclusion rule refuses (the caller checks first) and ``MemoryNotStored`` when the service
    did not store it (logged)."""
    rule = excluded(write)
    if rule is not None:
        raise ValueError(f"a calibration memory that trips {rule!r} is never stored")
    try:
        result = await memory_service().store_long_term(
            workspace_id=write.workspace_id, content=write.content, metadata=memory_metadata(write),
            subject_id=write.subject)
    except Exception as failure:
        logger.exception("[calibration] workspace %s: the memory of %s was not stored", write.workspace_id,
                         write.line_id)
        raise MemoryNotStored("the memory service failed") from failure
    if _refused(result):
        # Never the service's error text: only that it refused, so no content can reach the log.
        logger.warning("[calibration] workspace %s: the memory service did not store %s of %s", write.workspace_id,
                       write.line_id, write.calibration_id)
        raise MemoryNotStored("the memory service did not store it")


def remember(write: MemoryWrite) -> None:
    """``store`` on the app's event loop, from a plain ``def`` route's worker thread."""
    anyio.from_thread.run(store, write)
