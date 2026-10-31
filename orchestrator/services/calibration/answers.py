"""PRD-268 §6.11.5, K15, K16 — the person's answer to their calibration: Reject a line, or Sounds right.

Only the calibration's own person answers it, in its own workspace (``claim.own_record``):
anyone else's, another workspace's, a declined one and one not built yet (neither has lines) are
``NotFound``.

* **Reject** one line: it is marked rejected while the calibration is open, committed, then one
  correction is written (``services.calibration.memory``). Rejecting a line already rejected
  changes nothing and writes nothing; rejecting one once the calibration was answered is
  ``AlreadyAnswered``.
* **Sounds right**: every line not rejected is accepted and the calibration is answered,
  committed, then one memory per accepted line is written. Answering it again changes nothing
  and writes nothing.

The state change comes first, under the row's lock, so two answers at once write each memory
once; it is committed before any memory write, so no transaction is open while the memory
service is called. A write that fails gives the state change back (K16: the line's state does
not change) and raises ``NotRemembered`` (the route answers 503). A line whose content trips
an exclusion rule is never written: it is still rejected or accepted, and the answer lists it
under ``not_stored`` (only the rule's name is logged). Each answer that changed something writes
an audit row after its commit.

What committing first costs, knowingly (idle transactions across a network call are worse): for
the length of one memory write, a second answer sees the first's state change, so it answers as
if that change held (a second Sounds right answers ``answered``; an accept leaves out a line whose
rejection is still being written) even when the first write then fails and is given back. A
process that dies between the commit and the write leaves the change with no memory behind it.
Writes already made when a later one fails stay; a retry writes them again, and the durable store
keeps one copy of the same content per owner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Sequence, Tuple

from sqlalchemy.orm import Session

from core.models.proactive import CALIBRATION_DECLINED, CALIBRATION_OPEN, ProactiveCalibration
from services.calibration import claim, memory
from services.calibration.audit import ACCEPTED, REJECTED, audit
from services.calibration.model import Calibration

__all__ = ["AlreadyAnswered", "Answered", "CALIBRATION_NOT_FOUND", "LINE_NOT_FOUND", "NotFound", "NotRemembered",
           "accept", "reject_line"]

logger = logging.getLogger(__name__)

CALIBRATION_NOT_FOUND = "calibration_not_found"
LINE_NOT_FOUND = "line_not_found"


class NotFound(Exception):
    """No such calibration of the person's, or no such line in it; ``code`` says which."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class AlreadyAnswered(Exception):
    """The calibration was answered (Sounds right): its lines can no longer be rejected."""


class NotRemembered(Exception):
    """A memory was not written; the state change was given back."""


@dataclass(frozen=True)
class Answered:
    """The record after the answer, and the lines whose memory was not stored (an exclusion rule)."""

    record: Dict[str, Any]
    not_stored: Tuple[str, ...] = ()


def _own(db: Session, workspace_id: Any, user_id: int, calibration_id: Any) -> ProactiveCalibration:
    record = claim.own_record(db, workspace_id, user_id, calibration_id)
    if record is None or record.status == CALIBRATION_DECLINED or not record.calibration:
        raise NotFound(CALIBRATION_NOT_FOUND)
    return record


def _lines(record: ProactiveCalibration) -> Dict[str, str]:
    """The record's lines by id, as posted."""
    return {line.id: line.text for line in Calibration.model_validate(record.calibration).lines}


def _now_as_it_is(db: Session, workspace_id: Any, user_id: int, calibration_id: Any) -> Dict[str, Any]:
    return claim.view(_own(db, workspace_id, user_id, calibration_id))


def _kept(writes: Sequence[memory.MemoryWrite]) -> Tuple[List[memory.MemoryWrite], Tuple[str, ...]]:
    """The writes to store, and the line ids an exclusion rule keeps out (the rule logged, never the content)."""
    kept: List[memory.MemoryWrite] = []
    refused: List[str] = []
    for write in writes:
        rule = memory.excluded(write)
        if rule is None:
            kept = [*kept, write]
        else:
            logger.info("[calibration] %s of %s is not remembered: it trips %s", write.line_id, write.calibration_id,
                        rule)
            refused = [*refused, write.line_id]
    return kept, tuple(refused)


def _given_back(db: Session, give_back: Callable[[], None]) -> NotRemembered:
    db.rollback()
    give_back()
    return NotRemembered("a calibration memory was not written")


def _remember(db: Session, writes: Sequence[memory.MemoryWrite], give_back: Callable[[], None]) -> None:
    """Write each memory; when one fails, give the state change back and raise ``NotRemembered``
    (the store logged its own failure; anything else is logged here)."""
    try:
        for write in writes:
            memory.remember(write)
    except memory.MemoryNotStored as failure:
        raise _given_back(db, give_back) from failure
    except Exception as failure:
        logger.exception("[calibration] a memory write failed; the answer is given back")
        raise _given_back(db, give_back) from failure


def reject_line(db: Session, workspace_id: Any, user_id: int, calibration_id: Any, line_id: str) -> Answered:
    """Reject one line of the person's calibration and remember the correction."""
    record = _own(db, workspace_id, user_id, calibration_id)
    ident, text = record.id, _lines(record).get(line_id)
    if text is None:
        raise NotFound(LINE_NOT_FOUND)
    if line_id in (record.rejected_lines or []):
        return Answered(claim.view(record))
    if record.status != CALIBRATION_OPEN:
        raise AlreadyAnswered("the calibration was answered")
    if not claim.mark_rejected(db, workspace_id, user_id, ident, line_id):
        now = _now_as_it_is(db, workspace_id, user_id, ident)
        if line_id in now["rejected_lines"]:
            return Answered(now)
        raise AlreadyAnswered("the calibration was answered")
    kept, refused = _kept([memory.correction_write(workspace_id, user_id, ident, line_id, text)])
    _remember(db, kept, lambda: claim.unmark_rejected(db, workspace_id, user_id, ident, line_id))
    audit(db, workspace_id, user_id, REJECTED, ident, {"line_id": line_id, "stored": not refused})
    return Answered(_now_as_it_is(db, workspace_id, user_id, ident), refused)


def accept(db: Session, workspace_id: Any, user_id: int, calibration_id: Any, now: datetime) -> Answered:
    """Sounds right: accept every line not rejected and remember each."""
    record = _own(db, workspace_id, user_id, calibration_id)
    if record.status != CALIBRATION_OPEN:
        return Answered(claim.view(record))
    ident, lines = record.id, _lines(record)
    accepted = claim.mark_answered(db, workspace_id, user_id, ident, list(lines), now)
    if accepted is None:
        return Answered(_now_as_it_is(db, workspace_id, user_id, ident))
    writes = [memory.line_write(workspace_id, user_id, ident, line_id, lines[line_id]) for line_id in accepted]
    kept, refused = _kept(writes)
    _remember(db, kept, lambda: claim.unmark_answered(db, workspace_id, user_id, ident))
    audit(db, workspace_id, user_id, ACCEPTED, ident, {"accepted": len(accepted), "stored": len(kept)})
    return Answered(_now_as_it_is(db, workspace_id, user_id, ident), refused)
