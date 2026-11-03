"""PRD-268 §6.11.5, K15 — ``build_calibration``: one person's ``Calibration@1``.

It reads the titles and kinds the person may see (``services.calibration.sources``: the
workspace's Deliverables and Knowledge, the person's own conversations, the installed skills'
names), and never a body, a chunk or a file. Then:

* **Nothing to read** (none of the person's work: a skill alone is not): ``empty`` true, no
  lines, the three starters, and no model call.
* **Otherwise one model call** (Auto's model, the platform ``orchestrator_llm``, booked under
  the purpose ``calibration``: customer work, the person's own action, PRD-265). It is given
  the titles under refs, quoted as data (``services.calibration.prompt``), and what it answers
  is kept only where it holds (``services.calibration.answer``): at most 3 lines of at most 160
  characters, each citing only sources it was given and carrying no percentage or confidence
  number (§12-Q26), and 3 offers naming installed skills, topped up with the starters.

The language (W2's K7) is ``language``'s table: its own, its language's, else English. The id
is ``calibration_id`` when the caller has one (the claim's row, US-C22b), else a new one. It
writes nothing. A failed call raises (the credit gate's refusal, the provider's error) and an
answer with no JSON in it raises ``CalibrationFailed``: the caller releases its claim, and
answers with words of its own (the error's text is never the person's to read). It runs on a
worker thread (a plain ``def`` route's threadpool), so the call is the manager's synchronous one.

The reads run on ``db``. When they opened its transaction (none was open on entry), it is ended
before the model call, so no connection sits idle in a transaction while the model answers; a
transaction the caller already had open is the caller's, and is left as it is (a caller with
writes of its own commits them first, as US-C22b commits its claim).

Consent comes first, and is the caller's: nothing here asks; nothing is read until it is called.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

from sqlalchemy.orm import Session

from services.calibration.answer import kept_lines, kept_offers, parse_answer
from services.calibration.model import VERSION, Calibration
from services.calibration.prompt import calibration_messages, source_refs
from services.calibration.sources import Read, has_work, read_sources
from services.calibration.strings import calibration_language, starters

__all__ = ["AUTO_SERVICE", "CALIBRATION_PURPOSE", "CalibrationFailed", "build_calibration", "calibration_llm"]

logger = logging.getLogger(__name__)

# Auto's model: the platform orchestrator_llm setting.
AUTO_SERVICE = "orchestrator"
# The call's purpose (services/credits/classification.py CUSTOMER_PURPOSES).
CALIBRATION_PURPOSE = "calibration"


class CalibrationFailed(Exception):
    """The model's answer could not be read as a calibration."""


def calibration_llm(workspace_id: Any) -> Any:
    """The calibration's model: Auto's, its call booked as ``calibration``."""
    from core.llm.manager import LLMManager

    return LLMManager(service_name=AUTO_SERVICE, workspace_id=str(workspace_id), request_type=CALIBRATION_PURPOSE)


def build_calibration(
    db: Session, workspace_id: Any, user_id: int, language: Optional[str], *, calibration_id: Optional[str] = None,
) -> Calibration:
    """The person's ``Calibration@1`` in the workspace: empty with the starters when there is
    nothing of theirs to read, else what one model call says, as far as it holds."""
    lang = calibration_language(language)
    ident = str(calibration_id or uuid4())
    opened = not db.in_transaction()
    reads = read_sources(db, workspace_id, int(user_id))
    if opened:
        db.rollback()
    if not has_work(reads):
        return Calibration(v=VERSION, id=ident, lines=[], offers=starters(lang), empty=True)
    answer = _ask(workspace_id, lang, reads)
    refs = source_refs(reads)
    return Calibration(v=VERSION, id=ident, lines=kept_lines(answer, refs), offers=kept_offers(answer, refs, lang),
                       empty=False)


def _ask(workspace_id: Any, language: str, reads: Sequence[Read]) -> Mapping[str, Any]:
    """The model's one answer, as JSON."""
    response = calibration_llm(workspace_id).generate_response_sync(calibration_messages(language, reads))
    answer = parse_answer(getattr(response, "content", None))
    if answer is None:
        logger.warning("[calibration] workspace %s: the model's answer held no JSON object", workspace_id)
        raise CalibrationFailed("The model's answer could not be read as a calibration.")
    return answer
