"""F248 (night 7): a mission step that still holds a template's placeholders after
its revision fails. It never ends "verified".

Step #0126.3 was marked verified with "[Number]" and "[Your Name/Company Name]" in
it. Mission verification is advisory: a FAIL is revised once, and a FAIL after that
revision passes through to VERIFIED with its feedback noted. Placeholders are the
exception, because a step with slots left in has not done its work. The failure
names the slots, so the owner sees what was missing and can re-plan.
"""
from __future__ import annotations

import logging
from typing import Any, List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def placeholder_failures(result: Any) -> List[str]:
    """The verdict's deterministic failures that are placeholders left in the output."""
    from core.services.placeholders import UNFINISHED

    return [f for f in getattr(result, "deterministic_failures", None) or [] if str(f).startswith(UNFINISHED)]


def fail_unfinished(db: Session, task: Any, failures: List[str]) -> None:
    """The step fails, naming the slots still in it."""
    from core.models.orchestration_enums import ActorType, FailureReasonCode, TaskState
    from services.orchestration_board_bridge import sync_board_status
    from services.orchestration_state import transition_task

    detail = " ".join(failures)[:2000]
    task.failure_reason_code = FailureReasonCode.VERIFICATION_FAIL.value
    task.failure_detail = detail
    transition_task(db=db, task=task, new_state=TaskState.FAILED, actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler", reason=f"Verification failed after its revision: {detail[:300]}")
    sync_board_status(db, task)
    logger.warning("Task %s failed verification: placeholders left after its revision (%s)", task.id, detail[:200])


__all__ = ["fail_unfinished", "placeholder_failures"]
