"""F013 (night 1): a later run's result replaces a ticket's only when it says as much.

Night 1 (2026-09-18) lost ticket #255's six delivered files when a re-claim wrote a
488-character "I produced nothing" over the real write-up. Moved out of
``api/board_tasks.py`` unchanged (F380, 7 Oct), which is over the file-size limit.
"""
from __future__ import annotations

from typing import Optional

# A later turn's result is only an improvement if it says more.
RESULT_KEEP_RATIO = 0.5


def kept_result(existing: Optional[str], incoming: Optional[str]) -> Optional[str]:
    """Whichever of the two actually reports the work.

    An incoming result replaces the old one unless it is substantially shorter —
    then the longer account is kept and the newer one appended beneath it, so
    nothing is lost either way and the ticket still shows what the last turn said.
    """
    if not incoming:
        return existing
    if not existing:
        return incoming
    if len(incoming) >= len(existing) * RESULT_KEEP_RATIO:
        return incoming
    return f"{existing}\n\n---\n\n_A later run reported:_ {incoming}"
