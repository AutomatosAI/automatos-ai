"""Same-action-loop breaker for agent execution (PRD-161 S4).

Ported from the OpenHands StuckDetector heuristic: when an agent repeats the
exact same action (tool + arguments) several times in a row — typically a
failing call it keeps retrying — it is stuck. Detecting that and breaking the
loop stops the agent burning its whole tool-iteration budget (and the spend)
on a dead end.

Pure and dependency-free so it is trivially unit-testable and cheap to call on
every tool step.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

DEFAULT_STUCK_THRESHOLD = 3


def action_key(tool_name: str, tool_args: Any) -> str:
    """Stable fingerprint of one action (tool name + arguments).

    Args are JSON-normalised (sorted keys) so logically identical calls hash
    identically regardless of dict ordering; unserialisable args fall back to
    ``repr``.
    """
    try:
        arg_repr = json.dumps(tool_args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        arg_repr = repr(tool_args)
    digest = hashlib.sha1(f"{tool_name}:{arg_repr}".encode()).hexdigest()
    return digest[:16]


class StuckDetector:
    """Flags a stuck agent when the same action repeats ``threshold`` times.

    Usage inside a tool-iteration loop::

        detector = StuckDetector()
        for step in range(max_iters):
            ... pick tool ...
            detector.record(action_key(tool_name, tool_args))
            if detector.is_stuck():
                break  # same action N times in a row — abandon the loop
    """

    def __init__(self, threshold: int = DEFAULT_STUCK_THRESHOLD):
        if threshold < 2:
            raise ValueError("stuck threshold must be >= 2")
        self._threshold = threshold
        self._history: List[str] = []

    def record(self, key: str) -> None:
        self._history.append(key)

    def is_stuck(self) -> bool:
        """True once the last ``threshold`` recorded actions are all identical."""
        if len(self._history) < self._threshold:
            return False
        recent = self._history[-self._threshold:]
        return len(set(recent)) == 1

    def reset(self) -> None:
        self._history.clear()
