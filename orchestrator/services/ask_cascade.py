"""PRD-225 — the blocked cascade behind a parked ask.

When ``platform_ask_human`` parks a subject, the work that cannot move until it
is answered is the *cascade*: the subject's transitive dependents. Two edge
sources describe it — a board task's ``parent_task_id`` tree, and a mission's
``OrchestrationTaskDependency`` DAG. Both can (in corrupt data) contain a cycle,
so every traversal here carries a visited set and terminates.

The count feeds two decisions:
  - US-002: a cascade of ``URGENT_CASCADE_THRESHOLD`` or more bypasses quiet
    hours (the baked urgent rule);
  - US-004: the Questions tab renders the cascade list (capped, "+N more").

``reachable_from`` is a pure graph walk — unit-tested with a cyclic fixture. The
DB builders below turn live rows into an adjacency map, then hand off to it, so
cycle-safety is proven once and reused.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Iterable, Mapping, Optional

from core.models.approval_grants import SUBJECT_BOARD_TASK

logger = logging.getLogger(__name__)

# The baked urgent rule (Gerard 2026-08-27): a question that transitively blocks
# this many downstream tasks bypasses quiet hours.
URGENT_CASCADE_THRESHOLD = 3


def reachable_from(adjacency: Mapping[Any, Iterable[Any]], root: Any) -> list:
    """Every distinct node reachable from ``root`` (excluding ``root``), in BFS
    order. Cycle-safe: a node is visited at most once, so a cyclic adjacency
    terminates instead of looping forever.
    """
    seen = {root}
    order: list = []
    queue: deque = deque(adjacency.get(root, []) or [])
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        queue.extend(adjacency.get(node, []) or [])
    return order


def _board_subtree_adjacency(db: Any, workspace_id: Any, root_id: int) -> dict:
    """parent→children adjacency for the transitive descendants of ``root_id``.

    Built by BFS querying children by ``parent_task_id`` so we never scan the
    whole workspace, and guarded by a visited set so corrupt parent chains can't
    loop. Keys and values are string ids (stable for ``reachable_from``).
    """
    from core.models.core import BoardTask

    adjacency: dict = {}
    seen = {str(root_id)}
    frontier = [int(root_id)]
    while frontier:
        parents, frontier = frontier, []
        for pid in parents:
            children = (
                db.query(BoardTask)
                .filter(
                    BoardTask.workspace_id == workspace_id,
                    BoardTask.parent_task_id == pid,
                )
                .all()
            )
            kids = [str(c.id) for c in children]
            adjacency[str(pid)] = kids
            for c in children:
                cid = str(c.id)
                if cid not in seen:
                    seen.add(cid)
                    frontier.append(int(c.id))
    return adjacency


def board_task_cascade(db: Any, workspace_id: Any, task_id: int) -> list:
    """The ordered list of downstream board-task ids blocked behind ``task_id``.

    Cycle-safe. Empty when the task has no descendants. US-004 caps this for
    display; US-002 takes ``len(...)`` for the urgency decision.
    """
    try:
        root = int(task_id)
    except (TypeError, ValueError):
        return []
    adjacency = _board_subtree_adjacency(db, workspace_id, root)
    return reachable_from(adjacency, str(root))


def count_downstream_blocked(
    db: Any, workspace_id: Any, subject_type: str, subject_id: Any
) -> int:
    """How many tasks are transitively blocked behind this parked subject.

    Board-task subjects walk the ``parent_task_id`` tree. Other subject kinds
    (playbook_run / tool_call) have no board-shaped dependent tree in v1 and
    return 0 — they never trip the urgent bypass, which is the conservative
    default. Never raises: a traversal fault degrades to 0 (non-urgent).
    """
    try:
        if subject_type != SUBJECT_BOARD_TASK:
            return 0
        return len(board_task_cascade(db, workspace_id, subject_id))
    except Exception:  # noqa: BLE001 — cascade sizing must never break the ask
        logger.warning(
            "[ask_cascade] downstream count failed for %s:%s",
            subject_type, subject_id, exc_info=True,
        )
        return 0


def is_urgent_cascade(count: Optional[int]) -> bool:
    """True when a cascade of ``count`` warrants the quiet-hours bypass."""
    return bool(count is not None and count >= URGENT_CASCADE_THRESHOLD)
