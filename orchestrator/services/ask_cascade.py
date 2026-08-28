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

# A finished task is NOT blocked behind the ask and does not propagate blocking,
# so terminal descendants are excluded from the cascade (P225-RVW-4). Canonical
# board-task terminals are done/failed (api/board_tasks.py VALID_STATUSES);
# 'cancelled' is included defensively.
_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})


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


def _board_children(db: Any, workspace_id: Any, parent: Any) -> list:
    """The direct downstream board tasks of ``parent``, from BOTH edge sources.

    1. ``parent_task_id`` children — the board-task subtree.
    2. the mission ``OrchestrationTaskDependency`` DAG — tasks that DEPEND ON
       this task's orchestration task (``depends_on_task_id == parent's OT id``),
       mapped back to their board tasks via ``BoardTask.orchestration_task_id``.
       A mission's steps are FLAT ``parent_task_id`` siblings under the mission
       board task (orchestration_board_bridge), so step→step blocking lives ONLY
       here — without it the cascade is 0 for the primary mission ask (P225-RVW-3).

    Returned deduped by id, order stable (tree children first). Resolved with
    equality queries per edge (mission DAGs are small; this is the cold cascade
    path, not a request-hot loop).
    """
    from core.models.core import BoardTask

    kids: list = []
    seen_ids: set = set()

    def _add(task: Any) -> None:
        if task is None:
            return
        # A terminal (done/failed/cancelled) descendant isn't blocked behind the
        # ask, and doesn't propagate blocking — skip it AND its subtree so a
        # mission-level ask over N finished steps + 1 live one counts 1, not N
        # (P225-RVW-4). Flat mission siblings make this the common case.
        if getattr(task, "status", None) in _TERMINAL_STATUSES:
            return
        tid = str(task.id)
        if tid not in seen_ids:
            seen_ids.add(tid)
            kids.append(task)

    # Source 1 — the parent_task_id tree.
    for child in (
        db.query(BoardTask)
        .filter(
            BoardTask.workspace_id == workspace_id,
            BoardTask.parent_task_id == parent.id,
        )
        .all()
    ):
        _add(child)

    # Source 2 — the mission OrchestrationTaskDependency DAG.
    ot_id = getattr(parent, "orchestration_task_id", None)
    if ot_id is not None:
        from core.models.orchestration import OrchestrationTaskDependency

        edges = (
            db.query(OrchestrationTaskDependency)
            .filter(OrchestrationTaskDependency.depends_on_task_id == ot_id)
            .all()
        )
        for edge in edges:
            dep_board = (
                db.query(BoardTask)
                .filter(
                    BoardTask.workspace_id == workspace_id,
                    BoardTask.orchestration_task_id == edge.task_id,
                )
                .first()
            )
            _add(dep_board)

    return kids


def _board_subtree_adjacency(db: Any, workspace_id: Any, root_id: int) -> dict:
    """board→children adjacency for the transitive dependents of ``root_id``,
    merging BOTH edge sources — the ``parent_task_id`` tree AND the mission
    ``OrchestrationTaskDependency`` DAG (P225-RVW-3, see ``_board_children``).

    BFS over board-task rows (each row is needed to read its
    ``orchestration_task_id``), guarded by a single visited set so a cycle in
    EITHER source terminates. Keys and values are string ids (stable for
    ``reachable_from``). An absent root row ⇒ empty cascade.
    """
    from core.models.core import BoardTask

    root = (
        db.query(BoardTask)
        .filter(BoardTask.id == int(root_id), BoardTask.workspace_id == workspace_id)
        .first()
    )
    if root is None:
        return {}

    adjacency: dict = {}
    seen = {str(root.id)}
    frontier = [root]
    while frontier:
        parents, frontier = frontier, []
        for parent in parents:
            children = _board_children(db, workspace_id, parent)
            adjacency[str(parent.id)] = [str(c.id) for c in children]
            for child in children:
                cid = str(child.id)
                if cid not in seen:
                    seen.add(cid)
                    frontier.append(child)
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


def board_task_cascade_detail(
    db: Any, workspace_id: Any, task_id: Any, cap: int = 6
) -> dict:
    """The cascade behind a board-task question, shaped for the Questions tab:
    ``{"total": N, "tasks": [{id, title, status}, … up to ``cap``]}``.

    Order is preserved from the cycle-safe BFS; the tab renders the first ``cap``
    and shows "+N more". Never raises — a fault degrades to an empty cascade.

    The shown tasks load in ONE ``id.in_(...)`` query, not a per-id loop — this
    runs per question row (up to 200) on the 30s-polled grants list (P225-RVW-4).
    """
    from core.models.core import BoardTask

    try:
        ids = board_task_cascade(db, workspace_id, task_id)
    except Exception:  # noqa: BLE001
        logger.warning(
            "[ask_cascade] cascade detail failed for board_task %s", task_id,
            exc_info=True,
        )
        return {"total": 0, "tasks": []}

    shown = ids[: max(0, cap)]
    tasks: list = []
    if shown:
        try:
            int_ids = [int(t) for t in shown]
            rows = (
                db.query(BoardTask)
                .filter(
                    BoardTask.workspace_id == workspace_id,
                    BoardTask.id.in_(int_ids),
                )
                .all()
            )
            by_id = {str(r.id): r for r in rows}
            for tid in shown:  # preserve the cascade (BFS) order
                r = by_id.get(tid)
                if r is not None:
                    tasks.append({"id": r.id, "title": r.title, "status": r.status})
        except Exception:  # noqa: BLE001 — display detail must never break the list
            logger.warning(
                "[ask_cascade] cascade detail load failed for board_task %s", task_id,
                exc_info=True,
            )
            return {"total": len(ids), "tasks": []}
    return {"total": len(ids), "tasks": tasks}


def count_downstream_blocked(
    db: Any, workspace_id: Any, subject_type: str, subject_id: Any
) -> int:
    """How many tasks are transitively blocked behind this parked subject.

    Board-task subjects walk BOTH the ``parent_task_id`` tree and the mission
    ``OrchestrationTaskDependency`` DAG (P225-RVW-3). Other subject kinds
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
