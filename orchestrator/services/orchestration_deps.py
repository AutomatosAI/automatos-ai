"""
Orchestration Dependency Resolver — PRD-82A
============================================

DAG validation and topological ordering for task dependencies within a mission.

Uses stdlib `graphlib.TopologicalSorter` for cycle detection and ordering.
Provides `get_ready_tasks()` to find tasks whose dependencies are all met.

Source: PRD-82A Section 4, PRD-101 Section 5.5
"""

import logging
from graphlib import CycleError, TopologicalSorter
from typing import Any, List, Sequence
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from core.models.orchestration import OrchestrationTask, OrchestrationTaskDependency
from core.models.orchestration_enums import TaskState, TERMINAL_TASK_STATES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CyclicDependencyError(Exception):
    """Raised when task dependencies contain a cycle."""

    def __init__(self, detail: str = "Task dependency graph contains a cycle"):
        self.detail = detail
        super().__init__(detail)


class InvalidDependencyError(Exception):
    """Raised when a dependency references a non-existent task."""

    def __init__(self, task_id: Any, invalid_dep_id: Any):
        self.task_id = task_id
        self.invalid_dep_id = invalid_dep_id
        super().__init__(
            f"Task {task_id} depends on non-existent task {invalid_dep_id}"
        )


# ---------------------------------------------------------------------------
# DependencyResolver
# ---------------------------------------------------------------------------


class DependencyResolver:
    """
    Validates task dependency DAGs and resolves execution ordering.

    Stateless — all data comes from arguments or DB queries.
    """

    @staticmethod
    def validate_task_graph(
        task_ids: Sequence[UUID],
        deps: Sequence[OrchestrationTaskDependency],
    ) -> None:
        """
        Validate that tasks and dependencies form a valid DAG.

        Checks:
        - No cycles (via graphlib TopologicalSorter)
        - All dependency references point to valid task IDs
        - No orphan tasks (all tasks reachable or are roots)

        Args:
            task_ids: Sequence of task UUIDs in the mission.
            deps: Sequence of OrchestrationTaskDependency edges.

        Raises:
            CyclicDependencyError: If the graph contains a cycle.
            InvalidDependencyError: If a dep references a non-existent task ID.
        """
        valid_ids = frozenset(task_ids)

        # Build adjacency: node → set of predecessors (dependencies)
        graph: dict[UUID, set[UUID]] = {tid: set() for tid in task_ids}

        for dep in deps:
            # Validate both ends exist
            if dep.task_id not in valid_ids:
                raise InvalidDependencyError(dep.task_id, dep.depends_on_task_id)
            if dep.depends_on_task_id not in valid_ids:
                raise InvalidDependencyError(dep.task_id, dep.depends_on_task_id)

            graph[dep.task_id].add(dep.depends_on_task_id)

        # Cycle detection via TopologicalSorter
        sorter = TopologicalSorter(graph)
        try:
            sorter.prepare()
        except CycleError as e:
            logger.warning("Cyclic dependency detected: %s", e)
            raise CyclicDependencyError(str(e)) from e

        logger.debug(
            "Task graph validated: %d tasks, %d edges, no cycles",
            len(task_ids),
            len(deps),
        )

    @staticmethod
    def get_ready_tasks(db: Session, run_id: UUID) -> List[OrchestrationTask]:
        """
        Find tasks ready for dispatch: state='pending' with all dependencies met.

        A dependency is met when the upstream task is in the `verified` state
        (terminal success). For tasks with NO dependencies, they are immediately
        ready if in `pending` state.

        Args:
            db: SQLAlchemy session.
            run_id: The orchestration run to query.

        Returns:
            List of OrchestrationTask ordered by sequence_number.
        """
        # All pending tasks for this run
        pending_tasks = (
            db.query(OrchestrationTask)
            .filter(
                and_(
                    OrchestrationTask.run_id == run_id,
                    OrchestrationTask.state == TaskState.PENDING.value,
                )
            )
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        if not pending_tasks:
            return []

        # Get all dependencies for this run's tasks in one query
        pending_ids = [t.id for t in pending_tasks]
        deps = (
            db.query(OrchestrationTaskDependency)
            .filter(OrchestrationTaskDependency.task_id.in_(pending_ids))
            .all()
        )

        # Build dep map: task_id → set of depends_on_task_ids
        dep_map: dict[UUID, set[UUID]] = {}
        for dep in deps:
            dep_map.setdefault(dep.task_id, set()).add(dep.depends_on_task_id)

        # Check which upstream tasks are in verified state
        all_upstream_ids = set()
        for upstream_set in dep_map.values():
            all_upstream_ids.update(upstream_set)

        verified_ids: set[UUID] = set()
        if all_upstream_ids:
            verified_tasks = (
                db.query(OrchestrationTask.id)
                .filter(
                    and_(
                        OrchestrationTask.id.in_(all_upstream_ids),
                        OrchestrationTask.state == TaskState.VERIFIED.value,
                    )
                )
                .all()
            )
            verified_ids = {row[0] for row in verified_tasks}

        # A task is ready if ALL its dependencies are verified
        ready = []
        for task in pending_tasks:
            upstream = dep_map.get(task.id, set())
            if upstream.issubset(verified_ids):
                ready.append(task)

        logger.debug(
            "get_ready_tasks(run=%s): %d pending, %d ready",
            run_id,
            len(pending_tasks),
            len(ready),
        )

        return ready

    @staticmethod
    def get_topological_order(
        task_ids: Sequence[UUID],
        deps: Sequence[OrchestrationTaskDependency],
    ) -> List[UUID]:
        """
        Return task IDs in a valid topological execution order.

        Args:
            task_ids: Sequence of task UUIDs.
            deps: Sequence of dependency edges.

        Returns:
            List of task UUIDs in topological order.

        Raises:
            CyclicDependencyError: If the graph contains a cycle.
        """
        # Build adjacency: node → set of predecessors
        graph: dict[UUID, set[UUID]] = {tid: set() for tid in task_ids}
        for dep in deps:
            graph[dep.task_id].add(dep.depends_on_task_id)

        sorter = TopologicalSorter(graph)
        try:
            order = list(sorter.static_order())
        except CycleError as e:
            raise CyclicDependencyError(str(e)) from e

        return order
