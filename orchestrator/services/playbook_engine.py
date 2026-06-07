"""PRD-142 Wave 3 · WS-3R · W3-S12 — Playbook engine (consolidated seam).

Why this module exists
----------------------
Pre-W3-S12 the 7 backend launch sites for a Playbook (formerly "recipe")
each imported the legacy launch functions directly from
``api.recipe_executor``. That made the executor's call sites the only
consolidation seam — any future change to durability / retry / streaming
required edits at 7 places, and an out-of-band launch could silently bypass
the model (PRD §6 W3-S12: "scattered, not unified").

``PlaybookEngine`` is the **stable interface** the 7 sites collapse onto. It
wraps the existing entry points ``launch_recipe_task`` /
``execute_recipe_direct`` without rewriting them — the Wave 2 net guards
their behaviour, and the W3-S12 charter is consolidate-and-harden, never
rebuild (PRD §16, zero rewrites locked).

What it adds today
------------------
- One canonical seam every caller goes through (the strangler-fig).
- ``workspace_id`` is carried as an explicit kwarg end-to-end — never
  guessed or defaulted (A4).
- The Mission durability model is ported by piggybacking on
  ``core/boot/reaper.py`` (W3-S12 extends it with a ``RecipeExecution``
  surface — see ``_reap_recipe_executions``). On restart, in-flight rows
  past the staleness window are marked terminal instead of orphaned, so a
  user's playbook can no longer silently die fire-and-forget (§H DoD
  #3 Restart-safe).
- The executor (``api/recipe_executor.py``) emits the
  ``playbooks`` primitive heartbeat at its terminal transitions via
  ``services/playbook_engine_heartbeat.py`` — the W3-S1 helper. The
  playbooks tile reflects real-time outcome (§H DoD #4 Observable + #7
  Dashboard tile).

What it does NOT do (out of agent scope)
----------------------------------------
- **Does NOT delete** ``api/workflow_recipes.py`` (the legacy router) — that
  is the [HUMAN GATE] decision in PRD §12.6 (front-door choice + FE
  repoint of ~10 ``api-client.ts`` call sites).
- **Does NOT delete** the ``modules/workflows/`` twin — same human gate.
- **Does NOT promote** ``api/api_playbooks.py`` to an execution router —
  same human gate (the front-door decision).

Design constraint: import-light
-------------------------------
Importing ``api.recipe_executor`` eagerly defeats the seam — the heavy
chain (board bridge, learning service, memory service, S3 helpers) would
load any time the engine is touched. The methods import the executor
lazily so the seam module itself stays cheap.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PlaybookEngine:
    """The consolidated Playbook execution seam.

    Two methods cover every backend launch site:

    - :meth:`launch` — fire-and-track-row: schedules
      :func:`api.recipe_executor.execute_recipe_direct` as a background
      task and returns immediately. Used by the cron scheduler, the task
      reconciler retry path, the platform-tool handler, and the HTTP
      execute / webhook routes — anywhere the caller does NOT want to
      await the playbook end-to-end.
    - :meth:`execute_direct` — inline executor: awaits
      :func:`api.recipe_executor.execute_recipe_direct` directly. Used by
      the workspace webhook + the composio trigger dispatch, where the
      caller manages its own task lifecycle.

    Both methods forward the exact kwargs the legacy entry points already
    accept — no rename, no drop, no inject. That preserves behavioural
    parity (``tests/test_playbook_launch_parity.py``) so consolidating
    later (durability columns, retry-learning, SSE resume) lands in one
    place without breaking the seam.
    """

    def launch(
        self,
        *,
        recipe_execution_id: str,
        recipe_id: int,
        workspace_id: Any,
        input_data: dict,
    ) -> None:
        """Schedule a Playbook execution as a background task.

        Delegates to :func:`api.recipe_executor.launch_recipe_task`, which
        wraps :func:`execute_recipe_direct` with crash protection (a task
        that raises is marked ``failed`` instead of staying ``pending``
        forever). The caller is expected to have inserted the
        ``RecipeExecution`` row in ``status='pending'`` BEFORE calling this
        — the boot reaper (W1-S6 + the W3-S12 extension) sweeps rows still
        stuck in ``pending``/``running`` past the staleness window, so a
        process crash cannot silently lose the playbook.
        """
        from api.recipe_executor import launch_recipe_task

        launch_recipe_task(
            recipe_execution_id=recipe_execution_id,
            recipe_id=recipe_id,
            workspace_id=workspace_id,
            input_data=input_data,
        )

    async def execute_direct(
        self,
        *,
        recipe_execution_id: str,
        recipe_id: int,
        workspace_id: Any,
        input_data: dict,
        db_url: Optional[str] = None,
    ) -> None:
        """Await the Playbook executor inline.

        Delegates to :func:`api.recipe_executor.execute_recipe_direct` —
        the same inner loop ``launch_recipe_task`` schedules. ``db_url``
        is forwarded for the rare scheduled-retry callers that bind a
        different engine; production callers omit it.
        """
        from api.recipe_executor import execute_recipe_direct

        await execute_recipe_direct(
            recipe_execution_id=recipe_execution_id,
            recipe_id=recipe_id,
            workspace_id=workspace_id,
            input_data=input_data,
            db_url=db_url,
        )


_ENGINE: Optional[PlaybookEngine] = None


def get_playbook_engine() -> PlaybookEngine:
    """Module-level singleton accessor.

    A fresh instance per caller would defeat the strangler-fig (each caller
    would have its own state if future durability columns land here).
    Singleton means every call site shares the same seam — the harden-once
    promise (PRD §6 W3-S12).
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PlaybookEngine()
    return _ENGINE
