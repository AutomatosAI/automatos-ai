"""
MissionContextSection — Mission goal, plan, task statuses, budget.

Priority 2 (never dropped). Provides the coordinator with full mission
state so it can make dispatch and reconciliation decisions.

Source: PRD-82A Section 12 Phase 3, PRD-102 Section 7.2
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class MissionContextSection(BaseSection):
    """Renders current mission state for the coordinator's system prompt.

    Expects the following keys in ``ctx.kwargs``:
    - ``mission_run``: OrchestrationRun record (or dict with goal, state, etc.)
    - ``mission_tasks``: list of OrchestrationTask records (or dicts)

    If neither is present, renders an empty string (section is skipped).
    """

    name: str = "mission_context"
    priority: int = 2
    max_tokens: Optional[int] = 8000

    async def render(self, ctx: SectionContext) -> str:
        """Build the mission context block for the coordinator prompt."""
        try:
            return self._build(ctx)
        except Exception:
            logger.exception("MissionContextSection.render failed")
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        run = ctx.kwargs.get("mission_run")
        if run is None:
            return ""

        parts: list[str] = ["## Mission Context", ""]

        # --- Goal & state ---
        goal = _attr_or_key(run, "goal", "Unknown goal")
        state = _attr_or_key(run, "state", "unknown")
        run_id = _attr_or_key(run, "id", "?")
        parts.append(f"**Mission:** {goal}")
        parts.append(f"**State:** {state}")
        parts.append(f"**Run ID:** {run_id}")

        # --- Plan summary ---
        plan = _attr_or_key(run, "plan", None)
        if plan and isinstance(plan, dict):
            task_count = plan.get("task_count", "?")
            parts.append(f"**Planned tasks:** {task_count}")

        # --- Budget tracking ---
        budget_estimate = _attr_or_key(run, "token_budget_estimate", None)
        tokens_used = _attr_or_key(run, "tokens_used", 0)
        if budget_estimate is not None:
            pct = (tokens_used / budget_estimate * 100) if budget_estimate > 0 else 0
            parts.append(
                f"**Token budget:** {tokens_used:,} / {budget_estimate:,} ({pct:.0f}%)"
            )
        elif tokens_used:
            parts.append(f"**Tokens used:** {tokens_used:,}")

        # --- Task statuses ---
        tasks = ctx.kwargs.get("mission_tasks")
        if tasks:
            parts.append("")
            parts.append("### Task Statuses")
            parts.append("")
            for task in tasks:
                seq = _attr_or_key(task, "sequence_number", "?")
                title = _attr_or_key(task, "title", "Untitled")
                t_state = _attr_or_key(task, "state", "unknown")
                agent_role = _attr_or_key(task, "agent_role", None)
                assigned_id = _attr_or_key(task, "assigned_agent_id", None)
                attempt = _attr_or_key(task, "attempt_number", 0)

                line = f"{seq}. **{title}** — `{t_state}`"
                if agent_role:
                    line += f" (role: {agent_role})"
                if assigned_id:
                    line += f" [agent #{assigned_id}]"
                if attempt and attempt > 0:
                    line += f" (attempt {attempt})"
                parts.append(line)

        content = "\n".join(parts)
        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)
        return content


def _attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Read a value from an ORM object (attribute) or dict (key)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
