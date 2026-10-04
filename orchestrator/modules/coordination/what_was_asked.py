"""F283 (night 8): a template the owner asked for keeps its gaps.

#0433 was "a reusable welcome template with gaps", from the board's mission call. Both
of its written steps failed the mission's own check for the gaps the owner had asked
for ("It still has placeholders where its content belongs: [Owner first name]"), the
mission sat "running" with nothing moving, and a plain card wrote the same template
first time (#0438). The check (F248, ``deterministic_checks._placeholders_first``) saw
only the step's answer, never what was asked of it.

While a mission's step is checked, the check now knows what was asked: the step's
brief (its title and description, ``checked_against_its_brief`` around
``VerificationService.verify_task``) and its mission's goal (``checked_against_its_goal``
around ``MissionReconciler._verify_completed_tasks``). A placeholder the brief or the
goal already holds was asked for, and every placeholder was when the brief asks for a
template, placeholders, gaps or fill-in fields, or the goal asks for placeholders, gaps
or fill-in fields (a goal that only names a template, "use our welcome template for
Ruth", still wants Ruth's name in). Only the others fail the step
(``slots_not_asked_for``).
"""
from __future__ import annotations

import contextlib
import functools
import re
from contextvars import ContextVar
from typing import Any, Callable, Iterator, List, Optional

# What the owner asked of the step being checked, and of its mission.
_BRIEF: ContextVar[str] = ContextVar("checked_step_brief", default="")
_GOAL: ContextVar[str] = ContextVar("checked_mission_goal", default="")
# A brief that asks for a template or for gaps to fill: "a reusable welcome template
# with gaps", "keep the gaps", "fill-in fields", "placeholders for the café name".
_GAPS = (r"\bplaceholders?\b|\bfill[- ]?in\b"
         r"|\b(?:with|leave|leaving|keep|keeping|kept|as)\s+(?:the\s+|its\s+|some\s+)?gaps\b"
         r"|\bgaps?\s+(?:for|to\s+fill|to\s+be\s+filled|kept|left)\b|\bblanks?\s+(?:for|to\s+fill)\b")
GOAL_ASKS_FOR_GAPS = re.compile(_GAPS, re.IGNORECASE)
ASKS_FOR_GAPS = re.compile(r"\btemplates?\b|" + _GAPS, re.IGNORECASE)


@contextlib.contextmanager
def asking(*, brief: Optional[str] = None, goal: Optional[str] = None) -> Iterator[None]:
    """What the owner asked, while a step is checked: its brief, its mission's goal, or
    both. What is not given stays as it is."""
    tokens = [(var, var.set(text)) for var, text in ((_BRIEF, brief), (_GOAL, goal)) if text is not None]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def slots_not_asked_for(slots: List[str]) -> List[str]:
    """The placeholders in ``slots`` that neither the step's brief nor its mission's goal
    asked for."""
    if not slots:
        return []
    brief, goal = _BRIEF.get(), _GOAL.get()
    if ASKS_FOR_GAPS.search(brief) or GOAL_ASKS_FOR_GAPS.search(goal):
        return []
    asked = f"{brief}\n{goal}"
    held = asked.lower()
    return [slot for slot in slots if slot.lower() not in held]


def checked_against_its_brief(verify: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``VerificationService.verify_task``: its checks know the step's brief."""
    @functools.wraps(verify)
    async def wrapped(self: Any, task_title: str, task_description: str, *args: Any, **kwargs: Any) -> Any:
        with asking(brief=f"{task_title or ''}\n{task_description or ''}"):
            return await verify(self, task_title, task_description, *args, **kwargs)
    return wrapped


def checked_against_its_goal(verify_completed: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``MissionReconciler._verify_completed_tasks``: the checks of its steps know
    the mission's goal."""
    @functools.wraps(verify_completed)
    async def wrapped(db: Any, run: Any) -> Any:
        with asking(goal=str(getattr(run, "goal", None) or "")):
            return await verify_completed(db, run)
    return wrapped


__all__ = ["ASKS_FOR_GAPS", "GOAL_ASKS_FOR_GAPS", "asking", "checked_against_its_brief", "checked_against_its_goal", "slots_not_asked_for"]
