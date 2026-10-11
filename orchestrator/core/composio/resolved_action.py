"""PRD-256 P256-FIX-RVW-3: the owner's click is checked on the action that runs.

The owner's-click gate (``modules/tools/discovery/owner_only.asks_before_a_send``) judges
the name a call asks for, before the Composio executor runs. ``ComposioToolExecutor.execute``
may then resolve that name onto another action (a slug form, a suffix match, a display-name
rebuild, the auto-map of a near-miss) and checks the action that runs again, after the deny
list, through the Socials post gate (``post_action_refusal``; it is called on the name asked
for, then on the name resolved). So a send asked for under a non-send name ran with no card.

The executor's post gate is this module's :func:`post_action_refusal`: the Socials gate's
refusal first, which always wins; with none, inside a call the click gate is watching
(:func:`checks_the_resolved_action`), the click gate's own check of the action, which
raises the card and answers why the action did not run. Anywhere else it is the Socials
gate alone. Every action the post gate sees inside the watched call is checked, one an
executor call nested in it makes too: that fails toward asking. core/ never imports the
click gate (a feature module): the gate hands its check in.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Iterator, Optional

from core.composio import post_gate

# The click gate's check of one action: why it may not run now (a card was raised), or None.
ResolvedCheck = Callable[[str], Awaitable[Optional[str]]]

_CHECK: ContextVar[Optional[ResolvedCheck]] = ContextVar("composio_resolved_action_check", default=None)


@contextmanager
def checks_the_resolved_action(check: ResolvedCheck) -> Iterator[None]:
    """Run the body with ``check`` asked about every action the executor's post gate sees."""
    token = _CHECK.set(check)
    try:
        yield
    finally:
        _CHECK.reset(token)


async def post_action_refusal(action: Any, workspace_id: Any, *, way_through: Any = None) -> Optional[str]:
    """Why ``action`` may not run: the Socials post gate's refusal (D14b), else the watching
    click gate's (a card raised for a send the call resolved onto), else None."""
    refusal = await post_gate.post_action_refusal(action, workspace_id, way_through=way_through)
    check = _CHECK.get()
    if refusal or check is None:
        return refusal
    return await check(str(action or ""))


__all__ = ["ResolvedCheck", "checks_the_resolved_action", "post_action_refusal"]
