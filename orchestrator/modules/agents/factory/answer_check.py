"""F297 (night 8): an agent run's raw tool output is never its answer.

When a run ends without writing an answer, ``AgentFactory`` builds its result from
the last round's tool messages: "Based on the tool results:" over the raw JSON. On
night 8 that JSON was the whole answer on #0234, #0254 (a raw Composio error),
#0309, #0379, #0391, #0399, #0408.3, #0412 and #0430. ``said_plainly`` wraps the
factory's run so every lane that runs an agent (board cards, mission steps,
channels, webhooks, schedules) gets that result said plainly instead
(``services.result_substance.plain_no_answer``). Imported lazily: the factory loads
before the services.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)

Run = Callable[..., Awaitable[Dict[str, Any]]]


def said_plainly(run: Run) -> Run:
    """Wrap ``AgentFactory._execute_with_prompt_scoped``: a successful result that is
    only the run's raw tool output comes back said plainly; any other result as it
    was."""
    @functools.wraps(run)
    async def wrapped(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        out = await run(*args, **kwargs)
        if not isinstance(out, dict) or out.get("status") != "success":
            return out
        from services.result_substance import plain_no_answer, stopped_mid_step

        text = str(out.get("result") or "")
        plain = plain_no_answer(text) or stopped_mid_step(text)  # F306: a run that stopped mid-step too
        if plain is None:
            return out
        logger.warning("[F297/F306] the run wrote no answer; its result says so plainly")
        return {**out, "result": plain}
    return wrapped
