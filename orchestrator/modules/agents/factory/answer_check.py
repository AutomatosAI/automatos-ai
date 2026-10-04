"""F297 (night 8): an agent run's raw tool output is never its answer.

When a run ends without writing an answer, ``AgentFactory`` builds its result from
the last round's tool messages: "Based on the tool results:" over the raw JSON. On
night 8 that JSON was the whole answer on #0234, #0254 (a raw Composio error),
#0309, #0379, #0391, #0399, #0408.3, #0412 and #0430. ``said_plainly`` wraps the
factory's run so every lane that runs an agent (board cards, mission steps,
channels, webhooks, schedules) gets that result said plainly instead
(``services.result_substance.plain_no_answer``). Imported lazily: the factory loads
before the services.

F320 (night 9b): any other answer comes back as the work itself, without the agent's
working on top ("Perfect! Now I have all the information I need…" on #0046, #0054,
#0063, #0065, #0070) and with a draft's source line moved out of the draft
(``services.answer_working.the_answer_itself``).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)

Run = Callable[..., Awaitable[Dict[str, Any]]]


def plain_result(text: str) -> str:
    """A run's result as its card shows it: said plainly when the run wrote no answer
    (F297) or stopped mid-step (F306), else the answer without the working (F320)."""
    from services.answer_working import the_answer_itself
    from services.result_substance import plain_no_answer, stopped_mid_step

    plain = plain_no_answer(text) or stopped_mid_step(text)  # F306: a run that stopped mid-step too
    if plain is not None:
        logger.warning("[F297/F306] the run wrote no answer; its result says so plainly")
        return plain
    answer = the_answer_itself(text)
    if answer != text:
        logger.info("[F320] the agent's working was taken off its answer (%d to %d chars)", len(text), len(answer))
    return answer


def said_plainly(run: Run) -> Run:
    """Wrap ``AgentFactory._execute_with_prompt_scoped``: a successful result comes
    back as ``plain_result`` makes it; any other result as it was."""
    @functools.wraps(run)
    async def wrapped(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        out = await run(*args, **kwargs)
        if not isinstance(out, dict) or out.get("status") != "success":
            return out
        text = str(out.get("result") or "")
        result = plain_result(text)
        return out if result == text else {**out, "result": result}
    return wrapped
