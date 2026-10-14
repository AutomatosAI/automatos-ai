"""P256-FIX-N15-CI: what the claim check took out of an agent's text, kept for the run.

The tool loop's claim check (modules/tools/execution/claims_in_place.py) takes an unbacked claim out
of a reply. In Auto's own reply the not-done line takes its place; an agent's text may be for a
customer (#0245's email to Nadia), so there nothing does, and the owner is told beside the answer:
the sentences taken out are kept while the agent's run lasts (``TAKEN_OUT``, a list, as
``TRACKERS_MADE`` keeps a chat turn's trackers) and ride in its result's ``execution``
(``TAKEN_OUT_KEY``). A board card puts them in its note (services/claims_taken_out.py).

Stdlib only: the agent factory imports it at load, and the loop when a claim is taken out.
"""
from __future__ import annotations

import functools
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

Result = Dict[str, Any]
Run = Callable[..., Awaitable[Any]]

TAKEN_OUT_KEY = "claims_taken_out"     # in an agent run's result["execution"]: the sentences taken out
# The sentences taken out of an agent's text during its run, oldest first; None outside one.
TAKEN_OUT: ContextVar[Optional[List[str]]] = ContextVar("claims_taken_out", default=None)


def kept_for_the_run(sentences: Iterable[str]) -> None:
    """Keep ``sentences``, taken out of the run's text, for its result; nothing outside a run."""
    kept = TAKEN_OUT.get()
    if kept is not None:
        kept.extend(sentences)


def reports_the_claims_it_took_out(run: Run) -> Run:
    """Wraps ``AgentFactory._execute_with_prompt_scoped``: the sentences taken out of the run's text
    ride in its successful result's ``execution`` (``TAKEN_OUT_KEY``), each once. The result is a new
    dict; a run that took nothing out, or did not succeed, is returned as it came."""
    @functools.wraps(run)
    async def reported(*args: Any, **kwargs: Any) -> Any:
        taken: List[str] = []
        token = TAKEN_OUT.set(taken)
        try:
            out = await run(*args, **kwargs)
        finally:
            TAKEN_OUT.reset(token)
        if not taken or not isinstance(out, dict) or out.get("status") != "success":
            return out
        execution = {**(out.get("execution") or {}), TAKEN_OUT_KEY: list(dict.fromkeys(taken))}
        return {**out, "execution": execution}
    return reported


__all__ = ["TAKEN_OUT", "TAKEN_OUT_KEY", "kept_for_the_run", "reports_the_claims_it_took_out"]
