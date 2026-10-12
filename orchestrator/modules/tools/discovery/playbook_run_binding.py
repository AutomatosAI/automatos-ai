"""PRD-256 P256-FIX-T4 (Gerard, 11 Oct, benchmark F321-a): a playbook run from the owner's
chat is bound to one playbook, with inputs it can start on, before its card.

Running a playbook is owner-only now (``owner_only.OWNER_ONLY_ACTIONS``): the card names the
playbook and its inputs, and the click starts the run. The card must name the playbook the
click runs, so the ask binds the name the way ``execute_playbook`` finds it
(``playbook_lookup.finds_the_playbook(RUNS)``): by its id; else the one playbook with that
name, or the one whose name contains it; of namesakes, the only one whose every step has an
agent. None, or several, is refused before any grant, listing them with their steps' agents,
as the handler lists them (F091, F261).

A run the click could not start is refused before the card too, in the tool's own words:
inputs that are not key-value pairs (F113), details under names the playbook never reads
(F288) and an input it needs that the call did not give (F182). Auto asks the owner for
them in the chat, as it did before the card; a card the click could only fail is never raised.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from modules.tools.discovery.playbook_binding import NONE_CALLED, PLAYBOOK_ID, PLAYBOOK_NAME, RUN, _refused, _several

INPUT_DATA = "input_data"
INPUT_ALIASES = ("inputs", "input")
Bound = Tuple[Optional[Any], Optional[Dict[str, Any]]]


def run_target(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Bound:
    """(the playbook the click would run, None), or (None, the refusal): no such playbook,
    several, or a run it could not start with these inputs."""
    playbook, refusal = _found(db, workspace_id, params)
    if playbook is None:
        return None, refusal
    refusal = _cannot_start(db, workspace_id, playbook, params)
    return (None, refusal) if refusal is not None else (playbook, None)


def given_inputs(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """(the run's inputs, why they are not key-value pairs), read as ``execute_playbook``
    reads them: ``input_data``, else its ``inputs`` / ``input`` alias."""
    from core.services.playbook_inputs import playbook_inputs

    raw = params.get(INPUT_DATA)
    if raw is None:
        raw = next((params[key] for key in INPUT_ALIASES if key in params), None)
    return playbook_inputs(raw)


def _found(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Bound:
    """The playbook ``finds_the_playbook(RUNS)`` and the handler's lookup run, in this workspace only."""
    from modules.tools.discovery.playbook_lookup import playbooks_called
    from modules.tools.discovery.timer_target import by_id

    said_id = params.get(PLAYBOOK_ID)
    if said_id not in (None, ""):
        playbook = by_id(db, workspace_id, said_id)
        if playbook is None:
            return None, _refused(NONE_CALLED.format(said=f"#{said_id}", near="", action=RUN))
        return playbook, None
    found = playbooks_called(db, workspace_id, params[PLAYBOOK_NAME])
    if not found:
        return None, _several(db, workspace_id, RUN, params, found)
    chosen = found if len(found) == 1 else _the_staffed_namesake(found, str(params[PLAYBOOK_NAME]))
    if len(chosen) != 1:
        return None, _refused(_several_listed(db, found, str(params[PLAYBOOK_NAME]).strip()))
    return chosen[0], None


def _several_listed(db: Any, found: List[Any], name: str) -> str:
    """The handler's own refusal of several matches: each playbook with its steps' agents."""
    from modules.tools.discovery.playbook_lookup import DONE, RUNS, SEVERAL, _listed

    return SEVERAL.format(count=len(found), name=name, listed=_listed(db, found), done=DONE[RUNS])


def _the_staffed_namesake(found: List[Any], name: str) -> List[Any]:
    """Of playbooks that all carry the name, those whose every step has an agent; any
    other match of several is left as it is (refused, listing them)."""
    from modules.tools.discovery.playbook_lookup import staffed_steps

    wanted = name.strip().lower()
    if not all(str(playbook.name).strip().lower() == wanted for playbook in found):
        return found
    return [playbook for playbook in found if staffed_steps(playbook)]


def _cannot_start(db: Any, workspace_id: Any, playbook: Any, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The handler's own refusal for these inputs, or None when the run can start on them."""
    from core.services.playbook_inputs import contract_of, inputs_needed_error, missing_inputs, with_defaults
    from modules.tools.discovery.playbook_input_names import _other_names

    given, problem = given_inputs(params)
    if problem:
        return _refused(problem)
    other_names = _other_names(db, workspace_id, {**params, PLAYBOOK_ID: playbook.id})
    if other_names:
        return _refused(other_names)
    contract = contract_of(playbook)
    needed = missing_inputs(contract, with_defaults(contract, given))
    return _refused(inputs_needed_error(playbook.name, needed, contract)) if needed else None


__all__ = ["given_inputs", "run_target"]
