"""F288 (night 8): a playbook run's details under the playbook's own names.

Auto ran New Cafe Onboarding with the café's details under names of its own
(owner_name, owner_email, first_order) where the playbook's step named
contact_person, contact_email, usual_harbour_blend_kg and usual_decaf_kg. None of it
reached the run: #0366 asked the owner again in field names, and #0440 went out with
the template's own words (Thursday, "the crew", decaf).

A run of a playbook that declares its inputs (``contract_of``) is not started when
the call sends details under names the playbook never reads while it leaves one of
the playbook's own inputs to its default or, when required, unset. Auto is told the
playbook's names, what it sent, and to send the call again with each detail under
its own name, asking the owner only for a required one none of its details gives.
Nothing else is refused: a playbook with no declared or {input.…} inputs, one whose
steps read the whole input ({input}), a detail a step reads as a {{name}} blank, and
an extra detail sent alongside every input the playbook takes all run as before.

Playbook 102 itself declares no inputs (its steps ask for the details in words), so
this check cannot reach it; showing a run's details to every step is
fix/f249-f269-f297-lessons-and-answers (services/playbook_given.py).
"""
from __future__ import annotations

import functools
import json
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

# A {{name}} blank in a step reads the run's detail of that name (api.recipe_executor, F130).
_NAMED_BLANK = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
OTHER_NAMES = ("'{name}' was not started: it reads its details under its own names, and this call sent {unread} "
               "under names it never reads, leaving {left} to {fate}. Its inputs:")
SEND_AGAIN = ("Send the call again now with each detail under the name it belongs to, and leave out what the "
              "playbook doesn't take: {example}. Ask the owner only for a required input none of the details "
              "you have gives.")


def checks_the_input_names(handler: Handler) -> Handler:
    """platform_execute_playbook: details under names the playbook never reads are
    sent back before the run starts (see the module)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        refusal = _other_names(db, workspace_id, params or {})
        return {"success": False, "error": refusal} if refusal else await handler(db, workspace_id, params)
    return wrapped


def _other_names(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Optional[str]:
    from core.services.playbook_inputs import INPUT_KEY, contract_of, playbook_inputs

    playbook = _playbook(db, workspace_id, params.get("playbook_id"))
    contract = contract_of(playbook) if playbook is not None else {}
    if not contract or INPUT_KEY in contract:
        return None
    raw = params.get("input_data")
    given, problem = playbook_inputs(params.get("inputs", params.get("input")) if raw is None else raw)
    if problem or not given:
        return None
    read = set(contract) | _blanks(getattr(playbook, "steps", None))
    unread = sorted(k for k, v in given.items() if k not in read and not _empty(v))
    left = [name for name, spec in contract.items()
            if _empty(given.get(name)) and (spec.get("required") or not _empty(spec.get("default")))]
    if not unread or not left:
        return None
    return _refusal(playbook, contract, given, unread, left)


def _playbook(db: Session, workspace_id: Any, playbook_id: Any) -> Any:
    from core.models.core import WorkflowTemplate

    if playbook_id in (None, ""):
        return None
    return db.query(WorkflowTemplate).filter(
        WorkflowTemplate.id == playbook_id, WorkflowTemplate.workspace_id == workspace_id).first()


def _blanks(steps: Any) -> Set[str]:
    prompts = (s.get("prompt_template") or "" for s in (steps or []) if isinstance(s, dict))
    return {name for prompt in prompts for name in _NAMED_BLANK.findall(str(prompt))}


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value in ([], {})


def _refusal(playbook: Any, contract: Dict[str, Dict[str, Any]], given: Dict[str, Any], unread: List[str],
             left: List[str]) -> str:
    defaulted = [n for n in left if not _empty(contract[n].get("default"))]
    fate = "its default" if len(defaulted) == len(left) else "its default or unset" if defaulted else "unset"
    lines = [OTHER_NAMES.format(name=playbook.name, unread=_listed(unread), left=_listed(left), fate=fate)]
    lines += [f"- {name}{_spec(spec)}" for name, spec in contract.items()]
    example = {name: given.get(name) if not _empty(given.get(name)) else "…" for name in contract}
    call = json.dumps({"playbook_id": playbook.id, "input_data": example}, ensure_ascii=False, default=str)
    lines.append(SEND_AGAIN.format(example=call))
    return "\n".join(lines)


def _spec(spec: Dict[str, Any]) -> str:
    """" (required): what it is" or " (default 'Thursday')"."""
    said = "required" if spec.get("required") else (
        f"default {spec['default']!r}" if not _empty(spec.get("default")) else "optional")
    description = str(spec.get("description") or "").strip()
    return f" ({said})" + (f": {description}" if description else "")


def _listed(names: Iterable[str]) -> str:
    return ", ".join(names)


__all__ = ["checks_the_input_names"]
