"""PRD-193 S1/S2: the confirmation gate a platform call clears last, and (P256-FIX-RVW-9)
the ask it raises for an owner-only action.

Moved out of ``PlatformActionExecutor.clear``, the rule unchanged: an action that requires
confirmation, when neither the full-autonomy dial nor the instructing owner's or admin's
own turn skips the card, runs on a grant that said yes to this exact call (a destructive
grant is retired on use, single-use) and otherwise returns the ask with a pending grant
attached. Nothing is asked about a target that is not there (F091). Any error falls
closed to the ask.

The fix-wave review (P256-FIX-RVW-9): an editor's 'delete market' met this gate before
``owner_only.asks_the_owner_first``, so its card carried the raw ``agent_name``, no agent
bound and no question, and the click deleted the first agent whose name contained
'market'. What changed here: an owner-only call in a person's chat is asked about by
``owner_only.platform_ask`` (the agent and the mission bound, FX-008's question on the card,
a name two agents carry, or none, refused before any grant); on every lane, a call that
names its agent by name alone skips the grant consult (``names_the_agent_alone``) and its
plain card binds the agent's id, so the click runs on that id.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, NamedTuple, Optional

from modules.tools.discovery.agent_binding import bound_to_the_agent, names_the_agent_alone
from modules.tools.discovery.owner_only import human_driven, is_owner_only, platform_ask

logger = logging.getLogger(__name__)

DESCRIPTION_SHOWN = 100  # characters of the action's description on the plain card


class _Call(NamedTuple):
    """The call at the gate, in this workspace."""

    db: Any
    workspace_id: Any
    action: str
    params: Any
    caller_context: Optional[Dict[str, Any]]


def asks_at_the_confirmation_gate(clear: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap PlatformActionExecutor.clear: after its definition, super-admin and admin
    gates, a call that must be confirmed runs on its grant or returns the ask."""
    @functools.wraps(clear)
    def wrapped(self: Any, action_name: str, params: Any, caller_context: Optional[Dict[str, Any]] = None,
                *, card_subject: str = "") -> Any:
        cleared = clear(self, action_name, params, caller_context, card_subject=card_subject)
        if isinstance(cleared, dict):
            return cleared
        call = _Call(self.db, self.workspace_id, action_name, params, caller_context)
        try:
            return _the_gate(call, cleared, card_subject) if _the_gate_asks(cleared) else cleared
        except Exception:
            logger.exception("[PlatformExecutor] confirmation gate failed for %s — requiring confirmation",
                             action_name)
            return _could_not_verify(call)
    return wrapped


def _the_gate_asks(cleared: Any) -> bool:
    """The call must be confirmed: neither the dial nor the instructing admin's turn skips the card."""
    from modules.tools.discovery.platform_executor import _dial_skips_the_card

    action_def = cleared.action_def
    return bool(action_def and action_def.requires_confirmation
                and not _dial_skips_the_card(action_def, cleared.full_autonomy)
                and not cleared.human_directed)


def _the_gate(call: _Call, cleared: Any, card_subject: str) -> Any:
    """Consult first: a grant on this exact call opens the gate (S2). Otherwise the ask,
    with a pending grant to say yes to (S1). A call naming its agent by name alone has
    no click: it is bound and asked about."""
    from modules.tools.execution import tool_grants

    level = cleared.action_def.permission_level
    grant = None
    if not names_the_agent_alone(call.action, call.params):
        grant = tool_grants.consume_tool_grant(call.db, call.workspace_id, action=call.action,
                                               params=call.params, permission_level=level)
    if grant is not None:
        approved_via_grant_id = getattr(grant, "id", None)
        logger.info("[PlatformExecutor] '%s' authorised by approval grant %s — proceeding (workspace=%s)",
                    call.action, approved_via_grant_id, call.workspace_id)
        return cleared._replace(approved_via_grant_id=approved_via_grant_id)
    if isinstance(call.params, dict) and human_driven(call.caller_context) and is_owner_only(call.action, call.params):
        return platform_ask(call.db, call.workspace_id, call.action, call.params, call.caller_context,
                            permission_level=level)
    return _the_cards_ask(call, cleared.action_def, card_subject)


def _the_cards_ask(call: _Call, action_def: Any, card_subject: str) -> Dict[str, Any]:
    """The plain card (F091: never about something that is not there, naming what it acts
    on), its agent bound to the id the click will run on."""
    from modules.tools.discovery.platform_executor import _subject_line
    from modules.tools.execution import tool_grants
    from modules.tools.execution.subject_targets import missing_targets_error, named_subject, resolve_targets

    params, refused = bound_to_the_agent(call.db, call.workspace_id, call.action, call.params)
    if refused:
        return refused
    found, missing = resolve_targets(call.db, call.workspace_id, params, call.action)
    if missing:
        return missing_targets_error(call.action, missing)
    subject = card_subject or named_subject(found) or _subject_line(params)
    level = action_def.permission_level
    message = (f"This action ({level}) requires confirmation. Action: {call.action}{subject} — "
               f"{action_def.description[:DESCRIPTION_SHOWN]}")
    ask = {"success": False, "requires_confirmation": True, "action": call.action,
           "permission_level": level, "message": message, "params": params}
    return tool_grants.attach_ask_grant(call.db, call.workspace_id, action=call.action, params=params, ask=ask,
                                        permission_level=level, description=action_def.description,
                                        caller_context=call.caller_context, subject=subject)


def _could_not_verify(call: _Call) -> Dict[str, Any]:
    """Fail closed: the ask, with the grant loop, and nothing consumed."""
    from modules.tools.execution import tool_grants

    ask = {"success": False, "requires_confirmation": True, "action": call.action, "permission_level": "unknown",
           "message": f"Could not verify permissions for '{call.action}'. Confirmation required for safety.",
           "params": call.params}
    return tool_grants.attach_ask_grant(call.db, call.workspace_id, action=call.action, params=call.params,
                                        ask=ask, permission_level=None, description=None,
                                        caller_context=call.caller_context)


__all__ = ["asks_at_the_confirmation_gate"]
