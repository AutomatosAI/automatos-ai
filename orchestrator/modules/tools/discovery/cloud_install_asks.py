"""PRD-270 W2 US-270-14g (§5.6, §5.8, K17, K28-1): Auto's chat install of an Automatos Cloud item asks first.

Auto installs Marketplace items with three platform actions. When one names a row a Sync with Automatos Cloud
wrote (``chat_disclosure.cloud_target``), the call runs only on a single-use grant that said yes to this exact
call (``tool_grants.consume_tool_grant(..., single_use=True)``) after its card showed the item as it is now.
Otherwise it returns the existing ask: a pending tool_call grant (``tool_grants.attach_ask_grant``) whose
``question_md`` is the item's disclosure (``chat_disclosure.disclosure_markdown``), the card in the chat and the
approvals queue. Approve (api/approval_grants) re-runs the stored call, which meets its grant here and installs;
Deny runs nothing. The executor gives back a yes whose install did nothing (``give_back_unused``).

* Neither the full-autonomy dial nor the owner's own turn skips it, unlike the plain confirmation card
  (``confirmation_gate._the_gate_asks``): the disclosure is the point.
* The card binds the call to the row's own key (``CloudTarget.bound``), so the click installs the row it showed.
  A call in any other form (a name, a key the action doesn't take) asks with the bound card and never runs on a
  grant.
* A yes whose card no longer reads as the item does now (a Sync changed it since, or the card was not its
  disclosure) is spent, and a new card asks. A card still waiting with such text is closed as replaced
  (``stale_cards.SUPERSEDED_BY``: its click says so) before the ask, so no card shows what the item no longer is.
* Any error falls closed to an ask with no card: nothing runs, and nothing can be approved unseen.

Every other action, and a seeded or an operator item, passes as it cleared. The web's installs never come here:
their routes don't run platform actions (K17), and the web's own gate shows the disclosure (US-270-14d).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# The actions Auto installs a Marketplace item with (actions_packages.py, actions_marketplace.py).
AGENT_INSTALL = "platform_install_marketplace_agent"
PACKAGE_INSTALL = "platform_install_package"
PLUGIN_INSTALL = "platform_install_plugin"
CLOUD_INSTALLS = frozenset({AGENT_INSTALL, PACKAGE_INSTALL, PLUGIN_INSTALL})

ASKS = ("Waiting for approval to {act}. Nothing was installed: its approval card shows what it will use, and "
        "Approve installs it.")
CHANGED = ("The approval to {act} was for what its card showed, and the item has changed on this install since. "
           "Nothing was installed: a new card shows it as it is now.")
COULD_NOT_CHECK = ("Couldn't check whether this item came from Automatos Cloud, or read what it will use. Nothing "
                   "was installed: try again, or add it from the Marketplace.")
ACT = "install the {label} from Automatos Cloud"
QUESTION = "question_md"
UNKNOWN_LEVEL = "unknown"


def cloud_install_asks(call: Any, cleared: Any) -> Any:
    """``cleared`` for every call but an install of a Cloud item. For one: the cleared call, on a single-use grant
    for this exact call whose card showed the item as it is; otherwise the ask, its card the item's disclosure."""
    if isinstance(cleared, dict) or call.action not in CLOUD_INSTALLS:
        return cleared
    try:
        from services.marketplace_bundles import chat_disclosure

        target = chat_disclosure.cloud_target(call.db, call.action, call.params)
        if target is None:
            return cleared
        question = chat_disclosure.disclosure_markdown(call.db, target.kind, target.key)
        return _on_its_grant(call, cleared, target, question)
    except Exception:
        logger.exception("[cloud_install_asks] could not check %s: asking, nothing runs", call.action)
        return _could_not_check(call)


def _on_its_grant(call: Any, cleared: Any, target: Any, question: str) -> Any:
    """The cleared call when a single-use grant said yes to this exact call and its card is ``question``;
    otherwise the ask."""
    from modules.tools.execution import tool_grants

    level = cleared.action_def.permission_level
    if tool_grants.canonical_params(call.params) != tool_grants.canonical_params(target.bound):
        return _the_ask(call, level, target, question, ASKS)  # only the bound call's click runs
    grant = tool_grants.consume_tool_grant(call.db, call.workspace_id, action=call.action, params=call.params,
                                           permission_level=level, single_use=True)
    if grant is None:
        return _the_ask(call, level, target, question, ASKS)
    if getattr(grant, QUESTION, None) != question:
        logger.info("[cloud_install_asks] grant %s spent: its card no longer reads as the Cloud %s",
                    grant.id, target.kind)
        return _the_ask(call, level, target, question, CHANGED)
    logger.info("[cloud_install_asks] %s of a Cloud %s authorised by approval grant %s (workspace=%s)",
                call.action, target.kind, grant.id, call.workspace_id)
    return cleared._replace(approved_via_grant_id=grant.id)


def _the_ask(call: Any, level: str, target: Any, question: str, said: str) -> Dict[str, Any]:
    """The ask with its pending grant: the card shows ``question`` and its click runs the bound call."""
    from modules.tools.execution import tool_grants
    from modules.tools.execution.card_raised import ACT as ACT_KEY

    _close_an_outdated_card(call, target, question)
    act = ACT.format(label=target.label)
    message = said.format(act=act)
    ask = {"success": False, "requires_confirmation": True, "action": call.action, "permission_level": level,
           "message": message, "params": dict(target.bound), ACT_KEY: act, QUESTION: question}
    return tool_grants.attach_ask_grant(call.db, call.workspace_id, action=call.action, params=target.bound,
                                        ask=ask, permission_level=level, description=message,
                                        caller_context=call.caller_context, subject=f" on the {target.label}",
                                        question_md=question)


def _close_an_outdated_card(call: Any, target: Any, question: str) -> None:
    """Close this call's waiting card when its text is not ``question`` (the item changed since it was raised, or
    another path raised it), so the ask raises a fresh card instead of reusing it. Caller owns the transaction."""
    from core.models.approval_grants import SUBJECT_TOOL_CALL
    from core.services.approval_grants import expire_pending_grants, find_pending_grant
    from modules.tools.execution.stale_cards import SUPERSEDED_BY
    from modules.tools.execution.tool_grants import tool_call_subject_id

    subject_id = tool_call_subject_id(call.workspace_id, call.action, target.bound)
    waiting = find_pending_grant(call.db, call.workspace_id, subject_type=SUBJECT_TOOL_CALL, subject_id=subject_id)
    if waiting is None or waiting.question_md == question:
        return
    closed = expire_pending_grants(call.db, call.workspace_id, [waiting.id], revoked_by=SUPERSEDED_BY)
    call.db.expire(waiting)  # one UPDATE (synchronize_session=False): the next read sees it closed
    logger.info("[cloud_install_asks] card %s closed (%d): its text no longer reads as the Cloud %s",
                waiting.id, closed, target.kind)


def _could_not_check(call: Any) -> Dict[str, Any]:
    """Fail closed: the ask with no card, so nothing runs and nothing can be approved without its disclosure."""
    return {"success": False, "requires_confirmation": True, "action": call.action,
            "permission_level": UNKNOWN_LEVEL, "message": COULD_NOT_CHECK, "params": call.params}


__all__ = ["AGENT_INSTALL", "CLOUD_INSTALLS", "PACKAGE_INSTALL", "PLUGIN_INSTALL", "cloud_install_asks"]
