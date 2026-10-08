"""PRD-256 US-004 (FR-5): an owner-only action from a person's chat waits for their click.

Eleven customer nights: Auto approved #0329 "by you" when the owner had only named it,
and moved #0422 to Done when they said cancel. The night-8 fix (``follows_the_owner``)
read the owner's words for "approve", "cancel" and "yes" with regexes: words are not a
click, and those paths are gone. Now an action on ``OWNER_ONLY_ACTIONS`` (Decision D1,
scripts/ralph/prd-256w1.json; the owner trims it here) that a person's chat turn calls
(``caller_context.driving_user_id``) returns the confirmation ask the existing grant card
renders (``tool_grants.attach_ask_grant``), whatever the owner's role or the
full-autonomy dial. Their click grants it, and the resume (api/approval_grants
``_resume_tool_call``) runs the same call, which finds the grant
(``tool_grants.consume_tool_grant``). One click runs the call once: the grant is claimed
for the run (single-use), and given back when the call did nothing.

Agent runs, playbook steps and heartbeats carry no driving user and are unchanged; so
is the HARNESS's /approve, the admin's own decision (``OWNERS_OWN_DECISION``). The
policy plane (AUTOMATOS_POLICY_PLANE) is not needed, on either edition. The ask comes
after every other check of the call (the role gates, the hierarchy check, the rate limit),
so a call the owner could not run anyway is refused, not asked about. A Composio send or publish asks the same way (``asks_before_a_send``).

What a move writes on a card is signed by the user who clicked (``CLICKED_BY``, from the
grant), never by the owner on Auto's call alone (``signed_by``).
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from modules.tools.execution.card_raised import ACT
from modules.tools.execution.params_text import params_object

logger = logging.getLogger(__name__)

Execute = Callable[..., Awaitable[Dict[str, Any]]]

# Decision D1. The two card moves are owner-only only when they close the card.
OWNER_ONLY_ACTIONS = frozenset({
    "platform_update_task_status", "platform_update_task",
    "platform_assign_tool_to_agent", "platform_unassign_tool_from_agent", "platform_update_agent",
    "platform_update_system_setting", "platform_create_mission", "platform_approve_mission",
    "platform_cancel_mission", "platform_publish_blog_post", "platform_submit_social_post",
})
CARD_MOVES = frozenset({"platform_update_task_status", "platform_update_task"})
CLOSING_STATUSES = frozenset({"done", "cancelled"})
# D1's "every Composio send/publish action": a slug with one of these words and no read word.
COMPOSIO_SEND_WORDS = frozenset({"SEND", "SENDS", "PUBLISH", "POST", "REPLY", "FORWARD", "TWEET", "BROADCAST"})
COMPOSIO_READ_WORDS = frozenset({"GET", "LIST", "FETCH", "SEARCH", "FIND", "RETRIEVE", "READ", "LOOKUP",
                                 "COUNT", "DOWNLOAD"})

# Server-set keys, never the model's: who clicked (stripped from every call, set after a
# click), and a caller whose own decision is the click (the HARNESS's /approve).
CLICKED_BY = "_clicked_by"
OWNERS_OWN_DECISION = "owners_own_decision"
USER_ACTOR = "user:"
PERMISSION_LEVEL = "write"
MAX_CARDS_NAMED = 5
MORE_CARDS = " and {count} more"

ASK = ("Waiting for the owner's click: {act}. Nothing has been done: the approval card in the chat asks "
       "them, and their click runs it.")
CLOSING_VERBS = {"done": "approve (move to Done)", "cancelled": "cancel"}
VERBS = {
    "platform_assign_tool_to_agent": "give a tool to an agent",
    "platform_unassign_tool_from_agent": "take a tool from an agent",
    "platform_update_agent": "change an agent",
    "platform_update_system_setting": "change a system setting",
    "platform_create_mission": "start a mission",
    "platform_approve_mission": "approve a mission's plan",
    "platform_cancel_mission": "cancel a mission",
    "platform_publish_blog_post": "publish a blog post",
    "platform_submit_social_post": "submit a social post to publish",
}
SEND_VERB = "send or publish through"
QUESTION = "question_md"


def is_owner_only(action_name: str, params: Any, *, composio: bool = False) -> bool:
    """Whether the call is one only the owner's click may run (Decision D1)."""
    name = str(action_name or "")
    if composio:
        return is_composio_send(name)
    if name in CARD_MOVES:
        params = params_object(params)
        return isinstance(params, dict) and closing_status(params) is not None
    return name in OWNER_ONLY_ACTIONS


def is_composio_send(slug: str) -> bool:
    """A Composio action that sends or publishes: GMAIL_SEND_EMAIL, LINKEDIN_CREATE_LINKED_IN_POST."""
    words = set(str(slug or "").upper().split("_"))
    return bool(words & COMPOSIO_SEND_WORDS) and not words & COMPOSIO_READ_WORDS


def closing_status(params: Dict[str, Any]) -> Optional[str]:
    """"done" or "cancelled" when the call closes the card, read the board's way ("approved" is Done)."""
    from modules.tools.execution.call_effects import STATUS_WORDS

    status = str(params.get("status") or "").strip().lower()
    status = STATUS_WORDS.get(status, status)
    return status if status in CLOSING_STATUSES else None


def human_driven(caller_context: Any) -> bool:
    """A person drove the turn, and it is not a caller whose own decision is the click."""
    from core.security.driving_user import driving_user_id

    if isinstance(caller_context, dict) and caller_context.get(OWNERS_OWN_DECISION):
        return False
    return driving_user_id(caller_context) is not None


def signed_by(params: Any) -> Optional[str]:
    """Who signs what a chat-driven move writes on a card: the user who clicked, or None
    when no one did (Auto's call alone, or an agent's own run)."""
    signer = params.get(CLICKED_BY) if isinstance(params, dict) else None
    return str(signer) if signer else None


def asks_the_owner_first(run_cleared: Execute) -> Execute:
    """Wrap PlatformActionExecutor._run_cleared: an owner-only call in a person's chat runs
    its handler only on their click. Every other check of the call (the role gates before,
    the hierarchy check, the rate limit and the destructive backstop inside) comes first,
    so a call the owner could not run anyway is refused, not asked about. A caller-supplied
    signer is always dropped."""
    @functools.wraps(run_cleared)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any, cleared: Any,
                      handler: Execute) -> Dict[str, Any]:
        params = _without_a_signer(params)
        if isinstance(params, dict) and human_driven(caller_context) and is_owner_only(action_name, params):
            handler = _on_the_click(action_name, params, caller_context, handler)
        return await run_cleared(self, action_name, params, caller_context, cleared, handler)
    return wrapped


def _on_the_click(action: str, asked: Dict[str, Any], caller_context: Any, handler: Execute) -> Execute:
    """``handler`` run on the owner's click on this exact call (``asked``), signed by who
    clicked; without one, the ask."""
    async def on_the_click(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        grant = _the_click(db, workspace_id, action, asked)
        if grant is None:
            return platform_ask(db, workspace_id, action, asked, caller_context)
        clicker = _clicker(grant)
        signed = {**params, CLICKED_BY: clicker} if clicker else params
        return after_the_click(db, grant, await handler(db, workspace_id, signed))
    return on_the_click


def asks_before_a_send(execute_tool: Execute) -> Execute:
    """Wrap UnifiedToolExecutor.execute_tool: a Composio send or publish in a person's chat
    runs only on their click, through the same grant card."""
    signature = inspect.signature(execute_tool)

    @functools.wraps(execute_tool)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        call = signature.bind(self, *args, **kwargs)
        call.apply_defaults()
        tool, params, ctx = call.arguments["tool_name"], call.arguments["parameters"], call.arguments["caller_context"]
        workspace_id = call.arguments["workspace_id"]
        slug, inner, composio = self._resolve_effective_call(tool, params)
        if not (composio and human_driven(ctx) and is_owner_only(slug, params, composio=True)):
            return await execute_tool(*call.args, **call.kwargs)
        grant = _the_click(self.db, workspace_id, tool, params)
        if grant is None:
            return send_ask(self.db, workspace_id, tool, slug, params, ctx, sent=inner)
        return after_the_click(self.db, grant, await execute_tool(*call.args, **call.kwargs))
    return wrapped


def _the_click(db: Any, workspace_id: Any, action: str, params: Any) -> Any:
    """The owner's grant for this exact call, claimed for this one run (single-use, so a
    concurrent or repeated call asks again), or None (the ask stands)."""
    from modules.tools.execution import tool_grants

    return tool_grants.consume_tool_grant(db, workspace_id, action=action, params=params,
                                          permission_level=PERMISSION_LEVEL, single_use=True)


def after_the_click(db: Any, grant: Any, result: Any) -> Any:
    """One click, one run: a call that did nothing (``success: False``) gives the click
    back for its retry (tool_grants.give_back_unused, F193). The result records which
    grant said yes."""
    from modules.tools.execution import tool_grants

    tool_grants.give_back_unused(db, getattr(grant, "id", None), result)
    if isinstance(result, dict) and result.get("approved_via_grant_id") is None:
        return {**result, "approved_via_grant_id": getattr(grant, "id", None)}
    return result


def platform_ask(db: Any, workspace_id: Any, action: str, params: Dict[str, Any], caller_context: Any) -> Dict[str, Any]:
    """The ask for a platform action, naming the card by its number and the verb, and
    saying what the call changes (FX-008). A card that is not on the board is never
    asked about (F091)."""
    from modules.tools.discovery.card_question import platform_question
    from modules.tools.execution.subject_targets import missing_targets_error, named_subject

    found, missing = _targets(db, workspace_id, action, params)
    if missing:
        return missing_targets_error(action, missing)
    what = (named_subject(found).removeprefix(" on ") or _said_subject(params)) + _cards_not_named(params)
    status = closing_status(params) if action in CARD_MOVES else None
    act = f"{CLOSING_VERBS[status] if status else VERBS.get(action, action)} {what}".strip()
    asked = platform_question(db, workspace_id, action, params, act)
    return _ask(db, workspace_id, action, params, caller_context, act=act, what=what, asked=asked)


def send_ask(db: Any, workspace_id: Any, tool: str, slug: str, params: Any, caller_context: Any, *,
             sent: Any = None) -> Dict[str, Any]:
    """The ask for a Composio send or publish, naming the action, and to whom, about what
    and its first line (``sent``: the action's own params, FX-008)."""
    from modules.tools.discovery.card_question import send_question

    act = f"{SEND_VERB} {slug}".strip()
    asked = send_question(act, params_object(sent if sent is not None else params))
    return _ask(db, workspace_id, tool, params, caller_context, act=act, what=slug, asked=asked)


def _ask(db: Any, workspace_id: Any, action: str, params: Any, caller_context: Any, *, act: str,
         what: str, asked: str) -> Dict[str, Any]:
    from modules.tools.execution import tool_grants

    message = ASK.format(act=act)
    # ``act``: what the card asks, in the owner's words, for the receipt and the model (FX-004);
    # ``question_md``: what the card shows the owner, the subject and the change (FX-008).
    ask = {"success": False, "requires_confirmation": True, "owner_only": True, "action": action,
           "permission_level": PERMISSION_LEVEL, "message": message, "params": params, ACT: act,
           QUESTION: asked}
    logger.info("[owner_only] %s waits for the owner's click (%s)", action, what)
    return tool_grants.attach_ask_grant(db, workspace_id, action=action, params=params, ask=ask,
                                        permission_level=PERMISSION_LEVEL, description=message,
                                        caller_context=caller_context, subject=f" on {what}" if what else "",
                                        question_md=asked)


def _targets(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> tuple:
    """(found, missing) for the call's ids, each card of a bulk move too."""
    from modules.tools.execution.subject_targets import resolve_targets

    found, missing = resolve_targets(db, workspace_id, params, action)
    listed = params.get("task_ids") if isinstance(params.get("task_ids"), list) else []
    for ref in listed[:MAX_CARDS_NAMED]:
        more_found, more_missing = resolve_targets(db, workspace_id, {"task_id": ref}, action)
        found, missing = [*found, *more_found], [*missing, *more_missing]
    return found, missing


def _cards_not_named(params: Dict[str, Any]) -> str:
    """" and 3 more" when a bulk move carries more cards than the card names (FX-003)."""
    listed = params.get("task_ids") if isinstance(params.get("task_ids"), list) else []
    extra = len(listed) - MAX_CARDS_NAMED
    return MORE_CARDS.format(count=extra) if extra > 0 else ""


def _said_subject(params: Dict[str, Any]) -> str:
    """What the call names when no card or row does: mission_id='…', app_name='slack'."""
    from modules.tools.discovery.platform_executor import _subject_line

    return _subject_line(params).removeprefix(" on ")


def _clicker(grant: Any) -> Optional[str]:
    """The user who clicked, as a card signs them (board_consent.actor_from_user_id)."""
    from services.board_consent import actor_from_user_id

    who = str(getattr(grant, "granted_by", "") or "")
    return actor_from_user_id(who.removeprefix(USER_ACTOR)) if who.startswith(USER_ACTOR) else None


def _without_a_signer(params: Any) -> Any:
    """The call's params as an object (JSON text read as the object it holds), with no
    caller-supplied signer: only a click signs."""
    params = params_object(params)
    if isinstance(params, dict) and CLICKED_BY in params:
        return {key: value for key, value in params.items() if key != CLICKED_BY}
    return params


__all__ = ["CLICKED_BY", "OWNERS_OWN_DECISION", "OWNER_ONLY_ACTIONS", "after_the_click", "asks_before_a_send",
           "asks_the_owner_first", "human_driven", "is_composio_send", "is_owner_only", "signed_by"]
