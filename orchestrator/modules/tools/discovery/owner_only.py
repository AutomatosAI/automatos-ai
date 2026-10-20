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
import re
from typing import Any, Awaitable, Callable, Dict, Optional

from modules.tools.discovery.agent_binding import names_the_agent_alone
from modules.tools.discovery.said_instructions import as_the_owner_said
from modules.tools.discovery.send_words import LEAVES_THE_WORKSPACE, READ_WORDS, makes_or_times_a_message, slug_words
from modules.tools.execution.card_raised import ACT
from modules.tools.execution.params_text import params_object

logger = logging.getLogger(__name__)

Execute = Callable[..., Awaitable[Dict[str, Any]]]

# Decision D1. The two card moves are owner-only only when they close the card or send a worked one back.
OWNER_ONLY_ACTIONS = frozenset({
    "platform_update_task_status", "platform_update_task",
    "platform_assign_tool_to_agent", "platform_unassign_tool_from_agent", "platform_update_agent",
    "platform_update_system_setting", "platform_create_mission", "platform_approve_mission",
    "platform_cancel_mission", "platform_publish_blog_post", "platform_submit_social_post",
    # D1 amended 8 Oct (FX-010, night 12: a heartbeat changed, an agent deleted on one word,
    # eleven skills given, playbooks made, none with a card): every agent-setting change, and
    # a playbook made, timed or deleted.
    "platform_configure_agent_heartbeat", "platform_delete_agent", "platform_assign_skill_to_agent",
    "platform_unassign_skill_from_agent", "platform_assign_plugin_to_agent", "platform_create_playbook",
    "platform_schedule_playbook", "platform_delete_playbook",
    # P256-FIX-RVW-14: a timer set through an update, and a plugin or skill taken from every
    # agent (turned off, deleted, or forked with its agents moved onto the fork).
    "platform_update_playbook", "platform_uninstall_plugin", "platform_delete_workspace_skill",
    "platform_update_skill",
    # P256-FIX-RVW-23: an agent's timer is an agent-setting change, as a playbook's timer is.
    "platform_schedule_task",
    # P256-FIX-T4 (Gerard, 11 Oct, benchmark F321-a): a playbook run started from the chat.
    "platform_execute_playbook",
    # P256-FIX-N14-7 (F421): a skill made for the workspace, as a skill given or edited is.
    "platform_create_workspace_skill",
})
CARD_MOVES = frozenset({"platform_update_task_status", "platform_update_task"})
# An update is owner-only only when it sets the playbook's timer (P256-FIX-RVW-14).
TIMED_UPDATES = frozenset({"platform_update_playbook"})
# A create is owner-only only when it makes a cli agent, one that runs a CLI session on the
# owner's paired machine, as switching an agent to cli is (P256-FIX-RVW-28).
SESSION_CREATES = frozenset({"platform_create_agent"})
# P256-FIX-N13-7 (F414): a plan edit is owner-only when it changes a mission's steps or checks.
PLAN_EDITS = frozenset({"platform_update_mission_plan"})
SCHEDULE_CONFIG = "schedule_config"
# P256-FIX-RVW-40: "closed" is the board's close too (the status tool is the one for "close all …").
CLOSING_STATUSES = frozenset({"done", "cancelled", "closed"})
# P256-FIX-T3 (Gerard, 11 Oct): a card its agent has worked on, moved back for rework, is a send-back.
WORKED_STATUSES = frozenset({"review", "done"})
REWORK_STATUSES = frozenset({"assigned", "in_progress", "inbox"})
# D1's "every Composio send/publish action" and D7's order (send_words, shared with brief_sends):
# a slug with one of these words and no read word, split on any non-alphanumeric ('gmail-send-email').
COMPOSIO_SEND_WORDS = frozenset(word.upper() for word in LEAVES_THE_WORKSPACE)
COMPOSIO_READ_WORDS = frozenset(word.upper() for word in READ_WORDS)
_SLUG_WORD = re.compile(r"[^A-Z0-9]+")
MESSAGE_SEND = "message.send"  # the capability classifier's class for a message sent (P256-FIX-RVW-38)

# Server-set keys, never the model's: who clicked (stripped from every call, set after a
# click), and a caller whose own decision is the click (the HARNESS's /approve).
CLICKED_BY = "_clicked_by"
OWNERS_OWN_DECISION = "owners_own_decision"
USER_ACTOR = "user:"
PERMISSION_LEVEL = "write"
DESTRUCTIVE = "destructive"
MAX_CARDS_NAMED = 5
MORE_CARDS = " and {count} more"

ASK = ("Waiting for the owner's click: {act}. Nothing has been done: the approval card in the chat asks "
       "them, and their click runs it.")
# P256-FIX-N13-4 (F411): the card's last line, what approving does and that the decision is kept (services/click_memory).
REMEMBERED = "Approving will {act}; Auto will remember your decision, and a decline too."
CLOSING_VERBS = {"done": "approve (move to Done)", "cancelled": "cancel", "closed": "close"}
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
    "platform_configure_agent_heartbeat": "change an agent's heartbeat",
    "platform_delete_agent": "delete an agent",
    "platform_assign_skill_to_agent": "give a skill to an agent",
    "platform_unassign_skill_from_agent": "take a skill from an agent",
    "platform_assign_plugin_to_agent": "give a plugin to an agent",
    "platform_create_playbook": "create a playbook",
    "platform_schedule_playbook": "set a playbook's timer",
    "platform_delete_playbook": "delete a playbook",
    "platform_update_playbook": "set a playbook's timer",
    "platform_uninstall_plugin": "turn a plugin off and take it from every agent",
    "platform_delete_workspace_skill": "delete a skill and take it from every agent",
    "platform_update_skill": "edit a skill its agents use",
    "platform_create_workspace_skill": "create a skill for this workspace",
    "platform_schedule_task": "set an agent's timer",
    "platform_create_agent": "create an agent that runs as a session on the paired machine",
    "platform_execute_playbook": "run a playbook",
    "platform_update_mission_plan": "change a mission's plan",
}
SEND_VERB = "send, publish or order through"  # P256-FIX-RVW-3: an order asks too
QUESTION = "question_md"


def is_owner_only(action_name: str, params: Any, *, composio: bool = False, db: Any = None,
                  workspace_id: Any = None) -> bool:
    """Whether the call is one only the owner's click may run (Decision D1). A card move is
    when it closes the card, or (P256-FIX-T3) sends back a worked card, read from this
    workspace's board (``db``); without the board, by its status alone."""
    name = str(action_name or "")
    if composio:
        return is_composio_send(name)
    if name in CARD_MOVES:
        params = params_object(params)
        return isinstance(params, dict) and (closing_status(params) is not None
                                             or _sends_back_a_card(db, workspace_id, name, params))
    if name in TIMED_UPDATES:
        params = params_object(params)
        return isinstance(params, dict) and params.get(SCHEDULE_CONFIG) is not None
    if name in SESSION_CREATES:
        from modules.tools.discovery.agent_runtime import creates_a_session_agent

        params = params_object(params)
        return isinstance(params, dict) and creates_a_session_agent(params)
    if name in PLAN_EDITS:
        from modules.tools.discovery.card_question_plans import changes_the_plan

        return changes_the_plan(params_object(params))
    return name in OWNER_ONLY_ACTIONS


def is_composio_send(slug: str) -> bool:
    """A Composio action that sends, publishes or orders: GMAIL_SEND_EMAIL, gmail-send-email,
    LINKEDIN_CREATE_LINKED_IN_POST, SHOPIFY_CREATE_ORDER; a message made or timed and money paid
    out (P256-FIX-RVW-38: SLACK_SCHEDULE_MESSAGE, STRIPE_CREATE_REFUND), or a name the capability
    classifier reads as ``message.send``; and any action the Socials channel registry classes as
    ``publish`` (P256-FIX-RVW-19: a video upload carries no send word)."""
    words = [word for word in _SLUG_WORD.split(str(slug or "").upper()) if word]
    if set(words) & COMPOSIO_SEND_WORDS and not set(words) & COMPOSIO_READ_WORDS:
        return True
    if makes_or_times_a_message(slug_words(slug)) or _a_message_send("_".join(words).lower()):
        return True
    return _a_channel_publish("_".join(words))


def _a_message_send(name: str) -> bool:
    """Whether the capability classifier's ``message.send`` name patterns read the slug (its
    words in order, joined by '_') as a send (a notify, a direct message), and it carries no read word."""
    from modules.tools.capabilities.classifier import ActionClassifier

    if not name or slug_words(name) & READ_WORDS:
        return False
    patterns = ActionClassifier.CAPABILITY_HEURISTICS[MESSAGE_SEND]["name_patterns"]
    return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)


def _a_channel_publish(name: str) -> bool:
    """Whether a seeded channel's adapter (``modules/socials/channel_adapters.py``, read by
    the registry; its slugs live there alone) classes ``name`` as a publish step."""
    from modules.socials.capabilities import SEEDED, publish_candidate

    return bool(name) and publish_candidate(name) == SEEDED


def closing_status(params: Dict[str, Any]) -> Optional[str]:
    """"done", "cancelled" or "closed" when the call closes the card, read the board's way ("approved" is Done)."""
    from modules.tools.execution.call_effects import STATUS_WORDS

    status = str(params.get("status") or "").strip().lower()
    status = STATUS_WORDS.get(status, status)
    return status if status in CLOSING_STATUSES else None


def sends_back(card_status: Any, new_status: Any) -> bool:
    """Whether moving a card from ``card_status`` to ``new_status`` sends its agent's work back for rework."""
    return card_status in WORKED_STATUSES and new_status in REWORK_STATUSES


def _sends_back_a_card(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> bool:
    from modules.tools.discovery.send_backs import sends_back_a_card

    return sends_back_a_card(db, workspace_id, action, params)


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
    signer is always dropped, and an instruction the owner did not give (P256-FIX-N13-7)."""
    @functools.wraps(run_cleared)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any, cleared: Any,
                      handler: Execute) -> Dict[str, Any]:
        params = as_the_owner_said(getattr(self, "db", None), getattr(self, "workspace_id", None), action_name,
                                   _without_a_signer(params), caller_context)
        if isinstance(params, dict) and human_driven(caller_context) and is_owner_only(
                action_name, params, db=getattr(self, "db", None), workspace_id=getattr(self, "workspace_id", None)):
            handler = _on_the_click(action_name, params, caller_context, handler, _gate_claim(cleared))
        return await run_cleared(self, action_name, params, caller_context, cleared, handler)
    return wrapped


def _on_the_click(action: str, asked: Dict[str, Any], caller_context: Any, handler: Execute,
                  gate: Optional[int] = None) -> Execute:
    """``handler`` run on the owner's click on this exact call (``asked``), signed by who
    clicked; without one, the ask. ``gate``: the grant the confirmation gate claimed for
    this call (``Cleared.approved_via_grant_id``). A call naming its agent by name alone
    has no click: the ask binds it (P256-FIX-RVW-9)."""
    async def on_the_click(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        grant = None
        if not names_the_agent_alone(action, asked):
            grant = _the_click(db, workspace_id, action, asked) or _claimed_at_the_gate(db, workspace_id, action, gate)
        if grant is None:
            return platform_ask(db, workspace_id, action, asked, caller_context)
        clicker = _clicker(grant)
        signed = {**params, CLICKED_BY: clicker} if clicker else params
        return after_the_click(db, grant, await handler(db, workspace_id, signed))
    return on_the_click


def asks_before_a_send(execute_tool: Execute) -> Execute:
    """Wrap UnifiedToolExecutor.execute_tool: a Composio send or publish in a person's chat,
    or by an agent on a ticket Auto wrote (FX-011, Decision D7), runs only on the owner's
    click, through the same grant card. The call's kind is read first: any other call runs
    as it is, without touching the executor's session. A Composio call asked for under a
    name that is not a send is watched while it runs (P256-FIX-RVW-3, ``_ResolvedSend``):
    the executor may resolve it onto one."""
    signature = inspect.signature(execute_tool)

    @functools.wraps(execute_tool)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        call = signature.bind(self, *args, **kwargs)
        call.apply_defaults()
        tool, params, ctx = call.arguments["tool_name"], call.arguments["parameters"], call.arguments["caller_context"]
        workspace_id = call.arguments["workspace_id"]
        slug, inner, composio = self._resolve_effective_call(tool, params)
        if not composio:
            return await execute_tool(*call.args, **call.kwargs)
        if not is_composio_send(slug):
            return await _ResolvedSend(self, call.arguments, inner, slug).runs(lambda: execute_tool(*call.args, **call.kwargs))
        call, inner = _told(self, signature, call, slug, inner)  # PRD-264 EUAI-07: the AI line, before the card
        untold = _carries_no_line(call.arguments, slug)
        if untold is not None:
            return {"success": False, "error": untold}
        params = call.arguments["parameters"]
        db = getattr(self, "db", None)
        ticket = _whose_click(db, workspace_id, slug, ctx, composio)
        if ticket is None:
            return await execute_tool(*call.args, **call.kwargs)
        grant = _the_click(db, workspace_id, tool, params)
        if grant is None:
            return _send_card(db, call.arguments, slug, ticket, inner)
        return after_the_click(db, grant, await execute_tool(*call.args, **call.kwargs))
    return wrapped


def _told(executor: Any, signature: inspect.Signature, call: inspect.BoundArguments, slug: str,
          inner: Any) -> tuple:
    """PRD-264 EUAI-07: ``(call, inner)`` with the AI line at the top of the send's body
    (``send_notice.told_call``), so the card shows, and the grant's hash covers, what goes out;
    the call as it is when the line is already there or it has no body."""
    from modules.tools.discovery.send_notice import told_call

    arguments = call.arguments
    told = told_call(getattr(executor, "db", None), arguments["workspace_id"], arguments["tool_name"],
                     arguments["parameters"], slug, arguments["caller_context"])
    if told is arguments["parameters"]:
        return call, inner
    rebound = signature.bind(**{**arguments, "parameters": told})
    return rebound, executor._resolve_effective_call(arguments["tool_name"], told)[1]


def _carries_no_line(arguments: Dict[str, Any], slug: str) -> Optional[str]:
    """PRD-264 EUAI-07: why the told send may not go out (it has no text parameter to carry the AI
    line), refused before any card; None when what it sends starts with the line or needs none."""
    from modules.tools.discovery.send_notice import untold_send

    return untold_send(arguments["tool_name"], arguments["parameters"], slug, arguments["caller_context"])


class _ResolvedSend:
    """P256-FIX-RVW-3: a Composio call asked for under a name that is not a send, watched
    while it runs. Where the executor resolves that name onto another action (a slug form,
    a display name, the auto-map of a near-miss), it checks the action that runs through
    its post gate (core/composio/resolved_action), which asks :meth:`check`: a send there
    waits for the owner's click exactly as one asked for by name. Without the click the card
    is raised, the action never runs and the call's answer is the card; with it, it runs once."""

    def __init__(self, executor: Any, arguments: Dict[str, Any], inner: Any, asked: str = "") -> None:
        self.executor, self.arguments, self.inner, self.asked = executor, arguments, inner, asked
        self.card: Optional[Dict[str, Any]] = None
        self.grant: Any = None
        self.untold: Optional[str] = None   # PRD-264 EUAI-07: why a resolved send with words was refused

    async def runs(self, run: Callable[[], Awaitable[Dict[str, Any]]]) -> Dict[str, Any]:
        """The call's answer: the card when a resolved send raised one, else what ran."""
        from core.composio.resolved_action import checks_the_resolved_action

        with checks_the_resolved_action(self.check):
            result = await run()
        if self.card is not None:
            return self.card
        if self.untold is not None:
            return {"success": False, "error": self.untold}
        return result if self.grant is None else after_the_click(self._db(), self.grant, result)

    async def check(self, action: str) -> Optional[str]:
        """Why ``action``, the one the executor is about to run, may not run yet (the card
        was raised), or None (not a send, a ticket a person wrote, or the owner clicked)."""
        if self.card is not None:
            return self._refusal(action)
        if self.grant is not None or not is_composio_send(action):
            return None  # a read never touches the executor's session (F088)
        args, db = self.arguments, self._db()
        self.untold = self.untold or _untold(args, self.asked, action)  # PRD-264 EUAI-07: no AI line here
        if self.untold:
            return self.untold
        ticket = _whose_click(db, args["workspace_id"], action, args["caller_context"], True)
        if ticket is None:
            return None
        self.grant = _the_click(db, args["workspace_id"], args["tool_name"], args["parameters"])
        if self.grant is not None:
            return None
        self.card = _send_card(db, args, action, ticket, self.inner)
        return self._refusal(action)

    def _refusal(self, action: str) -> str:
        """Never empty: an empty refusal would let the executor run the action."""
        card = self.card or {}
        return str(card.get("message") or card.get("error") or ASK.format(act=f"{SEND_VERB} {action}"))

    def _db(self) -> Any:
        return getattr(self.executor, "db", None)


def _untold(arguments: Dict[str, Any], asked: str, action: str) -> Optional[str]:
    """Why a send the executor resolved onto from another name (``asked``) may not run (``send_notice``)."""
    from modules.tools.discovery.send_notice import untold_resolved_send

    return untold_resolved_send(asked or arguments["tool_name"], action, arguments["caller_context"])


def _send_card(db: Any, arguments: Dict[str, Any], slug: str, ticket: Dict[str, Any], inner: Any) -> Dict[str, Any]:
    """The card for a send that waits for the click: the ask in a person's chat (``ticket``
    is ``{}``), or the agent's on a ticket Auto wrote (FX-011)."""
    from modules.tools.discovery.agent_sends import waits_on_the_card

    workspace_id = arguments["workspace_id"]
    ask = send_ask(db, workspace_id, arguments["tool_name"], slug, arguments["parameters"],
                   arguments["caller_context"], sent=inner)
    if not ticket or not ask.get("requires_confirmation"):  # a refusal raised no card (P256-FIX-RVW-26)
        return ask
    return waits_on_the_card(db, workspace_id, ticket, ask, sent=params_object(inner),
                             agent_id=arguments.get("agent_id"))


def _whose_click(db: Any, workspace_id: Any, slug: str, caller_context: Any, composio: bool) -> Optional[Dict[str, Any]]:
    """Whose click a call waits for: None when it runs as it is (not a Composio send, or
    an agent's send on a ticket a person wrote); ``{}`` in a person's chat; the ticket
    (``agent_sends.autos_ticket``) when an agent sends on a ticket Auto wrote."""
    from modules.tools.discovery.agent_sends import autos_ticket

    if not (composio and is_composio_send(slug)):
        return None
    if human_driven(caller_context):
        return {}
    return autos_ticket(db, workspace_id, caller_context)


def _the_click(db: Any, workspace_id: Any, action: str, params: Any) -> Any:
    """The owner's grant for this exact call, claimed for this one run (single-use, so a
    concurrent or repeated call asks again), or None (the ask stands)."""
    from modules.tools.execution import tool_grants

    return tool_grants.consume_tool_grant(db, workspace_id, action=action, params=params,
                                          permission_level=PERMISSION_LEVEL, single_use=True)


def _gate_claim(cleared: Any) -> Optional[int]:
    """The grant the confirmation gate claimed for this call: only a destructive action's
    grant is single-use there; a write's stays granted for ``_the_click`` to claim."""
    if getattr(getattr(cleared, "action_def", None), "permission_level", None) != DESTRUCTIVE:
        return None
    return getattr(cleared, "approved_via_grant_id", None)


def _claimed_at_the_gate(db: Any, workspace_id: Any, action: str, grant_id: Optional[int]) -> Any:
    """The click the confirmation gate already claimed for this exact call, or None.

    FX-010: a destructive action (platform_delete_agent, platform_delete_playbook) asks at
    the gate too when an editor drives the turn (with this module's ask, P256-FIX-RVW-9:
    confirmation_gate), and the gate retires its single-use grant when it clears. That
    claim, made in this call for these params, is the owner's click: asking again would
    raise a card per click, for ever."""
    if db is None or grant_id is None:
        return None
    from core.models.approval_grants import ApprovalGrant
    from modules.tools.execution.tool_grants import GRANT_CONSUMED_BY

    grant = db.get(ApprovalGrant, grant_id)
    if grant is None or str(grant.workspace_id) != str(workspace_id) or grant.tool_name != action:
        return None
    return grant if grant.revoked_by == GRANT_CONSUMED_BY else None


def after_the_click(db: Any, grant: Any, result: Any) -> Any:
    """One click, one run: a call that did nothing (``success: False``) gives the click
    back for its retry (tool_grants.give_back_unused, F193). The result records which
    grant said yes."""
    from modules.tools.execution import tool_grants

    tool_grants.give_back_unused(db, getattr(grant, "id", None), result)
    if isinstance(result, dict) and result.get("approved_via_grant_id") is None:
        return {**result, "approved_via_grant_id": getattr(grant, "id", None)}
    return result


def platform_ask(db: Any, workspace_id: Any, action: str, params: Dict[str, Any], caller_context: Any, *,
                 permission_level: str = PERMISSION_LEVEL) -> Dict[str, Any]:
    """The ask for a platform action, naming the card by its number and the verb, and
    saying what the call changes (FX-008). A card that is not on the board is never
    asked about (F091). ``permission_level``: the grant's, the action's own when the
    confirmation gate asks (a destructive yes stays single-use there, P256-FIX-RVW-9)."""
    from modules.tools.discovery.agent_binding import bound_to_the_agent
    from modules.tools.discovery.agent_runtime import refused_before_the_card
    from modules.tools.discovery.card_question import platform_question
    from modules.tools.discovery.card_question_skills import bound_to_the_subject
    from modules.tools.discovery.mission_targets import bound_to_the_mission
    from modules.tools.discovery.playbook_binding import bound_to_the_playbook
    from modules.tools.discovery.ticket_edit_moves import rebrief_that_closes
    from modules.tools.execution.subject_targets import missing_targets_error, named_subject

    params = bound_to_the_mission(db, workspace_id, action, params)  # FX-009: the click runs on the mission shown
    params, refused = bound_to_the_agent(db, workspace_id, action, params)  # FX-010: and on the agent shown
    if not refused:  # RVW-14: and on the plugin or skill shown
        params, refused = bound_to_the_subject(db, workspace_id, action, params)
    if not refused:  # RVW-45: and on the playbook shown
        params, refused = bound_to_the_playbook(db, workspace_id, action, params)
    refused = refused or refused_before_the_card(db, workspace_id, action, params)  # FX-016: a runtime it can't set
    refused = refused or rebrief_that_closes(db, workspace_id, action, params)  # RVW-10: re-brief or close, not both
    if refused:
        return refused
    found, missing = _targets(db, workspace_id, action, params)
    if missing:
        return missing_targets_error(action, missing)
    what = (named_subject(found).removeprefix(" on ") or _said_subject(params)) + _cards_not_named(params)
    act = _act(db, workspace_id, action, params, what)
    asked = platform_question(db, workspace_id, action, params, act)
    return _ask(db, workspace_id, action, params, caller_context, act=act, what=what, asked=asked,
                level=permission_level)


def _act(db: Any, workspace_id: Any, action: str, params: Dict[str, Any], what: str) -> str:
    """What the card asks: approve, cancel or close a card, send it back (P256-FIX-T3), or the action's verb."""
    from modules.tools.discovery.send_backs import send_back_act

    status = closing_status(params) if action in CARD_MOVES else None
    if status:
        return f"{CLOSING_VERBS[status]} {what}".strip()
    sent_back = send_back_act(db, workspace_id, action, params, what) if action in CARD_MOVES else None
    return sent_back or f"{VERBS.get(action, action)} {what}".strip()


def send_ask(db: Any, workspace_id: Any, tool: str, slug: str, params: Any, caller_context: Any, *,
             sent: Any = None) -> Dict[str, Any]:
    """The ask for a Composio send or publish, naming the action, and to whom, about what
    and its first line (``sent``: the action's own params, FX-008). A recipient the card
    cannot show is refused before any grant (P256-FIX-RVW-26), as is a draft or a campaign
    sent by reference, whose contents the card cannot show (P256-FIX-RVW-38)."""
    from modules.tools.discovery.card_question import send_question
    from modules.tools.discovery.card_question_sends import refused_as_sent_by_reference, refused_before_the_send_card

    act, shown_params = f"{SEND_VERB} {slug}".strip(), params_object(sent if sent is not None else params)
    refused = refused_as_sent_by_reference(slug, shown_params) or refused_before_the_send_card(shown_params)
    if refused:
        return refused
    asked = send_question(act, shown_params)
    return _ask(db, workspace_id, tool, params, caller_context, act=act, what=slug, asked=asked)


def _ask(db: Any, workspace_id: Any, action: str, params: Any, caller_context: Any, *, act: str,
         what: str, asked: str, level: str = PERMISSION_LEVEL) -> Dict[str, Any]:
    from modules.tools.discovery.card_ticket_numbers import CARD_NUMBERS, numbers_on_the_card
    from modules.tools.execution import tool_grants

    message, asked = ASK.format(act=act), f"{asked}\n{REMEMBERED.format(act=act)}"
    # ``act``: what the card asks, in the owner's words, for the receipt and the model (FX-004);
    # ``question_md``: what the card shows the owner, the subject and the change (FX-008);
    # ``card_numbers``: each ticket the call names, by its board number (P256-FIX-N13-7).
    ask = {"success": False, "requires_confirmation": True, "owner_only": True, "action": action,
           "permission_level": level, "message": message, "params": params, ACT: act,
           QUESTION: asked, CARD_NUMBERS: numbers_on_the_card(db, workspace_id, action, params_object(params))}
    logger.info("[owner_only] %s waits for the owner's click (%s)", action, what)
    return tool_grants.attach_ask_grant(db, workspace_id, action=action, params=params, ask=ask,
                                        permission_level=level, description=message,
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
           "asks_the_owner_first", "human_driven", "is_composio_send", "is_owner_only", "sends_back", "signed_by"]
