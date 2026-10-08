"""What a call did, where its name alone does not say it (F261, night 8).

A reply's claim is backed by an action that succeeded this turn (F108; since PRD-256
FX-007, the receipts' rule, ``consumers/chatbot/claims_backed.py``, reads the effect
each receipt carries from here). Two of the board's actions do different things by
their arguments:

- platform_update_task_status moves a card. To "done" it approves the card, to
  "cancelled" it cancels it, to "assigned" it sends it back to its agent, to
  "in_progress" it starts it. Night 8: asked to cancel #0422, Auto moved it to
  done, then said "Task #0422 has been moved to 'cancelled'". By name, the move
  backed the cancel.
- platform_update_task changes a card's fields, and sends it back to its agent
  only with ``send_back``. Night 8: Auto changed #0451's brief, said it had sent
  it back, and #0451 stayed in Review.

Night 9 (F309): platform_update_task with a status moves the card the way
platform_update_task_status does (``ticket_edit_moves``), and is recorded as that move.

So a successful call of either is recorded with what it did too, as
``<action>:<status>`` or ``<action>:send_back``. A claim is backed by those
(``platform_update_task_status:done`` approves a card: its receipt says "moved to Done").

Night 9b (F319), succeeded calls the families read as nothing done:
- "I've assigned the Shopify Inventory Watchdog to both steps" (chat a0475f96) after two
  platform_update_playbook_step calls with its agent_id: no "assign" in the name. A call
  that writes an agent onto something is recorded with ``<action>:agent_set``.
- the board's words for a status ("send back", "rejected", "approved") were recorded as
  said, so ``…:send back`` backed no send-back: the status is read the board's way
  (``STATUS_WORDS``, which Auto's status tool uses too).
- a card sent back by any path (the status tool, the edit tool's ``send_back``, a mission
  step's own redo) says so in its answer (``sent_back``), recorded as ``send_back``.

Stdlib only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

STATUS_MOVE = "update_task_status"
CARD_EDIT = "update_task"
SENT_BACK = "send_back"
_TRUE = ("true", "yes", "1")
# Night 7b: the words Auto reaches for as a status, read the board's way
# (modules/tools/discovery/ticket_moves.py moves the card by them).
STATUS_WORDS = {
    "approved": "done", "approve": "done",
    "rejected": "assigned", "reject": "assigned",
    "send back": "assigned", "send_back": "assigned", "sent back": "assigned", "sent_back": "assigned",
}
# F319 (night 9b): what names the agent a call writes onto a card, a step or a mission.
AGENT_KEYS = ("agent_id", "agent_name", "assigned_agent_id", "assigned_agent_name")
AGENT_SET = "agent_set"
_WRITES = ("update_", "assign_", "create_", "add_", "set_", "configure_")
# Where a call's own answer sits inside the chat's envelope (tool_router.execute_and_format).
_ENVELOPES = ("raw_result", "data", "result")


def _is_set(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    return value is True


def _status(params: Dict[str, Any]) -> str:
    """The status a call asks for, the board's way ("send back" is "assigned")."""
    said = str(params.get("status") or "").strip().lower()
    return STATUS_WORDS.get(said, said)


def call_effects(action: str, params: Any) -> Tuple[str, ...]:
    """What ``action`` did with ``params``, beside its name: ``("<action>:done",)``
    for a card moved to done, ``("<action>:send_back",)`` for a card sent back
    with its edit, and ``<action>:agent_set`` beside them for a write that names an
    agent (F319); () when the name says it all."""
    if not isinstance(params, dict):
        return ()
    agent = (f"{action}:{AGENT_SET}",) if _sets_an_agent(action, params) else ()
    if action.endswith(STATUS_MOVE):
        status = _status(params)
        return ((f"{action}:{status}",) if status else ()) + agent
    if action.endswith(CARD_EDIT) and _is_set(params.get(SENT_BACK)):
        return (f"{action}:{SENT_BACK}",) + agent
    status = _status(params) if action.endswith(CARD_EDIT) else ""
    return ((f"{action}_status:{status}",) if status else ()) + agent   # F309 (9): its status is the card's move


def _sets_an_agent(action: str, params: Dict[str, Any]) -> bool:
    """A write whose call names an agent: it puts that agent on what it writes (F319)."""
    return any(stem in action for stem in _WRITES) and any(params.get(key) not in (None, "") for key in AGENT_KEYS)


# F308 (night 9): what a mission's answer said of its steps: #0033 was made and approved
# with no check of each step while the reply said "Each step will pause for your approval".
STEPS_CHECKED, STEPS_UNCHECKED = "mission_steps_checked", "mission_steps_unchecked"
# F319 (night 9b): the answer of a call that sent a card back says so.
SENT_BACK_SAID = "sent_back"
# FX-010 (D7): a card filed from the owner's chat whose brief sends or orders waits for their
# review: its answer carries this key (modules/tools/discovery/brief_sends).
REVIEWED_BY_YOU = "reviewed_by_you"
# F351 (night 10b): the calls that make a document the owner finds in Deliverables, and what a
# refused one leaves: "I've generated the letter" after generate_document answered DATA_BAD_JSON
# (chat 8ac5cf3a) is told as tried and refused, never as made. No family's stem is in its name.
DOCUMENT_MAKES = ("generate_document", "create_pdf", "create_docx", "create_xlsx", "create_pptx", "write_file",
                  "html_to_png")
MAKE_REFUSED = "make_refused"


def answers_in(result: Any) -> List[Dict[str, Any]]:
    """The call's answer and the answers inside its envelopes (the chat wraps it as
    ``raw_result``, an executor as ``data``), each a dict."""
    found: List[Dict[str, Any]] = []
    layer = [result]
    for _depth in range(len(_ENVELOPES)):
        layer = [answer for answer in layer if isinstance(answer, dict)]
        found += layer
        layer = [answer.get(key) for answer in layer for key in _ENVELOPES]
    return found


def result_effects(result: Any) -> Tuple[str, ...]:
    """What a call's answer says it did beyond its name: left a mission's steps waiting
    for the owner's check or running on unchecked (F308), sent a card back (F319), or
    filed a card that waits for the owner's review (FX-010); () when it says none."""
    effects: Tuple[str, ...] = ()
    for answer in answers_in(result):
        if answer.get(SENT_BACK_SAID) is True and SENT_BACK not in effects:
            effects += (SENT_BACK,)
        checks = answer.get("checks_each_step")
        if isinstance(checks, bool) and not {STEPS_CHECKED, STEPS_UNCHECKED} & set(effects):
            effects += (STEPS_CHECKED if checks else STEPS_UNCHECKED,)
        if answer.get(REVIEWED_BY_YOU) is True and REVIEWED_BY_YOU not in effects:
            effects += (REVIEWED_BY_YOU,)
    return effects


def refused_effects(action: str) -> Tuple[str, ...]:
    """What a refused call leaves for a reply to claim: that a document was tried and not made
    (F351), so "I've tried again" is backed and "I've generated it" is told as refused; ()
    for any other call."""
    return (MAKE_REFUSED,) if any(stem in action for stem in DOCUMENT_MAKES) else ()


def call_params(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """The parameters the action itself received: platform_execute's ``params``
    (or the keys beside ``action`` when it sent none), else the call's own."""
    if tool_name != "platform_execute" or not isinstance(tool_args, dict):
        return tool_args if isinstance(tool_args, dict) else {}
    inner = tool_args.get("params")
    if isinstance(inner, dict) and inner:
        return inner
    return {k: v for k, v in tool_args.items() if k not in ("action", "name", "params")}


__all__ = ["DOCUMENT_MAKES", "MAKE_REFUSED", "REVIEWED_BY_YOU", "SENT_BACK_SAID", "STATUS_WORDS", "STEPS_CHECKED", "STEPS_UNCHECKED",
           "answers_in", "call_effects", "call_params", "refused_effects", "result_effects"]
