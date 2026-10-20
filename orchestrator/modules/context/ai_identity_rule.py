"""PRD-264 EUAI-11a (EU AI Act Art. 50(1)) — every agent answers truthfully about being an AI.

Every agent's system prompt (Auto's, a marketplace agent's, one with a custom persona) ends with
one platform rule: asked whether they are talking to a person or an AI, the agent says it is an
AI agent acting for the workspace's public name, and never claims or implies it is human. The rule
is appended after everything else in the prompt, the persona text included, so no persona, custom
soul or seeded system prompt can remove or override it.

FR-15: the rule names its principal only by a public name, decided in one place in the shared
kernel, ``core/public_principal.py``, for every surface that names one. On a widget turn (a public
visitor) that is the Site's ``display_name``, never the workspace's name; otherwise the business
profile's ``company_name``, else the workspace's name when it is not the auto-created
"{name or email}'s Workspace" form and has no "@" in it. With no public name the rule carries none
("say that you are an AI agent."), so an owner's own name or email never reaches the prompt.

Three builders produce an agent's system prompt, and each ends with this rule:
``ContextService`` (every mode an agent works in, ``with_rule_for_mode``), the chat's short ATOM
path (``consumers/chatbot/atom_prompt.py``) and the Auto CTO prompt
(``consumers/chatbot/cto_prompt_builder.py``). The modes in ``MACHINE_READ_MODES`` are internal
calls whose output a program parses (a routing decision, SQL, a plan); no person talks to them,
so they carry no rule.
"""

from __future__ import annotations

from typing import Any, Optional

from core.public_principal import site_principal_name, workspace_principal_name
from modules.context.modes import ContextMode

AI_IDENTITY_RULE_HEADING = "## Platform rule: you are an AI"
AI_IDENTITY_RULE_TEMPLATE = (
    "If anyone asks whether they are talking to a person or an AI, say that you are an AI agent "
    "acting for {principal}. Never claim or imply that you are human."
)
# FR-15: with no public name the "acting for" clause is dropped.
AI_IDENTITY_RULE_UNNAMED = (
    "If anyone asks whether they are talking to a person or an AI, say that you are an AI agent. "
    "Never claim or imply that you are human."
)
# A public name is kept to one short line in the rule.
MAX_PRINCIPAL_NAME_CHARS = 120

# Internal stages whose output a program parses, not a person reads.
MACHINE_READ_MODES: frozenset[str] = frozenset(
    {
        ContextMode.ROUTER.value,
        ContextMode.NL2SQL.value,
        ContextMode.PLANNING.value,
        ContextMode.COORDINATOR.value,
        ContextMode.ORCHESTRATOR_STAGE.value,
    }
)


def _rule_line(name: Any) -> Optional[str]:
    """The name on one line, cut to the rule's length limit, or None when there is no name."""
    if not isinstance(name, str):
        return None
    line = " ".join(name.split())
    return line[:MAX_PRINCIPAL_NAME_CHARS] or None


def ai_identity_rule(principal: Optional[str]) -> str:
    """The rule block, naming ``principal`` (a name from :func:`principal_for_turn`) or no one."""
    name = _rule_line(principal)
    words = AI_IDENTITY_RULE_TEMPLATE.format(principal=name) if name else AI_IDENTITY_RULE_UNNAMED
    return f"{AI_IDENTITY_RULE_HEADING}\n{words}"


def with_ai_identity_rule(system_prompt: str, principal: Optional[str]) -> str:
    """A new prompt: ``system_prompt`` with the rule as its last block."""
    rule = ai_identity_rule(principal)
    body = (system_prompt or "").rstrip()
    return f"{body}\n\n{rule}" if body else rule


def principal_for_turn(db_session: Any, workspace_id: Any, widget_mode: bool) -> Optional[str]:
    """The public name the rule gives on this turn (FR-15), or None.

    A widget turn speaks for the workspace's Site, so only the Site's public ``display_name``;
    any other turn, the workspace's public name, read the way the channel notices read it.
    """
    if widget_mode:
        return site_principal_name(db_session, workspace_id)
    return workspace_principal_name(db_session, workspace_id)


def with_rule_for_mode(
    system_prompt: str, mode: Any, db_session: Any, workspace_id: Any, widget_mode: bool
) -> str:
    """ContextService's assembled prompt with the rule last, in every mode an agent works in."""
    mode_value = mode.value if isinstance(mode, ContextMode) else str(mode)
    if mode_value in MACHINE_READ_MODES:
        return system_prompt
    principal = principal_for_turn(db_session, workspace_id, widget_mode)
    return with_ai_identity_rule(system_prompt, principal)
