"""PRD-264 EUAI-11a (EU AI Act Art. 50(1)) — every agent answers truthfully about being an AI.

Every agent's system prompt (Auto's, a marketplace agent's, one with a custom persona) ends with
one platform rule: asked whether they are talking to a person or an AI, the agent says it is an
AI agent acting for the workspace's public name, and never claims or implies it is human. The rule
is appended after everything else in the prompt, the persona text included, so no persona, custom
soul or seeded system prompt can remove or override it.

FR-15: the rule names its principal only by a public name: the business profile's
``company_name``, else the workspace's name when it is not the auto-created "{name or email}'s
Workspace" form and has no "@" in it. Otherwise the rule carries no name ("say that you are an AI
agent."), so an owner's own name or email never reaches the prompt.

Three builders produce an agent's system prompt, and each ends with this rule:
``ContextService`` (every mode an agent works in, ``with_rule_for_mode``), the chat's short ATOM
path (``consumers/chatbot/atom_prompt.py``) and the Auto CTO prompt
(``consumers/chatbot/cto_prompt_builder.py``). The modes in ``MACHINE_READ_MODES`` are internal
calls whose output a program parses (a routing decision, SQL, a plan); no person talks to them,
so they carry no rule.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Optional

from modules.context.modes import ContextMode

logger = logging.getLogger(__name__)

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
# A workspace name with this in it may be the owner's email (FR-15).
EMAIL_MARK = "@"
# The name core/auth/hybrid.py gives a new personal workspace: "{name or email}'s Workspace".
AUTO_CREATED_WORKSPACE_NAME = re.compile(r"^.+['’]s workspace$", re.IGNORECASE)

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


def _one_line(name: Any) -> Optional[str]:
    """The name on one line, or None when there is no usable name."""
    if not isinstance(name, str):
        return None
    return " ".join(name.split()) or None


def _capped(name: Optional[str]) -> Optional[str]:
    """The name cut to the rule's length limit."""
    return name[:MAX_PRINCIPAL_NAME_CHARS] if name else None


def public_workspace_name(workspace_name: Any) -> Optional[str]:
    """The workspace's name when it is public (FR-15), else None.

    The auto-created "{name or email}'s Workspace" form and any name with an "@" carry the
    owner's name or email, so they are never used.
    """
    name = _one_line(workspace_name)
    if name is None or EMAIL_MARK in name or AUTO_CREATED_WORKSPACE_NAME.match(name):
        return None
    return _capped(name)


def public_principal_name(company_name: Any, workspace_name: Any) -> Optional[str]:
    """FR-15's public name: the business profile's company name, else a public workspace name."""
    return _capped(_one_line(company_name)) or public_workspace_name(workspace_name)


def ai_identity_rule(principal: Optional[str]) -> str:
    """The rule block, naming ``principal`` (a name from :func:`public_principal_name`) or no one."""
    name = _capped(_one_line(principal))
    words = AI_IDENTITY_RULE_TEMPLATE.format(principal=name) if name else AI_IDENTITY_RULE_UNNAMED
    return f"{AI_IDENTITY_RULE_HEADING}\n{words}"


def with_ai_identity_rule(system_prompt: str, principal: Optional[str]) -> str:
    """A new prompt: ``system_prompt`` with the rule as its last block."""
    rule = ai_identity_rule(principal)
    body = (system_prompt or "").rstrip()
    return f"{body}\n\n{rule}" if body else rule


def _workspace_key(workspace_id: Any) -> Optional[uuid.UUID]:
    """The workspace id as a UUID, or None when it is not one."""
    if not workspace_id:
        return None
    try:
        return uuid.UUID(str(workspace_id))
    except ValueError:
        return None


def _company_name(db_session: Any, key: uuid.UUID) -> Optional[str]:
    """The newest business profile's company name, or None (a failed read is logged)."""
    try:
        from core.models.business_profiles import BusinessProfile

        row = (
            db_session.query(BusinessProfile)
            .filter(BusinessProfile.workspace_id == key, BusinessProfile.company_name.isnot(None))
            .order_by(BusinessProfile.created_at.desc())
            .first()
        )
    except Exception:  # noqa: BLE001 -- logged; the rule then names no company and the turn goes on
        logger.exception("[PRD-264] business profile lookup failed for %s", key)
        return None
    return _one_line(getattr(row, "company_name", None))


def _workspace_row_name(db_session: Any, key: uuid.UUID) -> Optional[str]:
    """The workspace row's raw name, or None (a failed read is logged)."""
    try:
        from core.models.workspaces import Workspace

        row = db_session.query(Workspace).filter(Workspace.id == key).first()
    except Exception:  # noqa: BLE001 -- logged; the rule then names no workspace and the turn goes on
        logger.exception("[PRD-264] workspace name lookup failed for %s", key)
        return None
    return _one_line(getattr(row, "name", None))


def principal_for_workspace(
    db_session: Any, workspace_id: Any, workspace_name: Optional[str] = None
) -> Optional[str]:
    """The workspace's public name (FR-15), or None when it has none or it can't be read.

    ``workspace_name`` (the caller's, when it passed one) stands in for the workspace row's name;
    it is filtered the same way. With no session or no valid id only that name is considered.
    """
    key = _workspace_key(workspace_id)
    if db_session is None or key is None:
        return public_workspace_name(workspace_name)
    company = _company_name(db_session, key)
    if company:
        return public_principal_name(company, None)
    raw_name = _one_line(workspace_name) or _workspace_row_name(db_session, key)
    return public_workspace_name(raw_name)


def with_rule_for_mode(
    system_prompt: str, mode: Any, workspace_name: Optional[str], db_session: Any, workspace_id: Any
) -> str:
    """ContextService's assembled prompt with the rule last, in every mode an agent works in."""
    mode_value = mode.value if isinstance(mode, ContextMode) else str(mode)
    if mode_value in MACHINE_READ_MODES:
        return system_prompt
    principal = principal_for_workspace(db_session, workspace_id, workspace_name)
    return with_ai_identity_rule(system_prompt, principal)
