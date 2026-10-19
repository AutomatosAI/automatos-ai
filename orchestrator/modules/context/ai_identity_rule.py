"""PRD-264 EUAI-11a (EU AI Act Art. 50(1)) — every agent answers truthfully about being an AI.

Every agent's system prompt (Auto's, a marketplace agent's, one with a custom persona) ends with
one platform rule: asked whether they are talking to a person or an AI, the agent says it is an
AI agent acting for the workspace, and never claims or implies it is human. The rule is appended
after everything else in the prompt, the persona text included, so no persona, custom soul or
seeded system prompt can remove or override it.

Three builders produce an agent's system prompt, and each ends with this rule:
``ContextService`` (every mode an agent works in, ``with_rule_for_mode``), the chat's short ATOM
path (``consumers/chatbot/atom_prompt.py``) and the Auto CTO prompt
(``consumers/chatbot/cto_prompt_builder.py``). The modes in ``MACHINE_READ_MODES`` are internal
calls whose output a program parses (a routing decision, SQL, a plan); no person talks to them,
so they carry no rule.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from modules.context.modes import ContextMode

logger = logging.getLogger(__name__)

AI_IDENTITY_RULE_HEADING = "## Platform rule: you are an AI"
AI_IDENTITY_RULE_TEMPLATE = (
    "If anyone asks whether they are talking to a person or an AI, say that you are an AI agent "
    "acting for {workspace}. Never claim or imply that you are human."
)
# The words used when the workspace's name is not known to the builder (the ATOM and CTO paths).
UNNAMED_WORKSPACE = "this workspace"
# A workspace name is the owner's own text; it is kept to one short line in the rule.
MAX_WORKSPACE_NAME_CHARS = 120

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


def _clean_name(name: Any) -> Optional[str]:
    """The name on one line, capped, or None when there is no usable name."""
    if not isinstance(name, str):
        return None
    single_line = " ".join(name.split())
    return single_line[:MAX_WORKSPACE_NAME_CHARS] or None


def ai_identity_rule(workspace_name: Optional[str]) -> str:
    """The rule block, naming the workspace the agent acts for."""
    workspace = _clean_name(workspace_name) or UNNAMED_WORKSPACE
    return f"{AI_IDENTITY_RULE_HEADING}\n{AI_IDENTITY_RULE_TEMPLATE.format(workspace=workspace)}"


def with_ai_identity_rule(system_prompt: str, workspace_name: Optional[str]) -> str:
    """A new prompt: ``system_prompt`` with the rule as its last block."""
    rule = ai_identity_rule(workspace_name)
    body = (system_prompt or "").rstrip()
    return f"{body}\n\n{rule}" if body else rule


def workspace_display_name(db_session: Any, workspace_id: Any) -> Optional[str]:
    """The workspace's name from its row, or None when there is no session, no valid id or no row.

    A lookup failure is logged and answers None: the rule then says "this workspace", which is
    still true, and the turn goes on.
    """
    if db_session is None or not workspace_id:
        return None
    try:
        key = uuid.UUID(str(workspace_id))
    except ValueError:
        return None
    try:
        from core.models.workspaces import Workspace

        row = db_session.query(Workspace).filter(Workspace.id == key).first()
    except Exception:
        logger.exception("[PRD-264] workspace name lookup failed for %s", workspace_id)
        return None
    return _clean_name(getattr(row, "name", None))


def with_rule_for_mode(
    system_prompt: str, mode: Any, workspace_name: Optional[str], db_session: Any, workspace_id: Any
) -> str:
    """ContextService's assembled prompt with the rule last, in every mode an agent works in.

    ``workspace_name`` (the caller's, when it passed one) wins over the database lookup.
    """
    mode_value = mode.value if isinstance(mode, ContextMode) else str(mode)
    if mode_value in MACHINE_READ_MODES:
        return system_prompt
    name = _clean_name(workspace_name) or workspace_display_name(db_session, workspace_id)
    return with_ai_identity_rule(system_prompt, name)
