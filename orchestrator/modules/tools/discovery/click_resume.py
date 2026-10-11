"""P256-FIX-RVW-18: a call resumed from the owner's click is the owner's decision, not their latest words.

The click's resume (api/approval_grants ``_resume_tool_call``) replays the stored call with
the chat's ``caller_context``, so ``follows_the_owner`` and ``answers_the_question_first``
read the owner's LATEST message at the time of the click, not the turn that raised the
card. "Change GREEN BUYER's heartbeat to 15 min" raised the card; the owner then asked
"how's #0960?" and clicked; the click was refused ("The owner named #0960 … an agent
isn't what they asked about") and nothing ran.

The resume marks its context with the grant it runs on (``RESUMED_GRANT``, set by the
server, never a model key: the model writes params, not the context). The word rules skip
a call only when that grant is in the database as the owner's live yes for this exact
call (``on_the_click``): this workspace, this action, these params, still GRANTED.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from modules.tools.execution.params_text import params_object

logger = logging.getLogger(__name__)

RESUMED_GRANT = "_resumed_from_grant"


def resumed_context(caller_context: Any, grant_id: Any) -> Optional[Dict[str, Any]]:
    """The stored call's context for the click's resume, carrying the grant it runs on.
    No context stays none: a call made outside a chat is judged by no word rule anyway."""
    if not isinstance(caller_context, dict):
        return None
    return {**caller_context, RESUMED_GRANT: grant_id}


def on_the_click(db: Any, workspace_id: Any, action: str, params: Any, caller_context: Any) -> bool:
    """Whether this call is the click's resume of the owner's granted yes for exactly it.
    Fails closed: any doubt and the word rules judge the call as before."""
    grant_id = caller_context.get(RESUMED_GRANT) if isinstance(caller_context, dict) else None
    if db is None or grant_id is None:
        return False
    try:
        return _the_granted_call(db, workspace_id, action, params, grant_id)
    except Exception:
        logger.exception("[click_resume] could not read grant %s for %s; the word rules judge it", grant_id, action)
        return False


def _the_granted_call(db: Any, workspace_id: Any, action: str, params: Any, grant_id: Any) -> bool:
    from core.models.approval_grants import SUBJECT_TOOL_CALL, ApprovalGrant
    from core.services.approval_grants import is_authorising
    from modules.tools.execution.tool_grants import tool_call_subject_id

    grant = db.get(ApprovalGrant, int(grant_id))
    if grant is None or grant.subject_type != SUBJECT_TOOL_CALL or not is_authorising(grant):
        return False
    if str(grant.workspace_id) != str(workspace_id):
        return False
    return grant.subject_id == tool_call_subject_id(workspace_id, action, params_object(params))


__all__ = ["RESUMED_GRANT", "on_the_click", "resumed_context"]
