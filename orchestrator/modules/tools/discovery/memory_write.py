"""What ``platform_store_memory`` checks and writes (``handlers_workspace.store_memory``).

P256-FIX-N14-4 (night 14, F418): 'When you tell me to stop querying the database, I stop for
that turn' sat in the owner's standing rules, cut two answers short, and was never the owner's
rule: Auto wrote it, on a turn the owner drove, so it had an owner and counted. A memory is now
the owner's standing rule only in their own words: each of its sentences is one of the owner's
own sentences in this chat, word for word in their order (``rule_wording.in_their_words``), and
then it is stored ``stated_by: owner`` (``injection_filter``). A memory Auto composes, or one
stored on a turn no person drives, is kept and recalled, never a standing rule.
The owner's messages are read from the chat (``said_instructions.chat_owner_turns``: the turn's
own message is saved before the turn runs), never taken from what the model says they said.

The checks and the metadata were split out of ``store_memory`` (over the length limit) as they
were; their refusals are word for word the handler's.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from modules.memory.injection_filter import STATED_BY_KEY, STATED_BY_OWNER, is_standing_rule
from modules.tools.discovery.memory_source import SOURCE_TYPES

# A rule is wholly the owner's words in their order (``rule_wording``); Auto may leave out at most
# this share's complement of the owner's sentence ('Remember that from November we post on
# Tuesdays' stored as 'From November we post on Tuesdays' keeps 7 of 9 words).
STANDING_RULE_COVERAGE = 0.6
SOURCE = "platform_tool"
NOT_THE_OWNERS_RULE = ("Kept as a memory, not as the owner's standing rule: their messages in this chat do "
                       "not say it in these words. To make it their rule, store it as they said it.")


def memory_params_refusal(params: Dict[str, Any]) -> Optional[str]:
    """Why ``store_memory`` refuses these params, or None: a provenance, confidence, type or
    scope it does not take, or content the exclusion policy keeps out (never echoed back)."""
    from modules.memory.write_contract import MEMORY_FACT_TYPES, MEMORY_SCOPES, violates_exclusions

    if params.get("source_type", "inference") not in SOURCE_TYPES:
        return f"source_type must be one of: {', '.join(sorted(SOURCE_TYPES))}"
    if params.get("confidence") is not None and _unit_float(params["confidence"]) is None:
        return "confidence must be a number between 0 and 1"
    fact_type = params.get("type")
    if fact_type is not None and fact_type not in MEMORY_FACT_TYPES:
        return f"type must be one of: {', '.join(sorted(MEMORY_FACT_TYPES))}"
    scope = params.get("scope")
    if scope is not None and scope not in MEMORY_SCOPES:
        return f"scope must be one of: {', '.join(sorted(MEMORY_SCOPES))}"
    # PRD-206 S1 (Q3 silent-everything): the exclusion validator IS the consent gate.
    violation = violates_exclusions(str(params.get("content")))
    if violation:
        return (f"Refused by the memory exclusion policy (rule: {violation}). Secrets, credentials, "
                "payment and similar sensitive data are never stored as memory.")
    return None


def _unit_float(value: Any) -> Optional[float]:
    """``value`` as a number from 0 to 1, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None


def stated_by_the_owner(db: Optional[Session], workspace_id: Any, params: Dict[str, Any],
                        owner: Optional[str]) -> bool:
    """Whether the memory is in the words of its owner's own messages in this chat (an owned
    write only: a turn no person drives has none)."""
    from modules.tools.discovery.rule_wording import in_their_words
    from modules.tools.discovery.said_instructions import chat_owner_turns

    if not owner:
        return False
    said = chat_owner_turns(db, workspace_id, params)
    return in_their_words(str(params.get("content") or ""), said, STANDING_RULE_COVERAGE)


def memory_metadata(db: Optional[Session], workspace_id: str, params: Dict[str, Any],
                    owner: Optional[str]) -> Dict[str, Any]:
    """The canonical metadata ``store_memory`` writes (PRD-206 S1's write contract: type, scope,
    provenance, owner, chat link). ``_origin_chat_id`` is server-injected, never the caller's."""
    from modules.memory.write_contract import build_memory_metadata

    extra: Dict[str, Any] = {"workspace_id": workspace_id, "source": SOURCE,
                             "verified_at": datetime.now(timezone.utc).isoformat()}
    if params.get("evidence_uri"):
        extra["evidence_uri"] = params["evidence_uri"]
    if stated_by_the_owner(db, workspace_id, params, owner):
        extra[STATED_BY_KEY] = STATED_BY_OWNER
    return build_memory_metadata(
        fact_type=params.get("type") or "business_fact",
        importance=params.get("importance"),
        confidence=_unit_float(params.get("confidence")),
        source_type=params.get("source_type", "inference"),
        scope=params.get("scope"),
        owner=owner,
        chat_id=params.get("_origin_chat_id"),
        pinned=bool(params["pinned"]) if params.get("pinned") is not None else None,
        extra=extra,
    )


def rule_answer(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """What ``store_memory``'s answer says of the rule: whether this write is the owner's standing
    rule and, for one a person owns that is not, why (so Auto never tells them it is)."""
    standing = is_standing_rule({"metadata": metadata})
    if standing or not metadata.get("owner"):
        return {"standing_rule": standing}
    return {"standing_rule": False, "rule_note": NOT_THE_OWNERS_RULE}


__all__ = ["STANDING_RULE_COVERAGE", "memory_metadata", "memory_params_refusal", "rule_answer",
           "stated_by_the_owner"]
