"""P256-FIX-N13-5 (F412, night 13): what Auto recalls is an earlier conversation, never this chat's turns.

Six brand-new chats opened with "my last reply…" or "as I said" when the chat had no turn
before it. The prompt's memory block listed what was recalled from other chats as bare lines
("Autumn launch — where we left off: …", the thread checkpoint's summary), undated, so the
model read them as its own earlier turns in this chat.

Now each recalled memory reaches the model labelled with where it came from and when ("From
an earlier conversation on 11 Oct: …"), on both memory lanes: the SmartMemoryManager's block
(``recalled_block``, which ``SmartMemoryManager._format_memories_for_llm`` hands its memories
to) and the Context Router's bundle (``labels_the_bundle``, on
``MemorySection._format_context_bundle``). ``REMEMBERED_RULE`` (F316) tells the model that
none of it was said in this chat. The fresh chat's first reply is checked too
(``core/llm/fresh_reply.py``).
"""
from __future__ import annotations

import dataclasses
import functools
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

EARLIER = "From an earlier conversation"
# The SmartMemoryManager block's limits (moved here with its formatting, unchanged).
GLOBAL_LINES = 5
AGENT_LINES = 4
MAX_CHARS = 150
MIN_CHARS = 5
_DATE_KEYS = ("created_at", "updated_at")
# The Context Router bundle's recalled lists (modules/memory/context_router.ContextBundle).
_RECALLED_FIELDS = ("long_term_memories", "temporal_results")


def _day(mem: Dict[str, Any]) -> Optional[str]:
    """The day the memory was stored ("11 Oct"), or None when it carries no readable date."""
    for key in _DATE_KEYS:
        value = mem.get(key)
        if isinstance(value, datetime):
            stamp = value
        elif isinstance(value, str) and value:
            try:
                stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
        else:
            continue
        stamp = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
        return f"{stamp.day} {stamp:%b}"
    return None


def label(mem: Dict[str, Any]) -> str:
    """Where a recalled memory comes from, as the model reads it: "From an earlier conversation on 11 Oct"."""
    day = _day(mem)
    return f"{EARLIER} on {day}" if day else EARLIER


def recalled(mem: Dict[str, Any], text: str) -> str:
    """``text`` (the memory's words) under its label."""
    return f"{label(mem)}: {text}"


def _text(mem: Dict[str, Any]) -> str:
    return str(mem.get("memory") or mem.get("content") or "")


def labelled(memories: Sequence[Any]) -> tuple:
    """Copies of ``memories`` whose words carry their label; a string or an empty one is left as it is."""
    return tuple({**m, "memory": recalled(m, _text(m))} if isinstance(m, dict) and _text(m) else m
                 for m in memories)


def _lines(memories: Sequence[Dict[str, Any]]) -> List[str]:
    """The block's lines for one tier: each memory cut to ``MAX_CHARS``, then labelled."""
    lines = []
    for mem in memories:
        text = mem.get("memory", "")
        if text and len(text) > MIN_CHARS:
            cut = text[:MAX_CHARS] + "..." if len(text) > MAX_CHARS else text
            lines.append(f"  - {recalled(mem, cut)}")
    return lines


def recalled_block(memories: List[Dict[str, Any]], user_name: Optional[str]) -> str:
    """The SmartMemoryManager's memory block: the user's name, what is known about them (global
    memories first), then what is known with this agent, each line labelled."""
    if not memories and not user_name:
        return ""
    lines = [f"User's name: {user_name}"] if user_name else []
    tiers = (("\nAbout this user:", [m for m in memories if m.get("_tier") != "agent"][:GLOBAL_LINES]),
             ("\nWith this agent specifically:", [m for m in memories if m.get("_tier") == "agent"][:AGENT_LINES]))
    for heading, tier in tiers:
        if tier:
            lines.append(heading)
            lines.extend(_lines(tier))
    return "\n".join(lines)


Format = Callable[[Any, Any, Any], str]


def labels_the_bundle(format_bundle: Format) -> Format:
    """Wrap ``MemorySection._format_context_bundle(self, bundle, ctx)``: it formats a copy of the
    Context Router's bundle whose long-term memories and temporal results carry their labels."""
    @functools.wraps(format_bundle)
    def wrapped(self: Any, bundle: Any, ctx: Any) -> str:
        if dataclasses.is_dataclass(bundle) and not isinstance(bundle, type):
            names = {f.name for f in dataclasses.fields(bundle)} & set(_RECALLED_FIELDS)
            bundle = dataclasses.replace(bundle, **{name: labelled(getattr(bundle, name) or ()) for name in names})
        return format_bundle(self, bundle, ctx)
    return wrapped


__all__ = ["EARLIER", "label", "labelled", "labels_the_bundle", "recalled", "recalled_block"]
