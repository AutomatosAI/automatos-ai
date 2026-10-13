"""PRD-185 S11 — assembly-side memory-injection guard.

Sub-floor and noise-typed memories must never reach the prompt. The relevance
floor is already applied at the L3 search boundary
(``durable_store.filter_by_relevance_floor``, PRD-159 S3); this is the guard over
the **merged** candidate set at the one chokepoint that feeds
``_format_memories_for_llm`` — so it re-asserts the floor over every source
(L3 global + agent tiers, and any future L2) AND adds the content-type exclusion
the search layer lacks (heartbeat digests, playbook execution summaries).

F325 (night 9b): a memory that names an agent's report or output as where a fact came
from is dropped here too (``modules.memory.agent_sources``, by the names agents file
under), so Auto's recall and an agent's context never carry it as the owner's facts.

Pure — no I/O — so it unit-tests with plain dicts.
"""
from typing import Any, Dict, Iterable, List, Optional

from modules.memory.agent_sources import memory_text, names_an_agents_document

# ``content_type`` / ``metadata.type`` values that are operational noise, never
# user context: heartbeat digests and playbook/recipe execution summaries. These
# are excluded from prompt injection. ``recipe_summary`` is the legacy name for
# ``playbook_summary`` (pre-March rename) — kept so old rows are filtered too.
EXCLUDED_INJECTION_CONTENT_TYPES = frozenset({
    "heartbeat_log",
    "playbook_summary",
    "recipe_summary",
})

# F182 (night 6): raw chat transcripts (L2's record of a conversation, and the
# retired per-turn ``exchange`` rows), which L2 promotion copies into L3
# verbatim. Autonomous work never reads another conversation as its facts.
CHAT_TRANSCRIPT_CONTENT_TYPES = frozenset({"transcript", "exchange"})


def _content_type_of(mem: Dict[str, Any]) -> Optional[str]:
    """Best-effort content-type signal across the shapes a memory row takes.

    Mem0 search rows expose ``{id, memory, score, metadata, created_at}`` — the
    type, when present, rides in ``metadata`` (``content_type`` or ``type``);
    L2-shaped rows carry a top-level ``content_type`` (or ``category``). Checking
    every path means the filter bites wherever the tag lands instead of silently
    no-op-ing on a shape mismatch — the exact failure class this wave exists for.
    A row promoted from L2 keeps its type as ``metadata.category``
    (``store_long_term``), the path F182 found unread.
    """
    if not isinstance(mem, dict):
        return None
    meta = mem.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    return (
        mem.get("content_type")
        or meta.get("content_type")
        or meta.get("type")
        or mem.get("category")
        or meta.get("category")
    )


def visible_to_viewer(mem: Dict[str, Any], viewer_subject_id: Optional[str] = None) -> bool:
    """PRD-206 S1 (Q7) read-side rule: private memories belong to their owner.

    - No ``scope`` (legacy rows) or ``scope='workspace'`` → visible to everyone.
    - ``scope='private'`` → visible ONLY when the viewer's subject tag equals
      the memory's ``owner``. An unknown viewer (headless/background context)
      fails closed; an ownerless private row is visible to no one.
    """
    if not isinstance(mem, dict):
        return False
    meta = mem.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    if meta.get("scope") != "private":
        return True
    owner = meta.get("owner")
    return bool(viewer_subject_id) and owner == viewer_subject_id


def deletable_by(mem: Dict[str, Any], deleter_subject_id: Optional[str] = None) -> bool:
    """P256-FIX-N14-4: a private memory with an owner is deleted by that owner alone (a member's
    standing rule is theirs to forget); any other memory of the workspace by whoever may delete
    there. ``mem`` is a stored point's payload or a memory row (both carry ``metadata``)."""
    meta = mem.get("metadata") if isinstance(mem, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    if meta.get("scope") != "private" or not meta.get("owner"):
        return True
    return bool(deleter_subject_id) and meta["owner"] == deleter_subject_id


def filter_injectable_memories(
    memories: Iterable[Dict[str, Any]],
    *,
    floor: float,
    excluded_types: Iterable[str] = EXCLUDED_INJECTION_CONTENT_TYPES,
    viewer_subject_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Drop sub-floor, noise-typed and not-visible memories before prompt injection.

    - Scored-but-below-``floor`` rows are dropped; unscored rows are kept (cannot
      judge — same rule as ``filter_by_relevance_floor``). ``floor <= 0`` disables
      the score check.
    - Rows whose content-type signal is in ``excluded_types`` are dropped.
    - PRD-206 S7: rows not visible to the viewer (Q7 private scope) are dropped —
      pass ``viewer_subject_id`` (``user:{users.id}``) where the caller knows the
      human; None keeps legacy/workspace rows and drops only private ones.
    - F325: rows that name a document an agent wrote are dropped.
    """
    excluded = frozenset(excluded_types)
    out: List[Dict[str, Any]] = []
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        score = mem.get("score")
        if floor and floor > 0 and score is not None and (score or 0) < floor:
            continue
        if _content_type_of(mem) in excluded:
            continue
        if not visible_to_viewer(mem, viewer_subject_id):
            continue
        if names_an_agents_document(memory_text(mem)):
            continue
        out.append(mem)
    return out


# PRD-256 FX-015 (night 12, F393): the owner's standing rules, which every chat turn carries
# whatever its intent (``modules.context.sections.memory.StandingRulesSection``). A rule is a
# person's: it has an owner (the executor gives a ``store_memory`` row one only from the person
# driving the turn), so an agent's note on a ticket and a legacy row with no owner are never one.
# A rule rides only its own person's turns: one member's note is never put to another as what
# they asked for (recall still finds it).
# P256-FIX-N14-4 (night 14, F418): and it is in the person's own words. ``store_memory`` marks a
# row ``stated_by: owner`` only when that person's own messages in the chat say what it stores
# (``modules.tools.discovery.memory_write``); a memory Auto or an agent composed (night 14's 'When
# you tell me to stop querying the database, I stop for that turn'), and a fact the distiller
# drew from a chat, is recalled as memory but is never a standing rule.
STATED_BY_KEY = "stated_by"
STATED_BY_OWNER = "owner"
# The durable-store read: a row matching any of these is read, then ``is_standing_rule`` decides.
STANDING_RULE_FILTER: Dict[str, List[str]] = {f"metadata.{STATED_BY_KEY}": [STATED_BY_OWNER]}
# P256-FIX-RVW-11: the read holds only the viewer's rows (``metadata.owner`` == their subject id).
STANDING_RULE_OWNER_KEY = "metadata.owner"


# The keys only the platform writes: who a memory belongs to, and that it is in their words.
SERVER_SET_KEYS = ("owner", STATED_BY_KEY)


def without_server_keys(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """P256-FIX-N14-4: a caller's metadata with the platform's own keys taken off (a new dict), so
    no request can store a memory as another person's, or as their standing rule."""
    return {k: v for k, v in (metadata or {}).items() if k not in SERVER_SET_KEYS}


def is_standing_rule(mem: Any) -> bool:
    """A memory with an owner, stored in that person's own words (``stated_by: owner``)."""
    meta = mem.get("metadata") if isinstance(mem, dict) else None
    if not isinstance(meta, dict) or not meta.get("owner"):
        return False
    return meta.get(STATED_BY_KEY) == STATED_BY_OWNER


def standing_rule_rows(memories: Iterable[Any], viewer_subject_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """The rows of the standing rules ``viewer_subject_id`` stated, newest first (none for an
    unknown viewer); through the same guard as recall (no noise, no agent's document)."""
    rules = [m for m in memories if is_standing_rule(m) and viewer_subject_id
             and m["metadata"]["owner"] == viewer_subject_id]
    visible = filter_injectable_memories(rules, floor=0, viewer_subject_id=viewer_subject_id)
    return sorted(visible, key=lambda m: str(m.get("created_at") or ""), reverse=True)


def standing_rules(memories: Iterable[Any], viewer_subject_id: Optional[str] = None) -> List[str]:
    """The standing rules ``viewer_subject_id`` stated, newest first, each once."""
    return rule_texts(standing_rule_rows(memories, viewer_subject_id))


def rule_texts(newest: Iterable[Dict[str, Any]]) -> List[str]:
    """The rules' texts in order, each once (whitespace and case aside)."""
    seen: set = set()
    texts: List[str] = []
    for row in newest:
        text = " ".join(memory_text(row).split())
        if text and text.lower() not in seen:
            seen.add(text.lower())
            texts.append(text)
    return texts
