"""PRD-268 §6.2.3 — a check-in's read log: what each successful read lets a suggestion cite.

A suggestion's evidence must name reads its own run made and that succeeded
(``modules.proactive.provenance``). The runner keeps that log as ``{type, id}`` refs and
hands it to every call as ``caller_context['check_in_reads']`` (``check_in_lane``):

* Auto's workspace check-in starts with its signals' refs: the run read those itself
  (``services.proactive.signals``).
* Each read action that succeeds adds what its answer names, by the answer's structure and
  never by its text: each ticket it holds, by its board number (``number``, which the ticket
  tools add, ``services.ticket_refs``), and each id it holds, by the key that holds it
  (``report_id``; ``{"watch": {"id": ...}}``; ``{"missions": [{"id": ...}]}``). A
  ``ticket_id`` is a ticket's database id, so it is a ``task``; only a ``number`` is a
  ticket. An id given in an argument the action declares counts only when the answer names
  the same id, so a filter that matched nothing logs nothing.
* Only what the model was shown counts: a ref whose id is not in the answer as the model
  read it (cut to the loop's length) is left out.

A failed read adds nothing. Free-form blobs (``metadata``, ``inputs``, ``configuration`` and
the like) are not walked, and nothing written inside a value (a document's words, a
ticket's description) is ever read as a ref. The walk is breadth first, so an answer's own
items come before what they nest, and it is bounded in depth, in dicts and in refs.
"""
from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Mapping
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Optional, Tuple

from modules.proactive.dedupe import board_number
from modules.proactive.provenance import TICKET, ref_key
from services.ticket_refs import TICKET_LISTS

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_REFS_PER_READ",
    "MAX_RUN_READS",
    "declared_params",
    "read_succeeded",
    "refs_of_read",
    "with_reads",
]

# How deep the walk goes into an answer, how many dicts and items of one list it reads, and
# how many refs one read and one run may log.
MAX_DEPTH = 3
MAX_NODES = 200
MAX_LIST_ITEMS = 50
MAX_REFS_PER_READ = 50
MAX_RUN_READS = 300
MAX_ID_LENGTH = 64
ID_KEY = "id"
ID_SUFFIX = "_id"
# A ticket's database id names a task; only its board number names a ticket.
TASK = "task"
# A kind is a plain identifier: 'mission', 'scheduled_task'.
_KIND = re.compile(r"[a-z][a-z_]{0,31}")
# Where the ticket tools put a ticket: one under 'task', a list under TICKET_LISTS.
TICKET_KEYS = frozenset({TASK, *TICKET_LISTS})
NUMBER_KEY = "number"
# Every row names its workspace; that is never evidence.
NOT_EVIDENCE = frozenset({"workspace"})
# Free-form values a person or an agent wrote, never walked for refs.
FREE_FORM_KEYS = frozenset({
    "metadata", "meta", "custom_fields", "inputs", "input", "params", "parameters", "payload",
    "configuration", "config", "settings", "context", "extra", "raw",
})

Ref = Dict[str, str]
_Node = Tuple[str, str, Mapping]


def read_succeeded(result: Any) -> bool:
    """Whether a read's answer is a success: a dict that neither says ``success: false``
    nor carries an error."""
    return isinstance(result, Mapping) and result.get("success") is not False and not result.get("error")


def declared_params(action_name: str) -> FrozenSet[str]:
    """The argument names ``action_name`` declares; none when its definition cannot be read."""
    try:
        from modules.tools.discovery import get_action_registry

        action = get_action_registry().get(action_name)
        properties = ((getattr(action, "parameters", None) or {}).get("properties")) or {}
    except Exception:  # noqa: BLE001 — an unreadable definition logs no argument as a read
        logger.exception("[proactive] the arguments of %s could not be read; none is logged", action_name)
        return frozenset()
    return frozenset(name for name in properties if isinstance(name, str))


def _ref(kind: Any, value: Any) -> Optional[Ref]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    kind, ident = str(kind).strip().lower(), str(value).strip()
    if not _KIND.fullmatch(kind) or kind in NOT_EVIDENCE or not ident or len(ident) > MAX_ID_LENGTH:
        return None
    return {"type": kind, "id": ident}


def _id_kind(key: Any) -> Optional[str]:
    """The kind a ``<kind>_id`` key names (a ``ticket_id`` is a task's), else None."""
    if not isinstance(key, str) or not key.endswith(ID_SUFFIX) or len(key) <= len(ID_SUFFIX):
        return None
    kind = key[: -len(ID_SUFFIX)]
    return TASK if kind == TICKET else kind


def _singular(key: str) -> str:
    if key.endswith("ies"):
        return key[:-3] + "y"
    if key.endswith(("ches", "shes", "sses", "xes")):
        return key[:-2]
    return key[:-1] if key.endswith("s") else key


def _children(key: str, value: Any) -> Iterator[_Node]:
    """The dicts held under ``key``, each as (key, the kind it names, the dict)."""
    if isinstance(value, Mapping):
        yield key, key, value
    elif isinstance(value, list):
        for item in value[:MAX_LIST_ITEMS]:
            if isinstance(item, Mapping):
                yield key, _singular(key), item


def _nodes(answer: Mapping) -> List[_Node]:
    """The answer's dicts, breadth first from the answer itself; free-form blobs skipped."""
    found: List[_Node] = []
    queue = deque([("", "", answer, 1)])
    while queue and len(found) < MAX_NODES:
        key, kind, node, depth = queue.popleft()
        found.append((key, kind, node))
        if depth >= MAX_DEPTH:
            continue
        for child_key, value in node.items():
            if isinstance(child_key, str) and child_key not in FREE_FORM_KEYS:
                queue.extend((*child, depth + 1) for child in _children(child_key, value))
    return found


def _ticket(key: str, node: Mapping) -> Optional[Ref]:
    number = node.get(NUMBER_KEY)
    if key not in TICKET_KEYS or number is None:
        return None
    canonical = board_number(number)
    return {"type": TICKET, "id": canonical} if canonical else None


def _node_refs(key: str, kind: str, node: Mapping) -> Iterator[Optional[Ref]]:
    yield _ticket(key, node)
    if kind:
        yield _ref(kind, node.get(ID_KEY))
    for child_key, value in node.items():
        id_kind = _id_kind(child_key)
        if id_kind is not None:
            yield _ref(id_kind, value)


def _answer_ids(nodes: List[_Node], refs: Iterable[Optional[Ref]]) -> FrozenSet[Tuple[str, str]]:
    """Every id the answer holds, as ``ref_key`` compares them: each ref the walk found,
    and each dict's own ``id`` under every kind."""
    keys = {ref_key(ref) for ref in refs if ref is not None}
    bare = {str(node.get(ID_KEY)).strip() for _key, _kind, node in nodes if _ref(TASK, node.get(ID_KEY))}
    return frozenset(key for key in keys if key is not None) | frozenset(("*", ident) for ident in bare)


def _named_by(ref: Optional[Ref], answer_ids: FrozenSet[Tuple[str, str]]) -> bool:
    key = ref_key(ref) if ref is not None else None
    return key is not None and (key in answer_ids or ("*", key[1]) in answer_ids)


def _from_params(params: Any, declared: FrozenSet[str]) -> Iterator[Optional[Ref]]:
    if not isinstance(params, Mapping):
        return
    for key, value in params.items():
        if key not in declared:
            continue
        if key == TICKET:
            number = board_number(value) if value is not None else None
            yield {"type": TICKET, "id": number} if number else None
        elif _id_kind(key) is not None:
            yield _ref(_id_kind(key), value)


def _shown(ref: Ref, shown: str) -> bool:
    return ref["id"] in shown


def _unique(refs: Iterable[Optional[Ref]], limit: int) -> List[Ref]:
    seen, kept = set(), []
    for ref in refs:
        key = ref_key(ref) if ref is not None else None
        if key is None or key in seen:
            continue
        seen.add(key)
        kept.append(ref)
        if len(kept) >= limit:
            break
    return kept


def refs_of_read(action_name: str, params: Any, result: Any, shown: str) -> Tuple[Ref, ...]:
    """The refs one read of ``action_name`` logs: none when it failed; else what its answer
    names, and what its declared arguments name that the answer names too, each only when
    ``shown`` (the answer as the model read it) holds its id; at most ``MAX_REFS_PER_READ``."""
    if not read_succeeded(result):
        return ()
    nodes = _nodes(result)
    found = [ref for node in nodes for ref in _node_refs(*node)]
    answer_ids = _answer_ids(nodes, found)
    named = [ref for ref in _from_params(params, declared_params(action_name)) if _named_by(ref, answer_ids)]
    return tuple(_unique((ref for ref in (*named, *found) if ref is not None and _shown(ref, shown)),
                         MAX_REFS_PER_READ))


def with_reads(log: Iterable[Ref], refs: Iterable[Ref]) -> Tuple[Ref, ...]:
    """The run's read log with ``refs`` added (a new tuple): each ref once, at most
    ``MAX_RUN_READS``; the earliest are kept."""
    return tuple(_unique((*log, *refs), MAX_RUN_READS))
