"""PRD-268 §6.2.3 — provenance: a suggestion cites only what its own run read.

Every evidence ref of a suggestion must name a read the same run made and that
succeeded (the run's read log, ``caller_context['check_in_reads']``); otherwise the
suggestion tool refuses it. OpenDuet checks the same against its audit table
(``engine.ts:443-460``). A suggestion that cites nothing is refused too: a check-in
says what it saw.

Refs are ``{type, id}``. They match on the type, case folded, and the id as text, so a
read logged with id 846 and evidence citing "846" are the same read. A ticket's id is
its board number, read the way its subject key reads it: "#0846", "0846" and 846 match.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, FrozenSet, Iterable, Optional, Tuple

from modules.proactive.dedupe import board_number

__all__ = ["MAX_EVIDENCE", "TICKET", "provenance_ok", "ref_key"]

# A suggestion cites at most five reads (§6.2.4).
MAX_EVIDENCE = 5
# The ref type whose id is a board number.
TICKET = "ticket"


def _ident(kind: str, raw_id: Any) -> str:
    if raw_id is None:
        return ""
    if kind == TICKET:
        return board_number(raw_id) or str(raw_id).strip()
    return str(raw_id).strip()


def ref_key(ref: Any) -> Optional[Tuple[str, str]]:
    """A ``{type, id}`` ref as ``(type, id)`` to compare, or None when it is not one."""
    if not isinstance(ref, Mapping):
        return None
    kind = str(ref.get("type") or "").strip().casefold()
    ident = _ident(kind, ref.get("id"))
    return (kind, ident) if kind and ident else None


def _keys(refs: Iterable[Any]) -> FrozenSet[Tuple[str, str]]:
    return frozenset(key for key in map(ref_key, refs) if key is not None)


def provenance_ok(evidence_refs: Iterable[Any], run_reads: Iterable[Any]) -> bool:
    """True when the evidence names one to five refs and each is one of ``run_reads``.

    ``run_reads`` holds the refs of the run's reads that succeeded, and only those: the
    caller leaves a failed read out. A malformed ref, more than five, or none at all is
    not provenance.
    """
    evidence = list(evidence_refs or ())
    if not evidence or len(evidence) > MAX_EVIDENCE:
        return False
    keys = [ref_key(ref) for ref in evidence]
    if any(key is None for key in keys):
        return False
    return set(keys) <= _keys(run_reads or ())
