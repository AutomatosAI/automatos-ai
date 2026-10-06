"""The brand kit proposal card, as data (PRD-255 US-014, FR-11).

The Brand designer changes the kit only after the owner approves a proposal card.
The card is code, not prose the model must remember: ``platform_propose_brand_kit``
files it with the proposal and the Brand Board drawn from it in the grant's
``details`` (:data:`PROPOSAL_MARKER`), and ``platform_save_approved_brand_kit``
saves exactly that proposal, and only when the card's answer is :data:`APPROVE`.

Pure functions only: what changes, the card's text, the stored kit's fingerprint
(a kit changed since the proposal is not saved over), and whether a card's answer
lets its proposal be saved.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

APPROVE, REVISE = "Approve", "Revise"
CARD_OPTIONS = (APPROVE, REVISE)
# ApprovalGrant.details[PROPOSAL_MARKER] = {proposal, board_path, base, saved_at?}
PROPOSAL_MARKER = "brand_kit_proposal"
PENDING, ANSWERED = "pending", "granted"   # a question's status: open, answered (PRD-225)
# Who may approve a kit change: an answer from the Questions tab, whose route requires a
# workspace owner or admin (api/approval_grants.py, require_workspace_admin) and records
# "user:<id>". A Telegram reply ("telegram:<id>") proves no role: it never saves the kit.
ADMIN_ANSWER_PREFIX = "user:"
# The card shows every change in full (the owner approves what they saw, never a cut of it):
# a proposal whose card would be longer is refused, to be split. Under Telegram's 4096.
MAX_CARD_CHARS = 3500
MAX_WHY_CHARS = 300             # the designer's one line on why
NAME_HASH_CHARS = 12            # the board picture's name carries the proposal's hash

CARD_HEAD = "**Brand kit proposal** from {agent} (not saved yet)"
CHANGES_HEAD = "What changes:"
CHANGE_LINE = "- `{field}`: {old} → {new}"
# F364 (night 10c): approving a sign-off of "Automatos AI" replaced the owner's name on every
# letter and the card never said so. A change whose reach is wider than its name says it.
CHANGE_EFFECTS = {
    "voice.sign_off": ("this changes the signature on every letter and document that signs with the kit's sign-off "
                       "(a document that names its own signer keeps it)"),
}
BOARD_LINE = "The Brand Board drawn from it: `{path}`"
CARD_FOOT = ("**Approve** on the Questions tab in Automatos saves it to the brand kit. Anything else, such as "
             "\"less orange\", \"warmer\" or \"more space\", sends it back for a revision.")
UNSET = "(none)"
ELLIPSIS = "…"

STILL_WAITING = ("Your proposal card (question #{ask}) is still waiting for the owner: nothing was drawn or "
                 "asked. End your turn; the answer resumes you.")
NOT_ANSWERED = "The owner has not answered your proposal card (question #{ask}) yet: nothing was saved. End your turn; the answer resumes you."
REVISION = ("The owner sent your proposal back (question #{ask}): {answer!r}. Nothing was saved. Revise the "
            "proposal, look at the board again and propose again with propose_brand_kit.")
CLOSED = "Your proposal card (question #{ask}) was {status}, not approved: nothing was saved. Propose again."
NOT_IN_APP = ("The Approve on your proposal card (question #{ask}) came from {who}, not from the Questions tab, "
              "where only a workspace owner or admin answers: nothing was saved. Propose again and ask the owner "
              "to approve it in Automatos.")
ALREADY_SAVED = "The proposal the owner approved (question #{ask}) is already saved. Propose again to change more."


def fingerprint(stored_kit: Any) -> str:
    """The stored kit's fingerprint: a kit changed after the proposal was drawn is not saved over."""
    text = json.dumps(stored_kit or {}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def board_name(proposal: Mapping[str, Any]) -> str:
    """The board picture's bare file name: one per proposal, so an earlier card keeps its own board."""
    return f"brand-board-proposal-{fingerprint(dict(proposal))[:NAME_HASH_CHARS]}.png"


def _shown(value: Any) -> str:
    """A value as the card shows it: whole, never cut."""
    if value in (None, "", [], {}):
        return UNSET
    return f"`{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)}`"


def changes(current: Mapping[str, Any], proposed: Mapping[str, Any], fields: Sequence[str]) -> List[Dict[str, str]]:
    """Each field ``fields`` change, ``{field, old, new}``; a dict field by its keys (``palette.accent``)."""
    out: List[Dict[str, str]] = []
    for field in fields:
        old, new = current.get(field), proposed.get(field)
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            keys = sorted(set(old) | set(new))
            out += [{"field": f"{field}.{k}", "old": _shown(old.get(k)), "new": _shown(new.get(k))}
                    for k in keys if old.get(k) != new.get(k)]
        elif old != new:
            out.append({"field": field, "old": _shown(old), "new": _shown(new)})
    return out


def _change_line(change: Mapping[str, str]) -> str:
    """One change as the card lists it, with what it reaches when that is wider than its name (F364)."""
    effect = CHANGE_EFFECTS.get(change["field"])
    line = CHANGE_LINE.format(**change)
    return f"{line}: {effect}" if effect else line


def card_text(agent: str, why: str, changed: Sequence[Mapping[str, str]], board_path: str) -> str:
    """The card's markdown: who proposes, why, every change in full, where the board is, and what Approve does.

    The caller refuses a card longer than :data:`MAX_CARD_CHARS` (:func:`too_long`) rather than cut it.
    """
    lines = [CARD_HEAD.format(agent=agent)]
    why = " ".join(str(why or "").split())
    if why:
        lines.append(why if len(why) <= MAX_WHY_CHARS else why[:MAX_WHY_CHARS - 1] + ELLIPSIS)
    lines.append("\n".join([CHANGES_HEAD, *(_change_line(c) for c in changed)]))
    lines += [BOARD_LINE.format(path=board_path), CARD_FOOT]
    return "\n\n".join(lines)


def too_long(text: str) -> bool:
    """The card would not fit in full: the proposal must be split."""
    return len(text) > MAX_CARD_CHARS


def marker_of(grant: Any) -> Optional[Dict[str, Any]]:
    """The proposal a card carries, or ``None`` for any other question."""
    details = getattr(grant, "details", None)
    marker = details.get(PROPOSAL_MARKER) if isinstance(details, dict) else None
    return marker if isinstance(marker, dict) and isinstance(marker.get("proposal"), dict) else None


def is_approve(answer: Any) -> bool:
    """The owner's answer is exactly Approve (any case, a trailing full stop or "!" allowed).

    "Approve, but less orange" is a revision: only a plain Approve saves.
    """
    return str(answer or "").strip().rstrip(".!").strip().casefold() == APPROVE.casefold()


def status_of(grant: Any) -> str:
    """The card's status as text ("pending", "granted", ...), whether stored as text or as GrantStatus."""
    raw = getattr(grant, "status", "")
    return str(getattr(raw, "value", raw) or "")


def why_not_saved(grant: Any) -> Optional[str]:
    """Why the card's proposal may not be saved, or ``None`` when the owner approved it and it is unsaved."""
    ask, status = getattr(grant, "id", None), status_of(grant)
    if (marker_of(grant) or {}).get("saved_at"):
        return ALREADY_SAVED.format(ask=ask)
    if status == PENDING:
        return NOT_ANSWERED.format(ask=ask)
    if status != ANSWERED or not getattr(grant, "answered_by", None):
        return CLOSED.format(ask=ask, status=status or "closed")
    if not is_approve(getattr(grant, "answer_text", None)):
        return REVISION.format(ask=ask, answer=str(getattr(grant, "answer_text", "") or "").strip())
    who = str(grant.answered_by)
    return None if who.startswith(ADMIN_ANSWER_PREFIX) else NOT_IN_APP.format(ask=ask, who=who.split(":")[0])


def latest_proposal(grants: Sequence[Any]) -> Optional[Any]:
    """The ticket's newest proposal card (by id), or ``None`` when it has none."""
    cards = [g for g in grants if marker_of(g) is not None]
    return max(cards, key=lambda g: int(getattr(g, "id", 0) or 0)) if cards else None


__all__ = [
    "APPROVE", "CARD_OPTIONS", "CHANGE_EFFECTS", "MAX_CARD_CHARS", "PENDING", "PROPOSAL_MARKER", "REVISE", "STILL_WAITING", "board_name", "card_text", "changes",
    "fingerprint", "is_approve", "latest_proposal", "marker_of", "status_of", "too_long", "why_not_saved",
]
