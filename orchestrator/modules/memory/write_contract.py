"""PRD-206 S1 — the ONE memory-write contract.

Every L3 memory write (the distill path in ``consumers/chatbot/smart_memory``
and the tool path in ``tools/discovery/handlers_workspace``) builds its
metadata through :func:`build_memory_metadata`, so the split-brain the PRD-206
scout found — provenance on the tool path, only ``category``+``importance`` on
the distill path — ends here. This module is also the canonical home of the
memory taxonomy (previously defined in ``consumers/chatbot/smart_memory.py``;
that module now re-exports from here).

Consent model (Gerard's 2026-07-17 answers, spec §8):

- **Q3 = silent-everything.** Auto never asks before saving a memory, which
  means :func:`violates_exclusions` carries ALL the consent weight: secrets,
  credentials, payment data and other sensitive strings must never be stored,
  because no human confirmation step will catch them later. Transparency is
  the panel + "forget that", not a prompt.
- **Q7 = split sharing default.** ``user_fact``/``preference`` default to
  ``scope='private'`` (visible only to their owner); every other type defaults
  to ``scope='workspace'``. A caller may override per memory. A private scope
  needs an owner to mean anything, so the DEFAULT only resolves to private
  when an owner tag is present; an explicit ``scope='private'`` is honoured
  as given.

Pure — no I/O, no config reads — so it unit-tests with plain values.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Taxonomy (PRD-159 S1, extended by PRD-206 S1)
# ---------------------------------------------------------------------------

# The distiller emits a {fact, type, importance} object per durable fact;
# `type` is validated against this set and stored as both `type` and the
# legacy-named `category` key so recall, the injection filter and the Explorer
# can all filter operational knowledge by kind.
MEMORY_FACT_TYPES = frozenset({
    "tool_outcome",     # a tool/Composio call's notable result (failure, quirk, new id)
    "task_learning",    # what was learned from a mission/task succeeding or failing
    "playbook_pattern", # a reusable pattern surfaced while running a playbook
    "user_fact",        # stable fact about the user
    "business_fact",    # stable fact about their business/domain
    "preference",       # a stated preference
    "procedure",        # a how-to / standard operating procedure
    # PRD-206 S1 — the continuity types the auntie layer runs on:
    "decision",         # a decision that was made ("we decided X because Y")
    "open_loop",        # something left unresolved that should be picked back up
    "thread_summary",   # a checkpoint summary of a conversation thread
})
DEFAULT_FACT_TYPE = "task_learning"

# ---------------------------------------------------------------------------
# Sharing scope (Q7 — split default)
# ---------------------------------------------------------------------------

MEMORY_SCOPE_PRIVATE = "private"
MEMORY_SCOPE_WORKSPACE = "workspace"
MEMORY_SCOPES = frozenset({MEMORY_SCOPE_PRIVATE, MEMORY_SCOPE_WORKSPACE})

# Types that default to private-to-their-owner. Everything else defaults to
# workspace-shared (decisions, open loops and summaries are workspace objects,
# like tasks).
PRIVATE_DEFAULT_TYPES = frozenset({"user_fact", "preference"})


def default_scope_for_type(fact_type: str, *, has_owner: bool = False) -> str:
    """Q7 split default: personal types are private WHEN an owner is known.

    A private memory without an owner tag would be invisible to everyone
    (the injection guard fails closed), so the default only goes private when
    the write carries an owner. Explicit caller scope is handled by
    :func:`build_memory_metadata`, not here.
    """
    if fact_type in PRIVATE_DEFAULT_TYPES and has_owner:
        return MEMORY_SCOPE_PRIVATE
    return MEMORY_SCOPE_WORKSPACE


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

# The four tool-path provenance values (Wave 3) plus the distill lane's own.
SOURCE_TYPE_DISTILLED = "distilled"
MEMORY_SOURCE_TYPES = frozenset({
    "platform_verified",
    "claude_reports",
    "current_status",
    "inference",
    SOURCE_TYPE_DISTILLED,
})


# ---------------------------------------------------------------------------
# Exclusion validator (Q3 — this IS the consent gate)
# ---------------------------------------------------------------------------

# Each rule is (name, compiled regex). Ordered roughly most-specific-first so
# the reported rule name is the most useful one when several match.
_EXCLUSION_RULES: tuple = (
    ("pem_private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_assignment", re.compile(r"aws_secret_access_key\s*[:=]", re.I)),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("stripe_key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.")),
    ("bearer_token", re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{16,}", re.I)),
    (
        "password_assignment",
        # "password is hunter2" / "pwd: x" — the value must directly follow, so
        # "the password rotation policy is 90 days" does NOT match.
        re.compile(r"\b(?:password|passwd|pwd|passphrase)\s*(?:is|was|[:=])\s*\S+", re.I),
    ),
    (
        "credentialed_url",
        # scheme://user:pass@host — connection strings with inline credentials.
        re.compile(r"\b\w+://[^/\s:@]+:[^@\s]+@"),
    ),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    (
        "cvv",
        re.compile(r"\bcvv2?\b\s*(?:is|was|[:=])?\s*\d{3,4}\b", re.I),
    ),
    (
        "otp_code",
        re.compile(
            r"\b(?:otp|2fa|verification code|one[- ]time (?:pass)?code)\b.{0,40}\b\d{4,8}\b",
            re.I | re.S,
        ),
    ),
    ("seed_phrase", re.compile(r"\b(?:seed phrase|recovery phrase|mnemonic)\b", re.I)),
    (
        "generic_secret_assignment",
        re.compile(
            r"\b(?:api[ _-]?key|access[ _-]?token|secret[ _-]?key|client[ _-]?secret"
            r"|auth[ _-]?token|refresh[ _-]?token)\b\s*(?:is|was|[:=])\s*\S{8,}",
            re.I,
        ),
    ),
)

# Card numbers need a Luhn check on top of the digit-run regex, or every long
# numeric id would be refused.
_CARD_RUN_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def violates_exclusions(text: str) -> Optional[str]:
    """Return the name of the first exclusion rule *text* trips, else None.

    Q3 (silent-everything) makes this the ONLY consent gate on the write path:
    a hit means the content is never stored, on either write path. Callers log
    the rule name — never the matched content.
    """
    if not text:
        return None
    for name, pattern in _EXCLUSION_RULES:
        if pattern.search(text):
            return name
    for run in _CARD_RUN_RE.findall(text):
        digits = re.sub(r"[ -]", "", run)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return "card_number"
    return None


# ---------------------------------------------------------------------------
# The write contract
# ---------------------------------------------------------------------------

def _coerce_unit_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def build_memory_metadata(
    *,
    fact_type: str,
    importance: Any = None,
    confidence: Any = None,
    source_type: Optional[str] = None,
    scope: Optional[str] = None,
    owner: Optional[str] = None,
    page: Optional[str] = None,
    chat_id: Optional[str] = None,
    project_id: Optional[str] = None,
    pinned: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical L3 metadata dict — the one shape BOTH write paths store.

    Required keys, always present: ``type``, ``category`` (legacy alias, same
    value — the injection filter and Explorer read either), ``importance``,
    ``scope``, ``source_type``. Optional keys (``confidence``, ``owner``,
    ``page``, ``chat_id``, ``project_id``, ``pinned``) are included only when
    provided, so absent context never writes noise keys.

    ``extra`` entries are merged first and never override contract keys.
    Returns a NEW dict; nothing passed in is mutated.
    """
    ftype = fact_type if fact_type in MEMORY_FACT_TYPES else DEFAULT_FACT_TYPE

    if scope in MEMORY_SCOPES:
        resolved_scope = scope
    else:
        resolved_scope = default_scope_for_type(ftype, has_owner=bool(owner))

    resolved_source = (
        source_type if source_type in MEMORY_SOURCE_TYPES else SOURCE_TYPE_DISTILLED
    )

    meta: Dict[str, Any] = dict(extra) if extra else {}
    meta.update({
        "type": ftype,
        "category": ftype,
        "importance": _coerce_unit_float(importance, 0.5),
        "scope": resolved_scope,
        "source_type": resolved_source,
    })
    conf = _coerce_unit_float(confidence, None)
    if conf is not None:
        meta["confidence"] = conf
    if owner:
        meta["owner"] = str(owner)
    if page:
        meta["page"] = str(page)
    if chat_id:
        meta["chat_id"] = str(chat_id)
    if project_id:
        meta["project_id"] = str(project_id)
    if pinned is not None:
        meta["pinned"] = bool(pinned)
    return meta
