"""PRD-225 US-006 — the per-channel ingress trust gate.

Inbound channel traffic that *directs work* shouldn't drive an agent unreviewed.
Each channel carries a ``trigger_mode`` (stored in its ``config`` JSONB, no new
column):

  - ``strict`` (DEFAULT) — HOLD every inbound message for operator approval
    (nothing an untrusted channel says reaches the router without a human).
  - ``communication_only`` — route chatter, HOLD directives (a conservative
    classifier: anything not clearly social is a directive).
  - ``allow_all`` — route everything (today's behaviour).

Correlated answers (a reply / ``/answer`` matching a pending question) bypass the
gate entirely — they are *responses*, not directives — and are handled upstream
before this ever runs.

The classifier is a pure function, deliberately conservative: a false "directive"
merely asks a human (safe); a false "chatter" would auto-route an untrusted
instruction (unsafe), so ambiguity always resolves to *directive*.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# --- modes ------------------------------------------------------------------

TRIGGER_MODE_STRICT = "strict"
TRIGGER_MODE_COMMUNICATION_ONLY = "communication_only"
TRIGGER_MODE_ALLOW_ALL = "allow_all"
TRIGGER_MODES = (
    TRIGGER_MODE_STRICT,
    TRIGGER_MODE_COMMUNICATION_ONLY,
    TRIGGER_MODE_ALLOW_ALL,
)
DEFAULT_TRIGGER_MODE = TRIGGER_MODE_STRICT

# --- classifier verdicts ----------------------------------------------------

VERDICT_DIRECTIVE = "directive"
VERDICT_CHATTER = "chatter"

# Action verbs — a message containing any of these (as a word) directs work.
_DIRECTIVE_VERBS = frozenset({
    "delete", "remove", "drop", "wipe", "purge", "destroy", "cancel", "kill",
    "terminate", "send", "email", "message", "post", "publish", "tweet", "dm",
    "forward", "create", "make", "build", "generate", "draft", "write", "add",
    "set", "update", "change", "edit", "rename", "configure", "run", "execute",
    "launch", "start", "stop", "restart", "deploy", "migrate", "merge", "push",
    "install", "uninstall", "enable", "disable", "reset", "schedule", "book",
    "order", "buy", "purchase", "pay", "refund", "transfer", "invoice", "charge",
    "assign", "approve", "grant", "revoke", "invite", "ban", "kick", "fix",
    "upgrade", "rollback", "restore", "backup", "export", "import", "sync",
    "scrape", "download", "upload", "call", "notify", "alert", "remind",
    "escalate", "process", "analyze", "analyse", "summarize", "summarise",
})

# Request markers that imply an action even without a known verb.
_DIRECTIVE_MARKERS = (
    "please ", "can you", "could you", "would you", "i need", "we need",
    "help me", "make sure", "go ahead", "i want you", "set up", "sign up",
)

# Pure social tokens — a short message made only of these is chatter. Includes
# common filler ("so", "much", …) so realistic greetings/acks route under
# communication_only; a directive verb still wins (it is checked first).
_CHATTER_TOKENS = frozenset({
    "hi", "hello", "hey", "yo", "sup", "hiya", "howdy", "greetings", "there",
    "thanks", "thank", "thankyou", "thx", "ty", "cheers", "appreciate",
    "appreciated", "ok", "okay", "kk", "sure", "yep", "yes", "yeah", "no",
    "nope", "cool", "great", "nice", "awesome", "perfect", "good", "fine",
    "alright", "morning", "afternoon", "evening", "night", "gm", "gn", "bye",
    "goodbye", "later", "ciao", "welcome", "lol", "haha", "hah", "wow",
    "congrats", "congratulations", "team", "everyone", "all", "how", "are",
    "you", "doing", "today", "is", "it", "going", "hope", "well", "u", "r",
    # common social filler
    "so", "much", "very", "really", "for", "that", "this", "just", "here",
    "now", "back", "guys", "folks", "your", "my", "our", "mate", "friend",
    "and", "the", "a", "to", "of", "with", "im", "we", "were", "youre",
})


def classify_inbound_message(text: str) -> str:
    """Classify an inbound message as ``directive`` or ``chatter``.

    Deliberately conservative: only clearly-social short messages are chatter;
    anything with an action verb, a request marker, or any real ambiguity is a
    directive (which the gate holds).
    """
    t = (text or "").strip().lower()
    if not t:
        return VERDICT_DIRECTIVE
    words = re.findall(r"[a-z0-9']+", t)
    if not words:
        # Only emoji / punctuation — a social reaction.
        return VERDICT_CHATTER
    if any(w in _DIRECTIVE_VERBS for w in words):
        return VERDICT_DIRECTIVE
    if any(marker in t for marker in _DIRECTIVE_MARKERS):
        return VERDICT_DIRECTIVE
    if len(words) <= 6 and all(w in _CHATTER_TOKENS for w in words):
        return VERDICT_CHATTER
    # Unsure ⇒ directive (safe error: ask a human rather than auto-route).
    return VERDICT_DIRECTIVE


def should_hold(mode: Optional[str], text: str) -> bool:
    """Whether the gate holds this inbound message for operator approval.

    Unknown modes fail safe to ``strict`` (hold) — a misconfigured channel must
    not silently become ``allow_all``.
    """
    m = (mode or DEFAULT_TRIGGER_MODE).lower()
    if m == TRIGGER_MODE_ALLOW_ALL:
        return False
    if m == TRIGGER_MODE_COMMUNICATION_ONLY:
        return classify_inbound_message(text) == VERDICT_DIRECTIVE
    return True  # strict (and any unknown mode) holds everything


# --- config helpers (trigger_mode lives in the channel's config JSONB) -------

def normalize_trigger_mode(mode: Any) -> Optional[str]:
    """Return a valid trigger_mode or None (for validation at the API edge)."""
    m = str(mode or "").strip().lower()
    return m if m in TRIGGER_MODES else None


def trigger_mode_of(config: Any) -> str:
    """Read the trigger_mode out of a channel's config JSONB, defaulting to
    ``strict`` when unset or invalid."""
    cfg = config if isinstance(config, dict) else {}
    return normalize_trigger_mode(cfg.get("trigger_mode")) or DEFAULT_TRIGGER_MODE


def with_trigger_mode(config: Any, mode: str) -> dict:
    """Return a NEW config dict with trigger_mode set (rebuild-don't-mutate)."""
    base = dict(config) if isinstance(config, dict) else {}
    return {**base, "trigger_mode": mode}
