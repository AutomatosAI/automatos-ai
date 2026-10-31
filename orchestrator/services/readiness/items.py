"""PRD-268 §6.11.3 — the readiness vocabulary: states, items, what each is needed by, and the fixes.

A probe answers with a :class:`Finding`: its state, the strings key of its detail, the key of
its fix (when a person can fix it in the app) and the host it reached, which only the local
edition shows. A probe that does not answer in time is never asked what it found: the runner
(``services.readiness.service``) writes ``not_checked`` for it. Every fix is an app path
(``/settings?tab=…``), never an outside URL.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

READY = "ready"
NOT_CHECKED = "not_checked"
NEEDS_SETUP = "needs_setup"
STATES = (READY, NOT_CHECKED, NEEDS_SETUP)

# The server's items (§6.11.3); the client adds microphone, speech and the phone's own push.
CORE = "core"
MODEL_ROUTE = "model_route"
WEB_SEARCH = "web_search"
PUSH = "push"

# What an item is needed by.
FOR_BRIEF = "brief"
FOR_VOICE = "voice"
FOR_PUSH = "push"
FOR_CHECK_IN = "check_in"
NEEDED_BY = (FOR_BRIEF, FOR_VOICE, FOR_PUSH, FOR_CHECK_IN)

# Details: the strings keys a finding names (services/readiness/strings.py).
CORE_READY = "core_ready"
CORE_DOWN = "core_down"
MODEL_READY = "model_ready"
MODEL_UNSET = "model_unset"
MODEL_NO_KEY = "model_no_key"
SEARCH_OPENROUTER = "search_openrouter"
SEARCH_COMPOSIO = "search_composio"
SEARCH_SEARXNG = "search_searxng"
SEARCH_NONE = "search_none"
SEARCH_OFF = "search_off"
PUSH_READY = "push_ready"
PUSH_UNANSWERED = "push_unanswered"
PUSH_DECLINED = "push_declined"
NOT_ANSWERED = "not_answered"
UNREADABLE = "unreadable"

# Fixes: the strings key of the fix's label, and where it opens.
CHOOSE_MODEL = "fix_choose_model"
ADD_KEY = "fix_add_key"
ANSWER_PUSH = "fix_answer_push"


@dataclass(frozen=True)
class Finding:
    """What a probe found."""

    state: str
    detail: str
    fix: Optional[str] = None
    where: Optional[str] = None


@dataclass(frozen=True)
class Fix:
    """A fix's label (a strings key) and the app path it opens."""

    label: str
    href: str


FIXES: Mapping[str, Fix] = MappingProxyType({
    CHOOSE_MODEL: Fix(CHOOSE_MODEL, "/settings?tab=orchestrator"),
    ADD_KEY: Fix(ADD_KEY, "/settings?tab=api-keys"),
    ANSWER_PUSH: Fix(ANSWER_PUSH, "/settings?tab=phones-devices"),
})

Probe = Callable[[Session, UUID], Finding]


@dataclass(frozen=True)
class Item:
    """One server item: its id (also its label's strings key), what needs it, and its probe."""

    id: str
    needed_by: Tuple[str, ...]
    probe: Probe
