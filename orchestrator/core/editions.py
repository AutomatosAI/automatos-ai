"""PRD-277 US-102: what each edition does, as named capabilities.

``config.AUTH_EDITION`` names the edition (``local`` or ``saas``). Code that needs to
know whether billing is on, or whether a request with no credential is the operator,
asks for that capability here instead of comparing the edition. A new edition then
has to state every answer on purpose; it cannot inherit ``saas``'s answers by
falling through a ``== "local"`` check.

``docs/PRDS/PRD-277-EDITION-CHECKS.md`` lists every call site that compares the
edition today and the one capability it maps to. Call sites move here by directory,
in later pull requests. Each value below is today's behaviour for that edition.

An edition with no row is refused (``UnknownEdition``), never given another
edition's answers. ``config.py`` already turns an unknown ``AUTH_EDITION`` into
``saas`` before anything reads it here, so a running install never hits the refusal.

Importing this module reads no setting and imports no config, so ``config.py`` can
use it while it is being built.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

EDITION_LOCAL = "local"
EDITION_SAAS = "saas"

# How a person proves who they are (``EditionCapabilities.sign_in``).
SIGN_IN_NONE = "none"
SIGN_IN_CLERK = "clerk"

# How a sensitive action is confirmed (``EditionCapabilities.step_up``).
STEP_UP_SKIP = "skip"
STEP_UP_CLERK_FRESHNESS = "clerk_freshness"


class UnknownEdition(ValueError):
    """An edition with no row in the capability table."""


@dataclass(frozen=True)
class EditionCapabilities:
    """The answers one edition gives. Each field is one question the code asks.

    sign_in: how a person proves who they are (``SIGN_IN_*``).
    anonymous_operator: a request with no credential is the install's one operator,
        its super admin and the owner of every workspace (``LOCAL_REMOTE_ACCESS``
        and ``REQUIRE_AUTH`` still decide whether that lane is open).
    multi_person: several people share the install, each with their own role, so
        one person's action is not everyone's consent.
    billing: Automatos bills the workspace (billing routes, credits, plans, plan
        limits, quotas, the trial and the power-up). Off means the install's owner
        pays the AI providers directly.
    profile_editable: a person's name and email are edited in Automatos.
    step_up: how a sensitive action is confirmed (``STEP_UP_*``).
    device_pairing: a phone or browser may pair with a one-time code and hold an
        ``amd_`` token (``LOCAL_REMOTE_ACCESS`` still decides whether it must).
    session_mode: the operator's own CLI sessions may run as agents
        (``CLI_RUNTIME_ENABLED`` still switches it on).
    hosts_automatos_cloud: this install is Automatos Cloud: it serves the public
        Marketplace, the catalogue, bundles and the theme revocation feed, posts to
        the moderation inbox and runs the isolated theme review.
    calls_automatos_cloud: this install may call Automatos's hosted services
        (theme revocations, Cloud sync, Cloud items).
    push_needs_consent: notifications leave through Expo's push service only once
        the operator has agreed.
    operator_is_user: the person using the app runs the server, so host detail,
        ``docker compose`` fix commands and the operator's own folders are theirs.
    guards_user_endpoints: an AI endpoint a workspace saves must be public
        ``https`` and is called through the pinned client.
    """

    sign_in: str
    anonymous_operator: bool
    multi_person: bool
    billing: bool
    profile_editable: bool
    step_up: str
    device_pairing: bool
    session_mode: bool
    hosts_automatos_cloud: bool
    calls_automatos_cloud: bool
    push_needs_consent: bool
    operator_is_user: bool
    guards_user_endpoints: bool


_LOCAL = EditionCapabilities(
    sign_in=SIGN_IN_NONE,
    anonymous_operator=True,
    multi_person=False,
    billing=False,
    profile_editable=True,
    step_up=STEP_UP_SKIP,
    device_pairing=True,
    session_mode=True,
    hosts_automatos_cloud=False,
    calls_automatos_cloud=True,
    push_needs_consent=True,
    operator_is_user=True,
    guards_user_endpoints=False,
)

_SAAS = EditionCapabilities(
    sign_in=SIGN_IN_CLERK,
    anonymous_operator=False,
    multi_person=True,
    billing=True,
    profile_editable=False,
    step_up=STEP_UP_CLERK_FRESHNESS,
    device_pairing=False,
    session_mode=False,
    hosts_automatos_cloud=True,
    calls_automatos_cloud=False,
    push_needs_consent=False,
    operator_is_user=False,
    guards_user_endpoints=True,
)

CAPABILITIES: Mapping[str, EditionCapabilities] = MappingProxyType({
    EDITION_LOCAL: _LOCAL,
    EDITION_SAAS: _SAAS,
})
KNOWN_EDITIONS: Tuple[str, ...] = tuple(CAPABILITIES)


def capabilities_for(edition: str) -> EditionCapabilities:
    """The capabilities of ``edition``; ``UnknownEdition`` for one with no row.

    The name must match exactly: ``config.py`` has already trimmed and lower-cased it.
    """
    try:
        return CAPABILITIES[edition]
    except (KeyError, TypeError):
        raise UnknownEdition(f"no capabilities are defined for the edition {edition!r}") from None


def current() -> EditionCapabilities:
    """The running edition's capabilities.

    Reads ``config.AUTH_EDITION`` on every call rather than once at import, so a test
    that patches the edition sees the matching capabilities.
    """
    from config import config

    return capabilities_for(config.AUTH_EDITION)


__all__ = [
    "CAPABILITIES",
    "EDITION_LOCAL",
    "EDITION_SAAS",
    "KNOWN_EDITIONS",
    "SIGN_IN_CLERK",
    "SIGN_IN_NONE",
    "STEP_UP_CLERK_FRESHNESS",
    "STEP_UP_SKIP",
    "EditionCapabilities",
    "UnknownEdition",
    "capabilities_for",
    "current",
]
