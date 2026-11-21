"""PRD-277 US-102: what each edition does, as named capabilities.

``config.AUTH_EDITION`` names the edition (``local``, ``saas`` or ``enterprise``). Code
that needs to know whether billing is on, or whether a request with no credential is the
operator, asks for that capability here instead of comparing the edition. A new edition
then has to state every answer on purpose; it cannot inherit ``saas``'s answers by
falling through a ``== "local"`` check.

``enterprise`` (PRD-277 US-103) is a self-hosted install whose people sign in through the
organisation's own identity provider. Its row is decided (PRD-277 §5.1, D-7, D-12, D-13
and the owner's answers G-1 to G-8 in ``PRD-277-EDITION-CHECKS.md``). Start-up refuses it
only when it is half-configured, naming what is missing (``core/edition_start_up.py``).

``docs/PRDS/PRD-277-EDITION-CHECKS.md`` lists every call site that compares the
edition and the one capability it maps to. Outside ``config.py``, this module, the
migrations and the tests, no code compares the edition or reads its old helpers
(``tests/test_prd277_no_raw_edition_checks.py``). Each ``local`` and ``saas`` value below
is today's behaviour for that edition; each ``enterprise`` value is the decided one. Code
that only reports the edition's name reads ``current_name``.

``capabilities_for`` is strict: an edition with no row is refused (``UnknownEdition``),
never given another edition's answers. ``current`` follows the flag's own rule
(PRD-277 §5.1, ``config.py``): the name is trimmed and lower-cased, and a value with
no row is ``saas``, so a value a test patches in decides what the old comparisons did.

**Automatos's hosted services** (PRD-277 US-704, D-12). Each one the code calls has its own
capability: ``calls_theme_revocations``, ``calls_cloud_sync`` and ``calls_cloud_items``. In
``local`` and ``saas`` the row decides them, and no setting can change them. In an edition whose
row has ``hosted_services_by_setting`` (``enterprise``), the row holds each one off, and
``current`` gives each one its setting's value (``HOSTED_SERVICE_SETTINGS``): on only for a
plain true form (``SWITCH_ON``), off for anything else, so a malformed value never reads as on
(``core/edition_start_up.py`` refuses one at start). ``with_hosted_service_settings`` is that one
rule.

Importing this module reads no setting and imports no config, so ``config.py`` can
use it while it is being built.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, FrozenSet, Mapping, Tuple

EDITION_LOCAL = "local"
EDITION_SAAS = "saas"
EDITION_ENTERPRISE = "enterprise"

# How a person proves who they are (``EditionCapabilities.sign_in``).
SIGN_IN_NONE = "none"
SIGN_IN_CLERK = "clerk"
SIGN_IN_OIDC = "oidc"

# How a sensitive action is confirmed (``EditionCapabilities.step_up``).
STEP_UP_SKIP = "skip"
STEP_UP_CLERK_FRESHNESS = "clerk_freshness"
STEP_UP_REAUTH = "reauth"

# What Auto is told the install is (``EditionCapabilities.install_description``).
INSTALL_OWN_COMPUTER = "own_computer"
INSTALL_HOSTED = "hosted"
INSTALL_ORGANISATION = "organisation"


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
    calls_theme_revocations: this install pulls Automatos's theme revocation list
        (``THEME_REVOCATIONS_URL`` still names it): the scheduled pull, the admin's sync
        status and Check now. What a pull stored (its revocations and revoked signing keys)
        still applies while it is off: each of these three stops requests, never reads.
    calls_cloud_sync: this install syncs Marketplace themes from Automatos Cloud
        (``AUTOMATOS_CLOUD_URL`` still names it): the Cloud Sync routes and the catalogue
        pull, a Cloud theme's package on install and update, and its screenshots.
    calls_cloud_items: this install requests agents, Playbooks, packages and plugins from
        Automatos Cloud: the Sync's item and plugin passes and their bundle and package
        downloads. The "Items brought from Cloud" routes read what the install holds and
        follow ``hosts_automatos_cloud``. Items arrive only in a Sync, so it needs
        ``calls_cloud_sync``.
    hosted_services_by_setting: each of the three above is off in the row and is the
        operator's own setting (``HOSTED_SERVICE_SETTINGS``, D-12).
    push_needs_consent: notifications leave through Expo's push service only once
        the operator has agreed.
    operator_is_user: the person using the app runs the server, so host detail,
        ``docker compose`` fix commands and the operator's own folders are theirs.
    guards_user_endpoints: an AI endpoint a workspace saves must be public
        ``https`` and is called through the pinned client.
    seeds_first_run_content: every boot seeds the default workspace's first-run
        content: its workspace and operator rows, Auto, the starter roster, the demo
        Playbook and the welcome Deliverable (``DEFAULT_WORKSPACE_ID`` still names it).
    chat_seeds_user_name: chat seeds the driving person's name from their ``users``
        row into Auto's state.
    install_description: what Auto is told the install is, and the Settings tabs it
        names (``INSTALL_*``).
    live_agent_named_apart: a newly created Auto Live agent gets a name of its own,
        so it is never mistaken for the hosted product's agent in a shared Retell
        dashboard.
    last_workspace_purge_deletes_person: purging the last workspace a person owns or
        belongs to also deletes that person's account.
    issues_sessions: Automatos issues and resolves its own sign-in sessions (PRD-277 Wave 3):
        an ``ams_`` token, carried by a browser in a ``__Host-`` cookie, that the bearer
        registry resolves; a request the cookie authenticates that changes something must
        come from the install's own origin. Where it is off, an ``ams_`` token belongs to no
        lane and no cookie is read.
    agent_http_own_hosts_only: an agent's ``http_request`` tool may call only the install's own
        hosts (its internal API and web hosts, and the loopback names), never Automatos's public
        API and web hosts: the platform's API is the install's own (D-12). Where it is off, those
        two hosts are on the tool's list as well (``modules/tools/execution/http_hosts.py``).
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
    calls_theme_revocations: bool
    calls_cloud_sync: bool
    calls_cloud_items: bool
    hosted_services_by_setting: bool
    push_needs_consent: bool
    operator_is_user: bool
    guards_user_endpoints: bool
    seeds_first_run_content: bool
    chat_seeds_user_name: bool
    install_description: str
    live_agent_named_apart: bool
    last_workspace_purge_deletes_person: bool
    issues_sessions: bool
    agent_http_own_hosts_only: bool


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
    calls_theme_revocations=True,
    calls_cloud_sync=True,
    calls_cloud_items=True,
    hosted_services_by_setting=False,
    push_needs_consent=True,
    operator_is_user=True,
    guards_user_endpoints=False,
    seeds_first_run_content=True,
    chat_seeds_user_name=True,
    install_description=INSTALL_OWN_COMPUTER,
    live_agent_named_apart=True,
    last_workspace_purge_deletes_person=False,
    issues_sessions=False,
    agent_http_own_hosts_only=False,
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
    calls_theme_revocations=False,
    calls_cloud_sync=False,
    calls_cloud_items=False,
    hosted_services_by_setting=False,
    push_needs_consent=False,
    operator_is_user=False,
    guards_user_endpoints=True,
    seeds_first_run_content=False,
    chat_seeds_user_name=False,
    install_description=INSTALL_HOSTED,
    live_agent_named_apart=False,
    last_workspace_purge_deletes_person=True,
    issues_sessions=False,
    agent_http_own_hosts_only=False,
)

# PRD-277 §5.1 and its decisions. Each hosted service is off here, and the operator switches each
# one on with its own setting (D-12, US-704: hosted_services_by_setting). device_pairing is on, and
# LOCAL_REMOTE_ACCESS still decides whether codes are minted (D-7). issues_sessions is on: a
# browser holds a session Automatos issues, in a cookie scripts cannot read (D-3, §5.3).
_ENTERPRISE = EditionCapabilities(
    sign_in=SIGN_IN_OIDC,
    anonymous_operator=False,
    multi_person=True,
    billing=False,
    profile_editable=False,
    step_up=STEP_UP_REAUTH,
    device_pairing=True,
    session_mode=False,
    hosts_automatos_cloud=False,
    calls_theme_revocations=False,
    calls_cloud_sync=False,
    calls_cloud_items=False,
    hosted_services_by_setting=True,
    push_needs_consent=True,
    operator_is_user=False,
    guards_user_endpoints=True,
    seeds_first_run_content=False,
    chat_seeds_user_name=True,
    install_description=INSTALL_ORGANISATION,
    live_agent_named_apart=False,
    last_workspace_purge_deletes_person=False,
    issues_sessions=True,
    agent_http_own_hosts_only=True,
)

CAPABILITIES: Mapping[str, EditionCapabilities] = MappingProxyType({
    EDITION_LOCAL: _LOCAL,
    EDITION_SAAS: _SAAS,
    EDITION_ENTERPRISE: _ENTERPRISE,
})
KNOWN_EDITIONS: Tuple[str, ...] = tuple(CAPABILITIES)

# PRD-277 US-704: each hosted service's capability, and the setting that decides it where the
# row says ``hosted_services_by_setting``. Each setting names what it lets leave the install.
HOSTED_SERVICE_SETTINGS: Mapping[str, str] = MappingProxyType({
    "calls_theme_revocations": "ENTERPRISE_CALLS_THEME_REVOCATIONS",
    "calls_cloud_sync": "ENTERPRISE_CALLS_CLOUD_SYNC",
    "calls_cloud_items": "ENTERPRISE_CALLS_CLOUD_ITEMS",
})
# The forms config.py's other switches read as on ("1", "true", "yes", "on"), and their plain
# off forms. A setting's value is trimmed and lower-cased first; unset is off.
SWITCH_ON: FrozenSet[str] = frozenset({"1", "true", "yes", "on"})
SWITCH_OFF: FrozenSet[str] = frozenset({"", "0", "false", "no", "off"})


def _switch_text(value: Any) -> str:
    return str("" if value is None else value).strip().lower()


def is_switched_on(value: Any) -> bool:
    """True only for a plain on form (``SWITCH_ON``); off for anything else, a malformed value included."""
    return _switch_text(value) in SWITCH_ON


def is_switch_form(value: Any) -> bool:
    """True for a plain on or off form: what start-up accepts for a hosted service's setting."""
    return _switch_text(value) in SWITCH_ON | SWITCH_OFF


def with_hosted_service_settings(
    capabilities: EditionCapabilities, settings: Mapping[str, Any],
) -> EditionCapabilities:
    """``capabilities``, with each hosted service its setting's value where the row says so.

    The one place a setting and an edition combine. Where ``hosted_services_by_setting`` is off
    (``local``, ``saas``) the row is returned as it is, whatever ``settings`` hold. Elsewhere each
    capability in ``HOSTED_SERVICE_SETTINGS`` is on only when its setting is a plain on form.
    ``settings`` maps each setting's name to its value; a missing one is off.
    """
    if not capabilities.hosted_services_by_setting:
        return capabilities
    switched = {field: is_switched_on(settings.get(name)) for field, name in HOSTED_SERVICE_SETTINGS.items()}
    return replace(capabilities, **switched)


def capabilities_for(edition: str) -> EditionCapabilities:
    """The capabilities of ``edition``; ``UnknownEdition`` for one with no row.

    The name must match exactly: ``config.py`` has already trimmed and lower-cased it.
    """
    try:
        return CAPABILITIES[edition]
    except (KeyError, TypeError):
        raise UnknownEdition(f"no capabilities are defined for the edition {edition!r}") from None


def current_name() -> str:
    """The running edition's name, for code that reports it and switches nothing.

    Reads ``config.AUTH_EDITION`` on every call. As ``config.py`` does at load, the
    name is trimmed and lower-cased and a value with no row is ``saas``. Code that
    decides something asks ``current()`` for a capability instead of comparing this.
    """
    from config import config

    name = str(getattr(config, "AUTH_EDITION", "") or "").strip().lower()
    return name if name in CAPABILITIES else EDITION_SAAS


def current() -> EditionCapabilities:
    """The running edition's capabilities, with the hosted services' settings where its row takes them.

    Reads the edition, and those settings, on every call rather than once at import, so a test that
    patches ``config.AUTH_EDITION`` or a setting sees the matching capabilities (``current_name``).
    """
    row = CAPABILITIES[current_name()]
    if not row.hosted_services_by_setting:
        return row
    from config import config

    settings = {name: getattr(config, name, None) for name in HOSTED_SERVICE_SETTINGS.values()}
    return with_hosted_service_settings(row, settings)


__all__ = [
    "CAPABILITIES",
    "EDITION_ENTERPRISE",
    "EDITION_LOCAL",
    "EDITION_SAAS",
    "HOSTED_SERVICE_SETTINGS",
    "INSTALL_HOSTED",
    "INSTALL_ORGANISATION",
    "INSTALL_OWN_COMPUTER",
    "KNOWN_EDITIONS",
    "SIGN_IN_CLERK",
    "SIGN_IN_NONE",
    "SIGN_IN_OIDC",
    "STEP_UP_CLERK_FRESHNESS",
    "STEP_UP_REAUTH",
    "STEP_UP_SKIP",
    "SWITCH_OFF",
    "SWITCH_ON",
    "EditionCapabilities",
    "UnknownEdition",
    "capabilities_for",
    "current",
    "current_name",
    "is_switch_form",
    "is_switched_on",
    "with_hosted_service_settings",
]
