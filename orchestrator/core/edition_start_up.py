"""PRD-277 US-103: an edition whose people sign in through OIDC never starts half-configured.

``config.validate_auth_edition`` calls ``check_start_up`` at boot, in the hard-fail phase
(``main.py``'s lifespan, outside any swallowing stage). For an edition whose ``sign_in`` is
``oidc`` (``enterprise``) it refuses to start:

1. when any setting in ``OIDC_START_UP_SETTINGS`` is unset or blank. The refusal names every
   missing setting and never a value: these settings hold a client secret, and the issuer and
   client id are the customer's own. ``FRONTEND_URL`` is the install's public address (PRD-277
   §5.1): the web app and the API share it, and the sign-in redirect is built on it. Its
   built-in default is a developer's ``localhost``, so only a value set in the environment counts.
2. when a sign-in setting (``OIDC_SIGN_IN_SETTINGS``, PRD-277 Wave 4) cannot be used: the issuer
   and ``FRONTEND_URL`` must be ``https`` addresses with a host; ``OIDC_SCOPES`` must be scope
   tokens including ``openid``; ``OIDC_CLAIM_EMAIL`` and ``OIDC_CLAIM_NAME`` must name a claim;
   ``OIDC_CA_BUNDLE_FILE``, when set, must be a file this process can read; and
   ``ENTERPRISE_DEFAULT_ROLE`` must be ``none`` or a workspace role below owner, and a role needs
   ``DEFAULT_WORKSPACE_ID``. ``CLIENT_IP_HEADER`` must name the header the proxy in front sets to
   the caller's address: sign-in limits requests per address and writes it to the audit log, and
   without that header the address is whatever the caller wrote. ``OIDC_ENDPOINT_HOSTS``, when set,
   must be host names (no scheme, port or wildcard), and ``OIDC_PENDING_SIGN_INS_MAX`` a whole number
   of at least one. The refusal names the setting, never its value. ``config.py`` reads them so
   that no value can stop another edition from starting.
3. last, while ``OIDC_SIGN_IN_IN_THIS_BUILD`` is false: the edition is "not available in this
   build", so a half-built edition can never run. The pull request that finishes Wave 4 turns it
   on.

``check_session_settings`` (PRD-277 Wave 3) is called just before it. Where the edition issues its
own sessions (``issues_sessions``) it refuses to start unless both clocks are whole numbers of at
least one, the idle clock is at least ``SESSION_IDLE_MINUTES_MIN`` (the minutes between two writes
of a session's last-seen stamp: a shorter one would end the session of a person still at work) and
no longer than the absolute one, the cookie's name is a ``__Host-``
name (so the browser only accepts it Secure, on ``/``, for this host alone) and the header a
cookie-authenticated change must carry is a valid header name. The refusal names the setting,
never its value. ``config.py`` reads the clocks so that no value stops another edition: a value
that is not a number reaches this check as the text it was, and is refused here.

Every other edition passes untouched: nothing here reads a setting it does not use.

Imports no config and reads no setting itself: ``config.py`` hands the values in.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterator, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from core import editions

# The pull request that finishes PRD-277 Wave 4 turns this on; until then enterprise cannot start.
OIDC_SIGN_IN_IN_THIS_BUILD = False

# The issuer, client id and client secret (PRD-277 §5.10), then the install's public address.
OIDC_START_UP_SETTINGS: Tuple[str, ...] = ("OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "FRONTEND_URL")

MISSING_SETTINGS = (
    "AUTH_EDITION=enterprise cannot start: {names} {verb} unset. An install whose people sign in "
    "through the organisation's identity provider never starts half-configured; set {them} and "
    "start it again."
)
NOT_AVAILABLE = (
    "AUTH_EDITION=enterprise is not available in this build: signing people in through the "
    "organisation's identity provider is not built yet (PRD-277 Wave 4). Set AUTH_EDITION=local "
    "or AUTH_EDITION=saas."
)


# The session settings ``check_session_settings`` reads (PRD-277 §5.3, D-15).
SESSION_SETTINGS: Tuple[str, ...] = (
    "SESSION_IDLE_MINUTES", "SESSION_ABSOLUTE_HOURS", "SESSION_COOKIE_NAME", "SESSION_CSRF_HEADER",
)
SESSION_COOKIE_PREFIX = "__Host-"
# core.auth.device.LAST_SEEN_WRITE_INTERVAL in minutes, which the session lane uses too; this module
# imports nothing from core.auth, so a test pins the two together.
SESSION_IDLE_MINUTES_MIN = 5
MINUTES_PER_HOUR = 60
# RFC 9110 §5.6.2 / RFC 6265 §4.1.1: a header name and a cookie name are each a "token".
_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
BAD_SESSION_SETTING = "AUTH_EDITION=enterprise cannot start: {name} {problem}."


# The sign-in settings ``check_start_up`` reads after the missing ones (PRD-277 §5.10, Wave 4).
OIDC_SIGN_IN_SETTINGS: Tuple[str, ...] = (
    "OIDC_ISSUER", "FRONTEND_URL", "OIDC_SCOPES", "OIDC_CLAIM_EMAIL", "OIDC_CLAIM_NAME",
    "OIDC_CA_BUNDLE_FILE", "ENTERPRISE_DEFAULT_ROLE", "DEFAULT_WORKSPACE_ID",
    "CLIENT_IP_HEADER", "OIDC_ENDPOINT_HOSTS", "OIDC_PENDING_SIGN_INS_MAX",
)
CLIENT_IP_HEADER_PROBLEM = (
    "must name the header the proxy in front sets to the caller's address: sign-in limits requests "
    "by that address and writes it to the audit log, and without it the address is whatever the caller "
    "sent"
)
# RFC 1123 host names, lower-case: labels of letters, digits and hyphens. No scheme, port or wildcard.
_HOST_NAME = re.compile(r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$")
_HOST_LIST_SEPARATORS = re.compile(r"[\s,]+")
OPENID_SCOPE = "openid"
NO_DEFAULT_ROLE = "none"
# modules.policy.roles.WorkspaceRole below owner: sign-in never makes anyone a workspace's owner.
DEFAULT_ROLES: Tuple[str, ...] = (NO_DEFAULT_ROLE, "viewer", "editor", "admin")
# RFC 6749 §3.3: scope-token = 1*( %x21 / %x23-5B / %x5D-7E ).
_SCOPE_TOKEN = re.compile(r"^[\x21\x23-\x5B\x5D-\x7E]+$")


class EditionNotReady(RuntimeError):
    """Start-up refused: this edition is missing a setting, or is not in this build."""


def missing_settings(settings: Mapping[str, Optional[str]]) -> Tuple[str, ...]:
    """The names in ``OIDC_START_UP_SETTINGS`` whose value is unset or blank, in that order."""
    return tuple(name for name in OIDC_START_UP_SETTINGS if not str(settings.get(name) or "").strip())


def _missing_message(missing: Tuple[str, ...]) -> str:
    """The refusal for ``missing``: the names only, never a value."""
    plural = len(missing) > 1
    names = ", ".join(missing[:-1]) + " and " + missing[-1] if plural else missing[0]
    return MISSING_SETTINGS.format(names=names, verb="are" if plural else "is", them="them" if plural else "it")


def _is_https_address(value: Any) -> bool:
    try:
        parts = urlsplit(str(value or "").strip())
        _ = parts.port
    except ValueError:
        return False
    return parts.scheme == "https" and bool(parts.hostname)


def _scopes_problem(scopes: str) -> Optional[str]:
    tokens = scopes.split()
    if not tokens or not all(_SCOPE_TOKEN.match(token) for token in tokens):
        return "must be scope names separated by spaces"
    return None if OPENID_SCOPE in tokens else f"must include {OPENID_SCOPE}"


def _readable_file(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.R_OK)


def _sign_in_setting_problems(settings: Mapping[str, Any]) -> Iterator[Tuple[str, str]]:
    """Each ``(name, problem)`` among ``OIDC_SIGN_IN_SETTINGS``, in that order; the value is never quoted."""
    for name in ("OIDC_ISSUER", "FRONTEND_URL"):
        if not _is_https_address(settings.get(name)):
            yield name, "must be an https address with a host"
    scopes = _scopes_problem(str(settings.get("OIDC_SCOPES") or ""))
    if scopes:
        yield "OIDC_SCOPES", scopes
    for name in ("OIDC_CLAIM_EMAIL", "OIDC_CLAIM_NAME"):
        if not str(settings.get(name) or "").strip():
            yield name, "must name a claim"
    yield from _bundle_and_role_problems(settings)
    yield from _limit_and_host_problems(settings)


def _bundle_and_role_problems(settings: Mapping[str, Any]) -> Iterator[Tuple[str, str]]:
    bundle = str(settings.get("OIDC_CA_BUNDLE_FILE") or "")
    if bundle and not _readable_file(bundle):
        yield "OIDC_CA_BUNDLE_FILE", "must be a certificate file this process can read"
    role = settings.get("ENTERPRISE_DEFAULT_ROLE") or NO_DEFAULT_ROLE
    if role not in DEFAULT_ROLES:
        yield "ENTERPRISE_DEFAULT_ROLE", f"must be one of {', '.join(DEFAULT_ROLES)}"
    elif role != NO_DEFAULT_ROLE and not str(settings.get("DEFAULT_WORKSPACE_ID") or "").strip():
        yield "ENTERPRISE_DEFAULT_ROLE", "needs DEFAULT_WORKSPACE_ID, the workspace the role is in"


def listed_hosts(value: Any) -> Tuple[str, ...]:
    """The host names in an ``OIDC_ENDPOINT_HOSTS`` value (comma or space separated), lower-case."""
    return tuple(name.lower().rstrip(".") for name in _HOST_LIST_SEPARATORS.split(str(value or "")) if name)


def is_host_name(name: str) -> bool:
    """A DNS host name: not an address, a scheme, a port or a wildcard."""
    return bool(_HOST_NAME.match(name)) and not name.rsplit(".", 1)[-1].isdigit()


def _limit_and_host_problems(settings: Mapping[str, Any]) -> Iterator[Tuple[str, str]]:
    if not _TOKEN.match(str(settings.get("CLIENT_IP_HEADER") or "").strip()):
        yield "CLIENT_IP_HEADER", CLIENT_IP_HEADER_PROBLEM
    if not all(is_host_name(name) for name in listed_hosts(settings.get("OIDC_ENDPOINT_HOSTS"))):
        yield "OIDC_ENDPOINT_HOSTS", "must be host names separated by commas, with no scheme, port or wildcard"
    if not _whole_number_at_least_one(settings.get("OIDC_PENDING_SIGN_INS_MAX")):
        yield "OIDC_PENDING_SIGN_INS_MAX", "must be a whole number of at least 1"


def check_start_up(capabilities: editions.EditionCapabilities, settings: Mapping[str, Optional[str]]) -> None:
    """Raise ``EditionNotReady`` when an OIDC edition may not start; return for every other one.

    ``settings`` maps each name in ``OIDC_START_UP_SETTINGS`` and ``OIDC_SIGN_IN_SETTINGS`` to its
    value (``None`` when unset). The missing settings are checked first, then the sign-in settings;
    "not available in this build" is the last check.
    """
    if capabilities.sign_in != editions.SIGN_IN_OIDC:
        return
    missing = missing_settings(settings)
    if missing:
        raise EditionNotReady(_missing_message(missing))
    first = next(_sign_in_setting_problems(settings), None)
    if first is not None:
        name, problem = first
        raise EditionNotReady(BAD_SESSION_SETTING.format(name=name, problem=problem))
    if not OIDC_SIGN_IN_IN_THIS_BUILD:
        raise EditionNotReady(NOT_AVAILABLE)


def _whole_number_at_least_one(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _session_setting_problems(settings: Mapping[str, Any]):
    """Each ``(name, problem)`` among ``SESSION_SETTINGS``, in that order; the value is never quoted."""
    idle, absolute = settings.get("SESSION_IDLE_MINUTES"), settings.get("SESSION_ABSOLUTE_HOURS")
    for name, value in (("SESSION_IDLE_MINUTES", idle), ("SESSION_ABSOLUTE_HOURS", absolute)):
        if not _whole_number_at_least_one(value):
            yield name, "must be a whole number of at least 1"
    if _whole_number_at_least_one(idle) and idle < SESSION_IDLE_MINUTES_MIN:
        yield "SESSION_IDLE_MINUTES", (f"must be at least {SESSION_IDLE_MINUTES_MIN}, the minutes between two "
                                       "writes of a session's last-seen time")
    if _whole_number_at_least_one(idle) and _whole_number_at_least_one(absolute) \
            and idle > absolute * MINUTES_PER_HOUR:
        yield "SESSION_IDLE_MINUTES", "must not be longer than SESSION_ABSOLUTE_HOURS"
    cookie = str(settings.get("SESSION_COOKIE_NAME") or "")
    if not (cookie.startswith(SESSION_COOKIE_PREFIX) and _TOKEN.match(cookie)):
        yield "SESSION_COOKIE_NAME", f"must be a cookie name that starts with {SESSION_COOKIE_PREFIX}"
    if not _TOKEN.match(str(settings.get("SESSION_CSRF_HEADER") or "")):
        yield "SESSION_CSRF_HEADER", "must be a header name"


def check_session_settings(capabilities: editions.EditionCapabilities, settings: Mapping[str, Any]) -> None:
    """Raise ``EditionNotReady`` for the first bad session setting where the edition issues sessions.

    ``settings`` maps each name in ``SESSION_SETTINGS`` to its value. Every other edition returns
    at once, reading nothing.
    """
    if not capabilities.issues_sessions:
        return
    first = next(_session_setting_problems(settings), None)
    if first is not None:
        name, problem = first
        raise EditionNotReady(BAD_SESSION_SETTING.format(name=name, problem=problem))


__all__ = [
    "BAD_SESSION_SETTING",
    "MISSING_SETTINGS",
    "DEFAULT_ROLES",
    "NOT_AVAILABLE",
    "OIDC_SIGN_IN_IN_THIS_BUILD",
    "OIDC_SIGN_IN_SETTINGS",
    "OIDC_START_UP_SETTINGS",
    "SESSION_COOKIE_PREFIX",
    "SESSION_IDLE_MINUTES_MIN",
    "SESSION_SETTINGS",
    "EditionNotReady",
    "check_session_settings",
    "check_start_up",
    "is_host_name",
    "listed_hosts",
    "missing_settings",
]
