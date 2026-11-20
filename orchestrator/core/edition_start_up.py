"""PRD-277 US-103: an edition whose people sign in through OIDC never starts half-configured.

``config.validate_auth_edition`` calls ``check_start_up`` at boot, in the hard-fail phase
(``main.py``'s lifespan, outside any swallowing stage). For an edition whose ``sign_in`` is
``oidc`` (``enterprise``) it refuses to start:

1. when any setting in ``OIDC_START_UP_SETTINGS`` is unset or blank. The refusal names every
   missing setting and never a value: these settings hold a client secret, and the issuer and
   client id are the customer's own. ``FRONTEND_URL`` is the install's public address (PRD-277
   §5.1): the web app and the API share it, and the sign-in redirect is built on it. Its
   built-in default is a developer's ``localhost``, so only a value set in the environment counts.
2. last, while ``OIDC_SIGN_IN_IN_THIS_BUILD`` is false: the edition is "not available in this
   build", so a half-built edition can never run. Wave 4 (US-401 to US-408), which builds OIDC
   sign-in, deletes that switch and this check.

``check_session_settings`` (PRD-277 Wave 3) is called just before it. Where the edition issues its
own sessions (``issues_sessions``) it refuses to start unless both clocks are whole numbers of at
least one and the idle clock is no longer than the absolute one, the cookie's name is a ``__Host-``
name (so the browser only accepts it Secure, on ``/``, for this host alone) and the header a
cookie-authenticated change must carry is a valid header name. The refusal names the setting.

Every other edition passes untouched: nothing here reads a setting it does not use.

Imports no config and reads no setting itself: ``config.py`` hands the values in.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple

from core import editions

# Wave 4 of PRD-277 deletes this switch, and the check that reads it, when OIDC sign-in lands.
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
MINUTES_PER_HOUR = 60
# RFC 9110 §5.6.2 / RFC 6265 §4.1.1: a header name and a cookie name are each a "token".
_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
BAD_SESSION_SETTING = "AUTH_EDITION=enterprise cannot start: {name} {problem}."


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


def check_start_up(capabilities: editions.EditionCapabilities, settings: Mapping[str, Optional[str]]) -> None:
    """Raise ``EditionNotReady`` when an OIDC edition may not start; return for every other one.

    ``settings`` maps each name in ``OIDC_START_UP_SETTINGS`` to its value (``None`` when unset).
    The missing settings are checked first; "not available in this build" is the last check.
    """
    if capabilities.sign_in != editions.SIGN_IN_OIDC:
        return
    missing = missing_settings(settings)
    if missing:
        raise EditionNotReady(_missing_message(missing))
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
    "NOT_AVAILABLE",
    "OIDC_SIGN_IN_IN_THIS_BUILD",
    "OIDC_START_UP_SETTINGS",
    "SESSION_COOKIE_PREFIX",
    "SESSION_SETTINGS",
    "EditionNotReady",
    "check_session_settings",
    "check_start_up",
    "missing_settings",
]
