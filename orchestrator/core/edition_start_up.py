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

Every other edition passes untouched: nothing here reads a setting it does not use.

Imports no config and reads no setting itself: ``config.py`` hands the values in.
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

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


__all__ = [
    "MISSING_SETTINGS",
    "NOT_AVAILABLE",
    "OIDC_SIGN_IN_IN_THIS_BUILD",
    "OIDC_START_UP_SETTINGS",
    "EditionNotReady",
    "check_start_up",
    "missing_settings",
]
