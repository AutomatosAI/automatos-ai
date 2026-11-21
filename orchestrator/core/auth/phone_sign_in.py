"""PRD-277 US-603 (§5.7, D-8): what a phone's sign-in asks for, and where its answer goes.

A phone signs in through the one sign-in the browser uses (``GET /api/v1/auth/oidc/start`` and
``/callback``): the same redirect address, state, nonce and PKCE towards the provider, access rule,
first sign-in, first admin and audit. Only its two ends differ. It starts with ``START_PARAMS`` on the
start address, read here (``phone_start_from``) and refused whole when any is malformed, before the
person is sent to the provider. They are kept in the sign-in's own row (``oidc_pending``) and the
callback reads them from there, never from its own query. Its answer is a one-time code on one of
``APP_LINKS`` (``services.mobile.sign_in_codes``), never a session or a cookie.

* ``app_link``: a KEY of ``APP_LINKS``, never an address. The answer goes only to the link that key
  names, so no caller can choose where a code is sent. The one link is Automatos Go's own scheme:
  the code passes through no host but the install's (D-12), and PKCE makes a code another app
  catches on that scheme useless to it. ``APP_LINKS`` is the contract the app's PRD (US-604) matches.
* ``code_challenge``: the app's PKCE challenge, ``S256`` only (``code_challenge_method``, required):
  43 base64url characters. ``plain``, a missing method, another length or alphabet is refused.
* ``install_id`` and ``platform``: the device, as ``POST /api/v1/mobile/pair`` takes them; a phone
  is ``ios`` or ``android`` (a browser signs in with its own session).

The answer to the app carries ``code`` and nothing else, or ``error`` with one of ``APP_ERRORS``:
the callback's refusal code, never its detail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
from urllib.parse import urlencode

from core.models.mobile_devices import CODE_CHALLENGE_LENGTH, INSTALL_ID_PATTERN

# The app links a phone's answer may go to, by key. Automatos Go's scheme is ``automatos``
# (its ``app.json``); the path is the route the app opens for a sign-in's answer (US-604).
APP_LINKS: Mapping[str, str] = MappingProxyType({"go": "automatos://go/signed-in"})
CHALLENGE_METHOD = "S256"
PHONE_PLATFORMS = ("ios", "android")
START_PARAMS = ("app_link", "code_challenge", "code_challenge_method", "install_id", "platform")
_CHALLENGE = re.compile(rf"^[A-Za-z0-9_-]{{{CODE_CHALLENGE_LENGTH}}}$")
_INSTALL_ID = re.compile(INSTALL_ID_PATTERN)

# What the app is told when a sign-in it started ends without a code (besides the callback's own
# refusal codes, core.auth.sign_in_flow): the person is in no workspace a device can belong to, in
# more than one, or holds the most devices one person may.
NO_WORKSPACE = "no_workspace"
SEVERAL_WORKSPACES = "several_workspaces"
DEVICE_LIMIT_REACHED = "device_limit_reached"
PHONE_REFUSALS = (NO_WORKSPACE, SEVERAL_WORKSPACES, DEVICE_LIMIT_REACHED)


class InvalidPhoneStart(ValueError):
    """A start that asked to be a phone's, with a parameter missing, repeated or malformed."""


@dataclass(frozen=True)
class PhoneStart:
    """What a phone's start asked for: the app link's key, the app's S256 challenge, and the device."""

    app_link: str
    code_challenge: str = field(repr=False)
    install_id: str = field(repr=False)
    platform: str


def _one_each(params: Mapping[str, Tuple[str, ...]]) -> Mapping[str, str]:
    values = {name: params.get(name, ()) for name in START_PARAMS}
    if any(len(sent) != 1 for sent in values.values()):
        raise InvalidPhoneStart("each phone parameter is sent once")
    return {name: sent[0] for name, sent in values.items()}


def phone_start_from(params: Mapping[str, Tuple[str, ...]]) -> Optional[PhoneStart]:
    """The phone's start in ``params`` (each name with every value sent), ``None`` when none is named.

    ``InvalidPhoneStart`` when any of ``START_PARAMS`` is named and any is missing, repeated or
    malformed. Its text names the rule, never a value.
    """
    if not any(name in params for name in START_PARAMS):
        return None
    sent = _one_each(params)
    if sent["app_link"] not in APP_LINKS:
        raise InvalidPhoneStart("app_link is not a known app link")
    if sent["code_challenge_method"] != CHALLENGE_METHOD or not _CHALLENGE.match(sent["code_challenge"]):
        raise InvalidPhoneStart("the code challenge is not an S256 challenge")
    if not _INSTALL_ID.match(sent["install_id"]) or sent["platform"] not in PHONE_PLATFORMS:
        raise InvalidPhoneStart("install_id or platform is not a phone's")
    return PhoneStart(app_link=sent["app_link"], code_challenge=sent["code_challenge"],
                      install_id=sent["install_id"], platform=sent["platform"])


def app_answer(phone: PhoneStart, *, code: Optional[str] = None, error: Optional[str] = None) -> str:
    """The app link ``phone`` named, carrying the one-time ``code`` or the refusal's ``error`` and nothing else."""
    value = {"code": code} if code is not None else {"error": error}
    return f"{APP_LINKS[phone.app_link]}?{urlencode(value)}"


__all__ = [
    "APP_LINKS",
    "CHALLENGE_METHOD",
    "DEVICE_LIMIT_REACHED",
    "NO_WORKSPACE",
    "PHONE_PLATFORMS",
    "PHONE_REFUSALS",
    "SEVERAL_WORKSPACES",
    "START_PARAMS",
    "InvalidPhoneStart",
    "PhoneStart",
    "app_answer",
    "phone_start_from",
]
