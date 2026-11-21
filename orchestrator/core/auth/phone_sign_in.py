"""PRD-277 US-603 (§5.7, D-8): what a phone's sign-in asks for, where its answer goes, and the page between.

A phone signs in through the one sign-in the browser uses (``GET /api/v1/auth/oidc/start`` and
``/callback``): the same redirect address, state, nonce and PKCE towards the provider, access rule,
first sign-in, first admin and audit. It starts with ``START_PARAMS`` on the start address, read here
(``phone_start_from``) and refused whole when any is malformed, before the person is sent to the
provider. They are kept in the sign-in's own row (``oidc_pending``) and the callback reads them from
there, never from its own query. Towards the provider a phone's start differs in one way: it asks the
person to sign in again, as a step-up does (``oidc_sign_in.auth_time_values``), so no sign-in the
provider already holds in this browser answers for them unseen.

**What it can and cannot prove.** Nothing at the install can tell the real Automatos Go app from any
other program that sends the same start: the app holds no secret of its own. PKCE binds a code to
whoever STARTED the flow, not to the app: a code is useless without the verifier of the one who started
it, so another app that catches the code on the scheme cannot exchange it, but someone who starts a
flow with their own challenge and gets another person to sign in would hold a working pair. What stands
in for knowing the app is the person: after the callback nothing is made until they act on a page the
install serves (``CONFIRM_PAGE``), which says which app on which phone they are signing in to, as whom,
and asks them to continue only if they started this from the app. That record is bound to their browser
by ``CONFIRM_COOKIE`` (``services.mobile.sign_in_confirmation``); only Continue makes the one-time code.

* ``app_link``: a KEY of ``APP_LINKS``, never an address. The answer goes only to the link that key
  names, so no caller can choose where a code is sent. The one link is Automatos Go's own scheme: the
  code passes through no host but the install's (D-12). The scheme stays because every install runs on
  its own domain, and an app from a store can claim universal links only for domains fixed in its build.
* ``code_challenge``: the app's PKCE challenge, ``S256`` only (``code_challenge_method``, required):
  43 base64url characters. ``plain``, a missing method, another length or alphabet is refused.
* ``install_id`` and ``platform``: the device, as ``POST /api/v1/mobile/pair`` takes them; a phone
  is ``ios`` or ``android`` (a browser signs in with its own session).

The app is sent the link with ``code`` and nothing else, or ``error`` with a refusal's code from
before the person was known (the callback's, never its detail). A refusal once the person is known
that only a phone meets (``PHONE_REFUSALS``) is shown on ``CONFIRM_PAGE`` and the app is told nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import urlencode

from core.auth.session_cookie import COOKIE_PATH, COOKIE_SAME_SITE
from core.models.mobile_devices import CODE_CHALLENGE_LENGTH, INSTALL_ID_PATTERN

# The app links a phone's answer may go to, by key. Automatos Go's scheme is ``automatos``
# (its ``app.json``); the path is the route the app opens for a sign-in's answer (US-604).
APP_LINKS: Mapping[str, str] = MappingProxyType({"go": "automatos://go/signed-in"})
# What the confirmation page calls each app: from this fixed map, never from a request.
APP_NAMES: Mapping[str, str] = MappingProxyType({"go": "Automatos Go"})
CHALLENGE_METHOD = "S256"
PHONE_PLATFORMS = ("ios", "android")
START_PARAMS = ("app_link", "code_challenge", "code_challenge_method", "install_id", "platform")
_CHALLENGE = re.compile(rf"^[A-Za-z0-9_-]{{{CODE_CHALLENGE_LENGTH}}}$")
_INSTALL_ID = re.compile(INSTALL_ID_PATTERN)

# The web app's page a phone's sign-in waits on, with no value in its address (a refusal below adds
# ``?error=<code>``), and the cookie that binds what waits to this browser. ``__Host-`` requires
# ``Path=/`` and no ``Domain``, so ``/`` is the narrowest path the cookie can have.
CONFIRM_PAGE = "/sign-in/phone"
CONFIRM_COOKIE = "__Host-automatos_phone_sign_in"
# How long the person has to press Continue or Cancel; the cookie lives as long.
CONFIRM_TTL_SECONDS = 300

# The refusals only a phone meets, once the person is known: in no workspace a device can belong to,
# in more than one, or holding the most devices one person may. Shown on CONFIRM_PAGE, never sent to
# the app; each is audited as ``auth:sign_in_refused`` with itself as the reason.
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


def app_answer(app_link: str, *, code: Optional[str] = None, error: Optional[str] = None) -> str:
    """The app link the key ``app_link`` names, carrying the one-time ``code`` or ``error`` and nothing else."""
    value = {"code": code} if code is not None else {"error": error}
    return f"{APP_LINKS[app_link]}?{urlencode(value)}"


def refusal_page(code: str) -> str:
    """``CONFIRM_PAGE`` naming one of ``PHONE_REFUSALS``, which the page turns into its own sentence."""
    return f"{CONFIRM_PAGE}?{urlencode({'error': code})}"


def set_confirm_cookie(response: Any, token: str) -> None:
    """Bind what waits for the person to this browser: ``__Host-``, Secure, HttpOnly, SameSite=Lax."""
    response.set_cookie(key=CONFIRM_COOKIE, value=token, max_age=CONFIRM_TTL_SECONDS, path=COOKIE_PATH,
                        secure=True, httponly=True, samesite=COOKIE_SAME_SITE)


def clear_confirm_cookie(response: Any) -> None:
    """Tell the browser to drop ``CONFIRM_COOKIE`` (the same attributes it was set with)."""
    response.delete_cookie(key=CONFIRM_COOKIE, path=COOKIE_PATH, secure=True, httponly=True,
                           samesite=COOKIE_SAME_SITE)


__all__ = [
    "APP_LINKS",
    "APP_NAMES",
    "CHALLENGE_METHOD",
    "CONFIRM_COOKIE",
    "CONFIRM_PAGE",
    "CONFIRM_TTL_SECONDS",
    "DEVICE_LIMIT_REACHED",
    "NO_WORKSPACE",
    "PHONE_PLATFORMS",
    "PHONE_REFUSALS",
    "SEVERAL_WORKSPACES",
    "START_PARAMS",
    "InvalidPhoneStart",
    "PhoneStart",
    "app_answer",
    "clear_confirm_cookie",
    "phone_start_from",
    "refusal_page",
    "set_confirm_cookie",
]
