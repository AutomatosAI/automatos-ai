"""PRD-266 §4.1 (US-001) — the discovery document: what this server is, before any sign-in.

``GET /.well-known/automatos.json`` (``api/well_known.py``) answers ``build_discovery``, so one
app binary can talk to hosted Studio and to a self-hosted install alike:

* ``schema`` is ``automatos-discovery/v1``. When ``auth.modes`` later gains ``"oidc"``
  (PRD-257, §4.11) the schema stays v1: the app treats a mode it does not know as "not
  supported by this app version". This build never emits ``"oidc"``.
* ``edition`` is ``AUTH_EDITION``. saas signs in with Clerk: ``auth.modes ["clerk"]`` and
  ``clerk {publishable_key, frontend_api}``, the frontend API decoded from the publishable key
  (``clerk_frontend_api``; no setting of its own). local signs in with a paired device:
  ``auth.modes ["device"]``, ``clerk`` null, and ``pairing.enabled`` only while the install
  takes paired devices (``LOCAL_REMOTE_ACCESS=paired``, ``services.mobile.remote_access``).
* ``api_url`` and ``web_url`` are ``services.mobile.urls``'s, the helpers the pair link uses
  (K15): a local install never advertises the hosted API.
* ``push.expo`` says whether this server sends notifications through Expo's push service
  (``services.push.consent.expo_push_allowed``, W2 K13): always on SaaS; on a local install
  only once the operator agreed. It is read from the DB settings table on every call.
* ``capabilities`` lists only what this build implements, each name mapped in
  ``CAPABILITY_ROUTES`` to the live route that implements it (a test checks every entry
  against the app's routes). The app gates every feature on these names, never on a version.

Names and capabilities only, never a secret: the publishable key is the one key in the body,
and it is public by design (the web app ships it in its bundle). A misconfigured address or
publishable key raises ``DiscoveryUnavailable`` naming the setting, never its value.
"""
from __future__ import annotations

import base64
import binascii
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from config import config
from services.mobile.remote_access import MODE_PAIRED
from services.mobile.urls import MobileUrlError, api_url, web_url
from services.push.consent import expo_push_allowed

DISCOVERY_SCHEMA = "automatos-discovery/v1"
SERVER_NAME = "Automatos"
AUTH_MODE_CLERK = "clerk"
AUTH_MODE_DEVICE = "device"
# A Clerk publishable key is one of these prefixes, then base64 of "<frontend API host>$".
CLERK_KEY_PREFIXES = ("pk_test_", "pk_live_")
CLERK_KEY_TERMINATOR = "$"
BASE64_BLOCK = 4
PUBLISHABLE_KEY_SETTING = "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY"

# Capability name -> (method, path) of the live route that implements it. Only what this build
# implements. spoken_mode: ``spokenMode`` on the chat request (US-007a). voice_live: the Auto
# Live mint with Retell's transport fields (US-008a; Business-only on SaaS, D34). push: the
# phone registers its Expo token for push (W2 US-009). digest: Auto's Read is posted daily at
# the phone's digest time (W2 US-013); its switch is the daily_digest push preference row.
# mobile_theme: the member's installed theme as data for the phone (PRD-261 M-04: colours,
# fonts and mascot files, never code), which the app offers under Studio only when this is listed.
CAPABILITY_ROUTES: Mapping[str, Tuple[str, str]] = MappingProxyType({
    "digest": ("GET", "/api/notification-preferences"),
    "mobile_theme": ("GET", "/api/themes/mobile/{install_id}"),
    "push": ("POST", "/api/v1/mobile/devices"),
    "spoken_mode": ("POST", "/api/chat"),
    "voice_live": ("POST", "/api/voice/web-call"),
})


class DiscoveryUnavailable(RuntimeError):
    """This server cannot describe itself: a setting the document needs is misconfigured."""


def clerk_frontend_api(publishable_key: Optional[str]) -> Optional[str]:
    """The Clerk frontend API host a publishable key encodes, or None for anything else.

    ``pk_test_`` / ``pk_live_`` followed by base64 of ``"<host>$"`` (Clerk's format); a key
    without the prefix, with bad base64, without the trailing ``$`` or naming no host is None.
    """
    key = (publishable_key or "").strip()
    prefix = next((p for p in CLERK_KEY_PREFIXES if key.startswith(p)), None)
    if prefix is None:
        return None
    encoded = key[len(prefix):]
    padded = encoded + "=" * (-len(encoded) % BASE64_BLOCK)
    try:
        decoded = base64.b64decode(padded, validate=True).decode("ascii")
    except (binascii.Error, UnicodeDecodeError):
        return None
    host = decoded[:-len(CLERK_KEY_TERMINATOR)] if decoded.endswith(CLERK_KEY_TERMINATOR) else ""
    return host or None


def _clerk_block() -> Dict[str, str]:
    """``{publishable_key, frontend_api}`` for saas; refused when the key is not Clerk's."""
    key = (config.CLERK_PUBLISHABLE_KEY or "").strip()
    frontend_api = clerk_frontend_api(key)
    if frontend_api is None:
        raise DiscoveryUnavailable(f"{PUBLISHABLE_KEY_SETTING} is not a Clerk publishable key")
    return {"publishable_key": key, "frontend_api": frontend_api}


def _auth_and_pairing() -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """The ``auth`` and ``pairing`` blocks for this edition and remote-access mode."""
    if config.IS_LOCAL_EDITION:
        auth = {"modes": [AUTH_MODE_DEVICE], "clerk": None}
        return auth, {"enabled": config.LOCAL_REMOTE_ACCESS == MODE_PAIRED}
    return {"modes": [AUTH_MODE_CLERK], "clerk": _clerk_block()}, {"enabled": False}


def build_discovery(request_base_url: str) -> Dict[str, Any]:
    """The §4.1 discovery document for this server, read fresh from config on every call.

    ``request_base_url`` is the base URL the caller reached the API on; a local install that
    has not named its own ``PUBLIC_API_HOST`` answers with it (K15). Raises
    ``DiscoveryUnavailable`` when an address or the publishable key is misconfigured.
    """
    try:
        addresses = {"api_url": api_url(request_base_url), "web_url": web_url()}
    except MobileUrlError as misconfigured:
        raise DiscoveryUnavailable(str(misconfigured)) from misconfigured
    auth, pairing = _auth_and_pairing()
    return {
        "schema": DISCOVERY_SCHEMA,
        "name": SERVER_NAME,
        "edition": config.AUTH_EDITION,
        **addresses,
        "auth": auth,
        "pairing": pairing,
        "push": {"expo": expo_push_allowed()},
        "capabilities": sorted(CAPABILITY_ROUTES),
    }


__all__ = [
    "CAPABILITY_ROUTES",
    "DISCOVERY_SCHEMA",
    "DiscoveryUnavailable",
    "build_discovery",
    "clerk_frontend_api",
]
