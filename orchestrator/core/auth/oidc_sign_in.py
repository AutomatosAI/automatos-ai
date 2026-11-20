"""PRD-277 US-402 (§5.2, §5.11): signing in with the organisation's identity provider, over OIDC.

``OidcSignIn`` is core's one ``IdentityMethod`` (``core.auth.sign_in_methods``).

**start** reads the provider's metadata (``oidc_provider.current_metadata``), starts a sign-in
(``oidc_pending.begin``: a state, a nonce, a browser binding and a PKCE verifier, kept ten minutes)
and returns the
address of the provider's own sign-in page, with:

* ``response_type=code``, ``client_id``, ``scope`` (``OIDC_SCOPES``), ``state``, ``nonce``, and the
  PKCE ``code_challenge`` with ``code_challenge_method=S256``;
* ``redirect_uri``: exactly ``<FRONTEND_URL's origin>/api/v1/auth/oidc/callback``, the one address
  the customer registers (§5.2). The web app and the API share that origin (§5.3);
* for a step-up (US-406, §5.5) ``max_age=0`` as well, and the sign-in's row keeps the session that
  asked, which ``finish`` hands back with when the step-up started.

The browser binding is a secret of its own, never in an address: the route puts it in a
``__Host-`` cookie only this browser holds, so a callback carried to another browser (login
cross-site request forgery) is not a sign-in there, even by someone who has seen the state.

**finish** refuses at the first rule that fails, in this order:

1. The callback's one ``state`` and the browser's binding cookie must name a live sign-in, which is
   ended there and then (``oidc_pending.claim``: single use, atomic; the binding confirmed in
   constant time). A missing or other binding ends nothing. Else ``SignInExpired``. Everything after
   this point runs at most once per sign-in, so the code is exchanged at most once.
2. The provider must not have answered with ``error``; an RFC 9207 ``iss`` must equal
   ``OIDC_ISSUER``; a ``code`` must be present, once, and of a sane length. Else ``InvalidSignIn``.
3. The code is exchanged (``oidc_exchange``) and the ID token verified (``oidc_tokens``, every rule
   of §5.11, the nonce this sign-in's). A refusal is ``InvalidSignIn``.
4. The identity is read from the verified claims (``verified_identity``).

``InvalidSignIn``'s text names the rule broken and nothing else: it is for the log. What the
provider said in ``error`` or ``error_description`` is never read into a log line.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import urlencode
from uuid import UUID

from config import config
from core import editions
from core.auth import oidc_exchange, oidc_http, oidc_pending, oidc_provider, oidc_tokens
from core.auth.session_cookie import install_origin
from core.auth.sign_in_methods import (
    Callback,
    Finished,
    InvalidSignIn,
    SignInExpired,
    Started,
    VerifiedIdentity,
    safe_return_path,
)
from core.models.auth_sessions import IDP_SESSION_ID_MAX
from core.models.user_identities import ISSUER_MAX, SUBJECT_MAX
from core.auth.user_link import PROFILE_VALUE_MAX
from core.edition_start_up import EMAIL_CLAIM_VOUCHED, STANDARD_EMAIL_CLAIM

logger = logging.getLogger(__name__)

METHOD_NAME = "oidc"
CALLBACK_PATH = "/api/v1/auth/oidc/callback"
CODE_MAX_LENGTH = 4096
# A token without ``auth_time`` is recorded as signed in at the epoch, so a check of how recently
# the person proved who they are (step-up, US-406) can never pass on it.
NO_AUTH_TIME = datetime(1970, 1, 1, tzinfo=timezone.utc)
# OIDC Core §3.1.2.1: max_age=0 makes the provider authenticate the person again, as prompt=login
# does, and makes auth_time a required claim of the ID token it returns.
STEP_UP_MAX_AGE = "0"
# Where the provider sends the browser back after signing out there (§5.6): the sign-in page.
SIGNED_OUT_PATH = "/sign-in"


def callback_address() -> str:
    """The exact redirect address registered at the provider: the install's origin and ``CALLBACK_PATH``."""
    origin = install_origin()
    if origin is None:
        raise oidc_provider.ProviderMisconfigured("FRONTEND_URL has no origin; sign-in cannot redirect back.")
    return origin + CALLBACK_PATH


def _authorize_address(endpoint: str, values: Mapping[str, str]) -> str:
    joiner = "&" if "?" in endpoint else "?"
    return endpoint + joiner + urlencode(values)


def _one(params: Mapping[str, Tuple[str, ...]], name: str) -> Optional[str]:
    """The parameter's value when the callback carries it exactly once, else ``None``."""
    values = params.get(name) or ()
    return values[0] if len(values) == 1 else None


def _profile_value(value: Any, *, strip: bool) -> Optional[str]:
    """A claim as a ``users`` profile value, or ``None`` when it is not a clean, short string."""
    if not isinstance(value, str) or "\x00" in value:
        return None
    text = value.strip() if strip else value
    if not text or text != text.strip() or len(text) > PROFILE_VALUE_MAX:
        return None
    return text


def _is_seconds(value: Any) -> bool:
    """A JSON number that is a real count of seconds: not a boolean, NaN or an infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _auth_time(value: Any, now: float) -> datetime:
    """``auth_time`` as a moment: ``NO_AUTH_TIME`` when absent; ``InvalidSignIn`` when it is not a
    finite number of seconds, is before the epoch, or is later than ``now`` and the leeway."""
    if value is None:
        return NO_AUTH_TIME
    if not _is_seconds(value):
        raise InvalidSignIn("its auth_time is not a number")
    if value < 0:
        raise InvalidSignIn("its auth_time is before the epoch")
    if value > now + oidc_tokens.TIME_LEEWAY_SECONDS:
        raise InvalidSignIn("its auth_time is in the future")
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        raise InvalidSignIn("its auth_time is out of range") from None


def _fits(value: str, limit: int) -> bool:
    return "\x00" not in value and len(value.encode("utf-8")) <= limit


def _email_verified(claims: Mapping[str, Any]) -> bool:
    """Whether the address read from ``OIDC_CLAIM_EMAIL`` is one the provider verified.

    ``email_verified`` speaks only for the standard ``email`` claim, so it counts only when that is
    the claim read, and only as the JSON ``true``. An address from any other claim is unverified (and
    not kept), unless the operator says the provider puts only verified addresses there
    (``OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES``).
    """
    if config.OIDC_CLAIM_EMAIL == STANDARD_EMAIL_CLAIM:
        return claims.get("email_verified") is True
    return config.OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES == EMAIL_CLAIM_VOUCHED


def verified_identity(claims: Mapping[str, Any], now: Optional[float] = None) -> VerifiedIdentity:
    """Who the verified claims say the person is.

    The email is read from the claim ``OIDC_CLAIM_EMAIL`` names, and counts as verified only as
    ``_email_verified`` says. The name is read from ``OIDC_CLAIM_NAME``'s claim. A value that is
    not a clean string a ``users`` row can hold is no value. ``sid`` is kept when it fits its column.
    """
    issuer, subject = claims["iss"], claims["sub"]
    if not _fits(issuer, ISSUER_MAX) or not _fits(subject, SUBJECT_MAX):
        raise InvalidSignIn("its issuer or subject does not fit an identity link")
    sid = claims.get("sid")
    moment = datetime.now(timezone.utc).timestamp() if now is None else now
    return VerifiedIdentity(
        issuer=issuer, subject=subject,
        email=_profile_value(claims.get(config.OIDC_CLAIM_EMAIL), strip=False),
        email_verified=_email_verified(claims),
        name=_profile_value(claims.get(config.OIDC_CLAIM_NAME), strip=True),
        auth_time=_auth_time(claims.get("auth_time"), moment),
        idp_session_id=sid if isinstance(sid, str) and sid and len(sid) <= IDP_SESSION_ID_MAX else None,
        claims=claims,
    )


def _code_of(params: Mapping[str, Tuple[str, ...]]) -> str:
    """The one-time code, once the provider's answer is known to be a code from the right issuer."""
    if "error" in params:
        raise InvalidSignIn("the provider answered with an error")
    if "iss" in params and _one(params, "iss") != config.OIDC_ISSUER:
        raise InvalidSignIn("the callback's iss is not OIDC_ISSUER")
    code = _one(params, "code")
    if not code or len(code) > CODE_MAX_LENGTH:
        raise InvalidSignIn("the callback carries no usable code")
    return code


def _id_token_for(code: str, claimed: oidc_pending.ClaimedSignIn) -> str:
    metadata = oidc_provider.current_metadata()
    try:
        return oidc_exchange.exchange_code(metadata, code=code, code_verifier=claimed.code_verifier,
                                           redirect_uri=callback_address())
    except oidc_exchange.CodeExchangeRefused:
        raise InvalidSignIn("the token endpoint refused the code") from None


def _claims_of(id_token: str, claimed: oidc_pending.ClaimedSignIn) -> Mapping[str, Any]:
    try:
        return oidc_tokens.verify_id_token(id_token, nonce_hash=claimed.nonce_hash)
    except oidc_tokens.InvalidIdToken as broken:
        rule = broken.rule
    raise InvalidSignIn(f"the ID token was refused: {rule}")


def end_session_address() -> Optional[str]:
    """Where to send a browser that signed out, to sign out at the provider too (§5.6), or None.

    Only where people sign in with OIDC, and only when the provider's metadata lists an
    ``end_session_endpoint`` (an https address, ``oidc_provider.parse_metadata``) on the issuer's
    host or a host in ``OIDC_ENDPOINT_HOSTS``, as the token endpoint and the keys must be
    (``oidc_provider.on_allowed_host``); one anywhere else is treated as none. The address
    carries exactly two parameters (OpenID Connect RP-Initiated Logout 1.0 §2): ``client_id``, and
    ``post_logout_redirect_uri``, the install's own ``SIGNED_OUT_PATH``, which the customer
    registers at the provider. No ID token is kept, so none is sent as a hint. None when the
    provider cannot be read: the session has ended here all the same.
    """
    if editions.current().sign_in != editions.SIGN_IN_OIDC or not config.OIDC_ISSUER:
        return None
    origin = install_origin()
    try:
        endpoint = oidc_provider.current_metadata().end_session_endpoint
    except oidc_http.ProviderUnreachable:
        logger.warning("Signed out here; the provider could not be read, so not at the provider")
        return None
    if not endpoint or origin is None:
        return None
    if not oidc_provider.on_allowed_host(endpoint, config.OIDC_ISSUER):
        logger.warning("Signed out here only: the provider's end_session_endpoint is on a host that is "
                       "neither the issuer's nor in OIDC_ENDPOINT_HOSTS")
        return None
    return _authorize_address(endpoint, {"client_id": config.OIDC_CLIENT_ID,
                                         "post_logout_redirect_uri": origin + SIGNED_OUT_PATH})


class OidcSignIn:
    """Signing in with the organisation's OIDC provider: authorization code, PKCE S256, state, nonce."""

    def start(self, db: Any, return_to: str, client_address: str,
              step_up_session_id: Optional[UUID] = None) -> Started:
        """Begin a sign-in that returns to ``return_to`` (a path on this install, else ``/``), counted
        against ``client_address``'s pending sign-ins (``oidc_pending.begin``). For a step-up
        (``step_up_session_id``, US-406) the address also carries ``max_age=0``: the provider must
        make the person sign in again, and report when in ``auth_time``."""
        metadata = oidc_provider.current_metadata()
        redirect_uri = callback_address()
        started = oidc_pending.begin(db, safe_return_path(return_to), client_address,
                                     step_up_session_id=step_up_session_id)
        values = {
            "response_type": oidc_provider.CODE_FLOW, "client_id": config.OIDC_CLIENT_ID,
            "redirect_uri": redirect_uri, "scope": config.OIDC_SCOPES, "state": started.state,
            "nonce": started.nonce, "code_challenge": started.code_challenge,
            "code_challenge_method": oidc_provider.PKCE_S256,
        }
        if step_up_session_id is not None:
            values = {**values, "max_age": STEP_UP_MAX_AGE}
        address = _authorize_address(metadata.authorization_endpoint, values)
        return Started(redirect_to=address, browser_binding=started.binding)

    def finish(self, db: Any, callback: Callback) -> Finished:
        """The verified identity the callback proves, or ``SignInExpired`` / ``InvalidSignIn``."""
        claimed = oidc_pending.claim(db, _one(callback.params, "state"), callback.browser_binding)
        if claimed is None:
            raise SignInExpired("the callback names no live sign-in of this browser")
        code = _code_of(callback.params)
        claims = _claims_of(_id_token_for(code, claimed), claimed)
        return Finished(identity=verified_identity(claims), return_to=claimed.return_to,
                        step_up_session_id=claimed.step_up_session_id, started_at=claimed.started_at)


__all__ = [
    "CALLBACK_PATH",
    "CODE_MAX_LENGTH",
    "METHOD_NAME",
    "NO_AUTH_TIME",
    "SIGNED_OUT_PATH",
    "STEP_UP_MAX_AGE",
    "OidcSignIn",
    "callback_address",
    "end_session_address",
    "verified_identity",
]
