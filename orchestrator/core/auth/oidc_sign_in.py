"""PRD-277 US-402 (§5.2, §5.11): signing in with the organisation's identity provider, over OIDC.

``OidcSignIn`` is core's one ``IdentityMethod`` (``core.auth.sign_in_methods``).

**start** reads the provider's metadata (``oidc_provider.current_metadata``), starts a sign-in
(``oidc_pending.begin``: a state, a nonce and a PKCE verifier, kept ten minutes) and returns the
address of the provider's own sign-in page, with:

* ``response_type=code``, ``client_id``, ``scope`` (``OIDC_SCOPES``), ``state``, ``nonce``, and the
  PKCE ``code_challenge`` with ``code_challenge_method=S256``;
* ``redirect_uri``: exactly ``<FRONTEND_URL's origin>/api/v1/auth/oidc/callback``, the one address
  the customer registers (§5.2). The web app and the API share that origin (§5.3).

The state is also the browser binding: the route puts it in a ``__Host-`` cookie only this
browser holds, so a callback carried to another browser (login cross-site request forgery) is not
a sign-in there.

**finish** refuses at the first rule that fails, in this order:

1. The callback's ``state`` must equal the browser's cookie (constant time); else ``SignInExpired``.
2. The state must name a live sign-in, which is ended there and then (``oidc_pending.claim``:
   single use, atomic); else ``SignInExpired``. Everything after this point runs at most once per
   sign-in, so the code is exchanged at most once.
3. The provider must not have answered with ``error``; an RFC 9207 ``iss`` must equal
   ``OIDC_ISSUER``; a ``code`` must be present, once, and of a sane length. Else ``InvalidSignIn``.
4. The code is exchanged (``oidc_exchange``) and the ID token verified (``oidc_tokens``, every rule
   of §5.11, the nonce this sign-in's). A refusal is ``InvalidSignIn``.
5. The identity is read from the verified claims (``verified_identity``).

``InvalidSignIn``'s text names the rule broken and nothing else: it is for the log. What the
provider said in ``error`` or ``error_description`` is never read into a log line.
"""
from __future__ import annotations

import hmac
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import urlencode

from config import config
from core.auth import oidc_exchange, oidc_pending, oidc_provider, oidc_tokens
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

logger = logging.getLogger(__name__)

METHOD_NAME = "oidc"
CALLBACK_PATH = "/api/v1/auth/oidc/callback"
CODE_MAX_LENGTH = 4096
# A token without ``auth_time`` is recorded as signed in at the epoch, so a check of how recently
# the person proved who they are (step-up, US-406) can never pass on it.
NO_AUTH_TIME = datetime(1970, 1, 1, tzinfo=timezone.utc)


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


def _same_text(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _profile_value(value: Any, *, strip: bool) -> Optional[str]:
    """A claim as a ``users`` profile value, or ``None`` when it is not a clean, short string."""
    if not isinstance(value, str) or "\x00" in value:
        return None
    text = value.strip() if strip else value
    if not text or text != text.strip() or len(text) > PROFILE_VALUE_MAX:
        return None
    return text


def _auth_time(value: Any, now: float) -> datetime:
    if value is None:
        return NO_AUTH_TIME
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidSignIn("its auth_time is not a number")
    if value > now + oidc_tokens.TIME_LEEWAY_SECONDS:
        raise InvalidSignIn("its auth_time is in the future")
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _fits(value: str, limit: int) -> bool:
    return "\x00" not in value and len(value.encode("utf-8")) <= limit


def verified_identity(claims: Mapping[str, Any], now: Optional[float] = None) -> VerifiedIdentity:
    """Who the verified claims say the person is.

    The email counts as verified only when ``email_verified`` is the JSON ``true`` (absent, a
    string or anything else is not verified). The email and name are read from the claims
    ``OIDC_CLAIM_EMAIL`` and ``OIDC_CLAIM_NAME`` name; a value that is not a clean string a
    ``users`` row can hold is no value. ``sid`` is kept when it fits its column.
    """
    issuer, subject = claims["iss"], claims["sub"]
    if not _fits(issuer, ISSUER_MAX) or not _fits(subject, SUBJECT_MAX):
        raise InvalidSignIn("its issuer or subject does not fit an identity link")
    sid = claims.get("sid")
    moment = datetime.now(timezone.utc).timestamp() if now is None else now
    return VerifiedIdentity(
        issuer=issuer, subject=subject,
        email=_profile_value(claims.get(config.OIDC_CLAIM_EMAIL), strip=False),
        email_verified=claims.get("email_verified") is True,
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


class OidcSignIn:
    """Signing in with the organisation's OIDC provider: authorization code, PKCE S256, state, nonce."""

    def start(self, db: Any, return_to: str) -> Started:
        """Begin a sign-in that returns to ``return_to`` (a path on this install, else ``/``)."""
        metadata = oidc_provider.current_metadata()
        redirect_uri = callback_address()
        started = oidc_pending.begin(db, safe_return_path(return_to))
        address = _authorize_address(metadata.authorization_endpoint, {
            "response_type": oidc_provider.CODE_FLOW, "client_id": config.OIDC_CLIENT_ID,
            "redirect_uri": redirect_uri, "scope": config.OIDC_SCOPES, "state": started.state,
            "nonce": started.nonce, "code_challenge": started.code_challenge,
            "code_challenge_method": oidc_provider.PKCE_S256,
        })
        return Started(redirect_to=address, browser_binding=started.state)

    def finish(self, db: Any, callback: Callback) -> Finished:
        """The verified identity the callback proves, or ``SignInExpired`` / ``InvalidSignIn``."""
        state = _one(callback.params, "state")
        if not _same_text(state, callback.browser_binding):
            raise SignInExpired("the callback's state is not this browser's sign-in")
        claimed = oidc_pending.claim(db, state)
        if claimed is None:
            raise SignInExpired("the callback's state names no live sign-in")
        code = _code_of(callback.params)
        claims = _claims_of(_id_token_for(code, claimed), claimed)
        return Finished(identity=verified_identity(claims), return_to=claimed.return_to)


__all__ = [
    "CALLBACK_PATH",
    "CODE_MAX_LENGTH",
    "METHOD_NAME",
    "NO_AUTH_TIME",
    "OidcSignIn",
    "callback_address",
    "verified_identity",
]
