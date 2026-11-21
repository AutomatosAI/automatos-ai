"""PRD-277 §5.11, FR-6: an ID token is accepted only when every rule holds.

``verify_id_token`` refuses with ``InvalidIdToken`` at the first rule that fails:

1. **Shape.** A string of at most ``MAX_ID_TOKEN_LENGTH`` characters whose header can be read.
2. **Algorithm.** The header's ``alg`` is one of ``SIGNING_ALGORITHMS`` (RS256, PS256, ES256) and
   it names a key (``kid``). ``none`` and every HMAC algorithm are refused here, before any key is
   looked up, so a token "signed" with the provider's public key as an HMAC secret never reaches a
   verifier.
3. **Signature.** Verified by PyJWT against the provider's published key for that ``kid``, with
   that one algorithm allowed (``oidc_provider.signing_key``: an unknown ``kid`` reads the keys
   again once, rate limited).
4. **Issuer.** ``iss`` equals ``OIDC_ISSUER`` exactly.
5. **Audience.** ``aud`` contains ``OIDC_CLIENT_ID``. With more than one audience, ``azp`` must be
   present; an ``azp``, whenever present, must equal ``OIDC_CLIENT_ID``.
6. **Times.** ``exp`` and ``iat`` are required; ``exp``, ``iat`` and ``nbf`` are checked with
   ``TIME_LEEWAY_SECONDS`` (60) of leeway either way: expired, issued in the future and not yet
   valid are refused.
7. **Nonce.** Present, and its SHA-256 is the one kept for this sign-in at ``/start``
   (``core.auth.oidc_pending``), compared in constant time.
8. **Subject.** ``sub`` is a non-empty string.

PyJWT checks rules 3 to 6 itself, given the issuer, the audience, the leeway and the required
claims; rules 4 to 6 are checked again here, explicitly, so they hold whatever PyJWT's version
does. ``InvalidIdToken``'s text names only the rule, never a claim's value or the token: it is for
the log. PyJWT's own exception is caught and ``InvalidIdToken`` raised outside that handler.
"""
from __future__ import annotations

import copy
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import jwt

from config import config
from core.auth import oidc_provider
from core.auth.pairing import secrets_match

MAX_ID_TOKEN_LENGTH = 16 * 1024
TIME_LEEWAY_SECONDS = 60
REQUIRED_CLAIMS: Tuple[str, ...] = ("iss", "sub", "aud", "exp", "iat")


class InvalidIdToken(Exception):
    """The ID token broke a rule; ``rule`` says which, and never holds a value."""

    def __init__(self, rule: str) -> None:
        super().__init__(rule)
        self.rule = rule


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _header(id_token: Any) -> Mapping[str, Any]:
    if not isinstance(id_token, str) or not id_token or len(id_token) > MAX_ID_TOKEN_LENGTH:
        raise InvalidIdToken("not a token of a usable length")
    failure: Optional[str] = None
    try:
        header = jwt.get_unverified_header(id_token)
    except jwt.PyJWTError as unreadable:
        failure = type(unreadable).__name__
    if failure is not None:
        raise InvalidIdToken(f"its header cannot be read ({failure})")
    return header


def _algorithm_and_key_id(header: Mapping[str, Any]) -> Tuple[str, str]:
    alg, kid = header.get("alg"), header.get("kid")
    if alg not in oidc_provider.SIGNING_ALGORITHMS:
        raise InvalidIdToken("its algorithm is not allowed")
    if not isinstance(kid, str) or not kid:
        raise InvalidIdToken("its header names no key")
    return alg, kid


def _public_key(kid: str, alg: str) -> Any:
    try:
        return oidc_provider.signing_key(kid, alg)
    except oidc_provider.UnknownSigningKey:
        raise InvalidIdToken("no published key fits it") from None


def _signed_claims(id_token: str, key: Any, alg: str) -> dict:
    """The payload, once PyJWT has verified the signature, the issuer, the audience and the times."""
    failure: Optional[str] = None
    try:
        claims = jwt.decode(
            id_token, key=key, algorithms=[alg], audience=config.OIDC_CLIENT_ID, issuer=config.OIDC_ISSUER,
            leeway=TIME_LEEWAY_SECONDS, options={"require": list(REQUIRED_CLAIMS)},
        )
    except jwt.PyJWTError as refused:
        failure = type(refused).__name__
    if failure is not None:
        raise InvalidIdToken(f"PyJWT refused it ({failure})")
    if not isinstance(claims, dict):
        raise InvalidIdToken("its payload is not a JSON object")
    return claims


def _check_issuer_and_subject(claims: Mapping[str, Any]) -> None:
    issuer = claims.get("iss")
    if not isinstance(issuer, str) or issuer != config.OIDC_ISSUER:
        raise InvalidIdToken("its issuer is not OIDC_ISSUER")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise InvalidIdToken("it names no subject")


def _audiences(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise InvalidIdToken("its audience is not a string or a list of strings")


def _check_audience(claims: Mapping[str, Any]) -> None:
    client_id = config.OIDC_CLIENT_ID
    audiences = _audiences(claims.get("aud"))
    if client_id not in audiences:
        raise InvalidIdToken("its audience is not this client")
    if len(audiences) > 1 and "azp" not in claims:
        raise InvalidIdToken("it has several audiences and no azp")
    if "azp" in claims and claims.get("azp") != client_id:
        raise InvalidIdToken("its azp is not this client")


def _check_times(claims: Mapping[str, Any], now: float) -> None:
    expires, issued, not_before = claims.get("exp"), claims.get("iat"), claims.get("nbf", now)
    if not all(_is_number(value) for value in (expires, issued, not_before)):
        raise InvalidIdToken("a time in it is not a number")
    if expires <= now - TIME_LEEWAY_SECONDS:
        raise InvalidIdToken("it has expired")
    if issued > now + TIME_LEEWAY_SECONDS:
        raise InvalidIdToken("it was issued in the future")
    if not_before > now + TIME_LEEWAY_SECONDS:
        raise InvalidIdToken("it is not valid yet")


def _check_nonce(claims: Mapping[str, Any], nonce_hash: str) -> None:
    nonce = claims.get("nonce")
    if not isinstance(nonce, str) or not secrets_match(nonce_hash, nonce):
        raise InvalidIdToken("its nonce is not this sign-in's")


def verify_id_token(id_token: Any, *, nonce_hash: str, now: Optional[float] = None) -> Mapping[str, Any]:
    """The token's claims, read-only, when every rule in this module's docstring holds.

    ``InvalidIdToken`` for a broken rule; ``oidc_http.ProviderUnreachable`` when the keys cannot be read.
    """
    alg, kid = _algorithm_and_key_id(_header(id_token))
    claims = _signed_claims(id_token, _public_key(kid, alg), alg)
    _check_issuer_and_subject(claims)
    _check_audience(claims)
    _check_times(claims, time.time() if now is None else now)
    _check_nonce(claims, nonce_hash)
    return MappingProxyType(copy.deepcopy(claims))


__all__ = [
    "MAX_ID_TOKEN_LENGTH",
    "REQUIRED_CLAIMS",
    "TIME_LEEWAY_SECONDS",
    "InvalidIdToken",
    "verify_id_token",
]
