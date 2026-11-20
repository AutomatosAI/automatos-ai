"""PRD-277 US-401 (§5.9, §5.11): the identity provider's metadata and its signing keys.

**Metadata** is read from ``<OIDC_ISSUER>/.well-known/openid-configuration`` (OpenID Connect
Discovery 1.0 §4), through ``core.auth.oidc_http`` only: https, the address pinned, and the issuer's
own host the one host that may resolve to a private address. It is used only when:

* its ``issuer`` equals ``OIDC_ISSUER`` exactly (§5.11; Discovery §4.3);
* ``authorization_endpoint``, ``token_endpoint`` and ``jwks_uri`` are https addresses, and so is
  ``end_session_endpoint`` when it lists one (kept for signing out, §5.6);
* ``token_endpoint`` and ``jwks_uri``, the two this server calls, are on the issuer's own host or on
  a host the operator lists in ``OIDC_ENDPOINT_HOSTS`` (§5.9: some providers publish them elsewhere;
  ``on_allowed_host``). Sign-out sends a browser to ``end_session_endpoint`` only under the same
  rule (``core.auth.oidc_sign_in.end_session_address``);
* what it lists, where it lists it, includes the authorization code flow (``code``), PKCE ``S256``,
  one of ``SIGNING_ALGORITHMS``, and a client authentication this client uses
  (``client_secret_basic``, else ``client_secret_post``).

Two more things are read from it, never a reason to refuse it: whether the sign-in page may be sent
the ``claims`` parameter (``takes_claims_request``) and, on a step-up, ``prompt=login``
(``takes_prompt_login``); ``core.auth.oidc_sign_in`` sends them.

It is held for ``METADATA_LIFESPAN_SECONDS`` and read again after.

**Keys** follow ``core/auth/clerk.py``'s ``PyJWKClient`` pattern: the key set at ``jwks_uri`` is held
for ``KEYS_LIFESPAN_SECONDS`` (the same five minutes), and a token's key is the one its ``kid`` names
among keys whose ``use`` is ``sig`` or absent, built by PyJWT's ``PyJWK``. The set is fetched through
``oidc_http``, not by ``PyJWKClient`` itself, which opens the address with ``urllib``, where the pinned
connection and the private-address rule cannot be applied. A key must also fit the token's algorithm
(an RSA key for RS256 and PS256, a P-256 key for ES256, and the key's own ``alg`` when it states one).

**An unknown ``kid``** (the provider rotated its keys) reads the set again once, and at most once in
``KEY_REFETCH_INTERVAL_SECONDS`` in this process, so tokens naming keys that do not exist can never
make the server call the provider on every request.

**Unreachable.** Nothing here runs at import or blocks start-up. ``check_provider`` is the boot's
report stage (``core.auth.sign_in_start_up``): the API starts either way, ``/health/bootstrap`` says
whether the metadata and keys could be read, and sign-in answers with a clear error until they can.
A failed metadata read is held for ``FAILED_READ_HOLD_SECONDS``: sign-ins in that time are refused at
once instead of each waiting on the provider, so an unreachable provider cannot hold a worker per
request. The first sign-in after it asks again.

Each read replaces one frozen snapshot under a lock; nothing is changed in place.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

import jwt

from config import config
from core.auth import oidc_http
from core.edition_start_up import listed_hosts

logger = logging.getLogger(__name__)

METADATA_PATH = "/.well-known/openid-configuration"
METADATA_LIFESPAN_SECONDS = 300
# core/auth/clerk.py's PyJWKClient lifespan.
KEYS_LIFESPAN_SECONDS = 300
KEY_REFETCH_INTERVAL_SECONDS = 60
FAILED_READ_HOLD_SECONDS = 5

# §5.11: the only algorithms an ID token may be signed with. Never ``none``, never HMAC.
SIGNING_ALGORITHMS: Tuple[str, ...] = ("RS256", "PS256", "ES256")
KEY_TYPE_FOR = {"RS256": "RSA", "PS256": "RSA", "ES256": "EC"}
ES256_CURVE = "P-256"
SIGNING_USES = ("sig", None)

CODE_FLOW = "code"
PKCE_S256 = "S256"
AUTH_BASIC = "client_secret_basic"
AUTH_POST = "client_secret_post"

# What the sign-in page is asked for beyond the code flow (US-406, §5.5): the ``claims`` parameter
# (OIDC Core §5.5) and, on a step-up, ``prompt=login`` (§3.1.2.1). Discovery 1.0 §3 names the
# field that says whether the first is taken; Initiating User Registration 1.0 §4.2 the list of
# prompt values a provider takes.
CLAIMS_PARAMETER_FIELD = "claims_parameter_supported"
PROMPT_VALUES_FIELD = "prompt_values_supported"
PROMPT_LOGIN = "login"
# Issuers whose metadata lists no prompt values but whose own documentation does, without
# ``login``: Google's lists ``none``, ``consent`` and ``select_account``
# (developers.google.com/identity/openid-connect/openid-connect). OIDC Core §3.1.2.1 lets a
# provider answer an error for a prompt value it does not understand, so it is not sent to them.
PROMPT_LOGIN_UNLISTED_HOSTS: Tuple[str, ...] = ("accounts.google.com",)

MISCONFIGURED_MESSAGE = "The organisation's identity provider publishes settings sign-in cannot use."
UNKNOWN_KEY_MESSAGE = "The ID token names a signing key the identity provider does not publish."


class ProviderMisconfigured(oidc_http.ProviderUnreachable):
    """The provider answered with metadata or keys sign-in cannot use. Its text never holds a value."""


class UnknownSigningKey(Exception):
    """No published key fits the token's ``kid`` and algorithm, even after the one refetch."""


@dataclass(frozen=True)
class ProviderMetadata:
    """What sign-in uses of the provider's metadata."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    end_session_endpoint: Optional[str]
    token_auth_method: str
    takes_claims_request: bool
    takes_prompt_login: bool


@dataclass(frozen=True)
class _Snapshot:
    """Everything this process holds of the provider, replaced whole on every read."""

    issuer: str = ""
    metadata: Optional[ProviderMetadata] = None
    metadata_at: float = 0.0
    keys_uri: str = ""
    keys: Tuple[Mapping[str, Any], ...] = ()
    keys_at: float = 0.0
    refetched_at: Optional[float] = None
    failed_at: Optional[float] = None


_lock = threading.Lock()
_held = _Snapshot()
# The clock seam: tests move it; production is the monotonic clock.
_clock = time.monotonic


def _snapshot() -> _Snapshot:
    with _lock:
        return _held


def _store(snapshot: _Snapshot) -> None:
    global _held
    with _lock:
        _held = snapshot


def forget_provider() -> None:
    """Drop everything held, so the next sign-in reads the metadata and keys again."""
    _store(_Snapshot())


def metadata_address(issuer: str) -> str:
    """``<issuer>/.well-known/openid-configuration`` (Discovery §4: the issuer's path is kept)."""
    return issuer.rstrip("/") + METADATA_PATH


def _misconfigured(field: str) -> ProviderMisconfigured:
    logger.error("Identity provider metadata refused: %s", field)
    return ProviderMisconfigured(MISCONFIGURED_MESSAGE)


def _endpoint(doc: Mapping[str, Any], name: str, *, required: bool = True) -> Optional[str]:
    value = doc.get(name)
    if value is None and not required:
        return None
    if not oidc_http.is_https_address(value):
        raise _misconfigured(f"{name} is not an https address")
    return value


def on_allowed_host(address: Optional[str], issuer: str) -> bool:
    """Whether ``address`` is on the issuer's own host or a host listed in ``OIDC_ENDPOINT_HOSTS``."""
    allowed = (oidc_http.host_of(issuer), *listed_hosts(config.OIDC_ENDPOINT_HOSTS))
    return oidc_http.host_of(address) in allowed


def _called_endpoint(doc: Mapping[str, Any], name: str, issuer: str) -> str:
    """An endpoint this server calls: https, on the issuer's host or one in ``OIDC_ENDPOINT_HOSTS``."""
    value = _endpoint(doc, name)
    if not on_allowed_host(value, issuer):
        raise _misconfigured(f"{name} is on a host that is neither the issuer's nor in OIDC_ENDPOINT_HOSTS")
    return value


def _offers(doc: Mapping[str, Any], name: str, wanted: Tuple[str, ...]) -> bool:
    """Whether ``doc[name]`` lists one of ``wanted``; an absent list offers everything."""
    listed = doc.get(name)
    if listed is None:
        return True
    if not isinstance(listed, list):
        raise _misconfigured(f"{name} is not a list")
    return any(value in listed for value in wanted)


def _token_auth_method(doc: Mapping[str, Any]) -> str:
    """``client_secret_basic`` (OpenID Connect's default), else ``client_secret_post``."""
    listed = doc.get("token_endpoint_auth_methods_supported")
    if listed is None or (isinstance(listed, list) and AUTH_BASIC in listed):
        return AUTH_BASIC
    if isinstance(listed, list) and AUTH_POST in listed:
        return AUTH_POST
    raise _misconfigured("no client authentication with a secret is offered")


def _takes_claims_request(doc: Mapping[str, Any]) -> bool:
    """Whether to send the ``claims`` parameter: unless the metadata says the provider refuses it.

    Discovery 1.0 §3 reads an absent ``claims_parameter_supported`` as false, but OIDC Core says
    sending it then does no harm: a provider that does not support it "SHOULD return a set of
    Claims" it judges useful (§5.5), unrecognised parameters are ignored (§3.1.2.2), and a claim
    not returned, even an essential one, is never an error (§5.5.1). Google, Microsoft, Okta and
    Auth0 publish no such field, and Google documents the parameter as the way to ask for
    ``auth_time``. So only the provider's own word counts: any value but ``true`` there.
    """
    return CLAIMS_PARAMETER_FIELD not in doc or doc[CLAIMS_PARAMETER_FIELD] is True


def _takes_prompt_login(doc: Mapping[str, Any], issuer: str) -> bool:
    """Whether a step-up may send ``prompt=login``.

    A provider listing its prompt values takes it when ``login`` is listed (one that is not
    listed "SHOULD" be answered with ``invalid_request``, Initiating User Registration 1.0 §4.1);
    a value that is not a list is read as not listing it. A provider whose metadata lists no prompt
    values takes it, as OIDC Core §15.1 requires of every provider ("including ... none and
    login"), unless its issuer is in ``PROMPT_LOGIN_UNLISTED_HOSTS``.
    """
    if PROMPT_VALUES_FIELD in doc:
        listed = doc[PROMPT_VALUES_FIELD]
        return isinstance(listed, list) and PROMPT_LOGIN in listed
    return oidc_http.host_of(issuer) not in PROMPT_LOGIN_UNLISTED_HOSTS


def parse_metadata(doc: Any, issuer: str) -> ProviderMetadata:
    """The usable part of a metadata document, or ``ProviderMisconfigured`` naming what is wrong."""
    if not isinstance(doc, dict):
        raise _misconfigured("the document is not a JSON object")
    if not isinstance(doc.get("issuer"), str) or doc["issuer"] != issuer:
        raise _misconfigured("its issuer is not OIDC_ISSUER")
    for name, wanted in (("response_types_supported", (CODE_FLOW,)),
                         ("code_challenge_methods_supported", (PKCE_S256,)),
                         ("id_token_signing_alg_values_supported", SIGNING_ALGORITHMS)):
        if not _offers(doc, name, wanted):
            raise _misconfigured(f"{name} lists nothing sign-in uses")
    return ProviderMetadata(
        issuer=issuer,
        authorization_endpoint=_endpoint(doc, "authorization_endpoint"),
        token_endpoint=_called_endpoint(doc, "token_endpoint", issuer),
        jwks_uri=_called_endpoint(doc, "jwks_uri", issuer),
        end_session_endpoint=_endpoint(doc, "end_session_endpoint", required=False),
        token_auth_method=_token_auth_method(doc),
        takes_claims_request=_takes_claims_request(doc),
        takes_prompt_login=_takes_prompt_login(doc, issuer),
    )


def _held_for(issuer: str) -> _Snapshot:
    fresh = _snapshot()
    return fresh if fresh.issuer == issuer else _Snapshot(issuer=issuer)


def current_metadata() -> ProviderMetadata:
    """The provider's metadata, read again once ``METADATA_LIFESPAN_SECONDS`` have passed.

    A failed read is held for ``FAILED_READ_HOLD_SECONDS``; in that time this raises at once.
    """
    issuer, now, held = config.OIDC_ISSUER, _clock(), _snapshot()
    if held.issuer == issuer and held.metadata is not None and now - held.metadata_at < METADATA_LIFESPAN_SECONDS:
        return held.metadata
    if held.issuer == issuer and held.failed_at is not None and now - held.failed_at < FAILED_READ_HOLD_SECONDS:
        raise oidc_http.ProviderUnreachable(oidc_http.UNREACHABLE_MESSAGE)
    try:
        metadata = parse_metadata(oidc_http.get_json(metadata_address(issuer), "metadata"), issuer)
    except oidc_http.ProviderUnreachable:
        _store(replace(_held_for(issuer), failed_at=now))
        raise
    _store(replace(_held_for(issuer), metadata=metadata, metadata_at=now, failed_at=None))
    return metadata


def _parse_keys(doc: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(doc, dict) or not isinstance(doc.get("keys"), list):
        raise _misconfigured("the key set is not a JSON object with a list of keys")
    keys = tuple(copy.deepcopy(key) for key in doc["keys"] if isinstance(key, dict))
    if not keys:
        raise _misconfigured("the key set holds no key")
    return keys


def _key_set(metadata: ProviderMetadata, *, refresh: bool) -> Tuple[Mapping[str, Any], ...]:
    """The published keys: held ones while fresh, unless ``refresh``."""
    now, held = _clock(), _snapshot()
    fresh = held.keys_uri == metadata.jwks_uri and held.keys and now - held.keys_at < KEYS_LIFESPAN_SECONDS
    if fresh and not refresh:
        return held.keys
    keys = _parse_keys(oidc_http.get_json(metadata.jwks_uri, "keys"))
    _store(replace(_snapshot(), keys_uri=metadata.jwks_uri, keys=keys, keys_at=now))
    return keys


def _may_refetch() -> bool:
    """Whether an unknown ``kid`` may read the keys again now; claims the turn when it may."""
    global _held
    now = _clock()
    with _lock:
        last = _held.refetched_at
        if last is not None and now - last < KEY_REFETCH_INTERVAL_SECONDS:
            return False
        _held = replace(_held, refetched_at=now)
        return True


def _fits(jwk: Mapping[str, Any], kid: str, alg: str) -> bool:
    if jwk.get("kid") != kid or jwk.get("use") not in SIGNING_USES:
        return False
    if jwk.get("kty") != KEY_TYPE_FOR[alg] or jwk.get("alg") not in (None, alg):
        return False
    return alg != "ES256" or jwk.get("crv") == ES256_CURVE


def _matching_key(keys: Tuple[Mapping[str, Any], ...], kid: str, alg: str) -> Any:
    """The public key ``kid`` names that fits ``alg``, or ``None``."""
    for jwk in keys:
        if not _fits(jwk, kid, alg):
            continue
        try:
            return jwt.PyJWK(dict(jwk), algorithm=alg).key
        except (jwt.PyJWTError, ValueError, TypeError) as unreadable:
            logger.warning("A published signing key could not be read as %s (%s)", alg, type(unreadable).__name__)
    return None


def signing_key(kid: str, alg: str) -> Any:
    """The public key that must have signed a token with this ``kid`` and ``alg``.

    ``UnknownSigningKey`` when none is published, after the one rate-limited refetch;
    ``ProviderUnreachable`` when the keys cannot be read.
    """
    if alg not in SIGNING_ALGORITHMS or not isinstance(kid, str) or not kid:
        raise UnknownSigningKey(UNKNOWN_KEY_MESSAGE)
    metadata = current_metadata()
    key = _matching_key(_key_set(metadata, refresh=False), kid, alg)
    if key is None and _may_refetch():
        key = _matching_key(_key_set(metadata, refresh=True), kid, alg)
    if key is None:
        raise UnknownSigningKey(UNKNOWN_KEY_MESSAGE)
    return key


def check_provider() -> None:
    """Read the metadata and the keys once (the boot's report stage); ``ProviderUnreachable`` if not."""
    _key_set(current_metadata(), refresh=False)


__all__ = [
    "AUTH_BASIC",
    "AUTH_POST",
    "FAILED_READ_HOLD_SECONDS",
    "KEYS_LIFESPAN_SECONDS",
    "KEY_REFETCH_INTERVAL_SECONDS",
    "METADATA_LIFESPAN_SECONDS",
    "METADATA_PATH",
    "PROMPT_LOGIN",
    "PROMPT_LOGIN_UNLISTED_HOSTS",
    "SIGNING_ALGORITHMS",
    "ProviderMetadata",
    "ProviderMisconfigured",
    "UnknownSigningKey",
    "check_provider",
    "current_metadata",
    "forget_provider",
    "metadata_address",
    "on_allowed_host",
    "parse_metadata",
    "signing_key",
]
