"""PRD-277 US-407: a small OpenID Connect provider that runs inside the test process.

``FakeIdentityProvider`` serves, over TLS on ``127.0.0.1`` and a free port, with ``PyJWT`` and
``cryptography`` only:

* ``<issuer>/.well-known/openid-configuration``, the metadata document;
* ``<issuer>/keys``, the signing keys (one RSA key for RS256 and PS256, one P-256 key for ES256);
* ``/authorize``, which approves at once: it answers 302 to the ``redirect_uri`` with a one-time
  ``code``, the ``state`` and the RFC 9207 ``iss``;
* ``/token``, which exchanges a code once, checking the client's id and secret (HTTP Basic or the
  form), the exact ``redirect_uri`` and the PKCE S256 verifier, and answers with an ID token.

Everything secret-looking is made at run time: the keys, the TLS certificate (self-signed for
``127.0.0.1`` and ``localhost``, written to ``ca_file`` for ``OIDC_CA_BUNDLE_FILE``), the client
secret and every code. Nothing is a literal.

It can be told to misbehave (``misbehave``, which replaces the frozen ``Behaviour``): a wrong
signature, ``alg: none``, HS256 signed with the public key, a rotated key, a missing or extra claim,
a wrong or missing nonce, an issuer shared by many organisations, metadata that names another
issuer or another host, a status or a redirect in place of an answer, an ``auth_time`` sent always,
only when the sign-in page asked for it (``claims`` naming it, or ``max_age``) or never, and a
``prompt`` value it answers with ``invalid_request``. It records each token request
(``token_requests``) so a test can count exchanges, and each sign-in page's query
(``authorize_requests``) so a test can read exactly what was sent.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import ssl
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

import httpx
import jwt
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

LOOPBACK = "127.0.0.1"
ISSUER_PATH = "/realms/prd277"
TOKEN_LIFETIME_SECONDS = 300


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _new_kid(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


def _self_signed(tmp_path: Path, dns_names: Tuple[str, ...] = ("localhost",)) -> Tuple[Path, Path]:
    """A certificate for 127.0.0.1 and ``dns_names`` (its own authority: the CA file) and its key."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "prd277 fake identity provider")])
    now = datetime.now(timezone.utc)
    names = x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(LOOPBACK)),
                                         *(x509.DNSName(dns) for dns in dns_names)])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(hours=2)).add_extension(names, critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_file, key_file = tmp_path / "fake-idp-cert.pem", tmp_path / "fake-idp-key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return cert_file, key_file


@dataclass(frozen=True)
class Person:
    """Who signs in at the provider next."""

    subject: str = field(default_factory=lambda: "subject-" + secrets.token_hex(6))
    email: Optional[str] = field(default_factory=lambda: f"person-{secrets.token_hex(4)}@example.test")
    email_verified: Any = True
    name: Optional[str] = "A Person"
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Behaviour:
    """How the provider behaves. Every field's default is a well-behaved provider."""

    alg: str = "RS256"
    signature: str = "valid"            # valid | wrong_key | none | hs256_public | tampered | no_kid
    nonce: str = "echo"                 # echo | wrong | missing
    remove_claims: Tuple[str, ...] = ()
    claims: Mapping[str, Any] = field(default_factory=dict)
    shared_issuer_domain: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata_status: int = 200
    keys_status: int = 200
    token_status: int = 200
    token_body: Optional[Mapping[str, Any]] = None
    redirect_metadata_to: Optional[str] = None
    callback_iss: Optional[str] = None
    callback_error: Optional[str] = None
    auth_time: str = "always"           # always | when_asked (claims or max_age) | never
    refused_prompts: Tuple[str, ...] = ()  # prompt values answered with error=invalid_request


@dataclass
class _Code:
    client_id: str
    redirect_uri: str
    challenge: str
    nonce: Optional[str]
    person: Person
    asked_auth_time: bool = False
    used: bool = False


AUTH_TIME_ASKED = {"id_token": {"auth_time": {"essential": True}}}


def _asks_auth_time(query: Mapping[str, str]) -> bool:
    """Whether a sign-in page's query asks for ``auth_time``: a ``max_age``, or ``claims`` naming it."""
    if "max_age" in query:
        return True
    try:
        asked = json.loads(query.get("claims", "null"))
    except ValueError:
        return False
    return isinstance(asked, dict) and "auth_time" in (asked.get("id_token") or {})


class FakeIdentityProvider:
    """An OIDC provider on loopback TLS; ``close()`` stops it."""

    def __init__(self, tmp_path: Path, client_id: str = "prd277-client",
                 dns_names: Tuple[str, ...] = ("localhost",)) -> None:
        self.client_id = client_id
        self.client_secret = secrets.token_urlsafe(24)
        self.behaviour = Behaviour()
        self.person = Person()
        self.token_requests: List[Dict[str, Any]] = []
        self.authorize_requests: List[Dict[str, str]] = []  # each sign-in page's query, as sent
        self.issued: List[str] = []          # every code, ID token and access token it handed out
        self._lock = threading.Lock()
        self._codes: Dict[str, _Code] = {}
        self._rsa, self._rsa_kid = _rsa_key(), _new_kid("rsa")
        self._ec, self._ec_kid = ec.generate_private_key(ec.SECP256R1()), _new_kid("ec")
        cert_file, key_file = _self_signed(tmp_path, dns_names)
        self.ca_file = cert_file
        self._server = ThreadingHTTPServer((LOOPBACK, 0), _handler_for(self))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(cert_file), str(key_file))
        self._server.socket = context.wrap_socket(self._server.socket, server_side=True)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ addresses

    @property
    def origin(self) -> str:
        return f"https://{LOOPBACK}:{self.port}"

    @property
    def issuer(self) -> str:
        return self.origin + ISSUER_PATH

    def address(self, path: str, host: str = LOOPBACK) -> str:
        return f"https://{host}:{self.port}{path}"

    # ------------------------------------------------------------------ control

    def misbehave(self, **changes: Any) -> None:
        """Replace the behaviour with ``changes`` applied to a well-behaved provider."""
        self.behaviour = replace(Behaviour(), **changes)

    def sign_in_as(self, person: Person) -> None:
        self.person = person

    def rotate_keys(self) -> None:
        """Publish a new RSA key under a new ``kid``, and drop the old one."""
        self._rsa, self._rsa_kid = _rsa_key(), _new_kid("rsa")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    # ------------------------------------------------------------------ documents

    def metadata(self) -> Dict[str, Any]:
        return {
            "issuer": self.issuer,
            "authorization_endpoint": self.address("/authorize"),
            "token_endpoint": self.address("/token"),
            "jwks_uri": self.address(ISSUER_PATH + "/keys"),
            "end_session_endpoint": self.address("/logout"),
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
            "id_token_signing_alg_values_supported": ["RS256", "PS256", "ES256"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
            **dict(self.behaviour.metadata),
        }

    def keys(self) -> Dict[str, Any]:
        rsa_jwk = RSAAlgorithm.to_jwk(self._rsa.public_key(), as_dict=True)
        ec_jwk = ECAlgorithm.to_jwk(self._ec.public_key(), as_dict=True)
        return {"keys": [{**rsa_jwk, "kid": self._rsa_kid, "use": "sig"},
                         {**ec_jwk, "kid": self._ec_kid, "use": "sig", "alg": "ES256"}]}

    # ------------------------------------------------------------------ the ID token

    def claims_for(self, code: _Code) -> Dict[str, Any]:
        now, person, behaviour = int(time.time()), code.person, self.behaviour
        claims: Dict[str, Any] = {
            "iss": self.issuer, "sub": person.subject, "aud": self.client_id, "iat": now,
            "exp": now + TOKEN_LIFETIME_SECONDS, "auth_time": now, "sid": "sid-" + secrets.token_hex(6),
            "email": person.email, "email_verified": person.email_verified, "name": person.name,
            **dict(person.extra),
        }
        if behaviour.nonce == "echo" and code.nonce is not None:
            claims["nonce"] = code.nonce
        elif behaviour.nonce == "wrong":
            claims["nonce"] = secrets.token_urlsafe(32)
        if behaviour.shared_issuer_domain:
            claims["hd"] = behaviour.shared_issuer_domain
        if not self._sends_auth_time(code):
            del claims["auth_time"]
        claims.update(behaviour.claims)
        return {name: value for name, value in claims.items()
                if name not in behaviour.remove_claims and value is not None}

    def _sends_auth_time(self, code: _Code) -> bool:
        """Whether this code's ID token carries ``auth_time``, as ``Behaviour.auth_time`` says."""
        mode = self.behaviour.auth_time
        return mode == "always" or (mode == "when_asked" and code.asked_auth_time)

    def id_token(self, claims: Mapping[str, Any]) -> str:
        """``claims`` signed as the behaviour says."""
        behaviour, alg = self.behaviour, self.behaviour.alg
        if behaviour.signature == "none":
            return _unsigned({"alg": "none", "typ": "JWT", "kid": self._rsa_kid}, claims)
        if behaviour.signature == "hs256_public":
            public = self._rsa.public_key().public_bytes(serialization.Encoding.PEM,
                                                         serialization.PublicFormat.SubjectPublicKeyInfo)
            return _hmac_signed({"alg": "HS256", "typ": "JWT", "kid": self._rsa_kid}, claims, public)
        key, kid = (self._ec, self._ec_kid) if alg == "ES256" else (self._rsa, self._rsa_kid)
        if behaviour.signature == "wrong_key":
            key = ec.generate_private_key(ec.SECP256R1()) if alg == "ES256" else _rsa_key()
        headers = {} if behaviour.signature == "no_kid" else {"kid": kid}
        token = jwt.encode(dict(claims), key, algorithm=alg, headers=headers)
        return _tampered(token) if behaviour.signature == "tampered" else token

    # ------------------------------------------------------------------ the browser's step

    def approve(self, authorize_address: str) -> str:
        """Follow the provider's sign-in page as a browser would; the callback address it sends back to."""
        with httpx.Client(verify=str(self.ca_file), follow_redirects=False, trust_env=False) as browser:
            answer = browser.get(authorize_address)
        assert answer.status_code == 302, answer.status_code
        return answer.headers["location"]

    def authorize(self, query: Mapping[str, List[str]]) -> Tuple[int, Dict[str, str]]:
        one = {name: values[0] for name, values in query.items() if values}
        self.authorize_requests.append(dict(one))
        if one.get("response_type") != "code" or one.get("code_challenge_method") != "S256":
            return 400, {}
        code = secrets.token_urlsafe(24)
        self.issued.append(code)
        with self._lock:
            self._codes[code] = _Code(one.get("client_id", ""), one.get("redirect_uri", ""),
                                      one.get("code_challenge", ""), one.get("nonce"), self.person,
                                      asked_auth_time=_asks_auth_time(one))
        answer = {"code": code, "state": one.get("state", ""), "iss": self.behaviour.callback_iss or self.issuer}
        error = self.behaviour.callback_error
        if one.get("prompt") in self.behaviour.refused_prompts:
            error = "invalid_request"
        if error:
            answer = {"error": error, "state": one.get("state", "")}
        return 302, {"Location": one.get("redirect_uri", "") + "?" + urlencode(answer)}

    # ------------------------------------------------------------------ the server's step

    def _client_is_authenticated(self, form: Mapping[str, str], authorization: Optional[str]) -> bool:
        if authorization and authorization.startswith("Basic "):
            pair = base64.b64decode(authorization[len("Basic "):]).decode("utf-8")
            client_id, _, secret = pair.partition(":")
            return unquote(client_id) == self.client_id and unquote(secret) == self.client_secret
        return form.get("client_id") == self.client_id and form.get("client_secret") == self.client_secret

    def _claim_code(self, form: Mapping[str, str]) -> Optional[_Code]:
        with self._lock:
            found = self._codes.get(form.get("code", ""))
            if found is None or found.used:
                return None
            found.used = True
        verifier = form.get("code_verifier", "")
        if _b64url(hashlib.sha256(verifier.encode("ascii")).digest()) != found.challenge:
            return None
        return found if form.get("redirect_uri") == found.redirect_uri else None

    def token(self, form: Mapping[str, str], authorization: Optional[str]) -> Tuple[int, Dict[str, Any]]:
        authenticated = self._client_is_authenticated(form, authorization)
        self.token_requests.append({"fields": sorted(form), "basic": bool(authorization),
                                    "authenticated": authenticated})
        if self.behaviour.token_status != 200:
            return self.behaviour.token_status, {"error": "temporarily_unavailable"}
        if not authenticated:
            return 401, {"error": "invalid_client"}
        code = self._claim_code(form)
        if code is None or form.get("grant_type") != "authorization_code" or code.client_id != self.client_id:
            return 400, {"error": "invalid_grant"}
        if self.behaviour.token_body is not None:
            return 200, dict(self.behaviour.token_body)
        id_token, access_token = self.id_token(self.claims_for(code)), secrets.token_urlsafe(24)
        self.issued.extend((id_token, access_token))
        return 200, {"id_token": id_token, "token_type": "Bearer", "access_token": access_token,
                     "expires_in": TOKEN_LIFETIME_SECONDS}


def _unsigned(header: Mapping[str, Any], claims: Mapping[str, Any]) -> str:
    encoded = [_b64url(json.dumps(dict(part)).encode("utf-8")) for part in (header, claims)]
    return ".".join(encoded) + "."


def _hmac_signed(header: Mapping[str, Any], claims: Mapping[str, Any], secret: bytes) -> str:
    signing_input = _unsigned(header, claims)[:-1]
    signature = hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256).digest()
    return signing_input + "." + _b64url(signature)


def _tampered(token: str) -> str:
    """``token`` with one claim changed after signing, so its signature no longer fits."""
    header, payload, signature = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["sub"] = "someone-else"
    return ".".join((header, _b64url(json.dumps(claims).encode("utf-8")), signature))


def _answer_get(provider: FakeIdentityProvider, address: str) -> Tuple[int, Optional[Mapping[str, Any]],
                                                                     Optional[Mapping[str, str]]]:
    """The provider's answer to a GET: its status, its JSON body and its headers."""
    parts, behaviour = urlsplit(address), provider.behaviour
    if parts.path == ISSUER_PATH + "/.well-known/openid-configuration":
        if behaviour.redirect_metadata_to:
            return 302, None, {"Location": behaviour.redirect_metadata_to}
        return behaviour.metadata_status, provider.metadata(), None
    if parts.path == ISSUER_PATH + "/keys":
        return behaviour.keys_status, provider.keys(), None
    if parts.path == "/authorize":
        status, headers = provider.authorize(parse_qs(parts.query))
        return status, None, headers
    return 404, None, None


def _handler_for(provider: FakeIdentityProvider):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # the provider's own access lines stay quiet
            return None

        def _send(self, status: int, body: Optional[Mapping[str, Any]] = None,
                  headers: Optional[Mapping[str, str]] = None) -> None:
            raw = json.dumps(dict(body or {})).encode("utf-8")
            self.send_response(status)
            for name, value in {"Content-Type": "application/json", **dict(headers or {})}.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802 - the http.server name
            status, body, headers = _answer_get(provider, self.path)
            return self._send(status, body, headers)

        def do_POST(self) -> None:  # noqa: N802 - the http.server name
            if urlsplit(self.path).path != "/token":
                return self._send(404)
            length = int(self.headers.get("Content-Length") or 0)
            form = {name: values[0] for name, values in parse_qs(self.rfile.read(length).decode("utf-8")).items()}
            status, body = provider.token(form, self.headers.get("Authorization"))
            return self._send(status, body)

    return Handler


INSTALL = "https://automatos.example.test"


def install_settings(provider: FakeIdentityProvider, **overrides: Any) -> Dict[str, Any]:
    """The settings of an enterprise install whose people sign in at ``provider``."""
    return {
        "AUTH_EDITION": "enterprise", "REQUIRE_AUTH": True, "FRONTEND_URL": INSTALL,
        "OIDC_ISSUER": provider.issuer, "OIDC_CLIENT_ID": provider.client_id,
        "OIDC_CLIENT_SECRET": provider.client_secret, "OIDC_CA_BUNDLE_FILE": str(provider.ca_file),
        "OIDC_SCOPES": "openid profile email", "OIDC_CLAIM_EMAIL": "email", "OIDC_CLAIM_NAME": "name",
        "ENTERPRISE_DEFAULT_ROLE": "none", "OIDC_ENDPOINT_HOSTS": "", "OIDC_PENDING_SIGN_INS_MAX": 10_000,
        "OIDC_PENDING_SIGN_INS_PER_ADDRESS": 100, "OIDC_REQUIRED_CLAIMS": "", "ENTERPRISE_BOOTSTRAP_ADMINS": "",
        **overrides,
    }


def point_install_at(monkeypatch: Any, provider: FakeIdentityProvider, **overrides: Any) -> None:
    """Set the live config to ``install_settings`` and forget anything held of another provider."""
    from config import config
    from core.auth import oidc_provider

    for name, value in install_settings(provider, **overrides).items():
        monkeypatch.setattr(config, name, value)
    oidc_provider.forget_provider()


__all__ = [
    "INSTALL",
    "ISSUER_PATH",
    "LOOPBACK",
    "Behaviour",
    "FakeIdentityProvider",
    "Person",
    "install_settings",
    "point_install_at",
]
