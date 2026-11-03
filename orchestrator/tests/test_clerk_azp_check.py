"""The Clerk ``azp`` (authorized party) check on session tokens (core/auth/clerk_azp.py).

Tokens are real RS256 JWTs signed with a key made here; the JWKS lookup is stubbed so
``ClerkAuth.verify_token`` runs its real decode. Covers:

* ``azp`` present and allowed: the claims come back.
* ``azp`` present and wrong: ``verify_token`` returns None, and the request
  dependency answers 401.
* ``azp`` missing: accepted with a warning by default; refused when
  ``CLERK_AZP_REQUIRED`` is true.
* The native marker (PRD-266 US-020): an azp-less token with ``client:
  "automatos-go"`` is accepted under either setting, without the warning; a token
  without it, or with another value, is still refused when ``CLERK_AZP_REQUIRED`` is
  true; a token that has an ``azp`` is judged by the allow-list, marker or not.
* The ``mobile`` JWT template (core/auth/clerk_mobile_template.py): every claim
  clerk.py and clerk_azp.py read is in the template or listed as left out, and a
  token rendered from the template resolves the same user and role as the web's
  session token.
* Config parsing: the shipped default, normalisation, ``FRONTEND_URL``, and the two
  environment variables read by a fresh copy of config.py.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from core.auth.clerk_mobile_template import (
    CLAIMS_NOT_IN_TEMPLATE,
    CLERK_TEMPLATE_DEFAULT_CLAIMS,
    MOBILE_JWT_TEMPLATE_CLAIMS,
    MOBILE_JWT_TEMPLATE_NAME,
    NATIVE_CLIENT_CLAIM,
    NATIVE_CLIENT_MARKER,
)

# Dummy POSTGRES_* satisfies the config import chain without a live DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

PRODUCTION_FRONTEND = "https://ui.automatos.app"
LOCAL_DEV = "http://localhost:3000"
STUDIO = "https://studio.automatos.app"
CONFIG_PY = Path(__file__).resolve().parents[1] / "config.py"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(private_key, **claims) -> str:
    now = int(time.time())
    body = {"sub": "user_azp_test", "sid": "sess_azp_test", "iat": now - 5, "nbf": now - 5, "exp": now + 300}
    body.update(claims)
    return jwt.encode(body, private_key, algorithm="RS256")


def _native_token(private_key, **claims) -> str:
    """A token shaped like the ``mobile`` template's: Clerk's own claims (no ``sid``,
    no ``azp``, as from a native request) plus the marker, then ``claims``."""
    now = int(time.time())
    body = {
        "sub": "user_azp_test", "iat": now - 5, "nbf": now - 5, "exp": now + 300,
        "iss": "https://clerk.automatos.app", "jti": "jti_azp_test",
        NATIVE_CLIENT_CLAIM: NATIVE_CLIENT_MARKER,
    }
    body.update(claims)
    return jwt.encode(body, private_key, algorithm="RS256")


class _StubJWKS:
    """Stands in for PyJWKClient: always returns the test key's public half."""

    def __init__(self, public_key):
        self._public_key = public_key

    def get_signing_key_from_jwt(self, _token):
        return SimpleNamespace(key=self._public_key)


@pytest.fixture
def clerk(signing_key):
    """A ClerkAuth whose JWKS is the test key, under the default allow-list."""
    import core.auth.clerk as clerk_mod
    import core.auth.clerk_azp as azp_mod
    import config as config_mod

    auth = clerk_mod.ClerkAuth()
    auth._jwks_client = _StubJWKS(signing_key.public_key())
    with patch.object(clerk_mod.config, "CLERK_AUDIENCE", ""), \
         patch.object(azp_mod.config, "CLERK_AUTHORIZED_PARTIES", config_mod.CLERK_DEFAULT_AUTHORIZED_PARTIES), \
         patch.object(azp_mod.config, "FRONTEND_URL", LOCAL_DEV), \
         patch.object(azp_mod.config, "CLERK_AZP_REQUIRED", False):
        yield auth


# ---------------------------------------------------------------------------
# verify_token: the three azp cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("azp", [PRODUCTION_FRONTEND, LOCAL_DEV])
def test_allowed_azp_is_accepted(clerk, signing_key, azp):
    claims = clerk.verify_token(_token(signing_key, azp=azp))
    assert claims is not None
    assert claims["azp"] == azp


@pytest.mark.parametrize("azp", [STUDIO, "https://evil.example", "https://ui.automatos.app.evil.example", "ui.automatos.app"])
def test_wrong_azp_is_refused(clerk, signing_key, azp, caplog):
    with caplog.at_level(logging.WARNING, logger="core.auth.clerk_azp"):
        assert clerk.verify_token(_token(signing_key, azp=azp)) is None
    assert "not in CLERK_AUTHORIZED_PARTIES" in caplog.text


def test_missing_azp_is_accepted_and_logged_by_default(clerk, signing_key, caplog):
    with caplog.at_level(logging.WARNING, logger="core.auth.clerk_azp"):
        claims = clerk.verify_token(_token(signing_key))
    assert claims is not None and "azp" not in claims
    assert "no azp claim; accepted" in caplog.text


def test_empty_azp_counts_as_missing(clerk, signing_key):
    assert clerk.verify_token(_token(signing_key, azp="")) is not None


def test_missing_azp_is_refused_when_required(clerk, signing_key):
    import core.auth.clerk_azp as azp_mod

    with patch.object(azp_mod.config, "CLERK_AZP_REQUIRED", True):
        assert clerk.verify_token(_token(signing_key)) is None
        # Required only changes the missing case: an allowed azp still passes.
        assert clerk.verify_token(_token(signing_key, azp=PRODUCTION_FRONTEND)) is not None


def test_allow_list_comes_from_config_at_request_time(clerk, signing_key):
    import core.auth.clerk_azp as azp_mod

    with patch.object(azp_mod.config, "CLERK_AUTHORIZED_PARTIES", STUDIO):
        assert clerk.verify_token(_token(signing_key, azp=STUDIO)) is not None
        assert clerk.verify_token(_token(signing_key, azp=PRODUCTION_FRONTEND)) is None


def test_frontend_url_origin_is_always_allowed(clerk, signing_key):
    import core.auth.clerk_azp as azp_mod

    with patch.object(azp_mod.config, "CLERK_AUTHORIZED_PARTIES", ""), \
         patch.object(azp_mod.config, "FRONTEND_URL", "https://app.self-hosted.example/"):
        assert clerk.verify_token(_token(signing_key, azp="https://app.self-hosted.example")) is not None
        assert clerk.verify_token(_token(signing_key, azp=PRODUCTION_FRONTEND)) is None


# ---------------------------------------------------------------------------
# The native marker (PRD-266 US-020)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("no_azp", [{}, {"azp": ""}], ids=["azp-absent", "azp-empty"])
def test_native_marker_without_azp_is_accepted(clerk, signing_key, required, no_azp, caplog):
    import core.auth.clerk_azp as azp_mod

    with patch.object(azp_mod.config, "CLERK_AZP_REQUIRED", required), \
         caplog.at_level(logging.DEBUG, logger="core.auth.clerk_azp"):
        claims = clerk.verify_token(_native_token(signing_key, **no_azp))
    assert claims is not None and claims.get("azp") in (None, "")
    assert claims[NATIVE_CLIENT_CLAIM] == NATIVE_CLIENT_MARKER
    assert "accepted as Automatos Go" in caplog.text
    # Phones must not raise the warning the runbook watches for unknown azp-less clients.
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_no_marker_and_no_azp_is_still_refused_when_required(clerk, signing_key, caplog):
    import core.auth.clerk_azp as azp_mod

    with patch.object(azp_mod.config, "CLERK_AZP_REQUIRED", True), \
         caplog.at_level(logging.WARNING, logger="core.auth.clerk_azp"):
        # A browser-shaped session token whose azp is missing.
        assert clerk.verify_token(_token(signing_key)) is None
        # The mobile template's claims without its marker are no better.
        assert clerk.verify_token(_native_token(signing_key, **{NATIVE_CLIENT_CLAIM: None})) is None
    assert "no azp claim (CLERK_AZP_REQUIRED=true)" in caplog.text


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("azp", [STUDIO, "https://evil.example", "ui.automatos.app"])
def test_native_marker_does_not_rescue_a_disallowed_azp(clerk, signing_key, azp, required, caplog):
    import core.auth.clerk_azp as azp_mod

    with patch.object(azp_mod.config, "CLERK_AZP_REQUIRED", required), \
         caplog.at_level(logging.WARNING, logger="core.auth.clerk_azp"):
        assert clerk.verify_token(_native_token(signing_key, azp=azp)) is None
    assert "not in CLERK_AUTHORIZED_PARTIES" in caplog.text


@pytest.mark.parametrize("azp", [PRODUCTION_FRONTEND, LOCAL_DEV])
def test_native_marker_with_an_allowed_azp_is_accepted(clerk, signing_key, azp):
    import core.auth.clerk_azp as azp_mod

    with patch.object(azp_mod.config, "CLERK_AZP_REQUIRED", True):
        claims = clerk.verify_token(_native_token(signing_key, azp=azp))
    assert claims is not None and claims["azp"] == azp


@pytest.mark.parametrize("marker", [
    "automatos-web", "Automatos-Go", "automatos-go ", " automatos-go", "", None,
    ["automatos-go"], {"automatos-go": True}, True, 1,
])
def test_wrong_marker_value_is_refused_when_required(clerk, signing_key, marker):
    import core.auth.clerk_azp as azp_mod

    with patch.object(azp_mod.config, "CLERK_AZP_REQUIRED", True):
        assert clerk.verify_token(_native_token(signing_key, **{NATIVE_CLIENT_CLAIM: marker})) is None


def test_wrong_marker_value_counts_as_no_marker_by_default(clerk, signing_key, caplog):
    with caplog.at_level(logging.WARNING, logger="core.auth.clerk_azp"):
        claims = clerk.verify_token(_native_token(signing_key, **{NATIVE_CLIENT_CLAIM: "automatos-web"}))
    assert claims is not None
    assert "no azp claim; accepted" in caplog.text
    assert "client='automatos-web'" in caplog.text


# ---------------------------------------------------------------------------
# The request dependency: a wrong azp is a 401
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wrong_azp_bearer_gets_401_from_request_context(clerk, signing_key):
    import core.auth.hybrid as hybrid

    request = MagicMock()
    request.method = "GET"
    request.headers = {"authorization": f"Bearer {_token(signing_key, azp=STUDIO)}"}
    request.query_params = {}
    request.state = MagicMock(spec=[])

    with patch.object(hybrid, "get_clerk_auth", return_value=clerk), \
         patch.object(hybrid, "SessionLocal", MagicMock(return_value=MagicMock())):
        with pytest.raises(HTTPException) as exc:
            await hybrid.get_request_context_hybrid(request)
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# The mobile JWT template: claim set and parity with a session token
# ---------------------------------------------------------------------------

# The names clerk.py and clerk_azp.py give a verified token's claims.
CLAIMS_VARIABLES = frozenset({"claims", "payload"})


def _claim_key(node: ast.AST, module) -> str:
    """A claim name from a string literal, or from a ``str`` constant in ``module``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and isinstance(getattr(module, node.id, None), str):
        return getattr(module, node.id)
    raise AssertionError(f"{module.__name__} reads a claim this test cannot name: {ast.unparse(node)}")


def _claims_read(module) -> set:
    """Every claim ``module`` reads as ``claims.get(key)`` or ``claims[key]`` (or ``payload``)."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id in CLAIMS_VARIABLES and node.args):
            keys.add(_claim_key(node.args[0], module))
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id in CLAIMS_VARIABLES):
            keys.add(_claim_key(node.slice, module))
    return keys


def test_mobile_template_accounts_for_every_claim_the_api_reads():
    import core.auth.clerk as clerk_mod
    import core.auth.clerk_azp as azp_mod

    read = _claims_read(clerk_mod) | _claims_read(azp_mod)
    # The scan has to see the claims everyone knows are read, or it proves nothing.
    assert {"email", "metadata", "sub", "azp", NATIVE_CLIENT_CLAIM} <= read
    pinned = set(MOBILE_JWT_TEMPLATE_CLAIMS) | set(CLAIMS_NOT_IN_TEMPLATE)
    unaccounted = read - CLERK_TEMPLATE_DEFAULT_CLAIMS - pinned
    assert not unaccounted, (
        f"core/auth/clerk.py or clerk_azp.py now reads {sorted(unaccounted)}. Add each claim to "
        "MOBILE_JWT_TEMPLATE_CLAIMS (and to the Clerk dashboard's mobile template), or to "
        "CLAIMS_NOT_IN_TEMPLATE with the reason, in core/auth/clerk_mobile_template.py."
    )
    stale = pinned - read
    assert not stale, f"nothing reads {sorted(stale)} any more: drop it from core/auth/clerk_mobile_template.py"


def test_mobile_template_is_well_formed():
    # The contract with Automatos Go: getToken({template: "mobile"}) and this marker.
    assert MOBILE_JWT_TEMPLATE_NAME == "mobile"
    assert MOBILE_JWT_TEMPLATE_CLAIMS[NATIVE_CLIENT_CLAIM] == NATIVE_CLIENT_MARKER == "automatos-go"
    # PRD-266 US-020: verification and role resolution need these.
    assert {"email", "metadata"} <= set(MOBILE_JWT_TEMPLATE_CLAIMS)
    # Clerk sets its default claims itself; a template entry for one would be ignored.
    assert not set(MOBILE_JWT_TEMPLATE_CLAIMS) & CLERK_TEMPLATE_DEFAULT_CLAIMS
    assert not set(MOBILE_JWT_TEMPLATE_CLAIMS) & set(CLAIMS_NOT_IN_TEMPLATE)
    for claim, value in MOBILE_JWT_TEMPLATE_CLAIMS.items():
        if claim != NATIVE_CLIENT_CLAIM:
            assert re.fullmatch(r"\{\{(user|org)\.[a-z_]+\}\}", value), (claim, value)
    with pytest.raises(TypeError):
        MOBILE_JWT_TEMPLATE_CLAIMS["email"] = "changed at runtime"


def _render_mobile_template(shortcode_values) -> dict:
    """The claims Clerk would put in a mobile-template token, given each shortcode's value."""
    rendered = {claim: shortcode_values.get(code, code) for claim, code in MOBILE_JWT_TEMPLATE_CLAIMS.items()}
    unresolved = sorted(claim for claim, value in rendered.items() if isinstance(value, str) and "{{" in value)
    assert not unresolved, f"the test gives no value for the shortcodes of {unresolved}"
    return rendered


@pytest.mark.parametrize("role, email_domain, expected_role", [
    ("admin", "automatos.app", "admin"),
    ("admin", "customer.example", "user"),
    (None, "customer.example", "user"),
])
def test_native_token_resolves_the_same_user_and_role_as_a_session_token(
    clerk, signing_key, role, email_domain, expected_role,
):
    import core.auth.clerk as clerk_mod
    import core.auth.clerk_azp as azp_mod

    metadata = {"role": role} if role else {}
    profile = {"email": f"pat@{email_domain}", "metadata": metadata, "name": "Pat Doe",
               "first_name": "Pat", "last_name": "Doe", "image_url": "https://img.example/pat.png"}
    # The web: the session token the runbook's template produces (§2 step 4).
    session_token = _token(signing_key, azp=PRODUCTION_FRONTEND, fva=[0, -1], **profile)
    # The phone: the mobile template, rendered for the same person, with no active org.
    phone_token = _native_token(signing_key, **_render_mobile_template({
        "{{user.primary_email_address}}": profile["email"],
        "{{user.public_metadata}}": metadata,
        "{{user.full_name}}": profile["name"],
        "{{user.first_name}}": profile["first_name"],
        "{{user.last_name}}": profile["last_name"],
        "{{user.image_url}}": profile["image_url"],
        "{{org.id}}": None,
        "{{org.slug}}": None,
    }))
    with patch.object(azp_mod.config, "CLERK_AZP_REQUIRED", True), \
         patch.object(clerk_mod.config, "PLATFORM_STAFF_EMAIL_DOMAIN", "automatos.app"):
        web = clerk.verify_token(session_token)
        phone = clerk.verify_token(phone_token)
        assert web is not None and phone is not None
        web_user, phone_user = clerk.extract_user_info(web), clerk.extract_user_info(phone)
    assert phone_user == web_user
    assert phone_user["system_role"] == expected_role
    assert phone_user["email"] == profile["email"]


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def test_shipped_default_allows_production_frontend_and_local_dev_only():
    import config as config_mod
    from core.auth.clerk_azp import parse_authorized_parties

    allowed = parse_authorized_parties(config_mod.CLERK_DEFAULT_AUTHORIZED_PARTIES)
    assert allowed == frozenset({PRODUCTION_FRONTEND, LOCAL_DEV})
    assert STUDIO not in allowed


def test_parse_normalises_entries_and_adds_frontend_url():
    from core.auth.clerk_azp import parse_authorized_parties

    allowed = parse_authorized_parties(" https://UI.Example.com/ , ,http://localhost:3000/path", "https://app.example.org/x")
    assert allowed == frozenset({"https://ui.example.com", "http://localhost:3000", "https://app.example.org"})


def test_parse_drops_entries_without_scheme_and_host(caplog):
    from core.auth.clerk_azp import parse_authorized_parties

    with caplog.at_level(logging.WARNING, logger="core.auth.clerk_azp"):
        allowed = parse_authorized_parties("ui.no-scheme.example,https://ok.example")
    assert allowed == frozenset({"https://ok.example"})
    assert "ui.no-scheme.example" in caplog.text


def test_parse_drops_an_entry_it_cannot_parse_and_keeps_the_rest(caplog):
    from core.auth.clerk_azp import parse_authorized_parties

    with caplog.at_level(logging.WARNING, logger="core.auth.clerk_azp"):
        allowed = parse_authorized_parties("http://[bad,https://ok.example")
    assert allowed == frozenset({"https://ok.example"})
    assert "http://[bad" in caplog.text


def _fresh_config_copy():
    """Execute config.py as a private module, so the class body re-reads the env
    without touching ``sys.modules['config']`` or the shared singleton."""
    spec = importlib.util.spec_from_file_location("_config_azp_probe", CONFIG_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def test_env_unset_uses_default_and_azp_not_required(monkeypatch):
    monkeypatch.delenv("CLERK_AUTHORIZED_PARTIES", raising=False)
    monkeypatch.delenv("CLERK_AZP_REQUIRED", raising=False)
    cfg = _fresh_config_copy()
    assert PRODUCTION_FRONTEND in cfg.CLERK_AUTHORIZED_PARTIES.split(",")
    assert cfg.CLERK_AZP_REQUIRED is False


@pytest.mark.parametrize("raw, expected", [("true", True), ("1", True), (" ON ", True), ("false", False), ("no", False)])
def test_env_overrides_are_read(monkeypatch, raw, expected):
    monkeypatch.setenv("CLERK_AUTHORIZED_PARTIES", "https://a.example,https://b.example")
    monkeypatch.setenv("CLERK_AZP_REQUIRED", raw)
    cfg = _fresh_config_copy()
    assert cfg.CLERK_AUTHORIZED_PARTIES == "https://a.example,https://b.example"
    assert cfg.CLERK_AZP_REQUIRED is expected
