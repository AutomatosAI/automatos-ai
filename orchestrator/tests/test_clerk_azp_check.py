"""The Clerk ``azp`` (authorized party) check on session tokens (core/auth/clerk_azp.py).

Tokens are real RS256 JWTs signed with a key made here; the JWKS lookup is stubbed so
``ClerkAuth.verify_token`` runs its real decode. Covers:

* ``azp`` present and allowed: the claims come back.
* ``azp`` present and wrong: ``verify_token`` returns None, and the request
  dependency answers 401.
* ``azp`` missing: accepted with a warning by default; refused when
  ``CLERK_AZP_REQUIRED`` is true.
* Config parsing: the shipped default, normalisation, ``FRONTEND_URL``, and the two
  environment variables read by a fresh copy of config.py.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

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
