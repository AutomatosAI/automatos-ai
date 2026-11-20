"""PRD-277 US-401 and §5.11: the provider's metadata and keys, and every rule an ID token must meet.

Against the in-process fake provider (``tests/helpers_fake_oidc_provider.py``, US-407), over TLS
on loopback, trusted through ``OIDC_CA_BUNDLE_FILE``:

* **Metadata** is read from ``<issuer>/.well-known/openid-configuration`` and used only when its
  issuer is ``OIDC_ISSUER`` exactly, its endpoints are https, and it offers the code flow, PKCE
  S256, an allowed algorithm and a secret-based client authentication. Without the CA bundle, or
  answered with a redirect or an error, it is not read at all.
* **Keys**: a token's ``kid`` picks its key; an unknown ``kid`` (the provider rotated its keys)
  reads the keys once more, and a second unknown ``kid`` within a minute reads nothing.
* **The ID token** (``oidc_tokens.verify_id_token``): each rule of §5.11 has a test that fails
  without it: the signature (wrong key, tampered, ``alg: none``, HS256 with the public key, no
  ``kid``), the issuer, the audience and ``azp``, the three times with 60 seconds of leeway, the
  nonce, and the required claims. RS256, PS256 and ES256 are accepted.
"""
from __future__ import annotations

import base64
import json
import os
import time

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import oidc_http, oidc_provider, oidc_tokens  # noqa: E402
from core.auth.oidc_provider import ProviderMisconfigured, parse_metadata  # noqa: E402
from core.auth.oidc_tokens import InvalidIdToken, verify_id_token  # noqa: E402
from core.auth.pairing import hash_secret  # noqa: E402
from tests.helpers_fake_oidc_provider import FakeIdentityProvider, point_install_at  # noqa: E402

NONCE = "prd277-w4-nonce-" + "0" * 27


@pytest.fixture
def idp(tmp_path, monkeypatch):
    provider = FakeIdentityProvider(tmp_path)
    point_install_at(monkeypatch, provider)
    yield provider
    provider.close()
    oidc_provider.forget_provider()


@pytest.fixture
def fetches(monkeypatch):
    """Every address sign-in fetched, in order."""
    seen = []
    real = oidc_http.get_json

    def counting(url, purpose):
        seen.append(purpose)
        return real(url, purpose)

    monkeypatch.setattr(oidc_http, "get_json", counting)
    return seen


def _claims(idp, **changes):
    now = int(time.time())
    base = {"iss": idp.issuer, "sub": "subject-1", "aud": idp.client_id, "iat": now, "exp": now + 300,
            "nonce": NONCE, "email": "person@example.test", "email_verified": True}
    return {name: value for name, value in {**base, **changes}.items() if value is not None}


def _verify(token):
    return verify_id_token(token, nonce_hash=hash_secret(NONCE))


def _refused(token) -> str:
    with pytest.raises(InvalidIdToken) as refused:
        _verify(token)
    return refused.value.rule


# --------------------------------------------------------------------------- metadata


def test_the_metadata_is_read_from_the_issuers_well_known_address(idp, fetches):
    metadata = oidc_provider.current_metadata()

    assert oidc_provider.metadata_address(idp.issuer) == idp.issuer + "/.well-known/openid-configuration"
    assert metadata.issuer == idp.issuer and metadata.token_endpoint == idp.address("/token")
    assert metadata.token_auth_method == oidc_provider.AUTH_BASIC
    oidc_provider.current_metadata()
    assert fetches == ["metadata"]                                   # held, not read again


def test_without_the_ca_bundle_the_private_certificate_is_not_trusted(idp, monkeypatch):
    monkeypatch.setattr(config, "OIDC_CA_BUNDLE_FILE", "")

    with pytest.raises(oidc_http.ProviderUnreachable):
        oidc_provider.current_metadata()


@pytest.mark.parametrize("behaviour", [{"metadata_status": 500}, {"redirect_metadata_to": "https://127.0.0.1:1/x"}])
def test_an_error_or_a_redirect_in_place_of_metadata_is_not_followed(idp, behaviour):
    idp.misbehave(**behaviour)

    with pytest.raises(oidc_http.ProviderUnreachable):
        oidc_provider.current_metadata()


def test_metadata_naming_another_issuer_is_refused(idp):
    idp.misbehave(metadata={"issuer": idp.issuer + "/other"})

    with pytest.raises(ProviderMisconfigured):
        oidc_provider.current_metadata()


def _document(**changes):
    base = {"issuer": "https://idp.example.test", "authorization_endpoint": "https://idp.example.test/a",
            "token_endpoint": "https://idp.example.test/t", "jwks_uri": "https://idp.example.test/k"}
    return {name: value for name, value in {**base, **changes}.items() if value is not None}


@pytest.mark.parametrize("changes", [
    {"issuer": "https://idp.example.test/"}, {"issuer": None}, {"token_endpoint": "http://idp.example.test/t"},
    {"jwks_uri": None}, {"authorization_endpoint": "javascript:alert(1)"},
    {"end_session_endpoint": "http://idp.example.test/logout"},
    {"response_types_supported": ["id_token"]}, {"code_challenge_methods_supported": ["plain"]},
    {"id_token_signing_alg_values_supported": ["HS256", "none"]},
    {"token_endpoint_auth_methods_supported": ["private_key_jwt"]}, {"response_types_supported": "code"},
])
def test_metadata_sign_in_cannot_use_is_refused(changes):
    with pytest.raises(ProviderMisconfigured):
        parse_metadata(_document(**changes), "https://idp.example.test")


def test_metadata_that_lists_nothing_offers_the_standard_defaults():
    metadata = parse_metadata(_document(), "https://idp.example.test")
    assert metadata.token_auth_method == oidc_provider.AUTH_BASIC and metadata.end_session_endpoint is None
    posted = parse_metadata(_document(token_endpoint_auth_methods_supported=["client_secret_post"]),
                            "https://idp.example.test")
    assert posted.token_auth_method == oidc_provider.AUTH_POST


def test_a_token_endpoint_on_another_private_host_is_never_called(idp):
    idp.misbehave(metadata={"token_endpoint": idp.address("/token", host="localhost")})
    metadata = oidc_provider.current_metadata()

    with pytest.raises(oidc_http.ProviderUnreachable):
        oidc_http.call("POST", metadata.token_endpoint, "token", data={"grant_type": "authorization_code"})
    assert idp.token_requests == []


# --------------------------------------------------------------------------- keys


def test_a_rotated_key_is_found_with_one_more_read_of_the_keys(idp, fetches):
    assert _verify(idp.id_token(_claims(idp)))["sub"] == "subject-1"
    idp.rotate_keys()

    assert _verify(idp.id_token(_claims(idp)))["sub"] == "subject-1"
    assert fetches == ["metadata", "keys", "keys"]


def test_a_second_unknown_key_within_the_minute_reads_nothing(idp, fetches):
    _verify(idp.id_token(_claims(idp)))
    idp.rotate_keys()
    _verify(idp.id_token(_claims(idp)))
    idp.rotate_keys()

    assert "no published key" in _refused(idp.id_token(_claims(idp)))
    assert fetches == ["metadata", "keys", "keys"]


def test_the_provider_check_reads_the_metadata_and_the_keys(idp, fetches):
    oidc_provider.check_provider()
    assert fetches == ["metadata", "keys"]


# --------------------------------------------------------------------------- the ID token (§5.11)


@pytest.mark.parametrize("alg", ["RS256", "PS256", "ES256"])
def test_each_allowed_algorithm_is_accepted(idp, alg):
    idp.misbehave(alg=alg)
    claims = _verify(idp.id_token(_claims(idp)))
    assert claims["sub"] == "subject-1"
    with pytest.raises(TypeError):
        claims["sub"] = "changed"                                    # read-only


@pytest.mark.parametrize("signature", ["wrong_key", "tampered", "none", "hs256_public", "no_kid"])
def test_a_token_not_signed_by_a_published_key_is_refused(idp, signature):
    idp.misbehave(signature=signature)
    assert _refused(idp.id_token(_claims(idp)))


def test_none_and_hmac_are_refused_before_any_key_is_read(idp, fetches):
    for signature in ("none", "hs256_public"):
        idp.misbehave(signature=signature)
        assert _refused(idp.id_token(_claims(idp))) == "its algorithm is not allowed"
    assert fetches == []


def test_an_es256_token_naming_the_rsa_key_is_refused(idp):
    idp.misbehave(alg="ES256")
    token = idp.id_token(_claims(idp))
    rsa_kid = idp.keys()["keys"][0]["kid"]
    _header, rest = token.split(".", 1)
    forged = base64.urlsafe_b64encode(json.dumps({"alg": "ES256", "kid": rsa_kid}).encode()).decode().rstrip("=")
    assert _refused(forged + "." + rest)


@pytest.mark.parametrize("changes", [
    {"iss": "https://elsewhere.example.test/realms/prd277"}, {"aud": "another-client"},
    {"aud": ["another-client"]},
])
def test_the_issuer_and_audience_must_be_exact(idp, changes):
    assert _refused(idp.id_token(_claims(idp, **changes)))


def test_an_issuer_differing_only_by_a_trailing_slash_is_refused(idp):
    assert _refused(idp.id_token(_claims(idp, iss=idp.issuer + "/")))


def test_several_audiences_need_azp_and_azp_must_be_this_client(idp):
    both = [idp.client_id, "another-client"]
    assert "azp" in _refused(idp.id_token(_claims(idp, aud=both)))
    assert "azp" in _refused(idp.id_token(_claims(idp, aud=both, azp="another-client")))
    assert "azp" in _refused(idp.id_token(_claims(idp, azp="another-client")))
    assert _verify(idp.id_token(_claims(idp, aud=both, azp=idp.client_id)))["sub"] == "subject-1"


@pytest.mark.parametrize("changes", [
    {"exp": -61}, {"iat": 61}, {"nbf": 61},
])
def test_times_past_the_leeway_are_refused(idp, changes):
    now = int(time.time())
    shifted = {name: now + offset for name, offset in changes.items()}
    assert _refused(idp.id_token(_claims(idp, **shifted)))


def test_times_within_the_leeway_are_accepted(idp):
    now = int(time.time())
    assert oidc_tokens.TIME_LEEWAY_SECONDS == 60
    token = idp.id_token(_claims(idp, exp=now - 30, iat=now + 30, nbf=now + 30))
    assert _verify(token)["sub"] == "subject-1"


@pytest.mark.parametrize("nonce", [None, "another-nonce", ""])
def test_the_nonce_must_be_this_sign_ins(idp, nonce):
    assert "nonce" in _refused(idp.id_token(_claims(idp, nonce=nonce)))


@pytest.mark.parametrize("claim", ["sub", "exp", "iat", "aud", "iss"])
def test_a_required_claim_that_is_missing_is_refused(idp, claim):
    claims = {name: value for name, value in _claims(idp).items() if name != claim}
    assert _refused(idp.id_token(claims))


@pytest.mark.parametrize("token", ["", "not-a-token", "a.b", "x" * (oidc_tokens.MAX_ID_TOKEN_LENGTH + 1), None, 7])
def test_a_token_that_is_not_one_is_refused_before_anything_is_read(idp, fetches, token):
    assert _refused(token)
    assert fetches == []


def test_a_refusal_names_the_rule_and_never_a_value(idp):
    rule = _refused(idp.id_token(_claims(idp, aud="prd277-audience-marker")))
    assert "marker" not in rule and "subject-1" not in rule
