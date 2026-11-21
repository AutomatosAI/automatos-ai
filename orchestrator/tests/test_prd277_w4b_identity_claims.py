"""PRD-277 US-404 and US-406, pure: what ``oidc_sign_in.verified_identity`` reads from verified claims.

* **The email is verified** only when the claim read is the standard ``email`` and
  ``email_verified`` is the JSON ``true``. From any other claim it is unverified, unless
  ``OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES`` is ``true``.
* **``auth_time``**: absent is the epoch (``NO_AUTH_TIME``); a value that is not a finite number of
  seconds (a string, a boolean, NaN, an infinity), one before the epoch, one later than now by more
  than the leeway, or one too large for a date is ``InvalidSignIn``: a refused sign-in, never an
  exception of another kind.
* **The subject** is the claim ``OIDC_CLAIM_SUBJECT`` names (``sub`` unless set). Set to another claim,
  ``sub`` no longer says who the person is, and the first-admin list's ``sub:`` entries match the
  claim's value. A token without a string there that is not blank and holds no control character is
  ``InvalidSignIn``, whose text holds no value. The token's own ``sub`` is carried beside it
  (``standard_sub``) only when the subject is read from another claim, and only when a link could hold it.

DB-free: the claims are handed in, as ``oidc_tokens.verify_id_token`` hands them over.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import bootstrap_admins  # noqa: E402
from core.auth.oidc_sign_in import NO_AUTH_TIME, verified_identity  # noqa: E402
from core.auth.oidc_tokens import TIME_LEEWAY_SECONDS  # noqa: E402
from core.auth.sign_in_methods import InvalidSignIn  # noqa: E402
from core.models.user_identities import SUBJECT_MAX  # noqa: E402

NOW = 1_795_000_000.0
BASE = {"iss": "https://idp.example.test/realms/prd277-w4b", "sub": "subject-1"}


@pytest.fixture
def settings(monkeypatch):
    """The email settings as an install starts with them; each test changes what it needs."""
    monkeypatch.setattr(config, "OIDC_CLAIM_EMAIL", "email")
    monkeypatch.setattr(config, "OIDC_CLAIM_NAME", "name")
    monkeypatch.setattr(config, "OIDC_CLAIM_SUBJECT", "sub")
    monkeypatch.setattr(config, "OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES", "false")
    return monkeypatch


# --------------------------------------------------------------------------- the email


@pytest.mark.parametrize(("verified", "expected"), [(True, True), (False, False), ("true", False), (1, False),
                                                    (None, False)])
def test_the_standard_email_claim_is_verified_only_by_email_verified_true(settings, verified, expected):
    claims = {**BASE, "email": "a@example.test", **({} if verified is None else {"email_verified": verified})}
    identity = verified_identity(claims, now=NOW)
    assert (identity.email, identity.email_verified) == ("a@example.test", expected)


def test_another_claim_is_unverified_even_beside_email_verified_true(settings):
    settings.setattr(config, "OIDC_CLAIM_EMAIL", "mail")
    identity = verified_identity({**BASE, "mail": "a@example.test", "email_verified": True}, now=NOW)
    assert (identity.email, identity.email_verified) == ("a@example.test", False)


def test_another_claim_is_verified_when_the_operator_says_the_provider_verifies_it(settings):
    settings.setattr(config, "OIDC_CLAIM_EMAIL", "mail")
    settings.setattr(config, "OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES", "true")
    identity = verified_identity({**BASE, "mail": "a@example.test"}, now=NOW)
    assert (identity.email, identity.email_verified) == ("a@example.test", True)


def test_the_setting_never_vouches_for_the_standard_claim(settings):
    settings.setattr(config, "OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES", "true")
    identity = verified_identity({**BASE, "email": "a@example.test", "email_verified": False}, now=NOW)
    assert identity.email_verified is False


# --------------------------------------------------------------------------- auth_time


def test_no_auth_time_is_the_epoch(settings):
    assert verified_identity(dict(BASE), now=NOW).auth_time == NO_AUTH_TIME


@pytest.mark.parametrize("value", [NOW, NOW - 3600, NOW + TIME_LEEWAY_SECONDS, int(NOW), 0])
def test_an_auth_time_in_seconds_up_to_the_leeway_ahead_is_read(settings, value):
    identity = verified_identity({**BASE, "auth_time": value}, now=NOW)
    assert identity.auth_time == datetime.fromtimestamp(value, tz=timezone.utc)


@pytest.mark.parametrize("value", [
    "1795000000", "yesterday", True, False, [NOW], {"t": NOW},
    float("nan"), float("inf"), float("-inf"),
    -1, -1e300,
    NOW + TIME_LEEWAY_SECONDS + 1, 10 ** 30, 1e300,
])
def test_an_auth_time_that_is_not_a_real_moment_refuses_the_sign_in(settings, value):
    with pytest.raises(InvalidSignIn) as refused:
        verified_identity({**BASE, "auth_time": value}, now=NOW)
    assert "auth_time" in str(refused.value) and repr(value) not in str(refused.value)


def test_an_auth_time_too_large_for_a_date_is_refused_even_on_a_clock_that_allows_it(settings):
    with pytest.raises(InvalidSignIn):
        verified_identity({**BASE, "auth_time": 1e18}, now=1e19)


# --------------------------------------------------------------------------- the subject


def test_the_subject_is_sub_by_default(settings):
    identity = verified_identity({**BASE, "oid": "object-1"}, now=NOW)
    assert (identity.issuer, identity.subject) == (BASE["iss"], "subject-1")


def test_the_subject_is_the_claim_oidc_claim_subject_names(settings):
    settings.setattr(config, "OIDC_CLAIM_SUBJECT", "oid")
    identity = verified_identity({**BASE, "oid": "object-1"}, now=NOW)
    assert (identity.issuer, identity.subject) == (BASE["iss"], "object-1")


def test_a_first_admin_sub_entry_names_the_subject_claims_value(settings):
    settings.setattr(config, "OIDC_CLAIM_SUBJECT", "oid")
    identity = verified_identity({**BASE, "oid": "object-1"}, now=NOW)
    entries = bootstrap_admins.parse_entries("sub:object-1")
    assert bootstrap_admins.names_person(entries, identity.subject, identity.claims) == bootstrap_admins.SUBJECT_KIND
    assert bootstrap_admins.names_person(bootstrap_admins.parse_entries("sub:subject-1"), identity.subject,
                                         identity.claims) is None


@pytest.mark.parametrize("value", [None, "", " ", "\t", "object-1\n", "obj\x00ect-1", "object\x7f-1", 42, True,
                                   ["object-1"], {"id": "object-1"}])
def test_a_token_without_a_usable_subject_claim_refuses_the_sign_in(settings, value):
    settings.setattr(config, "OIDC_CLAIM_SUBJECT", "oid")
    claims = {**BASE, **({} if value is None else {"oid": value})}
    with pytest.raises(InvalidSignIn) as refused:
        verified_identity(claims, now=NOW)
    assert "subject" in str(refused.value) and "object-1" not in str(refused.value)


def test_the_standard_sub_is_carried_only_when_the_subject_is_another_claim(settings):
    assert verified_identity({**BASE, "oid": "object-1"}, now=NOW).standard_sub is None
    settings.setattr(config, "OIDC_CLAIM_SUBJECT", "oid")
    identity = verified_identity({**BASE, "oid": "object-1"}, now=NOW)
    assert (identity.subject, identity.standard_sub) == ("object-1", "subject-1")


def test_a_sub_no_link_could_hold_is_not_carried(settings):
    settings.setattr(config, "OIDC_CLAIM_SUBJECT", "oid")
    identity = verified_identity({**BASE, "sub": "s" * (SUBJECT_MAX + 1), "oid": "object-1"}, now=NOW)
    assert (identity.subject, identity.standard_sub) == ("object-1", None)
