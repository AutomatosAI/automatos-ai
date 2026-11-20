"""PRD-277 US-403 and US-405, pure: the access rule's language and the first-admin list.

* ``OIDC_REQUIRED_CLAIMS`` (``core.auth.access_rule``): ``name=value`` rules joined by ``;``, split at
  the first ``=``; a name is one top-level claim read literally, so ``cognito:groups`` and
  ``https://example.test/roles`` are nameable; a rule holds for an equal string or a list holding
  one, and every rule must hold (a null, or a value nested in a list, never does; spaces inside a
  value and non-ASCII are compared as written). A rule on a claim every token carries (``iss``,
  ``aud`` and the rest) is malformed. Anything else is malformed, and the refusal never quotes it.
* A shared issuer (Google, Microsoft's multi-tenant addresses) with no rule stops start-up, which
  says why; with a rule, or on an organisation's own issuer, it starts. A protocol-claim rule is
  never that rule. A default role with no rule stops start-up, naming both settings.
* ``OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES`` is ``true`` or ``false``, or unset.
* ``ENTERPRISE_BOOTSTRAP_ADMINS`` (``core.auth.bootstrap_admins``): ``sub:`` matches the subject
  exactly; ``email:`` matches only an ``email`` claim the token marks verified with ``true``; an
  entry that could name someone unintended (a pattern, an address without a dotted domain, no
  prefix) is malformed, and start-up refuses it naming the setting, never the value.
* ``local`` and ``saas`` are never stopped by either setting.

DB-free: ``check_start_up`` is handed the settings, as ``config.py`` hands them.
"""
from __future__ import annotations

import os
import secrets
from types import MappingProxyType

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core import edition_start_up  # noqa: E402
from core.auth import access_rule, bootstrap_admins  # noqa: E402
from core.auth.access_rule import MalformedRule, Rule  # noqa: E402
from core.auth.bootstrap_admins import EMAIL_KIND, SUBJECT_KIND, Entry, MalformedEntry  # noqa: E402
from core.edition_start_up import EditionNotReady  # noqa: E402
from core.editions import capabilities_for  # noqa: E402

OWN_ISSUER = "https://idp.example.test/realms/prd277-w4b"


def _settings(**overrides):
    """Every setting an enterprise install starts with, well configured, plus ``overrides``."""
    return {
        "OIDC_ISSUER": OWN_ISSUER, "OIDC_CLIENT_ID": "prd277-w4b-client",
        "OIDC_CLIENT_SECRET": secrets.token_urlsafe(24), "FRONTEND_URL": "https://automatos.example.test",
        "OIDC_SCOPES": "openid profile email", "OIDC_CLAIM_EMAIL": "email", "OIDC_CLAIM_NAME": "name",
        "OIDC_CA_BUNDLE_FILE": "", "ENTERPRISE_DEFAULT_ROLE": "none",
        "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000277e1", "CLIENT_IP_HEADER": "X-Real-IP",
        "OIDC_ENDPOINT_HOSTS": "", "OIDC_PENDING_SIGN_INS_MAX": 10_000, "OIDC_PENDING_SIGN_INS_PER_ADDRESS": 100,
        "OIDC_REQUIRED_CLAIMS": "", "ENTERPRISE_BOOTSTRAP_ADMINS": "",
        **overrides,
    }


def _refusal(**overrides) -> str:
    with pytest.raises(EditionNotReady) as refused:
        edition_start_up.check_start_up(capabilities_for("enterprise"), _settings(**overrides))
    return str(refused.value)


def _starts(**overrides) -> None:
    assert edition_start_up.check_start_up(capabilities_for("enterprise"), _settings(**overrides)) is None


# --------------------------------------------------------------------------- the rule language


def test_an_empty_setting_is_no_rule_and_lets_everyone_the_provider_let_in():
    assert access_rule.parse_rules("") == access_rule.parse_rules("   ") == access_rule.parse_rules(None) == ()
    assert access_rule.every_rule_holds((), {"sub": "anyone"}) is True


def test_rules_split_at_the_first_equals_and_drop_the_spaces_around_them():
    rules = access_rule.parse_rules(" hd = example.test ; groups=Automatos Users ;x=a=b,c:d.e")
    assert rules == (Rule("hd", "example.test"), Rule("groups", "Automatos Users"), Rule("x", "a=b,c:d.e"))


@pytest.mark.parametrize("name", ["cognito:groups", "https://example.test/roles", "realm_access.roles",
                                  "urn:example.test:claims:dept"])
def test_a_claim_name_with_a_colon_or_a_dot_is_one_top_level_claim(name):
    (rule,) = access_rule.parse_rules(f"{name}=automatos")
    assert rule.name == name
    assert access_rule.every_rule_holds((rule,), {name: "automatos"}) is True
    first, _, rest = name.partition(".") if "." in name else name.partition(":")
    assert access_rule.every_rule_holds((rule,), {first: {rest: "automatos"}}) is False  # never a path


@pytest.mark.parametrize("setting", [
    "hd", "hd=example.test;", ";hd=example.test", "hd=example.test;;groups=a", "=example.test", "hd=",
    "a b=c", "hd=exam\x01ple.test", "h\td=x", "x" * (access_rule.NAME_MAX + 1) + "=y",
    "x=" + "y" * (access_rule.VALUE_MAX + 1), ";".join(["a=b"] * (access_rule.RULES_MAX + 1)),
])
def test_anything_else_is_malformed_and_the_refusal_never_quotes_it(setting):
    with pytest.raises(MalformedRule) as malformed:
        access_rule.parse_rules(setting)
    assert setting.strip() not in str(malformed.value)


@pytest.mark.parametrize(("claims", "holds"), [
    ({"hd": "example.test"}, True),
    ({"hd": "Example.test"}, False),                 # exactly, case and all
    ({"hd": "example.test "}, False),
    ({"hd": ["other.test", "example.test"]}, True),  # a list that holds it
    ({"hd": ["other.test"]}, False),
    ({"hd": []}, False),
    ({}, False),                                     # missing
    ({"hd": None}, False),
    ({"hd": 1}, False),
    ({"hd": True}, False),
    ({"hd": {"example.test": True}}, False),
    ({"hd": [["example.test"]]}, False),
])
def test_a_rule_holds_for_an_equal_string_or_a_list_that_holds_one(claims, holds):
    rules = access_rule.parse_rules("hd=example.test")
    assert access_rule.every_rule_holds(rules, MappingProxyType(claims)) is holds


@pytest.mark.parametrize(("claims", "holds"), [
    ({"groups": None}, False),                          # a null claim is no value
    ({"groups": [None]}, False),
    ({"groups": [None, "automatos"]}, True),            # a null beside it in the list changes nothing
    ({"groups": [["automatos"]]}, False),               # never searched inside a nested list
    ({"groups": [["automatos"], "other"]}, False),
    ({"groups": ("automatos",)}, True),
])
def test_a_null_claim_and_a_nested_list_never_hold(claims, holds):
    rules = access_rule.parse_rules("groups=automatos")
    assert access_rule.every_rule_holds(rules, claims) is holds


def test_a_value_keeps_its_inner_spaces_and_is_compared_as_written():
    rules = access_rule.parse_rules("  dept =  Finance & Ops  ")
    assert rules == (Rule("dept", "Finance & Ops"),)
    assert access_rule.every_rule_holds(rules, {"dept": "Finance & Ops"}) is True
    for other in ("Finance  & Ops", " Finance & Ops", "Finance & Ops ", "Finance & Ops", "finance & ops"):
        assert access_rule.every_rule_holds(rules, {"dept": other}) is False


def test_a_non_ascii_name_and_value_match_exactly_with_no_folding_or_normalising():
    composed, decomposed = "équipe-rêve", "équipe-rêve"
    rules = access_rule.parse_rules(f"équipe={composed}")
    assert rules == (Rule("équipe", composed),)
    assert access_rule.every_rule_holds(rules, {"équipe": composed}) is True
    assert access_rule.every_rule_holds(rules, {"équipe": decomposed}) is False
    assert access_rule.every_rule_holds(rules, {"equipe": composed}) is False
    assert access_rule.every_rule_holds(rules, {"équipe": "EQUIPE-REVE"}) is False


@pytest.mark.parametrize("name", access_rule.PROTOCOL_CLAIMS)
def test_a_rule_on_a_claim_every_token_carries_is_malformed_and_stops_start_up(name):
    setting = f"{name}=prd277-w4b-value"
    with pytest.raises(MalformedRule):
        access_rule.parse_rules(setting)
    message = _refusal(OIDC_REQUIRED_CLAIMS=setting)
    assert message.startswith("AUTH_EDITION=enterprise cannot start: OIDC_REQUIRED_CLAIMS ")
    assert "prd277-w4b-value" not in message


def test_a_protocol_claim_rule_never_counts_as_the_rule_a_shared_issuer_needs():
    for setting in ("aud=prd277-w4b-client", "iss=https://accounts.google.com", "hd=example.test;azp=x"):
        message = _refusal(OIDC_ISSUER="https://accounts.google.com", OIDC_REQUIRED_CLAIMS=setting)
        assert message.startswith("AUTH_EDITION=enterprise cannot start: OIDC_REQUIRED_CLAIMS ")
    _starts(OIDC_ISSUER="https://accounts.google.com", OIDC_REQUIRED_CLAIMS="hd=example.test")


def test_every_rule_must_hold_and_a_name_given_twice_must_hold_both():
    rules = access_rule.parse_rules("tid=tenant-1;roles=Automatos.User;roles=Automatos.Admin")
    both = {"tid": "tenant-1", "roles": ["Automatos.User", "Automatos.Admin"]}
    assert access_rule.every_rule_holds(rules, both) is True
    assert access_rule.every_rule_holds(rules, {**both, "tid": "tenant-2"}) is False
    assert access_rule.every_rule_holds(rules, {**both, "roles": ["Automatos.User"]}) is False


# --------------------------------------------------------------------------- shared issuers


@pytest.mark.parametrize("issuer", [
    "https://accounts.google.com", "https://accounts.google.com/", " HTTPS://Accounts.Google.com ",
    "accounts.google.com",
    "https://login.microsoftonline.com/common/v2.0", "https://login.microsoftonline.com/organizations/v2.0",
    "https://login.microsoftonline.com/consumers/v2.0", "https://login.microsoftonline.com/common",
    "https://login.microsoftonline.com/9188040d-6c67-4c5b-b112-36a304b66dad/v2.0",
    "https://sts.windows.net/9188040d-6c67-4c5b-b112-36a304b66dad/",
])
def test_a_shared_issuer_with_no_rule_stops_start_up_and_says_why(issuer):
    assert access_rule.is_shared_issuer(issuer)
    message = _refusal(OIDC_ISSUER=issuer)

    assert message.startswith("AUTH_EDITION=enterprise cannot start: OIDC_REQUIRED_CLAIMS ")
    assert edition_start_up.SHARED_ISSUER_PROBLEM in message and "anyone" in message
    _starts(OIDC_ISSUER=issuer, OIDC_REQUIRED_CLAIMS="hd=example.test")


@pytest.mark.parametrize("issuer", [
    OWN_ISSUER, "https://login.microsoftonline.com/00000000-0000-0000-0000-0000000277e2/v2.0",
    "https://example.okta.test/oauth2/default", "https://accounts.google.com.example.test",
])
def test_an_organisations_own_issuer_starts_with_no_rule(issuer):
    assert not access_rule.is_shared_issuer(issuer)
    _starts(OIDC_ISSUER=issuer)


@pytest.mark.parametrize("setting", ["hd", "groups=", "a b=c", "hd=x;"])
def test_a_malformed_rule_stops_start_up_naming_the_setting_not_the_value(setting):
    message = _refusal(OIDC_REQUIRED_CLAIMS=setting)

    assert message.startswith("AUTH_EDITION=enterprise cannot start: OIDC_REQUIRED_CLAIMS ")
    assert setting not in message.split("OIDC_REQUIRED_CLAIMS", 1)[1]


# --------------------------------------------------------------------------- the first admins


def test_entries_are_subjects_and_addresses_in_order():
    entries = bootstrap_admins.parse_entries(" sub:248289761001 , email:First.Admin@Example.Test,sub:a|b ")
    assert entries == (Entry(SUBJECT_KIND, "248289761001"), Entry(EMAIL_KIND, "First.Admin@Example.Test"),
                       Entry(SUBJECT_KIND, "a|b"))
    assert bootstrap_admins.parse_entries("") == bootstrap_admins.parse_entries(None) == ()


@pytest.mark.parametrize("setting", [
    "someone@example.test", "SUB:abc", "Email:a@example.test", "user:abc", "sub:", "email:", "sub:abc,,",
    ",sub:abc", "email:*@example.test", "sub:abc*", "email:a?@example.test", "email:nobody@localhost",
    "email:a@@example.test", "email:a@b@example.test", "email:@example.test", "email:a@.example.test",
    "email:a@example.test.", "email:a b@example.test", "sub:a\x00b",
    "sub:" + "x" * bootstrap_admins.ENTRY_MAX, ",".join(["sub:a"] * (bootstrap_admins.ENTRIES_MAX + 1)),
])
def test_an_entry_that_could_name_someone_unintended_is_malformed(setting):
    with pytest.raises(MalformedEntry) as malformed:
        bootstrap_admins.parse_entries(setting)
    assert setting.strip() not in str(malformed.value)
    message = _refusal(ENTERPRISE_BOOTSTRAP_ADMINS=setting)
    assert message.startswith("AUTH_EDITION=enterprise cannot start: ENTERPRISE_BOOTSTRAP_ADMINS ")
    assert setting.strip() not in message.split("ENTERPRISE_BOOTSTRAP_ADMINS", 1)[1]


def test_a_subject_entry_matches_that_subject_exactly():
    entries = bootstrap_admins.parse_entries("sub:Subject-1")
    assert bootstrap_admins.names_person(entries, "Subject-1", {}) == SUBJECT_KIND
    for other in ("subject-1", "Subject-1 ", "Subject-12", ""):
        assert bootstrap_admins.names_person(entries, other, {}) is None


@pytest.mark.parametrize(("claims", "kind"), [
    ({"email": "First.Admin@example.test", "email_verified": True}, EMAIL_KIND),
    ({"email": "First.Admin@EXAMPLE.TEST", "email_verified": True}, EMAIL_KIND),   # the domain's case
    ({"email": "first.admin@example.test", "email_verified": True}, None),         # the local part's
    ({"email": "First.Admin@example.test"}, None),                                 # not marked verified
    ({"email": "First.Admin@example.test", "email_verified": False}, None),
    ({"email": "First.Admin@example.test", "email_verified": "true"}, None),
    ({"email": "First.Admin@example.test", "email_verified": 1}, None),
    ({"email": "First.Admin@example.test.evil.test", "email_verified": True}, None),
    ({"email": ["First.Admin@example.test"], "email_verified": True}, None),
    ({"mail": "First.Admin@example.test", "email_verified": True}, None),          # only the email claim
])
def test_an_email_entry_matches_only_an_email_the_token_marks_verified(claims, kind):
    entries = bootstrap_admins.parse_entries("email:First.Admin@Example.Test")
    assert bootstrap_admins.names_person(entries, "someone", claims) == kind


@pytest.mark.parametrize("role", ["viewer", "editor", "admin"])
def test_a_default_role_with_no_access_rule_stops_start_up_naming_both_settings(role):
    message = _refusal(ENTERPRISE_DEFAULT_ROLE=role)

    assert message.startswith("AUTH_EDITION=enterprise cannot start: ENTERPRISE_DEFAULT_ROLE ")
    assert edition_start_up.DEFAULT_ROLE_NEEDS_RULE_PROBLEM in message and "OIDC_REQUIRED_CLAIMS" in message
    _starts(ENTERPRISE_DEFAULT_ROLE=role, OIDC_REQUIRED_CLAIMS="groups=automatos-users")
    _starts(ENTERPRISE_DEFAULT_ROLE="none")


@pytest.mark.parametrize("value", [None, "", "true", "false", "TRUE", " false "])
def test_the_email_claim_setting_is_true_or_false(value):
    _starts(OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES=value)


@pytest.mark.parametrize("value", ["yes", "1", "maybe", "truthy", "enabled"])
def test_any_other_email_claim_setting_stops_start_up_never_quoting_it(value):
    message = _refusal(OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES=value)
    name = "OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES"
    assert message.startswith(f"AUTH_EDITION=enterprise cannot start: {name} ")
    assert value not in message.split(name, 1)[1]


def test_good_entries_and_rules_start_enterprise():
    _starts(OIDC_REQUIRED_CLAIMS="cognito:groups=automatos", ENTERPRISE_BOOTSTRAP_ADMINS="sub:1, email:a@example.test")


# --------------------------------------------------------------------------- the default workspace


@pytest.mark.parametrize("value", [None, "", "   ", "default", "00000000-0000-0000-0000"])
def test_enterprise_needs_its_default_workspace_as_a_uuid(value):
    message = _refusal(DEFAULT_WORKSPACE_ID=value)
    assert message.startswith("AUTH_EDITION=enterprise cannot start: DEFAULT_WORKSPACE_ID ")
    assert edition_start_up.DEFAULT_WORKSPACE_PROBLEM in message


# --------------------------------------------------------------------------- local and saas


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_neither_setting_ever_stops_local_or_saas(edition):
    for overrides in ({"OIDC_REQUIRED_CLAIMS": "hd"}, {"ENTERPRISE_BOOTSTRAP_ADMINS": "email:*@x"},
                      {"OIDC_ISSUER": "https://accounts.google.com"}, {"DEFAULT_WORKSPACE_ID": None}):
        assert edition_start_up.check_start_up(capabilities_for(edition), _settings(**overrides)) is None
