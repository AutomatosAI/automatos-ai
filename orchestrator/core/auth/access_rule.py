"""PRD-277 US-403 (§5.4, FR-8): the access rule, ``OIDC_REQUIRED_CLAIMS``.

The organisation's identity provider is the first gate: only people assigned to the application
there can finish signing in. The access rule is the second: claims of the VERIFIED ID token that
must hold given values, checked before anyone is created, linked or given a session
(``core.auth.sign_in_flow``).

**The language**, and nothing richer::

    OIDC_REQUIRED_CLAIMS = rule *( ";" rule )
    rule                 = name "=" value

* A rule is split at its FIRST ``=``: the name is what comes before it, the value everything after,
  so a value may itself hold ``=``, ``,``, ``:`` or ``.``. Spaces around a name and a value are
  dropped; a value may hold spaces inside it.
* A name is one top-level claim, read exactly as written: ``cognito:groups`` and
  ``https://example.test/roles`` are each one claim's name, never a path into nested claims. A name
  is 1 to ``NAME_MAX`` characters with no space, control character, ``=`` or ``;``.
* A value is 1 to ``VALUE_MAX`` characters with no control character and no ``;``.
* A name is never one of ``PROTOCOL_CLAIMS``: the claims every valid ID token for this install
  carries or that describe the token itself (``iss``, ``aud``, ``nonce`` and the rest), which say
  nothing about who the person is. A rule on one would hold for everyone, so it is malformed, and
  never counts as the rule a shared issuer needs.
* At most ``RULES_MAX`` rules. An empty setting is no rule. Anything else (an empty rule, as a
  doubled or trailing ``;`` makes, a rule with no ``=``, an empty name or value) is malformed, and
  start-up refuses it (``core.edition_start_up``), naming the setting and never its value.

**Matching.** Every rule must hold (a name given twice must hold both values). A rule holds when the
claim is a string equal to the value, exactly (case and all), or a list that holds a string equal to
the value. A missing claim, or one of any other kind (a number, a boolean, an object), does not hold.

``hd=example.com`` lets in only a Google Workspace domain's people; ``groups=automatos-users`` only
members of that group; ``tid=<tenant id>;roles=Automatos.User`` one Entra tenant's holders of one
app role.

**Shared issuers.** An issuer many organisations sign in through lets anyone with an account there
finish signing in, so start-up refuses one with no rule (``is_shared_issuer``, ``SHARED_ISSUERS``).
The list holds only issuers the provider's own documentation shows are shared by unrelated
organisations or the public. It cannot be complete: any provider may run such an issuer, and a
new one is not on it. The control is the access rule; the list only stops the plainest mistake.

Pure standard library: start-up imports it before anything else is loaded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

RULES_SEPARATOR = ";"
NAME_VALUE_SEPARATOR = "="
NAME_MAX = 256
VALUE_MAX = 1024
RULES_MAX = 32

# No whitespace or control character, no "=" or ";".
_NAME = re.compile(r"^[^\s\x00-\x1f\x7f=;]+$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

# Issuers that many organisations share (FR-26: data, never a branch on a provider in code). Each is
# compared after ``normalised_issuer``. NOT COMPLETE, and cannot be: the access rule is the control.
# * Google: "Always https://accounts.google.com or accounts.google.com for Google ID tokens"
#   (developers.google.com/identity/openid-connect/openid-connect, the ``iss`` claim). Only the
#   https form is listed: ``OIDC_ISSUER`` must be an https address, so start-up already refuses
#   the other for that (``core.edition_start_up``), and it can never be a configured issuer.
# * Microsoft: the ``common``, ``organizations`` and ``consumers`` authorities sign in people of
#   any tenant, or any personal account (learn.microsoft.com/entra/identity-platform/
#   v2-protocols-oidc), and 9188040d-6c67-4c5b-b112-36a304b66dad is the tenant of every personal
#   Microsoft account (.../id-token-claims-reference, ``iss`` and ``tid``), each in its v2.0 form.
#   Their v1 forms (no ``/v2.0``, and ``sts.windows.net``) are not shown on those pages; they stay
#   because an entry here can only make start-up stricter.
# Other providers are added only once their own documentation shows such an issuer.
_CONSUMER_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad"
SHARED_ISSUERS: Tuple[str, ...] = (
    "https://accounts.google.com",
    *(f"https://login.microsoftonline.com/{tenant}{version}"
      for tenant in ("common", "organizations", "consumers", _CONSUMER_TENANT) for version in ("", "/v2.0")),
    f"https://sts.windows.net/{_CONSUMER_TENANT}",
)
# Claims that describe the token, not the person (OpenID Connect Core 1.0 §2 and §3.1.3.6; RFC 7519
# §4.1): a rule on one holds for every valid token, so it is refused.
PROTOCOL_CLAIMS: Tuple[str, ...] = (
    "iss", "aud", "azp", "exp", "iat", "nbf", "nonce", "auth_time", "at_hash", "c_hash", "sid", "jti",
)
# Claims that say something about a person but never which one (PRD-277 §5.10): the same value for many
# people, or a value that changes and is reused. A rule may name one; ``OIDC_CLAIM_SUBJECT`` may not
# (``core.edition_start_up``), because the subject is the one value a person is found by.
NOT_ONE_PERSONS_ID_CLAIMS: Tuple[str, ...] = (
    "email_verified", "acr", "amr", "ver", "tid", "hd", "groups", "roles", "picture", "locale",
    "preferred_username", "upn",
)


class MalformedRule(ValueError):
    """``OIDC_REQUIRED_CLAIMS`` is not in the rule language. Its text never quotes the value."""


@dataclass(frozen=True)
class Rule:
    """One claim that must hold one value."""

    name: str
    value: str


def _rule(text: str) -> Rule:
    if NAME_VALUE_SEPARATOR not in text:
        raise MalformedRule("each rule must be a claim name, '=' and a value")
    name, value = (part.strip() for part in text.split(NAME_VALUE_SEPARATOR, 1))
    if not name or len(name) > NAME_MAX or not _NAME.match(name):
        raise MalformedRule(f"each claim name must be 1 to {NAME_MAX} characters with no space, '=' or ';'")
    if name in PROTOCOL_CLAIMS:
        raise MalformedRule("a rule may not name a claim every valid token carries "
                            f"({', '.join(PROTOCOL_CLAIMS)}): it would let everyone in")
    if not value or len(value) > VALUE_MAX or _CONTROL.search(value):
        raise MalformedRule(f"each value must be 1 to {VALUE_MAX} characters with no control character")
    return Rule(name=name, value=value)


def parse_rules(setting: Any) -> Tuple[Rule, ...]:
    """The rules ``setting`` names, in order; ``()`` for an empty one, ``MalformedRule`` otherwise."""
    text = str(setting or "").strip()
    if not text:
        return ()
    parts = text.split(RULES_SEPARATOR)
    if len(parts) > RULES_MAX:
        raise MalformedRule(f"at most {RULES_MAX} rules")
    return tuple(_rule(part) for part in parts)


def _holds(rule: Rule, claims: Mapping[str, Any]) -> bool:
    claim = claims.get(rule.name)
    if isinstance(claim, str):
        return claim == rule.value
    if isinstance(claim, (list, tuple)):
        return any(isinstance(item, str) and item == rule.value for item in claim)
    return False


def every_rule_holds(rules: Tuple[Rule, ...], claims: Mapping[str, Any]) -> bool:
    """Whether every rule holds for ``claims`` (the verified ID token's); no rule at all holds."""
    return all(_holds(rule, claims) for rule in rules)


def normalised_issuer(issuer: Any) -> str:
    """An issuer as ``SHARED_ISSUERS`` compares it: trimmed, no trailing ``/``, lower-case."""
    return str(issuer or "").strip().rstrip("/").lower()


def is_shared_issuer(issuer: Any) -> bool:
    """Whether many organisations sign in through ``issuer``, so an access rule is required."""
    return normalised_issuer(issuer) in SHARED_ISSUERS


__all__ = [
    "NAME_MAX",
    "NOT_ONE_PERSONS_ID_CLAIMS",
    "PROTOCOL_CLAIMS",
    "RULES_MAX",
    "SHARED_ISSUERS",
    "VALUE_MAX",
    "MalformedRule",
    "Rule",
    "every_rule_holds",
    "is_shared_issuer",
    "normalised_issuer",
    "parse_rules",
]
