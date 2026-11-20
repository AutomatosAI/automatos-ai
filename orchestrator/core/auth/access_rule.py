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
# compared after ``normalised_issuer``. Google's one issuer for every account; Microsoft's
# multi-tenant addresses (v2.0 and v1) and the tenant every personal Microsoft account shares.
_CONSUMER_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad"
SHARED_ISSUERS: Tuple[str, ...] = (
    "https://accounts.google.com",
    *(f"https://login.microsoftonline.com/{tenant}{version}"
      for tenant in ("common", "organizations", "consumers", _CONSUMER_TENANT) for version in ("", "/v2.0")),
    f"https://sts.windows.net/{_CONSUMER_TENANT}",
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
