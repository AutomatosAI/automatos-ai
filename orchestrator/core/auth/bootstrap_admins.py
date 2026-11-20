"""PRD-277 US-405 (§5.4, D-6): who ``ENTERPRISE_BOOTSTRAP_ADMINS`` names.

The people it names become install admins, and owners of the default workspace, the first time
they sign in (``core.auth.first_admin``). That is how the first admin gets in without anyone
touching the database.

**The list**::

    ENTERPRISE_BOOTSTRAP_ADMINS = entry *( "," entry )
    entry                       = "sub:" subject / "email:" address

* ``sub:<subject>`` matches the person whose verified ID token's ``sub`` is exactly ``<subject>``:
  every character, case and all. The subject is the provider's own id for the person, which never
  changes and is never reused, so it is the entry to prefer.
* ``email:<address>`` matches the person whose verified ID token has an ``email`` claim equal to
  ``<address>`` AND ``email_verified`` the JSON ``true`` (absent, ``false``, ``"true"`` or ``1`` is
  not verified, so the entry matches nobody). The part before the ``@`` is compared exactly; the
  domain after it is compared without regard to case. It reads the standard ``email`` claim
  itself, the one ``email_verified`` speaks for, whatever ``OIDC_CLAIM_EMAIL`` names.
* Spaces around an entry are dropped. The prefixes are lower-case, as written above.

**Malformed**, so start-up refuses it, naming the setting and never its value: an empty entry (a
doubled or trailing comma), an entry without one of the two prefixes, an empty subject or address,
a control character, a ``*`` or ``?`` anywhere (the list names people, never patterns), an address
without exactly one ``@``, or with a space, or whose domain has no dot, and more than
``ENTRIES_MAX`` entries. An empty setting names nobody.

Pure standard library: start-up imports it before anything else is loaded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

ENTRIES_SEPARATOR = ","
SUBJECT_PREFIX = "sub:"
EMAIL_PREFIX = "email:"
ENTRIES_MAX = 50
ENTRY_MAX = 320
SUBJECT_KIND = "sub"
EMAIL_KIND = "email"

_CONTROL_OR_PATTERN = re.compile(r"[\x00-\x1f\x7f*?]")


class MalformedEntry(ValueError):
    """``ENTERPRISE_BOOTSTRAP_ADMINS`` is not a list of entries. Its text never quotes the value."""


@dataclass(frozen=True)
class Entry:
    """One person the list names: by subject, or by verified email."""

    kind: str
    value: str


def _split_address(address: str) -> Optional[Tuple[str, str]]:
    """``(local part, domain)`` of an address with one ``@``, no space and a dotted domain, else None."""
    if address.count("@") != 1 or any(ch.isspace() for ch in address):
        return None
    local, domain = address.split("@")
    if not local or "." not in domain.strip(".") or domain.startswith(".") or domain.endswith("."):
        return None
    return local, domain


def _entry(text: str) -> Entry:
    if not text or len(text) > ENTRY_MAX or _CONTROL_OR_PATTERN.search(text):
        raise MalformedEntry(f"each entry must be sub:<subject> or email:<address>, at most {ENTRY_MAX} "
                             "characters, with no '*' or '?'")
    if text.startswith(SUBJECT_PREFIX) and text[len(SUBJECT_PREFIX):].strip():
        return Entry(kind=SUBJECT_KIND, value=text[len(SUBJECT_PREFIX):].strip())
    if text.startswith(EMAIL_PREFIX):
        address = text[len(EMAIL_PREFIX):].strip()
        if _split_address(address) is not None:
            return Entry(kind=EMAIL_KIND, value=address)
    raise MalformedEntry("each entry must be sub:<subject> or email:<address>")


def parse_entries(setting: Any) -> Tuple[Entry, ...]:
    """The entries ``setting`` names, in order; ``()`` for an empty one, ``MalformedEntry`` otherwise."""
    text = str(setting or "").strip()
    if not text:
        return ()
    parts = text.split(ENTRIES_SEPARATOR)
    if len(parts) > ENTRIES_MAX:
        raise MalformedEntry(f"at most {ENTRIES_MAX} entries")
    return tuple(_entry(part.strip()) for part in parts)


def _same_address(entry_address: str, claimed: Any) -> bool:
    if not isinstance(claimed, str):
        return False
    wanted, found = _split_address(entry_address), _split_address(claimed)
    if wanted is None or found is None:
        return False
    return wanted[0] == found[0] and wanted[1].casefold() == found[1].casefold()


def _matches(entry: Entry, subject: str, claims: Mapping[str, Any]) -> bool:
    if entry.kind == SUBJECT_KIND:
        return subject == entry.value
    return claims.get("email_verified") is True and _same_address(entry.value, claims.get("email"))


def names_person(entries: Tuple[Entry, ...], subject: str, claims: Mapping[str, Any]) -> Optional[str]:
    """The kind of the first entry that names this person (``sub`` or ``email``), or None.

    ``subject`` and ``claims`` are the verified ID token's.
    """
    for entry in entries:
        if _matches(entry, subject, claims):
            return entry.kind
    return None


__all__ = [
    "EMAIL_KIND",
    "ENTRIES_MAX",
    "SUBJECT_KIND",
    "Entry",
    "MalformedEntry",
    "names_person",
    "parse_entries",
]
