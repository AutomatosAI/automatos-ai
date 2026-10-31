"""PRD-261 C-05a — the Automatos theme signing keys, and the Ed25519 check of a signature.

A signature vouches for one theme version. It is the Ed25519 signature, by an Automatos signing
key, of the canonical text (``signing.canonical_text``) of the payload ``{id, version, digest,
files}`` (``payload_for``): the theme id, the version, the package digest
(``digest.package_digest``) and each file's sha256. The kit's sign workflow makes it
(automatos-design-kit ``scripts/sign/``, C-05m); the orchestrator never holds a signing key, it checks a
signature against the public keys it ships, and nothing here loads anything but a public key.

The keys file, ``data/signing-keys.json`` (the same bytes as the kit's ``keys/theme-signing.json``)::

    {"schema": 1, "keys": [{"keyId": ..., "spki": ..., "status": ..., "added": ...}]}

``spki`` is the public key's SubjectPublicKeyInfo DER in base64; ``keyId`` is derived from it
(``key_id_of``: ``ed25519:`` and the first 16 hex of the DER's sha256); ``status`` is
``active``, ``retired`` or ``revoked``; ``added`` a YYYY-MM-DD date. A file that breaks any of
that, or names a keyId twice, is refused with a plain message (``SigningKeysError``) that echoes
nothing from the file. It holds one key (C-05n): Automatos's theme-signing key,
``ed25519:e138b6d0fa7aaaf3``, ``active``, added 2026-10-26. Only a signature that key made verifies.

Verdicts (``Verdict``): ``verified`` when an active key verifies the signature, with that key's
id; ``invalid`` when no key does, or only a retired or revoked one does (that key's id is kept, so
a later check can name it); ``unsigned`` when there is no signature.

The shared test vector, ``tests/fixtures/themes/signing-vector.json``, is byte-identical to the
kit's ``tests/fixtures/signing-vector.json`` (RFC 8032 section 7.1 cases and canonical-text cases).

Synchronous and side-effect free: the keys file is read on each check, so a test can point
``KEYS_PATH`` at its own file.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key

from core.models.themes import ThemePackage
from services.themes import signing
from services.themes.digest import package_digest

KEYS_PATH = Path(__file__).resolve().parent / "data" / "signing-keys.json"
KEYS_SCHEMA = 1
STATUS_ACTIVE, STATUS_RETIRED, STATUS_REVOKED = "active", "retired", "revoked"
KEY_STATUSES = (STATUS_ACTIVE, STATUS_RETIRED, STATUS_REVOKED)
KEY_FIELDS = ("added", "keyId", "spki", "status")  # sorted
KEY_ID_PREFIX = "ed25519:"
KEY_ID_HEX_CHARS = 16
STATE_VERIFIED, STATE_INVALID, STATE_UNSIGNED = "verified", "invalid", "unsigned"
_ADDED_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

UNREADABLE = "The signing keys file can't be read as UTF-8 text."
NOT_JSON = "The signing keys file isn't JSON."
NOT_SCHEMA = 'The signing keys file must be {"schema": 1, "keys": [...]}.'
KEY_SHAPE = "Key {number} in the signing keys file must have exactly keyId, spki, status and added."
KEY_STATUS = "Key {number} in the signing keys file has a status other than active, retired or revoked."
KEY_ADDED = 'Key {number} in the signing keys file needs "added" as a YYYY-MM-DD date.'
KEY_SPKI = "Key {number} in the signing keys file has an spki that isn't an Ed25519 public key (SPKI DER, base64)."
KEY_ID_MISMATCH = "Key {number} in the signing keys file has a keyId that doesn't match its spki."
KEY_TWICE = "The signing keys file names {key_id} twice."


class SigningKeysError(ValueError):
    """The keys file can't be used; the message is plain and echoes nothing from the file."""


@dataclass(frozen=True)
class SigningKey:
    """One public key of the keys file."""

    key_id: str
    status: str
    public_key: Ed25519PublicKey


@dataclass(frozen=True)
class Verdict:
    """What a signature check found: ``state`` and the id of the key that verified, if any."""

    state: str
    key_id: Optional[str] = None

    @property
    def verified(self) -> bool:
        return self.state == STATE_VERIFIED

    def view(self) -> Dict[str, Optional[str]]:
        """``{state, keyId}``, as the listings and the admin report carry it."""
        return {"state": self.state, "keyId": self.key_id}


UNSIGNED = Verdict(STATE_UNSIGNED)


def key_id_of(spki_der: bytes) -> str:
    """``ed25519:`` and the first 16 hex of sha256(SPKI DER): a key's name in every keys file."""
    return KEY_ID_PREFIX + hashlib.sha256(spki_der).hexdigest()[:KEY_ID_HEX_CHARS]


def payload_for(theme_id: str, version: str, file_hashes: Mapping[str, str]) -> Dict[str, Any]:
    """``{id, version, digest, files}``: what a signature covers, the digest from the file hashes."""
    files = {path: file_hashes[path] for path in sorted(file_hashes)}
    return {"id": theme_id, "version": version, "digest": package_digest(files), "files": files}


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """The signed bytes: ``signing.canonical_text`` of the payload, as ASCII."""
    return signing.canonical_text(payload).encode("ascii")


def _public_key(spki: Any) -> Tuple[Optional[bytes], Optional[Ed25519PublicKey]]:
    """The DER and the Ed25519 key of a base64 ``spki``, or Nones for anything else."""
    if not isinstance(spki, str):
        return None, None
    try:
        der = base64.b64decode(spki, validate=True)
        key = load_der_public_key(der)
    except (binascii.Error, ValueError, UnsupportedAlgorithm):
        return None, None
    return (der, key) if isinstance(key, Ed25519PublicKey) else (None, None)


def _read_key(raw: Any, number: int) -> SigningKey:
    """One entry of the keys file (``number`` counts from 1), or ``SigningKeysError``."""
    if not isinstance(raw, dict) or tuple(sorted(raw)) != KEY_FIELDS:
        raise SigningKeysError(KEY_SHAPE.format(number=number))
    if not isinstance(raw["status"], str) or raw["status"] not in KEY_STATUSES:
        raise SigningKeysError(KEY_STATUS.format(number=number))
    if not isinstance(raw["added"], str) or _ADDED_DATE.fullmatch(raw["added"]) is None:
        raise SigningKeysError(KEY_ADDED.format(number=number))
    der, public_key = _public_key(raw["spki"])
    if der is None or public_key is None:
        raise SigningKeysError(KEY_SPKI.format(number=number))
    if raw["keyId"] != key_id_of(der):
        raise SigningKeysError(KEY_ID_MISMATCH.format(number=number))
    return SigningKey(key_id=raw["keyId"], status=raw["status"], public_key=public_key)


def parse_keys(text: str) -> Tuple[SigningKey, ...]:
    """The keys file's text: schema 1, every key well formed, no keyId twice."""
    try:
        raw = json.loads(text)
    except ValueError as bad:
        raise SigningKeysError(NOT_JSON) from bad
    schema = raw.get("schema") if isinstance(raw, dict) else None
    if isinstance(schema, bool) or schema != KEYS_SCHEMA or not isinstance(raw.get("keys"), list):
        raise SigningKeysError(NOT_SCHEMA)
    keys = []
    for number, entry in enumerate(raw["keys"], start=1):
        key = _read_key(entry, number)
        if any(held.key_id == key.key_id for held in keys):
            raise SigningKeysError(KEY_TWICE.format(key_id=key.key_id))
        keys.append(key)
    return tuple(keys)


def load_keys(path: Optional[Path] = None) -> Tuple[SigningKey, ...]:
    """The keys of the keys file (``KEYS_PATH`` unless ``path`` is given)."""
    target = KEYS_PATH if path is None else path
    try:
        text = Path(target).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as bad:
        raise SigningKeysError(UNREADABLE) from bad
    return parse_keys(text)


def _signature_bytes(signature: str) -> Optional[bytes]:
    """The 64 signature bytes of standard base64 text, or None."""
    try:
        raw = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == signing.SIGNATURE_BYTES else None


def _verifies(key: SigningKey, signature: bytes, message: bytes) -> bool:
    try:
        key.public_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def verify(
    payload: Mapping[str, Any], signature: Optional[str], keys: Optional[Sequence[SigningKey]] = None,
) -> Verdict:
    """The verdict on ``signature`` (standard base64, as stored) over the payload's canonical text.

    ``keys`` defaults to the keys file. Active keys are tried first, so ``verified`` names the
    active key that verifies; a signature only a retired or revoked key verifies is ``invalid``
    with that key's id; one no key verifies is ``invalid`` with none.
    """
    if not signature:
        return UNSIGNED
    raw = _signature_bytes(signature) if isinstance(signature, str) else None
    if raw is None:
        return Verdict(STATE_INVALID)
    message = canonical_bytes(payload)
    held = load_keys() if keys is None else keys
    ordered = sorted(held, key=lambda key: key.status != STATUS_ACTIVE)
    signer = next((key for key in ordered if _verifies(key, raw, message)), None)
    if signer is None:
        return Verdict(STATE_INVALID)
    return Verdict(STATE_VERIFIED if signer.status == STATUS_ACTIVE else STATE_INVALID, signer.key_id)


def package_verdict(package: ThemePackage, keys: Optional[Sequence[SigningKey]] = None) -> Verdict:
    """The verdict on a stored version's signature, over the payload rebuilt from its file list.

    A version whose file hashes don't add up to its digest can't be what was signed: ``invalid``.
    """
    if not package.signature:
        return UNSIGNED
    try:
        payload = signing.signing_payload(package)
    except signing.SigningError:
        return Verdict(STATE_INVALID)
    return verify(payload, package.signature, keys)


__all__ = [
    "KEYS_PATH", "KEY_STATUSES", "STATE_INVALID", "STATE_UNSIGNED", "STATE_VERIFIED", "STATUS_ACTIVE",
    "STATUS_RETIRED", "STATUS_REVOKED", "SigningKey", "SigningKeysError", "UNSIGNED", "Verdict",
    "canonical_bytes", "key_id_of", "load_keys", "package_verdict", "parse_keys", "payload_for", "verify",
]
