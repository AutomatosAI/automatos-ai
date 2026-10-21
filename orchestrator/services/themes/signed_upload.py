"""PRD-261 C-05d — the optional signature of a private upload: a detached ``.sig.json`` envelope.

QB-27 (detached, the plan's planned answer): a signed ``.autotheme`` travels with a small JSON file
made by the kit's sign workflow (automatos-play ``scripts/sign/envelope.ts``)::

    {form: "automatos-theme-signature/v1", algorithm: "ed25519", keyId, id, version, digest, signature}

at most ``ENVELOPE_MAX_BYTES``. ``POST /api/themes/import`` takes it as the optional multipart part
``SIGNATURE_PART``; ``check_upload_signature`` holds it to the kit's form, then to the upload: the
id, version and digest must be the uploaded package's (``SIGNATURE_MISMATCH``), and an active
Automatos signing key, the one the envelope names, must verify the signature over the payload
rebuilt from the upload's own file hashes (``signature_verify.payload_for``). Anything else is
``SIGNATURE_BAD``. Every check runs before the route stores anything, and no message echoes a value
from the file.

``signature_view`` is the ``{state, keyId}`` the installed list shows for a stored version.

In the local edition a key the hosted feed revoked (``revokedKeys``, as the pull stored them:
``revocation_sync_state.revoked_key_ids``) counts as revoked here too, as it does on load
(``signature_state``): an upload it verifies is refused with ``INACTIVE_KEY``, and the view reads
a stored signature it made as ``invalid`` with its key id.

Synchronous and side-effect free (it reads the database, never writes it); the keys file is read
on each check (``signature_verify``).
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from core.models.themes import ThemePackage
from services.themes import signature_verify
from services.themes.contract import KB, ThemeError, parse_json, theme_error
from services.themes.manifest import ID_PATTERN, VERSION_PATTERN
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.revocation_sync_state import revoked_key_ids

logger = logging.getLogger(__name__)

SIGNATURE_PART = "signature"
ENVELOPE_FORM = "automatos-theme-signature/v1"
ENVELOPE_ALGORITHM = "ed25519"
ENVELOPE_MAX_BYTES = 4 * KB
ENVELOPE_KEYS: Tuple[str, ...] = ("algorithm", "digest", "form", "id", "keyId", "signature", "version")  # sorted
KEY_ID_PATTERN = re.compile(r"ed25519:[0-9a-f]{16}")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
SIGNATURE_BYTES = 64
# The upload fields an envelope must name, as (envelope key, what the message calls it).
NAMED_FIELDS: Tuple[Tuple[str, str], ...] = (("id", "theme id"), ("version", "version"), ("digest", "package digest"))

TOO_LARGE_STATUS, BAD_STATUS = 413, 400
SIGNATURE_TOO_LARGE, SIGNATURE_BAD, SIGNATURE_MISMATCH = "SIGNATURE_TOO_LARGE", "SIGNATURE_BAD", "SIGNATURE_MISMATCH"
SIGNATURE_CODES: Tuple[str, ...] = (SIGNATURE_TOO_LARGE, SIGNATURE_BAD, SIGNATURE_MISMATCH)

TOO_LARGE = f"The signature file must be {ENVELOPE_MAX_BYTES // KB} KB or smaller."
NOT_JSON = "The signature file isn't a JSON signature envelope (.sig.json)."
NOT_AN_OBJECT = "The signature file must be one JSON object."
WRONG_KEYS = "The signature file must have exactly the keys form, algorithm, keyId, id, version, digest and signature."
INVALID = "The signature file is not valid: {reason}"
FOR_ANOTHER_PACKAGE = (
    "This signature is for another package: its {field} isn't the uploaded theme's. "
    "Upload the .sig.json made for this .autotheme."
)
INACTIVE_KEY = "This signature was made by an Automatos signing key that is retired or revoked, so nothing was installed."
NOT_VERIFIED = "No Automatos signing key verifies this signature for these files, so nothing was installed."
OTHER_KEY = "The signature file names a different key from the one that made the signature, so nothing was installed."


class SignatureRefused(ValueError):
    """The envelope was refused; ``status_code`` is the route's answer, ``error`` its ``{code, path, message}``."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error: ThemeError = theme_error(code, SIGNATURE_PART, message)


@dataclass(frozen=True)
class VerifiedSignature:
    """An envelope that verified for the upload: the signature to store and the key that made it."""

    signature: str
    key_id: str


def _bad(message: str) -> SignatureRefused:
    return SignatureRefused(BAD_STATUS, SIGNATURE_BAD, message)


def _is_signature_text(value: Any) -> bool:
    """Standard base64, exactly as it encodes, of a 64-byte Ed25519 signature (the kit's ``signatureBytes``)."""
    if not isinstance(value, str):
        return False
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return len(raw) == SIGNATURE_BYTES and base64.b64encode(raw).decode("ascii") == value


def _matches(pattern: "re.Pattern[str]") -> Callable[[Any], bool]:
    return lambda value: isinstance(value, str) and pattern.fullmatch(value) is not None


# Each field's rule and the plain reason given when it fails (the kit's FIELD_RULES, in its order).
FIELD_RULES: Tuple[Tuple[str, Callable[[Any], bool], str], ...] = (
    ("form", lambda value: value == ENVELOPE_FORM, f'"form" must be "{ENVELOPE_FORM}".'),
    ("algorithm", lambda value: value == ENVELOPE_ALGORITHM, f'"algorithm" must be "{ENVELOPE_ALGORITHM}".'),
    ("keyId", _matches(KEY_ID_PATTERN), '"keyId" must be "ed25519:" and 16 lower-case hex characters.'),
    ("id", _matches(ID_PATTERN), '"id" must be a theme id: "publisher/name".'),
    ("version", _matches(VERSION_PATTERN), '"version" must be MAJOR.MINOR.PATCH.'),
    ("digest", _matches(SHA256_HEX), '"digest" must be the package digest: 64 lower-case hex characters.'),
    ("signature", _is_signature_text, '"signature" must be the standard base64 of a 64-byte Ed25519 signature.'),
)


def parse_envelope(data: bytes) -> Dict[str, str]:
    """A ``.sig.json`` file's bytes held to the kit's form: 413 over the size, else ``SIGNATURE_BAD``."""
    if len(data) > ENVELOPE_MAX_BYTES:
        raise SignatureRefused(TOO_LARGE_STATUS, SIGNATURE_TOO_LARGE, TOO_LARGE)
    document, error = parse_json(data, SIGNATURE_PART)
    if error is not None:
        raise _bad(NOT_JSON)
    if not isinstance(document, dict):
        raise _bad(NOT_AN_OBJECT)
    if tuple(sorted(document)) != ENVELOPE_KEYS:
        raise _bad(WRONG_KEYS)
    failed = next((reason for key, holds, reason in FIELD_RULES if not holds(document[key])), None)
    if failed is not None:
        raise _bad(INVALID.format(reason=failed))
    return {key: document[key] for key in ENVELOPE_KEYS}


def _file_hashes(checked: CheckedThemePackage) -> Dict[str, str]:
    return {path: str(row["sha256"]) for path, row in checked.file_manifest.items()}


def _check_names_upload(envelope: Mapping[str, str], checked: CheckedThemePackage) -> None:
    """The envelope's id, version and digest are the upload's, or ``SIGNATURE_MISMATCH`` naming the first that isn't."""
    upload = {"id": checked.manifest["id"], "version": checked.manifest["version"], "digest": checked.digest}
    for key, called in NAMED_FIELDS:
        if envelope[key] != upload[key]:
            raise SignatureRefused(BAD_STATUS, SIGNATURE_MISMATCH, FOR_ANOTHER_PACKAGE.format(field=called))


def _refusal_for(verdict: signature_verify.Verdict) -> str:
    """Why a verdict that isn't ``verified`` refuses the upload."""
    return INACTIVE_KEY if verdict.key_id is not None else NOT_VERIFIED


def _feed_checked(db: Any, verdict: signature_verify.Verdict) -> signature_verify.Verdict:
    """``verdict``, or ``invalid`` with its key id when the key that verified is one the hosted feed revoked."""
    if verdict.verified and verdict.key_id in revoked_key_ids(db):
        return signature_verify.Verdict(signature_verify.STATE_INVALID, verdict.key_id)
    return verdict


def check_upload_signature(
    db: Any, checked: CheckedThemePackage, data: bytes, keys: Optional[Sequence[signature_verify.SigningKey]] = None,
) -> VerifiedSignature:
    """The envelope ``data`` checked against the read upload ``checked``; refuses with ``SignatureRefused``.

    ``keys`` defaults to the keys file; a keys file that does not load raises
    ``signature_verify.SigningKeysError`` (the route's 503). ``db`` gives the keys the hosted feed
    revoked (local edition): a signature one of them made is ``INACTIVE_KEY``.
    """
    envelope = parse_envelope(data)
    _check_names_upload(envelope, checked)
    manifest = checked.manifest
    payload = signature_verify.payload_for(manifest["id"], manifest["version"], _file_hashes(checked))
    verdict = _feed_checked(db, signature_verify.verify(payload, envelope["signature"], keys))
    if not verdict.verified:
        raise _bad(_refusal_for(verdict))
    if verdict.key_id != envelope["keyId"]:
        raise _bad(OTHER_KEY)
    return VerifiedSignature(signature=envelope["signature"], key_id=verdict.key_id)


def signature_view(db: Any, package: ThemePackage) -> Dict[str, Optional[str]]:
    """``{state, keyId}`` of a stored version's signature, as the installed list shows it.

    A keys file that does not load fails closed: ``invalid``, never ``verified`` (as the listings do).
    A key the hosted feed revoked (local edition) reads ``invalid`` with its key id.
    """
    try:
        verdict = signature_verify.package_verdict(package)
    except signature_verify.SigningKeysError as bad:
        logger.error("theme signing keys did not load; package %s reads as invalid: %s", package.id, bad)
        return signature_verify.Verdict(signature_verify.STATE_INVALID).view()
    return _feed_checked(db, verdict).view()


__all__ = [
    "ENVELOPE_ALGORITHM", "ENVELOPE_FORM", "ENVELOPE_KEYS", "ENVELOPE_MAX_BYTES", "SIGNATURE_BAD", "SIGNATURE_CODES",
    "SIGNATURE_MISMATCH", "SIGNATURE_PART", "SIGNATURE_TOO_LARGE", "SignatureRefused", "VerifiedSignature",
    "check_upload_signature", "parse_envelope", "signature_view",
]
