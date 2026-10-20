"""PRD-261 C-03a — signing an approved theme version through the kit's CI step (D3-c, 19 Oct).

The orchestrator never holds a signing key and never calls GitHub (AGENTS.md: no new
third-party client, no new secret). So signing is two hand-offs a person makes:

1. **Approve records a signature request** (``theme_review_events`` row, event
   ``signature_requested``, its id is the request id). Its payload is
   ``{id, version, digest, files}``: the theme id, the version, the package digest and the
   per-file sha256 map (PRD §6, "an Ed25519 signature over the per-file sha256 list").
   ``canonical`` is the exact text the kit signs: the payload as JSON with sorted keys, no
   spaces and ASCII only (``SIGNED_FORM``).
2. **A person runs the kit's sign workflow** with that payload. The key is a secret of the
   kit repository; the workflow returns a base64 Ed25519 signature (64 bytes), which the
   person pastes into the admin queue (``POST /api/admin/themes/{id}/signature``). This
   module checks its shape, that it answers the version's open request and that an active
   Automatos signing key verifies it over the canonical text (C-05b), and only then stores it
   in ``theme_packages.signature``; the ``signed`` event names that key. A signature no active
   key verifies is a 400 and nothing is stored. A version whose stored signature an active key
   still verifies refuses another (409); one whose stored signature no longer verifies (its key
   retired or revoked since, QB-28, or a row stored before C-05b checked) takes a new one for its
   open request, so it can be Official again without a database edit, and the ``signed`` event
   records what it replaced.

The check itself is ``signature_verify.verify`` (the keys file and the Ed25519 verifier, C-05a),
which ``review.sign`` hands to ``store_signature``: this module reads no key and imports no
cryptography, and ``signature_verify`` builds on its ``canonical_text``.

Synchronous; nothing here commits (the router does, once).
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from core.models.themes import ThemePackage, ThemeReviewEvent
from services.themes.choices import CONFLICT, ThemeInstallError
from services.themes.digest import package_digest, sha256_hex

if TYPE_CHECKING:  # the type only: signature_verify imports this module
    from services.themes.signature_verify import Verdict

logger = logging.getLogger(__name__)

EVENT_SIGNATURE_REQUESTED = "signature_requested"
EVENT_SIGNED = "signed"
SIGNED_FORM = "automatos-theme-sign/v1: canonical JSON of {id, version, digest, files}, sorted keys, no spaces, ASCII"
SIGNATURE_ALGORITHM = "ed25519"
SIGNATURE_BYTES = 64  # an Ed25519 signature is always 64 bytes
SIGNATURE_TEXT_MAX = 200  # base64 of 64 bytes is 88 characters; room for pasted padding variants
BAD_REQUEST = 400
STATUS_APPROVED = "approved"
_BASE64_TEXT = re.compile(r"[A-Za-z0-9+/_-]+={0,2}")

FILE_WITHOUT_HASH = "This version's file list has no sha256 for {path}, so it can't be signed. Reject it."
DIGEST_MISMATCH = "This version's file hashes don't add up to its digest, so it can't be signed. Reject it."
NOT_APPROVED = "Only an approved version can be signed. Approve it first."
ALREADY_SIGNED = "This version already has a signature an active Automatos signing key verifies."
NO_OPEN_REQUEST = "This version has no signature request. Approve it first."
WRONG_REQUEST = "That isn't this version's open signature request. Use the request id the approval showed."
BAD_SIGNATURE = (
    "That isn't an Ed25519 signature. Paste the base64 text the kit's sign workflow printed (64 bytes)."
)
NOT_VERIFIED = (
    "No Automatos signing key verifies that signature for this version, so it wasn't stored. "
    "Paste the signature the kit's sign workflow printed for this version's request."
)

# The check ``store_signature`` runs before storing: ``signature_verify.verify`` (payload, base64 text).
SignatureCheck = Callable[[Mapping[str, Any], str], "Verdict"]


class SigningError(ThemeInstallError):
    """A refusal on the signing path; the router answers its status and plain message."""


def now_utc() -> datetime:
    """An event's time, set here (microseconds) so "the latest" is well ordered on every database."""
    return datetime.now(timezone.utc)


def _file_hashes(package: ThemePackage) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path, row in sorted((package.file_manifest or {}).items()):
        sha = row.get("sha256") if isinstance(row, Mapping) else None
        if not isinstance(sha, str) or not sha:
            raise SigningError(CONFLICT, FILE_WITHOUT_HASH.format(path=path))
        hashes[path] = sha
    return hashes


def signing_payload(package: ThemePackage) -> Dict[str, Any]:
    """``{id, version, digest, files}`` for one version; refuses files that don't match its digest."""
    files = _file_hashes(package)
    if not files or package_digest(files) != package.digest:
        raise SigningError(CONFLICT, DIGEST_MISMATCH)
    return {"id": package.theme_id, "version": package.version, "digest": package.digest, "files": files}


def canonical_text(payload: Mapping[str, Any]) -> str:
    """The exact text the kit's sign workflow signs (``SIGNED_FORM``)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_signature(db: Session, package: ThemePackage, actor_pk: Optional[int]) -> Dict[str, Any]:
    """Record a signature request for an approved version and answer what the person hands the kit."""
    payload = signing_payload(package)
    canonical = canonical_text(payload)
    request_id = uuid4()
    detail = {
        "requestId": str(request_id), "payload": payload, "payloadSha256": sha256_hex(canonical.encode("ascii")),
        "algorithm": SIGNATURE_ALGORITHM, "form": SIGNED_FORM,
    }
    db.add(ThemeReviewEvent(
        id=request_id, package_id=package.id, event=EVENT_SIGNATURE_REQUESTED, created_at=now_utc(),
        from_status=package.status, to_status=package.status, actor_user_id=actor_pk, detail=detail,
    ))
    return {**detail, "canonical": canonical}


def open_request(db: Session, package_id: UUID) -> Optional[ThemeReviewEvent]:
    """The version's latest signature request, or None."""
    return (
        db.query(ThemeReviewEvent)
        .filter(ThemeReviewEvent.package_id == package_id, ThemeReviewEvent.event == EVENT_SIGNATURE_REQUESTED)
        .order_by(ThemeReviewEvent.created_at.desc(), ThemeReviewEvent.id.desc())
        .first()
    )


def request_view(request: Optional[ThemeReviewEvent]) -> Optional[Dict[str, Any]]:
    """What the admin queue shows of a request: its detail plus the canonical text to copy."""
    if request is None or not isinstance(request.detail, Mapping):
        return None
    detail = dict(request.detail)
    payload = detail.get("payload")
    return {**detail, "canonical": canonical_text(payload) if isinstance(payload, Mapping) else None}


def normalised_signature(text: str) -> str:
    """The pasted signature as standard base64 of exactly 64 bytes; anything else is refused."""
    compact = "".join(text.split())
    if not compact or len(compact) > SIGNATURE_TEXT_MAX or _BASE64_TEXT.fullmatch(compact) is None:
        raise SigningError(BAD_REQUEST, BAD_SIGNATURE)
    standard = compact.replace("-", "+").replace("_", "/")
    standard += "=" * (-len(standard) % 4)
    try:
        raw = base64.b64decode(standard, validate=True)
    except (binascii.Error, ValueError) as bad:
        raise SigningError(BAD_REQUEST, BAD_SIGNATURE) from bad
    if len(raw) != SIGNATURE_BYTES:
        raise SigningError(BAD_REQUEST, BAD_SIGNATURE)
    return base64.b64encode(raw).decode("ascii")


def _check_can_sign(
    db: Session, package: ThemePackage, request_id: UUID, check: SignatureCheck,
) -> Tuple[ThemeReviewEvent, Optional["Verdict"]]:
    """The open request a paste answers, and ``check``'s verdict on the stored signature it would
    replace (None when there is none). A stored signature an active key still verifies is never
    replaced (409); one that no longer verifies may be."""
    if package.status != STATUS_APPROVED:
        raise SigningError(CONFLICT, NOT_APPROVED)
    replaced = check(signing_payload(package), package.signature) if package.signature else None
    if replaced is not None and replaced.verified:
        raise SigningError(CONFLICT, ALREADY_SIGNED)
    request = open_request(db, package.id)
    if request is None:
        raise SigningError(CONFLICT, NO_OPEN_REQUEST)
    if request.id != request_id:
        raise SigningError(CONFLICT, WRONG_REQUEST)
    return request, replaced


def store_signature(
    db: Session, package: ThemePackage, request_id: UUID, signature_text: str, actor_pk: Optional[int],
    check: SignatureCheck,
) -> Dict[str, Any]:
    """Store the pasted signature for the version's open request once ``check`` finds an active key
    that verifies it, and record a ``signed`` event naming that key and the verdict on the stored
    signature it replaced (``replaces``, None for the first); otherwise 400, nothing stored."""
    request, replaced = _check_can_sign(db, package, request_id, check)
    signature = normalised_signature(signature_text)
    verdict = check(signing_payload(package), signature)
    if not verdict.verified:
        logger.warning("theme package %s: a pasted signature no active signing key verifies (%s)",
                       package.id, verdict.state)
        raise SigningError(BAD_REQUEST, NOT_VERIFIED)
    package.signature = signature
    detail = dict(request.detail) if isinstance(request.detail, Mapping) else {}
    db.add(ThemeReviewEvent(
        package_id=package.id, event=EVENT_SIGNED, from_status=package.status, to_status=package.status,
        actor_user_id=actor_pk, created_at=now_utc(),
        detail={"requestId": str(request_id), "payloadSha256": detail.get("payloadSha256"),
                "algorithm": SIGNATURE_ALGORITHM, "verified": True, "keyId": verdict.key_id,
                "replaces": None if replaced is None else replaced.view()},
    ))
    if replaced is not None:
        logger.info("theme package %s re-signed with %s; its stored signature no longer verified (%s, %s)",
                    package.id, verdict.key_id, replaced.state, replaced.key_id)
    return {"requestId": str(request_id), "signature": signature, "algorithm": SIGNATURE_ALGORITHM,
            "verified": True, "keyId": verdict.key_id}


__all__ = [
    "EVENT_SIGNATURE_REQUESTED", "EVENT_SIGNED", "NOT_VERIFIED", "SIGNATURE_ALGORITHM", "SIGNATURE_BYTES",
    "SIGNED_FORM", "SignatureCheck", "SigningError", "canonical_text", "normalised_signature", "open_request",
    "request_signature", "request_view", "signing_payload", "store_signature",
]
