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
   module checks its shape and that it answers the version's open request, and stores it in
   ``theme_packages.signature``.

Signature **verification** is PRD §10 C-05, outside Q23: until it lands no edition verifies a
signature, and on the hosted edition the serve-time digest check is the control (PRD §6).

Synchronous; nothing here commits (the router does, once).
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from core.models.themes import ThemePackage, ThemeReviewEvent
from services.themes.choices import CONFLICT, ThemeInstallError
from services.themes.digest import package_digest, sha256_hex

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
ALREADY_SIGNED = "This version already has a signature."
NO_OPEN_REQUEST = "This version has no signature request. Approve it first."
WRONG_REQUEST = "That isn't this version's open signature request. Use the request id the approval showed."
BAD_SIGNATURE = (
    "That isn't an Ed25519 signature. Paste the base64 text the kit's sign workflow printed (64 bytes)."
)


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


def _check_can_sign(db: Session, package: ThemePackage, request_id: UUID) -> ThemeReviewEvent:
    if package.status != STATUS_APPROVED:
        raise SigningError(CONFLICT, NOT_APPROVED)
    if package.signature:
        raise SigningError(CONFLICT, ALREADY_SIGNED)
    request = open_request(db, package.id)
    if request is None:
        raise SigningError(CONFLICT, NO_OPEN_REQUEST)
    if request.id != request_id:
        raise SigningError(CONFLICT, WRONG_REQUEST)
    return request


def store_signature(
    db: Session, package: ThemePackage, request_id: UUID, signature_text: str, actor_pk: Optional[int],
) -> Dict[str, Any]:
    """Store the pasted signature for the version's open request and record a ``signed`` event."""
    request = _check_can_sign(db, package, request_id)
    signature = normalised_signature(signature_text)
    package.signature = signature
    detail = dict(request.detail) if isinstance(request.detail, Mapping) else {}
    db.add(ThemeReviewEvent(
        package_id=package.id, event=EVENT_SIGNED, from_status=package.status, to_status=package.status,
        actor_user_id=actor_pk, created_at=now_utc(),
        detail={"requestId": str(request_id), "payloadSha256": detail.get("payloadSha256"),
                "algorithm": SIGNATURE_ALGORITHM, "verified": False},
    ))
    return {"requestId": str(request_id), "signature": signature, "algorithm": SIGNATURE_ALGORITHM, "verified": False}


__all__ = [
    "EVENT_SIGNATURE_REQUESTED", "EVENT_SIGNED", "SIGNATURE_ALGORITHM", "SIGNATURE_BYTES", "SIGNED_FORM",
    "SigningError", "canonical_text", "normalised_signature", "open_request", "request_signature",
    "request_view", "signing_payload", "store_signature",
]
