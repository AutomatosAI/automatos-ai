"""PRD-261 C-03a — the admin review queue for Marketplace theme versions.

A person reviews every version (Q8, D3 19 Oct). What this module does, for a
``theme_packages`` row with ``owner_type`` 'marketplace' (a workspace's private upload is
never in the queue and answers 404 here, exactly as a missing row does):

* **pending** and **report**: the versions waiting for review, oldest first, and one
  version's scan report, the reviewer's notes, its review history and its signature request;
* **report note**: the reviewer pastes the summary of the kit's isolated review job (C-02b, a
  person-run job with no network after install and no secrets) with its outcome. A ``pass``
  note must be that job's summary for THIS version: it must hold the kit's lines
  ``Package digest:  <this version's digest>`` and ``Outcome: pass`` exactly as
  ``scripts/review/summary.ts`` prints them, so a note for another version, or a failed run,
  can't approve this one. The orchestrator never runs submitter code and never polls GitHub;
* **approve**: refuses a ``blocked`` scan and a ``not_run`` scan (unlike
  ``admin_plugins.py:222-223``), a version that isn't pending, and a digest under a live
  ``disable`` revocation. On the hosted edition it also needs the latest isolated-job note to
  say ``pass`` (the reproducible build is required, D3); the local edition never has the
  isolated stage, so there the person's approval is the review. Approval sets ``approved``
  and ``reviewed``, offers the version to the listing installs that run an older one
  (``listing_updates.py``; an install never switches by itself) and records a signature
  request (``signing.py``); it never signs;
* **sign**: stores the signature the kit's sign workflow returned for the open request, only
  when an active Automatos signing key verifies it (``signature_verify``, C-05b); the report's
  ``signatureVerified`` and ``keyId`` are that check on the stored signature. A keys file that
  does not load reads ``invalid`` in the report (it still answers), and the paste raises
  ``SigningKeysError``, which the router answers as a 503 with its plain message;
* **reject** with a reason, for a pending version;
* **deactivate** an approved version with a reason: status ``deactivated`` and a ``disable``
  revocation of its digest, so every install falls back to its base on the next load.

The Marketplace's auto-publish rule for agents (``api/marketplace.py:798-805``, five approved
items make an author trusted) never applies to themes: nothing here, or anywhere else, moves a
theme version to ``approved`` without an admin calling approve.

Every action writes a ``theme_review_events`` row. Synchronous; nothing here commits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from config import config
from core.models.themes import THEME_REVOCATION_ORIGIN_LOCAL, ThemePackage, ThemeReviewEvent, ThemeRevocation
from services.themes import signature_verify, signing
from services.themes.choices import CONFLICT, NOT_FOUND, ThemeInstallError
from services.themes.listing_updates import offer_listed_update, withdraw_offers
from services.themes.revocations import REVOKE_DISABLE, is_disabled

logger = logging.getLogger(__name__)

MARKETPLACE_OWNER = "marketplace"
STATUS_PENDING, STATUS_APPROVED = "pending", "approved"
STATUS_REJECTED, STATUS_DEACTIVATED = "rejected", "deactivated"
VERDICT_BLOCKED, VERDICT_NOT_RUN = "blocked", "not_run"
EVENT_APPROVED, EVENT_REJECTED, EVENT_DEACTIVATED = "approved", "rejected", "deactivated"
EVENT_REPORT_NOTE = "report_note"
NOTE_PASS, NOTE_FAIL = "pass", "fail"
NOTE_OUTCOMES = (NOTE_PASS, NOTE_FAIL)
PENDING_LIMIT_DEFAULT, PENDING_LIMIT_MAX = 50, 200
HISTORY_LIMIT = 100
BAD_REQUEST = 400

PACKAGE_NOT_FOUND = "That theme version isn't in the Marketplace review queue."
NOT_PENDING = "This version is {status}, so it can't be {action}. Only a pending version can."
SCAN_BLOCKED = "The automated review blocked this version, so it can't be approved. Reject it with the reason."
SCAN_NOT_RUN = "The automated review hasn't run on this version yet, so it can't be approved."
REVOKED = "This version's files are revoked (disabled), so it can't be approved."
NOTE_MISSING = (
    "Run the isolated review job in the kit repo and paste its summary first: "
    "a version needs a passing reproducible build before approval."
)
NOTE_FAILED = "The isolated review job's latest summary says it failed. Re-run it, or reject this version."
NOT_APPROVED = "Only an approved version can be deactivated. Reject a pending one instead."
NUL_IN_TEXT = "The text has a NUL character in it. Paste plain text."
# The kit's summary lines (automatos-play scripts/review/summary.ts, ``header``): two spaces after
# "Package digest:" so the digests line up.
NOTE_DIGEST_LINE = "Package digest:  {digest}"
NOTE_OUTCOME_PASS_LINE = "Outcome: pass"
NOTE_OUTCOME_PREFIX = "Outcome:"
# The kit's header block ends at its first blank line, before "Checks:" (summary.ts ``summaryText``).
NOTE_CHECKS_LINE = "Checks:"
NOTE_NOT_THIS_VERSION = (
    "A passing note must be the isolated review job's summary for this version: paste it as the kit "
    "printed it, with the lines \"Package digest:  {digest}\" and \"Outcome: pass\"."
)
EMPTY_TEXT = "Write a reason or summary; it can't be blank."


class ThemeReviewError(ThemeInstallError):
    """A refusal in the review queue; the router answers its status and plain message."""


def plain_text(text: str) -> str:
    """Trimmed text the database can store (Postgres refuses NUL in text and JSONB)."""
    if "\x00" in text:
        raise ThemeReviewError(BAD_REQUEST, NUL_IN_TEXT)
    trimmed = text.strip()
    if not trimmed:
        raise ThemeReviewError(BAD_REQUEST, EMPTY_TEXT)
    return trimmed


def queued_package(db: Session, package_id: UUID, lock: bool = False) -> ThemePackage:
    """A Marketplace version by id; a missing row and a workspace's private upload both answer 404."""
    query = db.query(ThemePackage).filter(
        ThemePackage.id == package_id, ThemePackage.owner_type == MARKETPLACE_OWNER,
    )
    package = (query.with_for_update() if lock else query).first()
    if package is None:
        raise ThemeReviewError(NOT_FOUND, PACKAGE_NOT_FOUND)
    return package


def _iso(value: Any) -> Optional[str]:
    return None if value is None else value.isoformat()


def package_summary(package: ThemePackage) -> Dict[str, Any]:
    """One version as the queue lists it."""
    manifest = package.manifest if isinstance(package.manifest, Mapping) else {}
    return {
        "id": str(package.id), "themeId": package.theme_id, "version": package.version,
        "name": manifest.get("name"), "publisher": package.theme_id.split("/", 1)[0],
        "digest": package.digest, "status": package.status, "reviewed": bool(package.reviewed),
        "scanVerdict": package.scan_verdict, "submittedBy": package.submitted_by,
        "signed": bool(package.signature), "createdAt": _iso(package.created_at),
    }


def _events_query(db: Session, package_ids: List[UUID], event: str):
    return (
        db.query(ThemeReviewEvent)
        .filter(ThemeReviewEvent.package_id.in_(package_ids), ThemeReviewEvent.event == event)
        .order_by(ThemeReviewEvent.created_at.desc(), ThemeReviewEvent.id.desc())
    )


def latest_notes(db: Session, package_ids: List[UUID]) -> Dict[UUID, ThemeReviewEvent]:
    """Each version's latest isolated-job note, by package id."""
    latest: Dict[UUID, ThemeReviewEvent] = {}
    if package_ids:
        for row in _events_query(db, package_ids, EVENT_REPORT_NOTE).all():
            latest.setdefault(row.package_id, row)
    return latest


def note_view(note: Optional[ThemeReviewEvent]) -> Optional[Dict[str, Any]]:
    """The note as the queue shows it: summary, outcome, who and when."""
    if note is None:
        return None
    detail = note.detail if isinstance(note.detail, Mapping) else {}
    return {
        "summary": detail.get("summary"), "outcome": detail.get("outcome"),
        "by": note.actor_user_id, "at": _iso(note.created_at),
    }


def _note_outcome(note: Optional[ThemeReviewEvent]) -> Optional[str]:
    view = note_view(note)
    return None if view is None else view["outcome"]


def approval_refusal(db: Session, package: ThemePackage, note: Optional[ThemeReviewEvent]) -> Optional[str]:
    """Why approve would refuse this version, or None when it may be approved."""
    if package.status != STATUS_PENDING:
        return NOT_PENDING.format(status=package.status, action="approved")
    if package.scan_verdict == VERDICT_BLOCKED:
        return SCAN_BLOCKED
    if package.scan_verdict == VERDICT_NOT_RUN:
        return SCAN_NOT_RUN
    if is_disabled(db, package):
        return REVOKED
    if config.IS_LOCAL_EDITION:
        return None  # the local edition never has the isolated stage (C-02b)
    outcome = _note_outcome(note)
    if outcome is None:
        return NOTE_MISSING
    return None if outcome == NOTE_PASS else NOTE_FAILED


def _approval(db: Session, package: ThemePackage, note: Optional[ThemeReviewEvent]) -> Dict[str, Any]:
    reason = approval_refusal(db, package, note)
    return {"allowed": reason is None, "reason": reason}


def list_pending(db: Session, limit: int) -> List[Dict[str, Any]]:
    """Pending Marketplace versions, oldest first, each with its latest note and whether approve would pass."""
    rows = (
        db.query(ThemePackage)
        .filter(ThemePackage.owner_type == MARKETPLACE_OWNER, ThemePackage.status == STATUS_PENDING)
        .order_by(ThemePackage.created_at.asc(), ThemePackage.id.asc())
        .limit(limit)
        .all()
    )
    notes = latest_notes(db, [row.id for row in rows])
    return [
        {**package_summary(row), "reportNote": note_view(notes.get(row.id)),
         "approval": _approval(db, row, notes.get(row.id))}
        for row in rows
    ]


def _history(db: Session, package_id: UUID) -> List[Dict[str, Any]]:
    rows = (
        db.query(ThemeReviewEvent)
        .filter(ThemeReviewEvent.package_id == package_id)
        .order_by(ThemeReviewEvent.created_at.asc(), ThemeReviewEvent.id.asc())
        .limit(HISTORY_LIMIT)
        .all()
    )
    return [
        {"event": row.event, "from": row.from_status, "to": row.to_status, "by": row.actor_user_id,
         "reason": row.reason, "at": _iso(row.created_at)}
        for row in rows
    ]


def _stored_verdict(package: ThemePackage) -> signature_verify.Verdict:
    """The report's check on the stored signature. A keys file that does not load reads ``invalid``
    (as ``listings._signature_verdict`` does), so the report still answers; the paste refuses."""
    try:
        return signature_verify.package_verdict(package)
    except signature_verify.SigningKeysError as bad:
        logger.error("Theme signing keys did not load; the report shows %s unverified: %s", package.id, bad)
        return signature_verify.Verdict(signature_verify.STATE_INVALID)


def report(db: Session, package_id: UUID) -> Dict[str, Any]:
    """One version's report: summary, scan report, latest note, approval state, history, signing."""
    package = queued_package(db, package_id)
    note = latest_notes(db, [package.id]).get(package.id)
    verdict = _stored_verdict(package)
    return {
        **package_summary(package),
        "manifest": package.manifest, "scanReport": package.scan_report,
        "reportNote": note_view(note), "approval": _approval(db, package, note),
        "history": _history(db, package.id),
        "signatureRequest": signing.request_view(signing.open_request(db, package.id)),
        "signatureVerified": verdict.verified, "keyId": verdict.key_id,
    }


def _record(
    db: Session, package: ThemePackage, event: str, from_status: str, actor_pk: Optional[int],
    reason: Optional[str] = None, detail: Optional[Dict[str, Any]] = None,
) -> None:
    db.add(ThemeReviewEvent(
        package_id=package.id, event=event, from_status=from_status, to_status=package.status,
        actor_user_id=actor_pk, reason=reason, detail=detail, created_at=signing.now_utc(),
    ))


def note_header(summary: str) -> List[str]:
    """The kit's header block: the lines before the first blank line (or ``Checks:``), with trailing
    whitespace and CR dropped, dedented by the block's common indentation (``plain_text`` already
    trimmed the first line). Never per line: the kit indents text from the package by six spaces
    so that it can't pass for a header line, and a per-line trim would undo that."""
    header: List[str] = []
    for line in (raw.rstrip() for raw in summary.splitlines()):
        if not line or line.strip() == NOTE_CHECKS_LINE:
            break
        header.append(line)
    indent = min((len(line) - len(line.lstrip()) for line in header[1:]), default=0)
    return header[:1] + [line[indent:] for line in header[1:]]


def note_names_this_version(summary: str, digest: str) -> bool:
    """True when the kit's header block holds this digest's line and one ``Outcome:`` line, ``Outcome: pass``.

    Only the header counts (``note_header``): an ``Outcome: pass`` under a check, or a second
    ``Outcome:`` line of any other value in the header, never passes.
    """
    header = note_header(summary)
    outcomes = [line for line in header if line.startswith(NOTE_OUTCOME_PREFIX)]
    return NOTE_DIGEST_LINE.format(digest=digest) in header and outcomes == [NOTE_OUTCOME_PASS_LINE]


def add_report_note(
    db: Session, package_id: UUID, summary: str, outcome: str, actor_pk: Optional[int],
) -> Dict[str, Any]:
    """Store the reviewer's pasted isolated-job summary for a pending version; a ``pass`` must name this version."""
    package = queued_package(db, package_id, lock=True)
    if package.status != STATUS_PENDING:
        raise ThemeReviewError(CONFLICT, NOT_PENDING.format(status=package.status, action="given a review note"))
    if outcome not in NOTE_OUTCOMES:
        raise ThemeReviewError(BAD_REQUEST, "The outcome is 'pass' or 'fail'.")
    text = plain_text(summary)
    if outcome == NOTE_PASS and not note_names_this_version(text, package.digest):
        raise ThemeReviewError(BAD_REQUEST, NOTE_NOT_THIS_VERSION.format(digest=package.digest))
    detail = {"summary": text, "outcome": outcome}
    _record(db, package, EVENT_REPORT_NOTE, package.status, actor_pk, detail=detail)
    return {**package_summary(package), "reportNote": {**detail, "by": actor_pk}}


def approve(db: Session, package_id: UUID, actor_pk: Optional[int]) -> Dict[str, Any]:
    """Approve a pending version that passed review, and record its signature request."""
    package = queued_package(db, package_id, lock=True)
    note = latest_notes(db, [package.id]).get(package.id)
    refusal = approval_refusal(db, package, note)
    if refusal is not None:
        raise ThemeReviewError(CONFLICT, refusal)
    signing.signing_payload(package)  # refuse before any write when the files can't be signed
    package.status, package.reviewed = STATUS_APPROVED, True
    _record(db, package, EVENT_APPROVED, STATUS_PENDING, actor_pk,
            detail={"scanVerdict": package.scan_verdict, "isolatedJob": _note_outcome(note)})
    offered = offer_listed_update(db, package)
    request = signing.request_signature(db, package, actor_pk)
    return {**package_summary(package), "signatureRequest": request, "updatesOffered": offered}


def reject(db: Session, package_id: UUID, reason: str, actor_pk: Optional[int]) -> Dict[str, Any]:
    """Reject a pending version with the reason the submitter will see."""
    package = queued_package(db, package_id, lock=True)
    if package.status != STATUS_PENDING:
        raise ThemeReviewError(CONFLICT, NOT_PENDING.format(status=package.status, action="rejected"))
    text = plain_text(reason)
    package.status = STATUS_REJECTED
    _record(db, package, EVENT_REJECTED, STATUS_PENDING, actor_pk, reason=text)
    return {**package_summary(package), "reason": text}


def _has_live_disable(db: Session, digest: str) -> bool:
    """A live ``local`` disable of ``digest`` stands already. A pulled (``feed``) one doesn't count:
    the feed may lift it, and the deactivation would go with it. The row is locked: a lift of it
    in flight finishes first and this then finds it lifted, or waits and then refuses (409)."""
    row = (
        db.query(ThemeRevocation.id)
        .filter(ThemeRevocation.digest == digest, ThemeRevocation.level == REVOKE_DISABLE,
                ThemeRevocation.lifted_at.is_(None), ThemeRevocation.origin == THEME_REVOCATION_ORIGIN_LOCAL)
        .with_for_update()
        .first()
    )
    return row is not None


def deactivate(db: Session, package_id: UUID, reason: str, actor_pk: Optional[int]) -> Dict[str, Any]:
    """Take an approved version off the Marketplace, write a ``disable`` revocation of its digest (unless
    a live ``local`` one stands) and withdraw the update offers its approval made (``withdraw_offers``).
    ``revocation_admin.lift_revocation`` refuses to lift the last live ``local`` disable of a
    deactivated version, so the deactivation can't be undone by lifting a revocation."""
    package = queued_package(db, package_id, lock=True)
    if package.status != STATUS_APPROVED:
        raise ThemeReviewError(CONFLICT, NOT_APPROVED)
    text = plain_text(reason)
    package.status = STATUS_DEACTIVATED
    if not _has_live_disable(db, package.digest):
        db.add(ThemeRevocation(digest=package.digest, level=REVOKE_DISABLE, reason=text, created_by=actor_pk))
    db.flush()
    withdrawn = withdraw_offers(db, package)
    _record(db, package, EVENT_DEACTIVATED, STATUS_APPROVED, actor_pk, reason=text,
            detail={"revocation": REVOKE_DISABLE, "digest": package.digest, "offersWithdrawn": withdrawn})
    return {**package_summary(package), "reason": text, "revocation": REVOKE_DISABLE}


def sign(db: Session, package_id: UUID, request_id: UUID, signature: str, actor_pk: Optional[int]) -> Dict[str, Any]:
    """Store the signature the kit's sign workflow returned for this version's open request, once an
    active Automatos signing key verifies it (``signature_verify.verify``); otherwise 400, nothing stored."""
    package = queued_package(db, package_id, lock=True)
    stored = signing.store_signature(db, package, request_id, signature, actor_pk, signature_verify.verify)
    return {**package_summary(package), "signature": stored}


def pending_limit(requested: Optional[int]) -> int:
    """The page size for the pending list, clamped to 1..PENDING_LIMIT_MAX."""
    if requested is None:
        return PENDING_LIMIT_DEFAULT
    return max(1, min(int(requested), PENDING_LIMIT_MAX))


__all__ = [
    "NOTE_OUTCOMES", "PENDING_LIMIT_DEFAULT", "PENDING_LIMIT_MAX", "ThemeReviewError", "add_report_note",
    "approval_refusal", "approve", "deactivate", "list_pending", "package_summary", "pending_limit",
    "reject", "report", "sign",
]
