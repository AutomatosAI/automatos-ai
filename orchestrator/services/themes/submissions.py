"""PRD-261 C-01a — a theme submitted to the Marketplace, in the database.

A submission is a ``theme_packages`` row with ``owner_type`` 'marketplace' (no workspace),
``status`` 'pending' and ``reviewed`` false, plus its review events (``submitted`` and
``scanned``). A person reviews every version (D3, 19 Oct); nothing here publishes.

Rules, in the order they are checked:

* the platform switch ``THEMES_MARKETPLACE`` (``on|off``) must be ``on``; an unknown value is
  read as ``off`` (fail closed);
* publisher ownership (``theme_publishers``): a reserved publisher (``automatos``, or a row
  marked ``reserved``) takes submissions only from ``caller_is_admin``; any other publisher is
  claimed by its first submitter, and a later submission under it by someone else is a
  :class:`PublisherTaken` (409);
* Q9, one open submission per author, is the flag ``THEMES_ONE_OPEN_SUBMISSION`` (off until
  Gerard answers; C1-wire adds the name to ``config.py``): when on, an author with a pending
  submission of other bytes gets a :class:`SubmissionConflict`;
* a version the Marketplace already holds with other files, or one not higher than the
  theme's highest submitted version, is a :class:`SubmissionConflict`.

The same bytes submitted again return the existing row (``created`` false), never a 409. The
Marketplace's unique indexes are separate from a workspace's, so a private upload and a
submission of the same bytes are two rows that both succeed.

Identity: the submitter is ``users.id``. In SaaS a submission needs one (a person owns a
publisher); the local edition's single anonymous operator may have none, and then owns its
publishers as "no user", which only it can match.

Synchronous (a plain SQLAlchemy session): called from plain-``def`` routes only.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional

from sqlalchemy.orm import Session

from config import config
from core.models.themes import ThemePackage, ThemePublisher, ThemeReviewEvent
from services.themes.contract import RESERVED_PUBLISHERS, echo
from services.themes.package import ThemeFile
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.records import publisher_of, version_key

logger = logging.getLogger(__name__)

THEME_SUBMIT_ROUTE = "/api/marketplace/themes/submit"
MARKETPLACE_SETTING = "THEMES_MARKETPLACE"
ONE_OPEN_SUBMISSION_SETTING = "THEMES_ONE_OPEN_SUBMISSION"
ONE_OPEN_SUBMISSION_DEFAULT = False
SWITCH_ON, SWITCH_OFF = "on", "off"
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
OWNER_MARKETPLACE = "marketplace"
STATUS_PENDING = "pending"
EVENT_SUBMITTED, EVENT_SCANNED = "submitted", "scanned"

MARKETPLACE_OFF = "The theme Marketplace is turned off on this platform."
NO_PERSON = "Sign in as a person to submit a theme to the Marketplace."

# The automated review stage: the package's files and its size in bytes → the scan report,
# whose "verdict" is one of THEME_SCAN_VERDICTS. C-02a's stage plugs in here (the router's
# ``submission_review_stage``); until then the A-07a report, verdict "not_run".
ReviewStage = Callable[[Mapping[str, ThemeFile], int], Dict[str, Any]]


class SubmissionConflict(ValueError):
    """The submission clashes with what the Marketplace holds; the message says how to fix it (409)."""


class PublisherTaken(SubmissionConflict):
    """The publisher belongs to another submitter (409)."""


class PublisherReserved(PermissionError):
    """The publisher takes submissions only from an Automatos admin (403)."""


class SavedSubmission(NamedTuple):
    """The stored submission and what the call did."""

    package: ThemePackage
    created: bool  # False: these exact bytes were already submitted


class Submitter(NamedTuple):
    """Who submits: their ``users.id`` (None for the local operator) and whether they are a platform admin."""

    user_pk: Optional[int]
    is_admin: bool


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------


def marketplace_switch() -> str:
    """``THEMES_MARKETPLACE``; a value it doesn't know is read as ``off`` and logged."""
    raw = getattr(config, MARKETPLACE_SETTING, SWITCH_ON)
    value = str(raw or SWITCH_ON).strip().lower()
    if value not in (SWITCH_ON, SWITCH_OFF):
        logger.warning("[Themes] %s=%r is not on or off: the Marketplace is off", MARKETPLACE_SETTING, raw)
        return SWITCH_OFF
    return value


def one_open_submission() -> bool:
    """Q9's flag: True when an author may hold only one pending submission at a time."""
    raw = getattr(config, ONE_OPEN_SUBMISSION_SETTING, ONE_OPEN_SUBMISSION_DEFAULT)
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in TRUE_WORDS


# ---------------------------------------------------------------------------
# Reads (Marketplace rows only)
# ---------------------------------------------------------------------------


def _marketplace(db: Session):
    return db.query(ThemePackage).filter(ThemePackage.owner_type == OWNER_MARKETPLACE)


def submission_by_digest(db: Session, digest: str) -> Optional[ThemePackage]:
    """The Marketplace package with these exact bytes, or None."""
    return _marketplace(db).filter(ThemePackage.digest == digest).first()


def _same_user(column, user_pk: Optional[int]):
    return column.is_(None) if user_pk is None else column == user_pk


def _open_submission(db: Session, user_pk: Optional[int]) -> Optional[ThemePackage]:
    return _marketplace(db).filter(
        ThemePackage.status == STATUS_PENDING, _same_user(ThemePackage.submitted_by, user_pk),
    ).first()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _reserved_refusal(publisher: str) -> PublisherReserved:
    return PublisherReserved(f'Only an Automatos admin can submit a theme under the "{echo(publisher)}" publisher.')


def _publisher_row(db: Session, publisher: str, lock: bool) -> Optional[ThemePublisher]:
    query = db.query(ThemePublisher).filter(ThemePublisher.publisher == publisher)
    return (query.with_for_update() if lock else query).first()


def _check_publisher(row: Optional[ThemePublisher], publisher: str, submitter: Submitter) -> None:
    """Raise unless ``submitter`` may submit under ``publisher`` (``row`` is its record, if any)."""
    reserved = publisher in RESERVED_PUBLISHERS or (row is not None and row.reserved)
    if reserved:
        if not submitter.is_admin:
            raise _reserved_refusal(publisher)
        return
    if row is not None and row.owner_user_id != submitter.user_pk:
        raise PublisherTaken(
            f'The publisher "{echo(publisher)}" belongs to another author. '
            "Change the publisher part of the id in theme.json and submit again."
        )


def claim_publisher(db: Session, publisher: str, submitter: Submitter) -> None:
    """Check the caller may submit under ``publisher`` (row locked), recording it for its first submitter.

    Raises :class:`PublisherReserved` (a reserved publisher, caller not an admin) or
    :class:`PublisherTaken` (another submitter owns it).
    """
    row = _publisher_row(db, publisher, lock=True)
    _check_publisher(row, publisher, submitter)
    if row is None:
        reserved = publisher in RESERVED_PUBLISHERS
        db.add(ThemePublisher(publisher=publisher, owner_user_id=submitter.user_pk, reserved=reserved))
        db.flush()


def _check_open_submission(db: Session, submitter: Submitter) -> None:
    if not one_open_submission():
        return
    open_row = _open_submission(db, submitter.user_pk)
    if open_row is not None:
        raise SubmissionConflict(
            f"You already have {echo(open_row.theme_id)} {open_row.version} waiting for review. "
            "Submit again once it has been reviewed."
        )


def _check_new_version(db: Session, manifest: Mapping[str, Any]) -> None:
    """Refuse a version the Marketplace holds with other files, or one not higher than the highest held."""
    theme_id, version = manifest["id"], manifest["version"]
    held = [row.version for row in _marketplace(db).filter(ThemePackage.theme_id == theme_id).all()]
    if version in held:
        raise SubmissionConflict(
            f"{echo(theme_id)} {version} has already been submitted with different files. "
            "Raise the version in theme.json and submit again."
        )
    highest = max(held, key=version_key, default=None)
    if highest is not None and version_key(version) < version_key(highest):
        raise SubmissionConflict(
            f"{echo(theme_id)} {highest} has already been submitted. Submit a higher version."
        )


def check_submission(db: Session, manifest: Mapping[str, Any], digest: str, submitter: Submitter) -> None:
    """Every refusal :func:`save_submission` would make, read-only, so the review stage never runs for nothing.

    These exact bytes submitted already pass when the caller may submit under their publisher.
    Raises :class:`PublisherReserved` or :class:`SubmissionConflict`.
    """
    publisher = publisher_of(manifest["id"])
    _check_publisher(_publisher_row(db, publisher, lock=False), publisher, submitter)
    if submission_by_digest(db, digest) is not None:
        return
    _check_open_submission(db, submitter)
    _check_new_version(db, manifest)


def submission_conflict(db: Session, manifest: Mapping[str, Any], digest: str, submitter: Submitter) -> Optional[str]:
    """After a concurrent submission won a unique index: the clash it left, as a message, or None.

    The losing call's own checks ran before the winner committed; run again, they see its rows.
    PostgreSQL raises the unique violation only once the winner has committed (an insert that
    meets an uncommitted key waits for that transaction), so the re-check never races it.
    """
    try:
        check_submission(db, manifest, digest, submitter)
    except (SubmissionConflict, PublisherReserved) as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _new_submission(checked: CheckedThemePackage, prefix: str, report: Dict[str, Any],
                    user_pk: Optional[int]) -> ThemePackage:
    manifest = checked.manifest
    return ThemePackage(
        theme_id=manifest["id"], version=manifest["version"], owner_type=OWNER_MARKETPLACE, workspace_id=None,
        digest=checked.digest, manifest=manifest, file_manifest=checked.file_manifest, storage_prefix=prefix,
        status=STATUS_PENDING, reviewed=False, scan_report=report, scan_verdict=report["verdict"],
        submitted_by=user_pk,
    )


def _events(row: ThemePackage, user_pk: Optional[int]) -> List[ThemeReviewEvent]:
    return [
        ThemeReviewEvent(package_id=row.id, event=EVENT_SUBMITTED, to_status=STATUS_PENDING, actor_user_id=user_pk),
        ThemeReviewEvent(
            package_id=row.id, event=EVENT_SCANNED, from_status=STATUS_PENDING, to_status=STATUS_PENDING,
            actor_user_id=user_pk, detail={"verdict": row.scan_verdict},
        ),
    ]


def _commit_after_store(db: Session, store: Callable[[], None]) -> None:
    """Write the files once the rows are flushed, then commit; a failed write commits nothing."""
    try:
        store()
    except Exception:
        logger.exception("[Themes] Storing a submission's files failed: rolling its rows back")
        db.rollback()
        raise
    db.commit()


def save_submission(
    db: Session, checked: CheckedThemePackage, submitter: Submitter, prefix: str,
    report: Dict[str, Any], store: Callable[[], None],
) -> SavedSubmission:
    """Store a checked package as a pending Marketplace submission with its stage verdict.

    ``store`` writes the files once the rows are flushed (a concurrent duplicate is already
    refused by the unique indexes) and before the commit. Raises :class:`PublisherReserved`
    or :class:`SubmissionConflict`; the caller rolls back.
    """
    existing = submission_by_digest(db, checked.digest)
    if existing is not None:
        claim_publisher(db, publisher_of(existing.theme_id), submitter)
        return SavedSubmission(existing, created=False)
    claim_publisher(db, publisher_of(checked.manifest["id"]), submitter)
    _check_open_submission(db, submitter)
    _check_new_version(db, checked.manifest)
    row = _new_submission(checked, prefix, report, submitter.user_pk)
    db.add(row)
    db.flush()
    db.add_all(_events(row, submitter.user_pk))
    db.flush()
    _commit_after_store(db, store)
    db.refresh(row)
    return SavedSubmission(row, created=True)
