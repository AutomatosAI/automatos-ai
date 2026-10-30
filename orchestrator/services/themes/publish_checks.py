"""PRD-261 — what refuses a theme publish (``publish.py``), and the shape of each refusal.

Every refusal is a :class:`PublishRefused`: an HTTP status and a code, answered in the Marketplace's
error shape, ``{code, message, errors: [{code, path, message}]}`` (the package checks' ``errors``
list, as ``POST /api/marketplace/themes/submit`` answers them, plus the code at the top). The
checks reuse the rules the rest of the Marketplace applies and never restate them:

* the publisher is one of ``THEME_PUBLISH_PUBLISHERS`` (403), and the key's service user may submit
  under it (``submissions.check_publisher``: someone else's publisher is 403);
* the service user's ``users`` row exists and is active (else the endpoint isn't set up: 503);
* the envelope verifies and names this package (``signed_upload.check_upload_signature``): 422;
* the summary passes the hosted approval note's rule (``review.note_names_this_version``): 422;
* no live ``disable`` revocation covers the version (``revocations.revocation_level``): 422;
* the version is above the highest held (``submissions.check_submission``): 409;
* the day's publishes (``review_limits.publish_quota_refusal``): 429;
* core's verdict is ``pass`` or ``review_required`` (D3, this endpoint only): ``blocked`` and
  ``not_run`` are 422.

Read-only: nothing here writes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from services.themes import review, review_limits, submissions
from services.themes.choices import ThemeInstallError
from services.themes.contract import ThemeError, echo, theme_error
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.publish_key import ACTOR_SETTING, PublishSettings
from services.themes.review_stage import VERDICT_BLOCKED, VERDICT_PASS, VERDICT_REVIEW
from services.themes.revocations import REVOKE_DISABLE, revocation_level
from services.themes.signature_verify import SigningKeysError
from services.themes.signed_upload import SIGNATURE_PART, SignatureRefused, VerifiedSignature, check_upload_signature

logger = logging.getLogger(__name__)

UNAUTHORIZED, FORBIDDEN, CONFLICT, UNPROCESSABLE, TOO_MANY, UNAVAILABLE = 401, 403, 409, 422, 429, 503
PUBLISHABLE_VERDICTS = (VERDICT_PASS, VERDICT_REVIEW)  # D3: core's own verdict approves on this endpoint only
STATUS_PENDING, STATUS_APPROVED = "pending", "approved"
ID_PATH, SUMMARY_PATH = "theme.json#id", "review_summary"

NOT_CONFIGURED_CODE = "PUBLISH_NOT_CONFIGURED"
KEY_REFUSED_CODE = "PUBLISH_KEY_REFUSED"
PUBLISHER_NOT_LISTED_CODE = "PUBLISHER_NOT_LISTED"
PUBLISHER_TAKEN_CODE = "PUBLISHER_TAKEN"
VERSION_NOT_HIGHER_CODE = "VERSION_NOT_HIGHER"
VERSION_CLOSED_CODE = "VERSION_CLOSED"
PUBLISH_BUSY_CODE = "PUBLISH_BUSY"
SUMMARY_CODE = "SUMMARY_NOT_THIS_VERSION"
REVIEW_BLOCKED_CODE = "REVIEW_BLOCKED"
REVIEW_NOT_RUN_CODE = "REVIEW_NOT_RUN"
REVIEW_REFUSED_CODE = "REVIEW_REFUSED"
DIGEST_DISABLED_CODE = "DIGEST_DISABLED"
QUOTA_CODE = "PUBLISH_QUOTA"
REVIEWS_BUSY_CODE = "REVIEWS_BUSY"
STORE_FAILED_CODE = "STORE_FAILED"
SIGNING_KEYS_CODE = "SIGNING_KEYS_UNREADABLE"

NOT_CONFIGURED = "Theme publishing isn't set up on this server."
ACTOR_MISSING = "Theme publishing isn't set up on this server: the user it acts as is missing."
KEY_REFUSED = "That isn't the theme publish key."
PUBLISHER_NOT_LISTED = 'The publish key can\'t publish under the "{publisher}" publisher.'
VERSION_CLOSED = "{theme_id} {version} with these files was {status} in the review queue. Raise the version."
PUBLISH_BUSY = "This version is being published or submitted already. Try again in a moment."
SUMMARY_NOT_THIS_VERSION = (
    "The review summary must be the kit review's summary for this package, with the lines "
    '"Package digest:  {digest}" and "Outcome: pass".'
)
REVIEW_BLOCKED = "Core's automated review blocked this version, so it wasn't published."
REVIEW_NOT_RUN = "Core's automated review didn't run on this version, so it wasn't published."
DIGEST_DISABLED = "This package's files are revoked (disabled), so it can't be published."
STORE_FAILED = "The theme's files couldn't be stored. Try again."

_ACTOR_ROW = text("SELECT is_active FROM users WHERE id = :user_id")


class PublishRefused(Exception):
    """A refused publish: the HTTP status, a code and a plain message; ``errors`` in the package checks' shape."""

    def __init__(self, status_code: int, code: str, message: str, path: str = "",
                 errors: Optional[List[ThemeError]] = None, headers: Optional[Mapping[str, str]] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.errors = list(errors) if errors else [theme_error(code, path, message)]
        self.headers = dict(headers) if headers else None

    @property
    def detail(self) -> Dict[str, Any]:
        """The answer's ``detail``: ``{code, message, errors}``."""
        return {"code": self.code, "message": self.message, "errors": self.errors}


def not_configured(message: str = NOT_CONFIGURED) -> PublishRefused:
    """503: the endpoint's settings, or its service user, are missing."""
    return PublishRefused(UNAVAILABLE, NOT_CONFIGURED_CODE, message)


def key_refused() -> PublishRefused:
    """401: no bearer, or not the publish key."""
    return PublishRefused(UNAUTHORIZED, KEY_REFUSED_CODE, KEY_REFUSED, headers={"WWW-Authenticate": "Bearer"})


def busy() -> PublishRefused:
    """409: another publish or submission of these bytes holds the Marketplace claim."""
    return PublishRefused(CONFLICT, PUBLISH_BUSY_CODE, PUBLISH_BUSY)


def reviews_busy() -> PublishRefused:
    """503 with ``Retry-After``: every review slot in this process is taken."""
    return PublishRefused(UNAVAILABLE, REVIEWS_BUSY_CODE, review_limits.REVIEWS_BUSY,
                          headers=review_limits.busy_headers())


def store_failed() -> PublishRefused:
    """503: the files couldn't be written; nothing was committed."""
    return PublishRefused(UNAVAILABLE, STORE_FAILED_CODE, STORE_FAILED)


def signing_keys_refusal(broken: SigningKeysError) -> PublishRefused:
    """503: the signing keys file doesn't load (its message echoes nothing from the file)."""
    return PublishRefused(UNAVAILABLE, SIGNING_KEYS_CODE, str(broken))


def submission_refusal(clash: Exception) -> PublishRefused:
    """A ``submissions`` refusal: someone else's publisher is 403, any other clash 409."""
    if isinstance(clash, (submissions.PublisherTaken, submissions.PublisherReserved)):
        return PublishRefused(FORBIDDEN, PUBLISHER_TAKEN_CODE, str(clash), path=ID_PATH)
    return PublishRefused(CONFLICT, VERSION_NOT_HIGHER_CODE, str(clash), path=ID_PATH)


def review_refusal(refusal: ThemeInstallError) -> PublishRefused:
    """422: the review queue or the signing path refused the version (``review.approve``, ``review.sign``)."""
    return PublishRefused(UNPROCESSABLE, REVIEW_REFUSED_CODE, refusal.message)


def require_listed(publisher: str, settings: PublishSettings) -> None:
    """403 unless ``publisher`` is one of ``THEME_PUBLISH_PUBLISHERS``."""
    if publisher not in settings.publishers:
        raise PublishRefused(FORBIDDEN, PUBLISHER_NOT_LISTED_CODE,
                             PUBLISHER_NOT_LISTED.format(publisher=echo(publisher)), path=ID_PATH)


def require_actor(db: Session, actor_pk: int) -> None:
    """503 unless ``THEME_PUBLISH_ACTOR`` names an active ``users`` row (a NULL ``is_active`` is not active)."""
    row = db.execute(_ACTOR_ROW, {"user_id": actor_pk}).fetchone()
    if row is None or not row[0]:
        logger.warning("[Themes] %s names no active users row: theme publishing is off", ACTOR_SETTING)
        raise not_configured(ACTOR_MISSING)


def require_owner(db: Session, publisher: str, submitter: submissions.Submitter) -> None:
    """403 when the publisher belongs to someone else (read-only; the save claims it under a row lock)."""
    try:
        submissions.check_publisher(db, publisher, submitter)
    except (submissions.PublisherTaken, submissions.PublisherReserved) as taken:
        raise submission_refusal(taken) from taken


def verified_signature(db: Session, checked: CheckedThemePackage, envelope: bytes) -> VerifiedSignature:
    """The envelope, verified by an active key over this package and naming its id, version and digest; else 422."""
    try:
        return check_upload_signature(db, checked, envelope)
    except SignatureRefused as refused:
        error = refused.error
        raise PublishRefused(UNPROCESSABLE, error["code"], error["message"], path=SIGNATURE_PART,
                             errors=[error]) from refused
    except SigningKeysError as broken:
        raise signing_keys_refusal(broken) from broken


def checked_summary(summary: str, digest: str) -> str:
    """The kit review's summary, trimmed, when it names this digest with ``Outcome: pass`` (the hosted note's rule)."""
    refusal = PublishRefused(UNPROCESSABLE, SUMMARY_CODE, SUMMARY_NOT_THIS_VERSION.format(digest=digest),
                             path=SUMMARY_PATH)
    try:
        trimmed = review.plain_text(summary)
    except review.ThemeReviewError as bad:
        raise refusal from bad
    if not review.note_names_this_version(trimmed, digest):
        raise refusal
    return trimmed


def require_not_disabled(db: Session, checked: CheckedThemePackage) -> None:
    """422 when a live ``disable`` revocation covers this digest or this version."""
    manifest = checked.manifest
    if revocation_level(db, checked.digest, manifest["id"], manifest["version"]) == REVOKE_DISABLE:
        raise PublishRefused(UNPROCESSABLE, DIGEST_DISABLED_CODE, DIGEST_DISABLED)


def require_new_version(db: Session, checked: CheckedThemePackage, submitter: submissions.Submitter) -> None:
    """409 unless the version is above every version of the theme held (Q9 never applies to the service user)."""
    try:
        submissions.check_submission(db, checked.manifest, checked.digest, submitter)
    except (submissions.SubmissionConflict, submissions.PublisherReserved) as clash:
        raise submission_refusal(clash) from clash


def require_quota(db: Session, actor_pk: int) -> None:
    """429 past ``THEME_PUBLISH_PER_DAY`` versions in 24 hours."""
    refusal = review_limits.publish_quota_refusal(db, actor_pk)
    if refusal is not None:
        raise PublishRefused(TOO_MANY, QUOTA_CODE, refusal)


def require_verdict(verdict: Optional[str]) -> None:
    """422 unless core's verdict is ``pass`` or ``review_required`` (D3)."""
    if verdict in PUBLISHABLE_VERDICTS:
        return
    if verdict == VERDICT_BLOCKED:
        raise PublishRefused(UNPROCESSABLE, REVIEW_BLOCKED_CODE, REVIEW_BLOCKED)
    raise PublishRefused(UNPROCESSABLE, REVIEW_NOT_RUN_CODE, REVIEW_NOT_RUN)


def require_open(package: Any) -> None:
    """For bytes already held: a pending version goes on when its verdict allows; an approved one is signed;
    a rejected or deactivated one is 409."""
    if package.status == STATUS_PENDING:
        require_verdict(package.scan_verdict)
        return
    if package.status != STATUS_APPROVED:
        message = VERSION_CLOSED.format(theme_id=echo(package.theme_id), version=package.version, status=package.status)
        raise PublishRefused(CONFLICT, VERSION_CLOSED_CODE, message, path=ID_PATH)
