"""PRD-266 §4.3 (K11) — mint the first pairing code from the server console.

With ``LOCAL_REMOTE_ACCESS=paired`` a request with no credential is refused, so the
operator's first browser (or phone) cannot ask Studio for a code. This mints one for the
local operator (``LOCAL_OPERATOR_EMAIL``'s ``users`` row) in ``DEFAULT_WORKSPACE_ID``,
through ``services.mobile.pairing``::

    docker compose exec backend python -m scripts.mint_pairing_code [--name NAME] [--review --ttl-hours N]

It prints the code, when it expires (ten minutes) and its pair link, the universal link
``{FRONTEND_URL}/go/pair?api=…&code=…`` (never a custom scheme). The link names
``https://<PUBLIC_API_HOST>`` once the operator has set it, else ``BACKEND_URL``: there is no
request here to read a base URL from (K15). ``--name`` labels the device until it pairs with
a name of its own. Once a browser is paired, it mints codes from Settings → Phones & devices.

``--review --ttl-hours N`` (D13) mints App Review's code: one code, single-use like every
other, that lasts N hours (1 to ``REVIEW_CODE_MAX_HOURS``, two weeks) so the reviewer can pair
a phone with the public demo install without asking for a fresh one. It writes one
``audit_logs`` row in the same commit (the operator, the expiry, the device name; never the
code or its hash). ``--ttl-hours`` without ``--review`` is refused: every other code lasts
ten minutes, and nothing but this script can make one last longer.

Refuses (exit 2) on the saas edition, where phones sign in with Clerk, without
``DEFAULT_WORKSPACE_ID``, and on a local install that is not in paired mode, saying how to
turn it on: ``POST /api/v1/mobile/pair`` answers only in paired mode, so a code minted
there could never be used. Exit 1 when the operator's row is missing or the database
fails. A database error is printed by its type only, because its text carries the
statement's bound parameters (the code's hash among them); the code itself is printed once,
to this console, and never logged.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence, TextIO
from urllib.parse import urlsplit
from uuid import UUID

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.pairing import PAIRING_CODE_TTL_SECONDS  # noqa: E402
from core.auth.principal import resolve_user_pk  # noqa: E402
from core.database.database import SessionLocal  # noqa: E402
from core.models.mobile_devices import NAME_LENGTH  # noqa: E402
from core.workspaces.audit import AuditService  # noqa: E402
from services.mobile import pairing  # noqa: E402
from services.mobile.remote_access import MODE_PAIRED, is_loopback  # noqa: E402
from services.mobile.urls import MobileUrlError, api_url  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
OPERATOR_ROLE = "super_admin"
SECONDS_PER_HOUR = 3600
# D13: the reviewer pairs days after the build is submitted; a review code lasts two weeks at most.
REVIEW_CODE_MAX_HOURS = 336
REVIEW_CODE_ACTION = "mobile:review_pairing_code_issued"
REVIEW_CODE_RESOURCE = "pairing_code"
REVIEW_CODE_SOURCE = "console"

SAAS_REFUSAL = (
    "Pairing codes are a local-edition feature (AUTH_EDITION=local). On the hosted edition "
    "phones sign in with their Automatos account instead."
)
NO_WORKSPACE_REFUSAL = "DEFAULT_WORKSPACE_ID is unset or not a UUID: there is no local workspace to pair with."
NOT_PAIRED_REFUSAL = (
    "This install runs LOCAL_REMOTE_ACCESS={mode}, and POST /api/v1/mobile/pair answers only "
    "in paired mode, so a code minted now could not be used. To turn pairing on, set "
    "LOCAL_REMOTE_ACCESS=paired in .env, restart the backend (docker compose up -d backend), "
    "then run this again."
)
NO_OPERATOR = (
    "The local operator's users row (LOCAL_OPERATOR_EMAIL) is missing: the first-run seed "
    "has not run. Start the stack once (make up), then run this again."
)
DATABASE_FAILED = "Could not mint a pairing code: the database refused it ({kind}). Nothing was written."
AUDIT_FAILED = "Could not mint a review code: its audit row could not be written. Nothing was written."
TTL_HOURS_RANGE = f"--ttl-hours must be a whole number of hours from 1 to {REVIEW_CODE_MAX_HOURS}"
REVIEW_NEEDS_TTL = "--review needs --ttl-hours N: how many hours the review code lasts"
TTL_NEEDS_REVIEW = "--ttl-hours is only for --review: every other code lasts {minutes} minutes"
REPORT = (
    "Pairing code: {code}\n"
    "Expires at:   {expires_at} (in {lifetime}; it works once)\n"
    "Pair link:    {link}\n"
    "\n"
    "Type the code into \"Pair this browser\" on the web app, or open the link on your phone."
)
REVIEW_NOTE = (
    "\nThis is App Review's code: put the code and the pair link in the review notes. It "
    "pairs one phone, once, and minting it is in the audit log."
)
LOOPBACK_LINK_HINT = (
    "\nThe link's API address ({api}) is this machine only, so a phone cannot open it: set "
    "PUBLIC_API_HOST to the https host phones reach this API on, restart the backend, and "
    "mint again. The code itself still pairs a browser on this machine."
)


class OperatorMissing(RuntimeError):
    """``LOCAL_OPERATOR_EMAIL`` has no ``users`` row to mint the code for."""


class ReviewAuditFailed(RuntimeError):
    """A review code's audit row could not be written, so the code was rolled back with it."""


def _device_name(value: str) -> str:
    name = value.strip()
    if not name or len(name) > NAME_LENGTH:
        raise argparse.ArgumentTypeError(f"--name must be 1 to {NAME_LENGTH} characters")
    return name


def _ttl_hours(value: str) -> int:
    try:
        hours = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(TTL_HOURS_RANGE) from None
    if not 1 <= hours <= REVIEW_CODE_MAX_HOURS:
        raise argparse.ArgumentTypeError(TTL_HOURS_RANGE)
    return hours


def _args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mint a one-time code that pairs a browser or phone")
    parser.add_argument("--name", type=_device_name, default=None,
                        help="A label for the device, until it pairs with a name of its own")
    parser.add_argument("--review", action="store_true",
                        help="Mint App Review's long-lived code (needs --ttl-hours)")
    parser.add_argument("--ttl-hours", type=_ttl_hours, default=None, metavar="N",
                        help=f"How many hours the review code lasts, 1 to {REVIEW_CODE_MAX_HOURS}")
    args = parser.parse_args(argv)
    if args.review and args.ttl_hours is None:
        parser.error(REVIEW_NEEDS_TTL)
    if args.ttl_hours is not None and not args.review:
        parser.error(TTL_NEEDS_REVIEW.format(minutes=PAIRING_CODE_TTL_SECONDS // 60))
    return args


def _workspace_id() -> Optional[UUID]:
    try:
        return UUID(str(config.DEFAULT_WORKSPACE_ID or "").strip())
    except ValueError:
        return None


def refusal() -> Optional[str]:
    """Why this install may not mint a code now, or ``None`` when it may."""
    if not config.IS_LOCAL_EDITION:
        return SAAS_REFUSAL
    if _workspace_id() is None:
        return NO_WORKSPACE_REFUSAL
    if config.LOCAL_REMOTE_ACCESS != MODE_PAIRED:
        return NOT_PAIRED_REFUSAL.format(mode=config.LOCAL_REMOTE_ACCESS)
    return None


def _operator_pk(db: Session, workspace_id: UUID) -> int:
    """The local operator's ``users.id``, through the one resolver (``resolve_user_pk``)."""
    operator = RequestContext(
        workspace_id=workspace_id,
        user=UserContext(email=config.LOCAL_OPERATOR_EMAIL, system_role=OPERATOR_ROLE),
    )
    user_pk = resolve_user_pk(db, operator)
    if user_pk is None:
        raise OperatorMissing(NO_OPERATOR)
    return user_pk


def _audit_review_code(db: Session, workspace_id: UUID, operator_pk: int, issued: pairing.PairingCode,
                       name: Optional[str], ttl_hours: int) -> None:
    """Write a review code's one ``audit_logs`` row: the operator, its expiry, the device name.

    Never the code or its hash. ``AuditService.log`` commits the code's row with it; when it
    cannot, it has rolled both back and answers ``None``, and this raises ``ReviewAuditFailed``.
    """
    entry = AuditService(db).log(
        workspace_id=str(workspace_id),
        user_id=operator_pk,
        action=REVIEW_CODE_ACTION,
        resource_type=REVIEW_CODE_RESOURCE,
        resource_name=name,
        details={
            "expires_at": issued.expires_at.isoformat(),
            "ttl_hours": ttl_hours,
            "device_name": name,
            "source": REVIEW_CODE_SOURCE,
        },
    )
    if entry is None:
        raise ReviewAuditFailed(AUDIT_FAILED)


def mint(db: Session, name: Optional[str] = None, ttl_hours: Optional[int] = None) -> pairing.PairingCode:
    """Mint and commit a code for the local operator in ``DEFAULT_WORKSPACE_ID``.

    ``ttl_hours`` (``--review``) makes it App Review's code: it lasts that many hours, and its
    audit row is committed with it. Without it the code lasts ``PAIRING_CODE_TTL_SECONDS``.
    A database error rolls back and is re-raised for ``main`` to report by its type.
    """
    workspace_id = _workspace_id()
    try:
        operator_pk = _operator_pk(db, workspace_id)
        issued = pairing.create_pairing_code(
            db, workspace_id, operator_pk, request_base_url=config.BACKEND_URL, name=name,
            ttl_seconds=None if ttl_hours is None else ttl_hours * SECONDS_PER_HOUR,
        )
        if ttl_hours is not None:
            _audit_review_code(db, workspace_id, operator_pk, issued, name, ttl_hours)
        db.commit()
        return issued
    except SQLAlchemyError:
        db.rollback()
        raise


def _lifetime(ttl_hours: Optional[int]) -> str:
    """How long the code lasts, in words: ten minutes, or a review code's hours."""
    if ttl_hours is None:
        return f"{PAIRING_CODE_TTL_SECONDS // 60} minutes"
    return f"{ttl_hours} hour" if ttl_hours == 1 else f"{ttl_hours} hours"


def report(issued: pairing.PairingCode, ttl_hours: Optional[int] = None) -> str:
    """The console's lines for a minted code: the code, its expiry and its pair link.

    A review code (``ttl_hours``) says so. A link whose API address is loopback
    (``BACKEND_URL``'s default, while ``PUBLIC_API_HOST`` is unset) gets a line saying a phone
    cannot use it, and how to fix that.
    """
    lines = REPORT.format(
        code=issued.code,
        expires_at=issued.expires_at.isoformat(timespec="seconds"),
        lifetime=_lifetime(ttl_hours),
        link=issued.qr_payload,
    )
    if ttl_hours is not None:
        lines += REVIEW_NOTE
    api = api_url(config.BACKEND_URL)
    if is_loopback(urlsplit(api).hostname):
        return lines + LOOPBACK_LINK_HINT.format(api=api)
    return lines


def main(argv: Optional[Sequence[str]] = None, out: TextIO = sys.stdout, err: TextIO = sys.stderr) -> int:
    """Mint one code and print it; the exit code says whether it was refused or failed."""
    args = _args(argv)
    refused = refusal()
    if refused:
        print(refused, file=err)
        return EXIT_REFUSED
    db = SessionLocal()
    try:
        issued = mint(db, args.name, args.ttl_hours)
    except (OperatorMissing, MobileUrlError, ReviewAuditFailed) as failure:
        print(str(failure), file=err)
        return EXIT_FAILED
    except SQLAlchemyError as failure:
        print(DATABASE_FAILED.format(kind=type(failure).__name__), file=err)
        return EXIT_FAILED
    finally:
        db.close()
    print(report(issued, args.ttl_hours), file=out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
