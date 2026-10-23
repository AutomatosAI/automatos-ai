"""PRD-266 §4.3 (K11) — mint the first pairing code from the server console.

With ``LOCAL_REMOTE_ACCESS=paired`` a request with no credential is refused, so the
operator's first browser (or phone) cannot ask Studio for a code. This mints one for the
local operator (``LOCAL_OPERATOR_EMAIL``'s ``users`` row) in ``DEFAULT_WORKSPACE_ID``,
through ``services.mobile.pairing``::

    docker compose exec backend python -m scripts.mint_pairing_code [--name NAME]

It prints the code, when it expires (ten minutes) and its pair link, the universal link
``{FRONTEND_URL}/go/pair?api=…&code=…`` (never a custom scheme). The link names
``https://<PUBLIC_API_HOST>`` once the operator has set it, else ``BACKEND_URL``: there is no
request here to read a base URL from (K15). ``--name`` labels the device until it pairs with
a name of its own. Once a browser is paired, it mints codes from Settings → Phones & devices.

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
from services.mobile import pairing  # noqa: E402
from services.mobile.remote_access import MODE_PAIRED, is_loopback  # noqa: E402
from services.mobile.urls import MobileUrlError, api_url  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
OPERATOR_ROLE = "super_admin"

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
REPORT = (
    "Pairing code: {code}\n"
    "Expires at:   {expires_at} (in {minutes} minutes; it works once)\n"
    "Pair link:    {link}\n"
    "\n"
    "Type the code into \"Pair this browser\" on the web app, or open the link on your phone."
)
LOOPBACK_LINK_HINT = (
    "\nThe link's API address ({api}) is this machine only, so a phone cannot open it: set "
    "PUBLIC_API_HOST to the https host phones reach this API on, restart the backend, and "
    "mint again. The code itself still pairs a browser on this machine."
)


class OperatorMissing(RuntimeError):
    """``LOCAL_OPERATOR_EMAIL`` has no ``users`` row to mint the code for."""


def _device_name(value: str) -> str:
    name = value.strip()
    if not name or len(name) > NAME_LENGTH:
        raise argparse.ArgumentTypeError(f"--name must be 1 to {NAME_LENGTH} characters")
    return name


def _args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mint a one-time code that pairs a browser or phone")
    parser.add_argument("--name", type=_device_name, default=None,
                        help="A label for the device, until it pairs with a name of its own")
    return parser.parse_args(argv)


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


def mint(db: Session, name: Optional[str] = None) -> pairing.PairingCode:
    """Mint and commit a code for the local operator in ``DEFAULT_WORKSPACE_ID``.

    A database error rolls back and is re-raised for ``main`` to report by its type.
    """
    workspace_id = _workspace_id()
    try:
        issued = pairing.create_pairing_code(
            db, workspace_id, _operator_pk(db, workspace_id),
            request_base_url=config.BACKEND_URL, name=name,
        )
        db.commit()
        return issued
    except SQLAlchemyError:
        db.rollback()
        raise


def report(issued: pairing.PairingCode) -> str:
    """The console's lines for a minted code: the code, its expiry and its pair link.

    A link whose API address is loopback (``BACKEND_URL``'s default, while ``PUBLIC_API_HOST``
    is unset) gets a line saying a phone cannot use it, and how to fix that.
    """
    lines = REPORT.format(
        code=issued.code,
        expires_at=issued.expires_at.isoformat(timespec="seconds"),
        minutes=PAIRING_CODE_TTL_SECONDS // 60,
        link=issued.qr_payload,
    )
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
        issued = mint(db, args.name)
    except (OperatorMissing, MobileUrlError) as failure:
        print(str(failure), file=err)
        return EXIT_FAILED
    except SQLAlchemyError as failure:
        print(DATABASE_FAILED.format(kind=type(failure).__name__), file=err)
        return EXIT_FAILED
    finally:
        db.close()
    print(report(issued), file=out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
