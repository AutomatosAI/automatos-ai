"""PRD-266 §4.3 — the device lane: a paired phone's (or browser's) ``Authorization: Bearer amd_…``.

Reached only through the bearer-prefix registry (``core/auth/bearer_registry.py``), which
runs it BEFORE the Clerk lane, so a device token is never shown to Clerk.

* Local edition only. On saas an ``amd_`` bearer is 401 and nothing is looked up: a paired
  row never exists there (§8).
* The row is found by the SHA-256 of the token (``token_hash``) and confirmed in constant
  time (``core.auth.pairing.secrets_match``), or by ``prev_token_hash`` while the rotation
  overlap lasts (strictly before ``prev_token_valid_until``). A revoked row is
  ``401 {"code": "device_revoked"}``, so the app knows to pair again; every other miss is a
  plain 401.
* The workspace is the ROW's. A request that names another one (``X-Workspace-ID``,
  ``X-Workspace`` or ``?workspace_id=``, the inputs hybrid reads) is 403; naming the same one,
  or none, is fine. The workspace must be usable by hybrid's own rules
  (``_workspace_exists``, ``_assert_workspace_usable``), reused, not copied.
* The person is the row's ``users`` row (a deactivated one is a plain 401), bound as the
  local-operator lane binds it: ``id`` is the EMAIL (hybrid's
  ``_local_operator_user_context`` says why), the operator is ``super_admin`` and anyone
  else a plain ``user``, and ``raw_claims`` carries ``{source: "mobile_device", device_id,
  user_id}`` so ``core.auth.principal.resolve_user_pk`` reads the integer PK without a query.
* ``last_seen_at`` is written at most once every ``LAST_SEEN_WRITE_INTERVAL``, never on
  every request; a failed write is logged and never costs the caller the request.
* The lookup runs in the threadpool on its own session, always closed. Nothing here logs a
  token or a hash.

``core.auth.hybrid`` imports the registry, which imports this module, so hybrid's helpers are
imported at call time (``_hybrid``), never at module level.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.pairing import hash_secret, secrets_match, within_rotation_overlap
from core.auth.principal import MOBILE_DEVICE_CLAIM_SOURCE
from core.database.database import SessionLocal
from core.models.core import User
from core.models.mobile_devices import MobileDevice, MobileDeviceKind, MobileDeviceStatus

logger = logging.getLogger(__name__)

DEVICE_AUTH_TYPE = "device"
DEVICE_REVOKED = "device_revoked"
LAST_SEEN_WRITE_INTERVAL = timedelta(minutes=5)
LOCAL_OPERATOR_ROLE = "super_admin"
PERSON_ROLE = "user"
# Every input through which a request names a workspace: the ones hybrid's
# _get_workspace_id_from_request reads (test_prd266_device_lane keeps the two in step).
WORKSPACE_HEADERS = ("x-workspace-id", "x-workspace")
WORKSPACE_QUERY_PARAM = "workspace_id"
INVALID_TOKEN = "Invalid or expired token"
INVALID_WORKSPACE = "Invalid workspace_id"
OTHER_WORKSPACE = "This device is paired to another workspace."


def _hybrid() -> ModuleType:
    """``core.auth.hybrid``, imported at call time (it imports the registry that imports us)."""
    from core.auth import hybrid

    return hybrid


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _unauthorized(reason: str) -> HTTPException:
    """A plain 401 for any bearer that is not a live device token; the reason goes to the log only."""
    logger.warning("Device lane: bearer refused (%s)", reason)
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=INVALID_TOKEN)


def _find_device(db: Session, token: str, now: datetime) -> Optional[MobileDevice]:
    """The paired row this token opens: its current token, or its previous one inside the overlap."""
    digest = hash_secret(token)
    paired = db.query(MobileDevice).filter(MobileDevice.kind == MobileDeviceKind.PAIRED.value)
    current = paired.filter(MobileDevice.token_hash == digest).first()
    if current is not None and secrets_match(current.token_hash, token):
        return current
    previous = paired.filter(MobileDevice.prev_token_hash == digest).first()
    if previous is not None and secrets_match(previous.prev_token_hash, token) \
            and within_rotation_overlap(previous.prev_token_valid_until, now):
        return previous
    return None


def _refuse_unless_active(device: MobileDevice) -> None:
    """401 ``device_revoked`` for a revoked row; 401 for any other state but active."""
    if device.status == MobileDeviceStatus.REVOKED.value:
        logger.warning("Device lane: revoked device %s presented its token", device.id)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"code": DEVICE_REVOKED})
    if device.status != MobileDeviceStatus.ACTIVE.value:
        raise _unauthorized(f"device {device.id} is {device.status}")


def _named_workspaces(request: Request) -> Tuple[str, ...]:
    """Every workspace the request names, as sent (blank values name none)."""
    sent = [request.headers.get(header) for header in WORKSPACE_HEADERS]
    sent.append(request.query_params.get(WORKSPACE_QUERY_PARAM))
    return tuple(value.strip() for value in sent if isinstance(value, str) and value.strip())


def _assert_bound_workspace(request: Request, device: MobileDevice, hybrid: ModuleType) -> None:
    """403 if the request names any workspace other than the device's (unparsable included)."""
    for named in _named_workspaces(request):
        if hybrid._parse_uuid(named) != device.workspace_id:
            logger.warning("Device lane: device %s named another workspace", device.id)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=OTHER_WORKSPACE)


def _assert_workspace_usable(db: Session, workspace_id: UUID, hybrid: ModuleType) -> None:
    """Hybrid's rules for a credential bound to a workspace: deleted is 400, paused is 403."""
    if not hybrid._workspace_exists(db, workspace_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_WORKSPACE)
    hybrid._assert_workspace_usable(db, workspace_id, is_admin=False)


def _bound_person(db: Session, user_pk: int) -> Optional[Tuple[int, str]]:
    """The device's person as ``(users.id, email)``, or ``None`` when the row is gone or deactivated."""
    row = db.query(User.id, User.email).filter(User.id == user_pk, User.is_active.isnot(False)).first()
    return (int(row[0]), row[1]) if row is not None and row[1] else None


def _user_context(device: MobileDevice, person: Tuple[int, str]) -> UserContext:
    """The device's person, bound as the local-operator lane binds the operator."""
    user_pk, email = person
    role = LOCAL_OPERATOR_ROLE if email == config.LOCAL_OPERATOR_EMAIL else PERSON_ROLE
    return UserContext(
        id=email,
        email=email,
        system_role=role,
        raw_claims={"source": MOBILE_DEVICE_CLAIM_SOURCE, "device_id": str(device.id), "user_id": user_pk},
    )


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _touch_last_seen(db: Session, device_id: UUID, seen: Optional[datetime], now: datetime) -> None:
    """Stamp ``last_seen_at`` when the last stamp (``seen``) is ``LAST_SEEN_WRITE_INTERVAL`` old or more.

    Bookkeeping, never fatal: a failed write rolls back and logs by type (never the
    statement's text), and the request goes on.
    """
    if seen is not None and now - _as_utc(seen) < LAST_SEEN_WRITE_INTERVAL:
        return
    try:
        db.query(MobileDevice).filter(MobileDevice.id == device_id).update(
            {MobileDevice.last_seen_at: now}, synchronize_session=False,
        )
        db.commit()
    except SQLAlchemyError as failure:
        db.rollback()
        logger.warning("Device lane: last_seen_at write failed for %s: %s", device_id, type(failure).__name__)


def _resolve_in_own_session(request: Request, token: str) -> RequestContext:
    """Find, check and bind the device on a session of its own, always closed."""
    hybrid = _hybrid()
    now = _now()
    db = SessionLocal()
    try:
        device = _find_device(db, token, now)
        if device is None:
            raise _unauthorized("no live device holds this token")
        _refuse_unless_active(device)
        _assert_bound_workspace(request, device, hybrid)
        _assert_workspace_usable(db, device.workspace_id, hybrid)
        person = _bound_person(db, device.user_id)
        if person is None:
            raise _unauthorized(f"device {device.id} has no person")
        context = RequestContext(
            workspace_id=device.workspace_id,
            user=_user_context(device, person),
            auth_type=DEVICE_AUTH_TYPE,
        )
        _touch_last_seen(db, device.id, device.last_seen_at, now)
        return context
    finally:
        db.close()


async def resolve_device_context(request: Request, token: str) -> RequestContext:
    """The registry's resolver for ``amd_`` bearers: the caller's ``RequestContext``, or 401/403.

    Local edition only (saas: 401, nothing looked up). The lookup runs off the event loop;
    the log context is set here, on the request's own context.
    """
    if not config.IS_LOCAL_EDITION:
        raise _unauthorized("device tokens are a local-edition credential")
    context = await run_in_threadpool(_resolve_in_own_session, request, token)
    _hybrid()._enrich_log_context(context)
    return context


__all__ = ["DEVICE_AUTH_TYPE", "DEVICE_REVOKED", "LAST_SEEN_WRITE_INTERVAL", "resolve_device_context"]
