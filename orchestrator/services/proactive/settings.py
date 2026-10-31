"""PRD-268 §6.9, FR-13 — the settings behind ``/api/proactive/settings`` and ``/api/proactive/me``.

* **The workspace's**: ``settings.proactive`` on the workspace row: ``enabled``, the
  workspace switch (``services.proactive.switches``), and ``caps``, each over its config
  default (``services.proactive.limits``), shown beside the platform's. Owners and admins
  read and change them; a workspace can only tighten the platform's caps. A change locks the
  row, builds a new ``settings`` value (the stored one is never mutated), refuses caps it
  would loosen or leave wrong before anything is written, marks the column modified and commits.
* **The person's**: ``proactive_person_settings``, a member's own in a workspace: ``paused``
  ("Pause suggestions") and the quiet hours, ``quiet_start``, ``quiet_end`` and ``timezone``
  (read with the workspace's window by ``services.proactive.quiet.is_person_quiet``). A
  change writes only what it names, in one upsert, and answers with the row as it then is.

Every change commits, then writes its workspace audit row (K15: ``AuditService.log``
commits the session it is given, and logs a write that fails without raising).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from core.models.proactive import ProactivePersonSettings
from modules.proactive import Caps
from services.proactive.limits import CAPS_KEY, caps_as_dict, caps_from, default_caps, looser_than_platform
from services.proactive.quiet import clock, parse_clock
from services.proactive.switches import KEY_ENABLED, WORKSPACE_KEY, enabled_in

__all__ = [
    "PAUSED_ACTION",
    "QUIET_ACTION",
    "RESUMED_ACTION",
    "SETTINGS_ACTION",
    "CapsLooser",
    "CapsWrong",
    "NoWorkspace",
    "PersonSettings",
    "WorkspaceSettings",
    "change_person",
    "change_workspace",
    "read_person",
    "read_workspace",
    "with_changes",
]

SETTINGS_ACTION = "proactive:settings_changed"
PAUSED_ACTION = "proactive:paused"
RESUMED_ACTION = "proactive:resumed"
QUIET_ACTION = "proactive:quiet_hours_changed"
WORKSPACE_RESOURCE = "workspace"
PERSON_RESOURCE = "proactive_person_settings"
SETTINGS_COLUMN = "settings"
PLATFORM_CAPS_KEY = "platform_caps"
# proactive_person_settings' primary key, the upsert's conflict target.
PERSON_KEY_WORKSPACE, PERSON_KEY_USER = "workspace_id", "user_id"
PAUSED, QUIET_START, QUIET_END, TIMEZONE = "paused", "quiet_start", "quiet_end", "timezone"
QUIET_FIELDS = (QUIET_START, QUIET_END, TIMEZONE)
PERSON_FIELDS = (PAUSED, *QUIET_FIELDS)


class NoWorkspace(LookupError):
    """The caller's workspace row is not there."""


class CapsWrong(ValueError):
    """Caps that do not fit together (``modules.proactive.Caps`` refuses them)."""


class CapsLooser(ValueError):
    """Caps a change would set looser than the platform's: their names."""

    def __init__(self, names: Any) -> None:
        super().__init__(", ".join(names))
        self.names = tuple(names)


@dataclass(frozen=True)
class PersonSettings:
    """A person's own proactive settings in a workspace: the pause and the quiet hours (a
    time zone of None is the workspace's)."""

    paused: bool = False
    quiet_start: Optional[time] = None
    quiet_end: Optional[time] = None
    timezone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """``{paused, quiet_start, quiet_end, timezone}``, the times as ``HH:MM``."""
        return {PAUSED: self.paused, QUIET_START: clock(self.quiet_start), QUIET_END: clock(self.quiet_end),
                TIMEZONE: self.timezone}


@dataclass(frozen=True)
class WorkspaceSettings:
    """A workspace's proactive settings: its switch, its caps (defaults filled in) and the
    platform's, which its own can only tighten."""

    enabled: bool
    caps: Caps
    platform_caps: Caps

    def to_dict(self) -> Dict[str, Any]:
        """``{enabled, caps, platform_caps}``, every cap a workspace may set by name."""
        return {KEY_ENABLED: self.enabled, CAPS_KEY: caps_as_dict(self.caps),
                PLATFORM_CAPS_KEY: caps_as_dict(self.platform_caps)}


def _platform() -> Caps:
    """The platform's caps (config's); config that the caps refuse raises ``CapsWrong``."""
    try:
        return default_caps()
    except ValueError as wrong:
        raise CapsWrong(str(wrong)) from wrong


def _of(settings: Any) -> WorkspaceSettings:
    """The proactive settings a workspace's ``settings`` state; caps that are wrong raise ``CapsWrong``."""
    platform = _platform()
    try:
        return WorkspaceSettings(enabled=enabled_in(settings), caps=caps_from(settings), platform_caps=platform)
    except ValueError as wrong:
        raise CapsWrong(str(wrong)) from wrong


def _workspace(db: Session, workspace_id: UUID, *, for_update: bool = False) -> Any:
    from core.models.workspaces import Workspace

    query = db.query(Workspace).filter(Workspace.id == workspace_id)
    workspace = (query.with_for_update() if for_update else query).first()
    if workspace is None:
        raise NoWorkspace(str(workspace_id))
    return workspace


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def with_changes(settings: Any, enabled: Optional[bool], caps: Mapping[str, Optional[int]]) -> Dict[str, Any]:
    """A new ``settings`` value: ``settings.proactive`` with ``enabled`` (when given) and each
    cap in ``caps`` set, a cap given as None back at its default; the rest as it was."""
    proactive = _mapping(_mapping(settings).get(WORKSPACE_KEY))
    defaulted = {name for name, value in caps.items() if value is None}
    kept = {name: value for name, value in _mapping(proactive.get(CAPS_KEY)).items() if name not in defaulted}
    changed = {**proactive, CAPS_KEY: {**kept, **{name: value for name, value in caps.items() if value is not None}}}
    if enabled is not None:
        changed[KEY_ENABLED] = enabled
    return {**_mapping(settings), WORKSPACE_KEY: changed}


def read_workspace(db: Session, workspace_id: UUID) -> WorkspaceSettings:
    """The workspace's proactive settings. ``NoWorkspace`` when it is not there; caps stored
    wrong raise ``CapsWrong``."""
    return _of(_workspace(db, workspace_id).settings)


def _audit(db: Session, workspace_id: UUID, user_id: Optional[int], action: str, resource_type: str,
           resource_id: str, details: Dict[str, Any]) -> None:
    """The change's audit row, after its commit (K15)."""
    from core.workspaces.audit import AuditService

    AuditService(db).log(workspace_id=str(workspace_id), user_id=user_id, action=action,
                         resource_type=resource_type, resource_id=resource_id, details=details)


def change_workspace(db: Session, workspace_id: UUID, enabled: Optional[bool], caps: Mapping[str, Optional[int]],
                     user_id: Optional[int], actor: Any) -> WorkspaceSettings:
    """Switch the workspace's check-ins and change its caps, commit, then audit. Caps the
    change would loosen raise ``CapsLooser``, and caps it would leave wrong ``CapsWrong``,
    before anything is written; the row is locked until the commit, so no other write of
    ``settings`` is lost."""
    looser = looser_than_platform(caps, _platform())
    if looser:
        raise CapsLooser(looser)
    workspace = _workspace(db, workspace_id, for_update=True)
    changed = with_changes(workspace.settings, enabled, caps)
    try:
        after = _of(changed)
    except CapsWrong:
        db.rollback()  # the row lock goes with nothing written
        raise
    workspace.settings = changed
    flag_modified(workspace, SETTINGS_COLUMN)
    db.commit()
    _audit(db, workspace_id, user_id, SETTINGS_ACTION, WORKSPACE_RESOURCE, str(workspace_id),
           {"actor": actor, "changed": {KEY_ENABLED: enabled, CAPS_KEY: dict(caps)}, "settings": after.to_dict()})
    return after


_PERSON_COLUMNS = (ProactivePersonSettings.paused, ProactivePersonSettings.quiet_start,
                   ProactivePersonSettings.quiet_end, ProactivePersonSettings.timezone)


def _person(row: Any) -> PersonSettings:
    if row is None:
        return PersonSettings()
    paused, quiet_start, quiet_end, timezone = row
    return PersonSettings(paused=bool(paused), quiet_start=quiet_start, quiet_end=quiet_end, timezone=timezone)


def read_person(db: Session, workspace_id: UUID, user_id: int) -> PersonSettings:
    """The person's own settings in this workspace; a person who set none has the defaults."""
    return _person(db.query(*_PERSON_COLUMNS).filter(
        ProactivePersonSettings.workspace_id == workspace_id, ProactivePersonSettings.user_id == user_id).first())


def _stored(change: Mapping[str, Any]) -> Dict[str, Any]:
    """The columns a change writes: the quiet times from ``HH:MM``, the rest as given. A name
    that is not one of ``PERSON_FIELDS`` raises ``ValueError``."""
    unknown = sorted(set(change) - set(PERSON_FIELDS))
    if unknown:
        raise ValueError(f"not a person setting: {', '.join(unknown)}")
    return {name: parse_clock(value) if name in (QUIET_START, QUIET_END) else value for name, value in change.items()}


def _audit_person(db: Session, workspace_id: UUID, user_id: int, change: Mapping[str, Any], after: PersonSettings,
                  actor: Any) -> None:
    """One audit row for a pause or a resume, one for a quiet-hours change (K15: after the commit)."""
    if PAUSED in change:
        _audit(db, workspace_id, user_id, PAUSED_ACTION if after.paused else RESUMED_ACTION, PERSON_RESOURCE,
               str(user_id), {"actor": actor, PAUSED: after.paused})
    quiet = {name: change[name] for name in QUIET_FIELDS if name in change}
    if quiet:
        shown = after.to_dict()
        _audit(db, workspace_id, user_id, QUIET_ACTION, PERSON_RESOURCE, str(user_id),
               {"actor": actor, "changed": quiet, "quiet_hours": {name: shown[name] for name in QUIET_FIELDS}})


def change_person(db: Session, workspace_id: UUID, user_id: int, change: Mapping[str, Any],
                  actor: Any) -> PersonSettings:
    """Change the person's own settings in this workspace, commit, then audit. ``change`` names
    only what changes: ``paused``, and the quiet hours (``quiet_start``/``quiet_end`` as
    ``HH:MM`` or None, ``timezone`` an IANA name or None); the rest stays."""
    values = _stored(change)
    statement = insert(ProactivePersonSettings).values(workspace_id=workspace_id, user_id=user_id, **values)
    row = db.execute(statement.on_conflict_do_update(
        index_elements=[PERSON_KEY_WORKSPACE, PERSON_KEY_USER],
        set_={**values, "updated_at": func.now()}).returning(*_PERSON_COLUMNS)).one()
    db.commit()
    after = _person(row)
    _audit_person(db, workspace_id, user_id, change, after, actor)
    return after
