"""Partner tools: installed packs, the credential read and the audit row (PRD-275).

An installed pack is one row of ``workspace_tool_config``, the table that already holds a
workspace's configuration for an integration and its link to a credential: ``tool_id`` is
``partner:<pack name>``, ``configuration.pack`` is the pack as it was checked, and
``credential_id`` is one of the workspace's ``generic_api`` credentials (base URL, API key,
how the key is sent). No new table, and no new kind of credential.

The key is read only through ``CredentialStore.get_decrypted_credential``, which refuses an
inactive or expired credential and audits the access; another workspace's credential is "not
found", as a missing one is. A call that read the key then writes one more
``credential_audit_logs`` row, ``partner_call``: the tool, the operation, the partner's
status code and how long the call took.

The manager who installs a pack names the agents that may use it (``configuration.agents``).
An agent is offered a pack's tools only when it lists the pack *and* the install allows it
(``may_use``): editing an agent (``agents:update``, which an editor holds) cannot reach a
partner's records that installing (``workspace:manage``) did not allow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID

from modules.tools.partner.pack import PackError, PartnerPack, parse_pack
from modules.tools.partner.request import PartnerRefusal

logger = logging.getLogger(__name__)

TOOL_ID_PREFIX = "partner:"
PACK_KEY = "pack"
AGENTS_KEY = "agents"
MAX_AGENTS_PER_PACK = 50
CREDENTIAL_TYPE = "generic_api"
SERVICE_NAME = "partner_tools"
AUDIT_ACTION = "partner_call"
NO_CREDENTIAL = "partner_no_credential"


@dataclass(frozen=True)
class InstalledPack:
    """A pack a workspace has installed, and the credential its calls use."""

    pack: PartnerPack
    credential_id: Optional[int]
    agents: Tuple[int, ...] = ()  # the agents the install allows; none means the pack is offered to no one


def _uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def same_workspace(left: Any, right: Any) -> bool:
    """Both name the same workspace; False when either is missing or is not an id."""
    try:
        return left is not None and right is not None and _uuid(left) == _uuid(right)
    except ValueError:
        return False


def _agent_ids(raw: Any) -> Tuple[int, ...]:
    """The agent ids an installed pack allows; anything that is not an id is no agent."""
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(sorted({value for value in raw if isinstance(value, int) and not isinstance(value, bool)}))


def may_use(installed: InstalledPack, agent_id: Any) -> bool:
    """The install allows ``agent_id`` to use the pack."""
    try:
        return agent_id is not None and int(agent_id) in installed.agents
    except (TypeError, ValueError):
        return False


def tool_id_for(pack_name: str) -> str:
    """The ``workspace_tool_config.tool_id`` an installed pack is kept under."""
    return f"{TOOL_ID_PREFIX}{pack_name}"


def on_own_session(work: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``work(db, ...)`` on a session of its own, closed afterwards. A call's key read and
    audit row commit here, never on the turn's session, and run off the event loop."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return work(db, *args, **kwargs)
    finally:
        db.close()


def _rows(db: Any, workspace_id: Any) -> Any:
    from core.models.tool_assignments import WorkspaceToolConfig

    return db.query(WorkspaceToolConfig).filter(
        WorkspaceToolConfig.workspace_id == _uuid(workspace_id),
        WorkspaceToolConfig.tool_id.like(f"{TOOL_ID_PREFIX}%"),
    )


def _installed(row: Any) -> Optional[InstalledPack]:
    try:
        pack = parse_pack((row.configuration or {}).get(PACK_KEY))
    except PackError:
        logger.warning("[partner-tools] the pack stored as %s no longer fits the format; it is not offered", row.tool_id)
        return None
    if tool_id_for(pack.name) != row.tool_id:
        logger.warning("[partner-tools] the pack stored as %s names itself %s; it is not offered", row.tool_id, pack.name)
        return None
    return InstalledPack(pack, row.credential_id, _agent_ids((row.configuration or {}).get(AGENTS_KEY)))


def installed_packs(db: Any, workspace_id: Any, names: Optional[Iterable[str]] = None) -> List[InstalledPack]:
    """The workspace's installed, enabled packs (only those in ``names`` when given), by name."""
    from core.models.tool_assignments import WorkspaceToolConfig

    query = _rows(db, workspace_id).filter(WorkspaceToolConfig.enabled.is_(True))
    if names is not None:
        query = query.filter(WorkspaceToolConfig.tool_id.in_([tool_id_for(name) for name in names]))
    found = [_installed(row) for row in query.order_by(WorkspaceToolConfig.tool_id, WorkspaceToolConfig.id).all()]
    return [pack for pack in found if pack is not None]


def _check_credential(db: Any, workspace_id: Any, credential_id: int) -> None:
    from core.models.credentials import Credential, CredentialType

    found = (db.query(Credential, CredentialType.name)
             .join(CredentialType, Credential.credential_type_id == CredentialType.id)
             .filter(Credential.id == int(credential_id)).first())
    if found is None or not same_workspace(found[0].workspace_id, workspace_id) or not found[0].is_active:
        raise PackError("That credential is not one of this workspace's active credentials.")
    if found[1] != CREDENTIAL_TYPE:
        raise PackError(f"A pack's credential must be of the type '{CREDENTIAL_TYPE}': a base URL and an API key.")


def _check_agents(db: Any, workspace_id: Any, agent_ids: Tuple[int, ...]) -> None:
    from core.models import Agent

    if len(agent_ids) > MAX_AGENTS_PER_PACK:
        raise PackError(f"A pack may be allowed to at most {MAX_AGENTS_PER_PACK} agents.")
    if not agent_ids:
        return
    found = {row.id for row in db.query(Agent.id).filter(Agent.id.in_(agent_ids),
                                                         Agent.workspace_id == _uuid(workspace_id)).all()}
    unknown = [agent_id for agent_id in agent_ids if agent_id not in found]
    if unknown:
        raise PackError(f"Agents {unknown} are not this workspace's agents.")


def install_pack(db: Any, workspace_id: Any, pack: PartnerPack, credential_id: int,
                 agent_ids: Sequence[int] = ()) -> InstalledPack:
    """Install ``pack`` in the workspace, or replace the one of its name, for the agents named in
    ``agent_ids``. Raises :class:`PackError` when the credential is not this workspace's or not a
    ``generic_api`` one, or an agent is not this workspace's."""
    from core.models.tool_assignments import WorkspaceToolConfig

    agents = _agent_ids(list(agent_ids))
    _check_credential(db, workspace_id, credential_id)
    _check_agents(db, workspace_id, agents)
    tool_id = tool_id_for(pack.name)
    row = _rows(db, workspace_id).filter(WorkspaceToolConfig.tool_id == tool_id).first()
    if row is None:
        row = WorkspaceToolConfig(workspace_id=_uuid(workspace_id), tool_id=tool_id, display_name=pack.title)
        db.add(row)
    row.display_name, row.enabled, row.credential_id = pack.title, True, int(credential_id)
    row.configuration = {PACK_KEY: pack.as_stored(), AGENTS_KEY: list(agents)}
    row.cached_metadata = {"version": pack.version, "tools": [tool.name for tool in pack.tools]}
    db.commit()
    logger.info("[partner-tools] pack %s %s installed in workspace %s for %d agent(s)", pack.name, pack.version,
                workspace_id, len(agents))
    return InstalledPack(pack, int(credential_id), agents)


def remove_pack(db: Any, workspace_id: Any, name: str) -> bool:
    """Remove the workspace's pack of that name; False when there was none."""
    from core.models.tool_assignments import WorkspaceToolConfig

    removed = (_rows(db, workspace_id).filter(WorkspaceToolConfig.tool_id == tool_id_for(name))
               .delete(synchronize_session=False))
    db.commit()
    return bool(removed)


def credential_fields(db: Any, workspace_id: Any, credential_id: Optional[int]) -> Dict[str, Any]:
    """The pack's credential, decrypted for one call. The store audits the read. Raises
    :class:`PartnerRefusal` when the credential is gone, another workspace's, inactive or expired."""
    from core.credentials.service import CredentialNotFoundError, CredentialStore, CredentialValidationError

    missing = PartnerRefusal(NO_CREDENTIAL, "The pack has no credential in this workspace: install it again with one.")
    if credential_id is None:
        raise missing
    store = CredentialStore(db)
    credential = store.get_credential(int(credential_id))
    if credential is None or not same_workspace(credential.workspace_id, workspace_id):
        raise missing
    try:
        return store.get_decrypted_credential(credential.id, service_name=SERVICE_NAME)
    except CredentialNotFoundError:
        raise missing from None
    except CredentialValidationError:  # inactive or expired; the store's words name the credential
        raise PartnerRefusal(NO_CREDENTIAL, "The pack's credential is inactive or has expired.") from None


def record_call(db: Any, credential_id: int, *, tool: str, operation: str, status_code: Optional[int],
                duration_ms: int, success: bool, agent_id: Any, error_code: Optional[str] = None) -> None:
    """The audit row for one call that read the key: which tool, the partner's status code and
    how long it took. Never the key, a header or the arguments."""
    from core.models.credentials import CredentialAuditLog

    db.add(CredentialAuditLog(
        credential_id=int(credential_id), action=AUDIT_ACTION, success=bool(success), error_message=error_code,
        audit_metadata={"service": SERVICE_NAME, "tool": tool, "operation": operation, "status_code": status_code,
                        "duration_ms": int(duration_ms), "agent_id": agent_id},
        created_at=datetime.utcnow(),
    ))
    db.commit()


__all__ = ["AGENTS_KEY", "AUDIT_ACTION", "CREDENTIAL_TYPE", "InstalledPack", "MAX_AGENTS_PER_PACK", "NO_CREDENTIAL",
           "SERVICE_NAME", "TOOL_ID_PREFIX", "credential_fields", "install_pack", "installed_packs", "may_use",
           "on_own_session", "record_call", "remove_pack", "same_workspace", "tool_id_for"]
