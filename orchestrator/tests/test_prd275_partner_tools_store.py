"""PRD-275: an installed partner pack, its credential and its audit rows. Real database.

A pack is one ``workspace_tool_config`` row beside the workspace's ``generic_api`` credential.
Only an agent of that workspace whose configuration lists the pack resolves its tools; the key
is read through the credential store (which audits the read) and never for another workspace;
each call adds a ``partner_call`` row to the credential's audit log.
"""
from __future__ import annotations

# CI collection-order guard (see PR #434): an earlier test can leave a stale,
# spec-less `modules`/`services` entry in sys.modules that breaks these imports.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import json  # noqa: E402
import uuid  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from modules.tools.partner import runner, store  # noqa: E402
from modules.tools.partner.offer import granted_packs, schema_name, with_partner_tools  # noqa: E402
from modules.tools.partner.pack import PackError, parse_pack  # noqa: E402
from modules.tools.partner.request import PartnerRefusal  # noqa: E402

KEY = "partnerkey" * 3  # not a real key
BASE = "https://hub.partner.example"
TOOL = "acme_find_orders"
PACK = {
    "schema": "automatos-partner-pack/v1", "name": "acme", "title": "Acme Hub", "version": "0.1.0",
    "tools": [{"name": "find_orders", "description": "Look up one order.", "scope": "orders:read", "operations": [
        {"name": "get", "description": "One order by its id.", "path": "/api/v1/orders/{order_id}",
         "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
         "response": {"root": "data", "keep": ["id", "status"]}}]}],
}


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


def _credential_type(session, name: str) -> tuple:
    """The id of the credential type ``name``, made when the test database has none; and whether it was made."""
    from core.models.credentials import CredentialType

    found = session.query(CredentialType).filter(CredentialType.name == name).first()
    if found is not None:
        return found.id, False
    made = CredentialType(name=name, display_name=name, category="api", schema_definition=[], is_system=False,
                          is_active=True)
    session.add(made)
    session.commit()
    return made.id, True


def _credential(session, workspace: str, type_id: int, name: str) -> int:
    from core.credentials.service import CredentialStore
    from core.models.credentials import CredentialCreate

    fields = {"api_key": KEY, "base_url": BASE, "auth_method": "bearer"}
    made = CredentialStore(session).create_credential(
        CredentialCreate(name=name, credential_type_id=type_id, credential_data=fields),
        workspace_id=uuid.UUID(workspace))
    return made.id


@pytest.fixture
def seeded(new_session):
    """Two workspaces. The first has an agent that lists the pack, an agent that does not, a
    ``generic_api`` credential and a credential of another type; the second has its own credential."""
    from sqlalchemy import text

    ours, theirs = str(uuid.uuid4()), str(uuid.uuid4())
    s = new_session()
    for workspace, name in ((ours, "prd275-ours"), (theirs, "prd275-theirs")):
        s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :name)"),
                  {"id": workspace, "name": name})
    agents = [s.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) VALUES "
                             "(:name, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
                        {"name": name, "w": ours, "c": json.dumps(configuration)}).scalar()
              for name, configuration in (("Partner Ops", {"partner_packs": ["acme"]}), ("Writer", {}))]
    s.commit()
    api_type, made_api = _credential_type(s, store.CREDENTIAL_TYPE)
    other_type, made_other = _credential_type(s, "prd275_other_type")
    tag = ours[:8]
    seed = SimpleNamespace(
        ours=ours, theirs=theirs, lists_it=agents[0], does_not=agents[1],
        credential=_credential(s, ours, api_type, f"acme-{tag}"),
        wrong_type=_credential(s, ours, other_type, f"other-{tag}"),
        their_credential=_credential(s, theirs, api_type, f"theirs-{tag}"),
    )
    s.close()
    yield seed
    s = new_session.sweep()
    scope = {"a": ours, "b": theirs}
    for table in ("workspace_tool_config", "credentials", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id IN (CAST(:a AS uuid), CAST(:b AS uuid))"), scope)  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id IN (CAST(:a AS uuid), CAST(:b AS uuid))"), scope)
    for made, name in ((made_api, store.CREDENTIAL_TYPE), (made_other, "prd275_other_type")):
        if made:
            s.execute(text("DELETE FROM credential_types WHERE name = :name"), {"name": name})
    s.commit()
    s.close()


def _audit(new_session, credential_id: int):
    from core.models.credentials import CredentialAuditLog

    s = new_session()
    rows = (s.query(CredentialAuditLog).filter(CredentialAuditLog.credential_id == credential_id)
            .order_by(CredentialAuditLog.id).all())
    s.close()
    return rows


def test_a_pack_installs_as_one_row_and_a_second_install_replaces_it(new_session, seeded):
    s = new_session()
    store.install_pack(s, seeded.ours, parse_pack(PACK), seeded.credential)
    newer = parse_pack({**PACK, "version": "0.2.0", "title": "Acme Hub (new)"})
    store.install_pack(s, seeded.ours, newer, seeded.credential)
    found = store.installed_packs(s, seeded.ours)
    assert [(i.pack.name, i.pack.version, i.credential_id) for i in found] == [("acme", "0.2.0", seeded.credential)]
    assert store.installed_packs(s, seeded.theirs) == []                    # another workspace has none
    assert store.installed_packs(s, seeded.ours, ["other"]) == []
    assert store.remove_pack(s, seeded.ours, "acme") is True and store.installed_packs(s, seeded.ours) == []
    assert store.remove_pack(s, seeded.ours, "acme") is False
    s.close()


def test_a_pack_installs_only_with_this_workspaces_own_generic_api_credential(new_session, seeded):
    s = new_session()
    for credential_id, said in ((seeded.their_credential, "not one of this workspace's"),
                                (seeded.wrong_type, "generic_api"), (2_000_000_000, "not one of this workspace's")):
        with pytest.raises(PackError, match=said):
            store.install_pack(s, seeded.ours, parse_pack(PACK), credential_id)
    assert store.installed_packs(s, seeded.ours) == []
    s.close()


def test_only_an_agent_of_the_workspace_that_lists_the_pack_resolves_its_tools(new_session, seeded):
    s = new_session()
    store.install_pack(s, seeded.ours, parse_pack(PACK), seeded.credential)
    assert granted_packs(s, seeded.lists_it, seeded.ours) == ("acme",)
    assert granted_packs(s, seeded.does_not, seeded.ours) == ()
    assert granted_packs(s, seeded.lists_it, seeded.theirs) == ()            # the agent is not that workspace's

    resolved = runner.resolve(s, seeded.ours, seeded.lists_it, TOOL)
    assert (resolved.pack_name, resolved.name, resolved.credential_id) == ("acme", TOOL, seeded.credential)
    assert runner.resolve(s, seeded.ours, seeded.does_not, TOOL) is None
    assert runner.resolve(s, seeded.theirs, seeded.lists_it, TOOL) is None
    assert runner.resolve(s, seeded.ours, seeded.lists_it, "acme_delete_orders") is None
    s.close()


def test_the_tool_list_carries_a_listed_agents_partner_tools(new_session, seeded, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "PARTNER_TOOLS_ENABLED", True, raising=False)
    s = new_session()
    store.install_pack(s, seeded.ours, parse_pack(PACK), seeded.credential)
    assert [schema_name(schema) for schema in with_partner_tools([], s, seeded.lists_it, seeded.ours)] == [TOOL]
    assert with_partner_tools([], s, seeded.does_not, seeded.ours) == []
    s.close()


def test_the_key_is_read_through_the_store_for_its_own_workspace_only(new_session, seeded):
    s = new_session()
    fields = store.credential_fields(s, seeded.ours, seeded.credential)
    assert fields["api_key"] == KEY and fields["base_url"] == BASE and fields["auth_method"] == "bearer"
    for workspace, credential_id in ((seeded.theirs, seeded.credential), (seeded.ours, seeded.their_credential),
                                     (seeded.ours, None), (seeded.ours, 2_000_000_000)):
        with pytest.raises(PartnerRefusal) as refused:
            store.credential_fields(s, workspace, credential_id)
        assert refused.value.code == store.NO_CREDENTIAL
    s.close()
    read = [row for row in _audit(new_session, seeded.credential) if row.action == "accessed"]
    assert len(read) == 1 and read[0].audit_metadata["service"] == store.SERVICE_NAME    # the one read, audited
    assert not [row for row in _audit(new_session, seeded.their_credential) if row.action == "accessed"]


def test_a_call_writes_its_audit_row_on_the_credential(new_session, seeded):
    store.on_own_session(store.record_call, seeded.credential, tool=TOOL, operation="get", status_code=404,
                         duration_ms=87, success=False, agent_id=seeded.lists_it, error_code="partner_not_found")
    rows = [row for row in _audit(new_session, seeded.credential) if row.action == store.AUDIT_ACTION]
    assert len(rows) == 1 and rows[0].success is False and rows[0].error_message == "partner_not_found"
    assert rows[0].audit_metadata == {"service": store.SERVICE_NAME, "tool": TOOL, "operation": "get",
                                      "status_code": 404, "duration_ms": 87, "agent_id": seeded.lists_it}
    assert KEY not in json.dumps(rows[0].audit_metadata)


def test_a_stored_pack_that_no_longer_fits_the_format_is_not_offered(new_session, seeded):
    from core.models.tool_assignments import WorkspaceToolConfig

    s = new_session()
    store.install_pack(s, seeded.ours, parse_pack(PACK), seeded.credential)
    row = s.query(WorkspaceToolConfig).filter(WorkspaceToolConfig.workspace_id == uuid.UUID(seeded.ours)).one()
    row.configuration = {"pack": {**PACK, "tools": []}}
    s.commit()
    assert store.installed_packs(s, seeded.ours) == []
    assert runner.resolve(s, seeded.ours, seeded.lists_it, TOOL) is None
    s.close()
