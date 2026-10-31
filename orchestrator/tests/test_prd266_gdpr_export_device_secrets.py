"""PRD-266 US-011a (K16) — the GDPR export never hands out a phone's secrets.

``services.gdpr_service.export_workspace`` (behind ``GET /api/v1/gdpr/export`` and the verticals
export) reads every workspace table with ``SELECT *``. ``EXPORT_REDACTED_COLUMNS`` drops, from
``mobile_devices``, W1's ``SECRET_COLUMNS`` (the pairing code hash, the token hash, the previous
token hash and the push key) and the Expo push token. Proven here on Postgres:

* a paired phone holding all five is exported with its id, name and platform and none of them;
* another table's rows (the phone's push ticket) are exported unchanged;
* the map names W1's tuple itself, not a copy of it.

The derived stores and the audit row are stubbed: this is the SQL leg.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from core.models.mobile_devices import SECRET_COLUMNS
from services import gdpr_service

FIVE_SECRETS = ("pairing_code_hash", "token_hash", "prev_token_hash", "push_key", "expo_push_token")
PHONE_NAME = "Ana's iPhone"


@pytest.fixture
def export_sql_only(monkeypatch):
    """``export_workspace`` with the field memory, durable memory and audit legs stubbed."""
    monkeypatch.setattr(gdpr_service, "_export_field_memory", lambda workspace_id: [])
    monkeypatch.setattr(gdpr_service, "_export_durable_memory", lambda workspace_id: [])
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda *args, **kwargs: None)
    return gdpr_service.export_workspace


def _seed_user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-{tag}", "e": f"prd266-{tag}@t.test"},
    ).scalar()


def _secrets() -> dict:
    """Fresh values for the five fields: unique per run, and never written out in the source."""
    return {
        "pairing_code_hash": uuid.uuid4().hex * 2,
        "token_hash": uuid.uuid4().hex * 2,
        "prev_token_hash": uuid.uuid4().hex * 2,
        "push_key": f"pk-{uuid.uuid4().hex[:6]}",
        "expo_push_token": f"ExponentPushToken[{uuid.uuid4().hex[:8]}]",
    }


def _seed_device(db, workspace_id: str, user_id: int, secrets: dict) -> str:
    device_id = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO mobile_devices (id, workspace_id, user_id, kind, status, install_id, name, platform, "
        "pairing_code_hash, token_hash, prev_token_hash, push_key, expo_push_token, push_enabled) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :u, 'paired', 'active', :install, :name, 'ios', "
        ":pairing_code_hash, :token_hash, :prev_token_hash, :push_key, :expo_push_token, true)"),
        {"id": device_id, "ws": workspace_id, "u": user_id, "install": uuid.uuid4().hex, "name": PHONE_NAME,
         **secrets})
    return device_id


def _seed_ticket(db, device_id: str, workspace_id: str) -> str:
    ticket_id = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO mobile_push_tickets (ticket_id, device_id, workspace_id, push_type) "
        "VALUES (:t, CAST(:d AS uuid), CAST(:ws AS uuid), 'needs_answer')"),
        {"t": ticket_id, "d": device_id, "ws": workspace_id})
    return ticket_id


def test_the_map_drops_w1s_secret_columns_and_the_expo_token():
    redacted = gdpr_service.EXPORT_REDACTED_COLUMNS["mobile_devices"]
    assert redacted[:len(SECRET_COLUMNS)] == SECRET_COLUMNS
    assert set(redacted) == set(FIVE_SECRETS)


def test_a_paired_phone_is_exported_without_its_secrets(db_session, seed_workspace, export_sql_only):
    ws, user = seed_workspace(), _seed_user(db_session)
    secrets = _secrets()
    device_id = _seed_device(db_session, ws, user, secrets)

    (row,) = export_sql_only(db_session, ws)["sql"]["mobile_devices"]

    assert str(row["id"]) == device_id
    assert (row["name"], row["platform"], row["kind"]) == (PHONE_NAME, "ios", "paired")
    assert not set(FIVE_SECRETS) & set(row)
    assert not set(secrets.values()) & {value for value in row.values() if isinstance(value, str)}


def test_another_tables_rows_are_exported_unchanged(db_session, seed_workspace, export_sql_only):
    ws, user = seed_workspace(), _seed_user(db_session)
    device_id = _seed_device(db_session, ws, user, _secrets())
    ticket_id = _seed_ticket(db_session, device_id, ws)
    stored = dict(db_session.execute(
        text("SELECT * FROM mobile_push_tickets WHERE ticket_id = :t"), {"t": ticket_id},
    ).mappings().one())

    assert export_sql_only(db_session, ws)["sql"]["mobile_push_tickets"] == [stored]


def test_another_workspaces_phone_is_not_in_the_export(db_session, seed_workspace, export_sql_only):
    ws, other_ws, user = seed_workspace(), seed_workspace(), _seed_user(db_session)
    _seed_device(db_session, other_ws, user, _secrets())

    assert export_sql_only(db_session, ws)["sql"]["mobile_devices"] == []
