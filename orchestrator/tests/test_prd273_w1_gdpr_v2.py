"""PRD-273 US-273-04 — the GDPR export v2: no secret tables or columns, and the workspace's settings.

``GET /api/v1/gdpr/export`` (``services.gdpr_service.export_workspace``) carried the
credentials, API keys and connections of a workspace as stored, and none of its settings. On
CI's Postgres, with the memory legs and the audit row stubbed:

* the format is ``automatos.gdpr.export/v2``;
* the secret tables (credentials, SDK and user API keys, channel and Composio connections) are
  not in ``sql``, and no planted secret is anywhere in the bundle;
* a paired machine's row is there without its hashes; a secret-looking column added to a table
  is left out;
* ``workspace`` carries the name and the allow-listed settings (brand kit, Orchestrator block
  without its model, voice, push), never the BYOK keys or integrations;
* the redacted-columns map is the move's table map, and keeps PRD-266's phone secrets.
"""
from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text

from config import config
from core.models.mobile_devices import SECRET_COLUMNS
from services import gdpr_service
from services.workspace_move.table_map import SECRET_TABLES, TABLE_RULES
from tests.helpers_prd273_workspace import seed_workspace_contents


@pytest.fixture
def export_sql_and_settings(monkeypatch):
    monkeypatch.setattr(gdpr_service, "_export_field_memory", lambda workspace_id: [])
    monkeypatch.setattr(gdpr_service, "_export_durable_memory", lambda workspace_id: [])
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda *args, **kwargs: None)
    return gdpr_service.export_workspace


@pytest.fixture
def seeded(db_session, seed_workspace, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DOCUMENT_STORAGE_DIR", str(tmp_path))
    return seed_workspace_contents(db_session, seed_workspace(name="Harbour Coffee"), tmp_path)


def test_the_format_is_v2():
    assert gdpr_service.EXPORT_FORMAT == "automatos.gdpr.export/v2"


def test_the_secret_tables_and_every_planted_secret_are_gone(db_session, seeded, export_sql_and_settings):
    bundle = export_sql_and_settings(db_session, seeded.workspace_id)
    assert bundle["format"] == "automatos.gdpr.export/v2"
    assert not SECRET_TABLES & set(bundle["sql"])
    assert {"user_api_keys", "sdk_api_keys"} <= SECRET_TABLES
    dumped = json.dumps(bundle, default=str)
    for secret in seeded.secrets:
        assert secret not in dumped
    (playbook,) = bundle["sql"]["workflow_recipes"]
    assert playbook["schedule_config"] == {"type": "webhook"}
    assert bundle["sql"]["agents"], "the workspace's other tables are still exported"


def test_a_paired_machine_is_exported_without_its_hashes(db_session, seeded, export_sql_and_settings):
    (host,) = export_sql_and_settings(db_session, seeded.workspace_id)["sql"]["cli_hosts"]
    assert host["name"] == "laptop"
    assert not {"token_hash", "pairing_code_hash"} & set(host)


def test_a_secret_looking_column_is_left_out_named_or_not(db_session, seeded, export_sql_and_settings):
    db_session.execute(text("ALTER TABLE teams ADD COLUMN prd273_api_key text"))
    db_session.execute(text("UPDATE teams SET prd273_api_key = :s WHERE workspace_id = :ws"),
                       {"s": uuid.uuid4().hex, "ws": seeded.workspace_id})
    (team,) = export_sql_and_settings(db_session, seeded.workspace_id)["sql"]["teams"]
    assert team["name"] == "Growth" and "prd273_api_key" not in team


def test_the_workspace_settings_are_allow_listed(db_session, seeded, export_sql_and_settings):
    workspace = export_sql_and_settings(db_session, seeded.workspace_id)["workspace"]
    assert workspace["name"] == "Harbour Coffee"
    assert set(workspace["settings"]) == {"brand_kit", "orchestrator", "voice_live", "push"}
    assert workspace["settings"]["brand_kit"]["primary_color"] == "#1F6FEB"
    assert {"llm", "model"}.isdisjoint(workspace["settings"]["orchestrator"])


def test_a_gone_workspace_has_no_settings(db_session, export_sql_and_settings):
    assert export_sql_and_settings(db_session, str(uuid.uuid4()))["workspace"] is None


def test_the_redacted_columns_are_the_table_maps_and_keep_the_phone_secrets():
    redacted = gdpr_service.EXPORT_REDACTED_COLUMNS
    assert redacted["mobile_devices"][:len(SECRET_COLUMNS)] == SECRET_COLUMNS
    assert "expo_push_token" in redacted["mobile_devices"]
    assert redacted == {table: rule.secret_columns for table, rule in TABLE_RULES.items() if rule.secret_columns}
