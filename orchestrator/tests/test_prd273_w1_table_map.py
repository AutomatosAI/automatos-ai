"""PRD-273 US-273-01 — the classification map, its guard and the settings allow-list.

* Every rule has a reason; a moved rule has one of §5.2's categories; a child table names its
  parent and column; the tables §5.3 says never move are NOT_MOVED, and the five that are
  nothing but secrets are secret tables.
* On CI's Postgres (the schema ``scripts/init_test_db.py`` builds): every table
  ``_discover_scoped_tables`` finds is classified, and so is every table that hangs off one by
  foreign key; every secret-looking column of a classified table is named as secret.
* A new workspace table, a new child table or a new secret-looking column (made inside the
  test's rolled-back transaction) fails the guard.
* The settings allow-list carries ``brand_kit``, ``orchestrator`` (without ``llm`` and ``model``),
  ``voice_live`` and ``push``, never ``byok_overrides``, ``integrations`` or an unnamed key.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from services.workspace_move.auto_part import (
    ALLOWED_SETTINGS_KEYS, NEVER_SETTINGS_KEYS, ORCHESTRATOR_DROPPED_KEYS, allowed_settings,
)
from services.workspace_move.rules import (
    AUTO, BOARD, CATEGORIES, DEFAULT_CATEGORIES, looks_secret, resolve_categories,
)
from services.workspace_move.schema import read_schema, unclassified_tables, unnamed_secret_columns
from services.workspace_move.table_map import SECRET_TABLES, TABLE_RULES, rule_for

# §5.3's list of what never moves, and §3's telemetry.
NEVER_MOVED = (
    "credentials", "sdk_api_keys", "user_api_keys", "channel_connections", "composio_entities", "mobile_devices",
    "mobile_digest_posts", "approval_grants", "credit_ledger", "credit_balances", "workspace_members", "audit_logs",
    "substrate_metric_events", "heartbeat_results", "llm_usage", "tool_execution_logs", "notifications",
    "routing_decisions", "codegraph_projects", "codegraph_symbols",
)
SECRETS_ONLY = ("credentials", "sdk_api_keys", "user_api_keys", "channel_connections", "composio_entities")
# §5.2's tables, by the category that carries them.
PRD_CATEGORIES = {
    "business_profiles": "auto", "personas": "auto", "memory_short_term": "memory", "documents": "knowledge",
    "knowledge_items": "knowledge", "workspace_graphs": "knowledge", "agents": "agents", "skills": "agents",
    "workspace_enabled_skills": "agents", "teams": "agents", "routing_rules": "agents", "workflow_recipes": "agents",
    "chats": "chats", "messages": "chats", "deliverables": "deliverables", "board_tasks": "board",
    "agent_reports": "board",
}


def test_every_rule_has_a_reason_and_a_moved_one_a_category():
    for table, rule in TABLE_RULES.items():
        assert rule.table == table and rule.reason.strip(), table
        assert (rule.category in CATEGORIES) if rule.moved else rule.category is None, table
        if rule.parent is not None:
            assert len(rule.parent) == 2 and all(rule.parent), table


def test_the_prds_tables_sit_in_their_categories():
    for table, category in PRD_CATEGORIES.items():
        rule = rule_for(table)
        assert rule is not None and rule.moved and rule.category == category, table


def test_what_never_moves_is_not_moved_and_the_secret_tables_are_named():
    for table in NEVER_MOVED:
        assert rule_for(table) is not None and not rule_for(table).moved, table
    assert set(SECRETS_ONLY) <= SECRET_TABLES
    assert all(not TABLE_RULES[table].moved for table in SECRET_TABLES)


def test_a_moved_tables_secret_columns_are_named():
    assert TABLE_RULES["agent_tool_assignments"].secret_columns == ("credential_id",)
    assert TABLE_RULES["workflow_recipes"].redacted_keys["schedule_config"] == ("webhook_id",)
    assert "embedding" in TABLE_RULES["knowledge_items"].derived_columns


def test_secret_looking_names():
    for name in ("encrypted_key", "token_hash", "pairing_code_hash", "expo_push_token", "token", "push_key",
                 "credential_id", "secrets", "api_key", "webhook_url", "private_key"):
        assert looks_secret(name), name
    for name in ("tokens_used", "token_budget_estimate", "content_hash", "key_fingerprint", "token_rotated_at",
                 "name", "workspace_id", "estimated_tokens"):
        assert not looks_secret(name), name


def test_categories_default_without_the_board_and_auto_always():
    assert resolve_categories(None) == DEFAULT_CATEGORIES and BOARD not in DEFAULT_CATEGORIES
    assert resolve_categories(("chats",)) == (AUTO, "chats")
    assert resolve_categories(("board", "auto", "memory")) == (AUTO, "memory", BOARD)
    with pytest.raises(ValueError):
        resolve_categories(("telemetry",))


def test_the_settings_allow_list():
    stored = {
        "brand_kit": {"primary_color": "#123456"},
        "orchestrator": {"communication_style": "direct", "llm": {"model_id": "m"}, "model": "m", "custom_soul": "s"},
        "voice_live": {"voice": "v"}, "push": {"digest": True},
        "byok_overrides": {"openai": "a-key"}, "integrations": {"slack": {"token": "t"}}, "power_mode": True,
    }
    kept = allowed_settings(stored)
    assert set(kept) == set(ALLOWED_SETTINGS_KEYS)
    assert kept["orchestrator"] == {"communication_style": "direct", "custom_soul": "s"}
    assert not set(ORCHESTRATOR_DROPPED_KEYS) & set(kept["orchestrator"])
    assert not set(NEVER_SETTINGS_KEYS) & set(kept) and "power_mode" not in kept
    assert "llm" in stored["orchestrator"]  # the stored settings are not changed
    kept["brand_kit"]["primary_color"] = "#000000"
    assert stored["brand_kit"]["primary_color"] == "#123456"
    assert allowed_settings(None) == {}


# --------------------------------------------------------------------------- the guard, on Postgres


def test_every_workspace_table_and_child_table_in_this_schema_is_classified(db_session):
    schema = read_schema(db_session)
    assert schema.scoped, "no workspace tables found: the schema probe is broken"
    assert unclassified_tables(schema) == ()


def test_every_secret_looking_column_in_this_schema_is_named(db_session):
    assert unnamed_secret_columns(read_schema(db_session)) == ()


def test_a_new_workspace_table_fails_the_guard(db_session):
    probe = f"prd273_probe_{uuid.uuid4().hex[:10]}"
    db_session.execute(text(f"CREATE TABLE {probe} (id serial PRIMARY KEY, workspace_id uuid)"))
    assert probe in unclassified_tables(read_schema(db_session))


def test_a_new_child_table_fails_the_guard(db_session):
    probe = f"prd273_child_{uuid.uuid4().hex[:10]}"
    db_session.execute(text(f"CREATE TABLE {probe} (id serial PRIMARY KEY, agent_id integer REFERENCES agents(id))"))
    assert probe in unclassified_tables(read_schema(db_session))


def test_a_new_secret_looking_column_fails_the_guard(db_session):
    db_session.execute(text("ALTER TABLE teams ADD COLUMN prd273_api_key text"))
    assert ("teams", "prd273_api_key") in unnamed_secret_columns(read_schema(db_session))
