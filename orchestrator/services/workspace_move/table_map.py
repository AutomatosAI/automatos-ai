"""PRD-273 §5.3 — every workspace table, classified: MOVED (in a §5.2 category) or NOT_MOVED.

Allow-list, never deny-list. A table ``_discover_scoped_tables`` finds that no rule names is
not exported (the manifest lists it under ``skipped_tables``), and the guard test
(``tests/test_prd273_w1_table_map.py``) fails in CI until someone classifies it. The same
holds for a table without ``workspace_id`` that hangs off a workspace table by foreign key:
its rule names the parent and the column, and its rows are those of the workspace's parents.

Keyed by table name, because schemas drift: ``skill_files`` has ``workspace_id`` on Automatos
Cloud and is a child of ``skills`` on a fresh local edition, and some tables exist on one
schema only (``kb_entities``, ``harness_prescriptions``). One rule serves every schema.

Secret columns are named here and removed from every export, the GDPR export included
(§5.6); the export also drops any column whose name ``rules.SECRET_COLUMN_NAME`` flags that a
rule does not name. Derived columns (vectors, search indexes) stay behind: the target
re-indexes documents and memories with its own embeddings (§1).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional

from core.models.mobile_devices import SECRET_COLUMNS as MOBILE_DEVICE_SECRET_COLUMNS
from services.workspace_move.rules import (
    AGENTS, AUTO, BOARD, CHATS, DELIVERABLES, KNOWLEDGE, MEMORY, TableRule, child, moved, not_moved,
)

_EMBEDDING = ("embedding",)
_CREDENTIAL_LINK = ("credential_id",)

_MOVED = (
    # ── Auto ────────────────────────────────────────────────────────────────
    moved("business_profiles", AUTO, "The business profile Auto works from."),
    moved("personas", AUTO, "The workspace's own personas; a shared persona travels by slug in the agent bundles."),
    moved("proactive_person_settings", AUTO, "Each person's quiet hours and pause for Auto's check-ins."),
    # ── Agents and skills ───────────────────────────────────────────────────
    moved("agents", AGENTS, "The workspace's agents, Auto's row included (auto.json names it); "
          "bundles/agents.jsonl carries what each one references.",
          derived_columns=("semantic_embedding", "semantic_text_hash")),
    moved("agent_blueprints", AGENTS, "The rules the workspace's agents are created with."),
    moved("skills", AGENTS, "The workspace's own skills; a global skill an agent uses is named in refs/skills.jsonl."),
    moved("workspace_enabled_skills", AGENTS, "Which skills the workspace has switched on."),
    moved("workspace_enabled_plugins", AGENTS, "Which plugins the workspace has switched on (by slug in refs/)."),
    moved("teams", AGENTS, "The workspace's agent teams."),
    moved("routing_rules", AGENTS, "Which agent or Playbook an incoming message goes to."),
    moved("workflow_recipes", AGENTS, "The workspace's Playbooks; bundles/playbooks.jsonl carries their portable form. "
          "A trigger's webhook id (its URL's only key) and signing secret stay behind.",
          redacted_keys=MappingProxyType({"schedule_config": ("webhook_id", "webhook_secret")})),
    moved("workflows", AGENTS, "Workflow definitions api/workflows.py still serves."),
    moved("patterns", AGENTS, "Coordination patterns the workspace defined (api/patterns.py)."),
    moved("voice_profiles", AGENTS, "The voices the workspace's agents speak with."),
    child("agent_skills", "agents", "agent_id", AGENTS, "Which skills each agent uses, in priority order."),
    child("agent_assigned_plugins", "agents", "agent_id", AGENTS, "Which plugins each agent uses (by slug in refs/)."),
    child("agent_app_assignments", "agents", "agent_id", AGENTS,
          "Which Composio apps each agent may use, by app name; the owner reconnects the apps."),
    child("agent_tool_assignments", "agents", "agent_id", AGENTS,
          "Each agent's app assignments by name; the credential each one used stays behind.",
          secret_columns=_CREDENTIAL_LINK),
    child("agent_tool_assignments_v2", "agents", "agent_id", AGENTS, "Each agent's platform tool assignments."),
    child("skill_files", "skills", "skill_id", AGENTS, "A skill's files (scoped on Automatos Cloud, a child locally)."),
    child("skill_versions", "skills", "skill_id", AGENTS, "A skill's version history."),
    child("workflow_agents", "workflows", "workflow_id", AGENTS, "Which agents a workflow uses."),
    # ── Knowledge ───────────────────────────────────────────────────────────
    moved("documents", KNOWLEDGE, "Documents; each file goes in files/ by content hash and is re-indexed on the target."),
    moved("knowledge_items", KNOWLEDGE, "Knowledge items; re-embedded on the target.", derived_columns=_EMBEDDING),
    moved("kb_formulas", KNOWLEDGE, "Formulas extracted into knowledge items."),
    moved("kb_images", KNOWLEDGE, "Images extracted into knowledge items.", derived_columns=("visual_embedding",)),
    moved("kb_tables", KNOWLEDGE, "Tables extracted into knowledge items."),
    moved("kb_entities", KNOWLEDGE, "Knowledge Graph entities (Automatos Cloud's schema)."),
    moved("kb_types", KNOWLEDGE, "Knowledge item types (Automatos Cloud's schema)."),
    moved("entity_relationships", KNOWLEDGE, "Knowledge Graph relationships (Automatos Cloud's schema)."),
    moved("workspace_graphs", KNOWLEDGE, "The workspace's Knowledge Graph files."),
    # ── Memory ──────────────────────────────────────────────────────────────
    moved("memory_short_term", MEMORY, "Short-term memory (L2); durable and field memory travel in memory/."),
    moved("proactive_calibrations", MEMORY, "What each person taught Auto's check-ins."),
    moved("social_voice_examples", MEMORY, "Drafts and the approved text Auto learns the brand voice from."),
    moved("memory_profiles", MEMORY, "The workspace's own memory profiles (what distillation keeps, by domain); "
          "the seeded ones are the platform's and have no workspace."),
    moved("end_users", MEMORY, "A product's end users and the consent recorded for them; their memories travel "
          "in memory/ with the rest. The key that recorded it stays behind, as keys do.",
          secret_columns=("api_key_id",)),
    # ── Chats ───────────────────────────────────────────────────────────────
    moved("chats", CHATS, "Chats; the Auto thread becomes the target's Auto thread."),
    moved("messages", CHATS, "The chats' messages."),
    moved("pinned_documents", CHATS, "Documents pinned to a chat."),
    child("votes", "chats", "chat_id", CHATS, "Thumbs up and down on messages."),
    # ── Deliverables ────────────────────────────────────────────────────────
    moved("deliverables", DELIVERABLES, "Deliverables; each file goes in files/ by content hash."),
    moved("document_templates", DELIVERABLES, "The templates the workspace's documents are rendered from."),
    moved("blog_posts", DELIVERABLES, "Blog posts the workspace wrote."),
    moved("content_provenance", DELIVERABLES, "The AI-generated marks of the workspace's files, by hash."),
    # ── Board (off by default, §8-Q3) ───────────────────────────────────────
    moved("board_tasks", BOARD, "Board tickets."),
    moved("agent_reports", BOARD, "Agents' reports."),
    moved("workspace_ticket_counters", BOARD, "The next ticket number, so moved tickets keep theirs."),
    moved("agent_scheduled_tasks", BOARD, "Agents' scheduled jobs; with the Board because they run again on the target."),
    moved("watches", BOARD, "Watches; with the Board because they act again on the target."),
    moved("social_campaigns", BOARD, "The social plan; with the Board because approved posts publish again on the target."),
    moved("social_posts", BOARD, "Social posts; with the Board because approved posts publish again on the target."),
    moved("social_topics", BOARD, "The social plan's topics."),
)

_TELEMETRY = "Telemetry or history the new install rebuilds or does not need (§3)."
_CONNECTION = "A connection or credential of this install; the owner reconnects on the new one (§3)."
_BILLING = "Billing and credits stay with the install that holds the payment (§3)."
_PEOPLE = "People and access of this install; the importing person owns what arrives."
_THEMES = "Installed themes are signed packages of this install; the owner installs the theme on the new one."
_REBUILT = "Rebuilt on the target when it re-indexes the documents from their files."

_NOT_MOVED = (
    not_moved("credentials", _CONNECTION, secret_table=True, secret_columns=("encrypted_data",)),
    not_moved("sdk_api_keys", _CONNECTION, secret_table=True, secret_columns=("key_hash",)),
    not_moved("workspace_signing_secrets", "The secret this install signs callbacks with; the new install makes "
              "its own, and the receiver is given it.", secret_table=True, secret_columns=("secret_encrypted",)),
    not_moved("run_callbacks", "Where this install POSTs a run's ending, and how the delivery went."),
    not_moved("user_api_keys", _CONNECTION, secret_table=True, secret_columns=("encrypted_key",)),
    not_moved("channel_connections", _CONNECTION, secret_table=True, secret_columns=("webhook_url",)),
    not_moved("composio_entities", _CONNECTION, secret_table=True),
    not_moved("mobile_devices", "A phone paired to this install; it pairs again with the new one.",
              secret_columns=(*MOBILE_DEVICE_SECRET_COLUMNS, "expo_push_token")),
    not_moved("mobile_digest_posts", "This install's record of which phone digests it posted."),
    not_moved("mobile_push_tickets", _TELEMETRY),
    not_moved("cli_hosts", "A machine paired to this install for session mode.",
              secret_columns=("pairing_code_hash", "token_hash")),
    not_moved("sites", _CONNECTION, secret_columns=("secrets",)),
    not_moved("workspace_tool_config", _CONNECTION, secret_columns=_CREDENTIAL_LINK),
    not_moved("database_knowledge_sources", _CONNECTION, secret_columns=_CREDENTIAL_LINK),
    not_moved("nl2sql_benchmark_runs", "Benchmarks of a database connection that stays behind."),
    not_moved("nl2sql_training_examples", "Examples for a database connection that stays behind."),
    not_moved("cloud_documents", "Sync state of a cloud-drive connection; the documents themselves move."),
    not_moved("cloud_sync_jobs", _TELEMETRY),
    not_moved("approval_grants", "Approvals are decisions made on this install (§5.3)."),
    # In its own schema since PRD-276 §10-7, so the public-schema probe no longer sees it; the rule
    # stays so that the table is still classified wherever it sits.
    not_moved("jobs", "Work queued on this install finishes here; the new install rebuilds its schedules "
              "from what it imports (PRD-276 §6.8)."),
    not_moved("credit_ledger", _BILLING),
    not_moved("credit_balances", _BILLING),
    not_moved("billing_accounts", _BILLING),
    not_moved("billing_events", _BILLING),
    not_moved("workspace_members", _PEOPLE),
    not_moved("workspace_invitations", _PEOPLE, secret_columns=("token",)),
    not_moved("workspace_shares", _PEOPLE),
    not_moved("workspace_models", "The models this install offers; the target keeps its own (§8-Q7)."),
    not_moved("llm_models", "The model catalogue of this install; the target keeps its own (§8-Q7)."),
    not_moved("audit_logs", _TELEMETRY),
    not_moved("permission_bypass_log", _TELEMETRY),
    not_moved("error_events", _TELEMETRY),
    not_moved("heartbeat_results", _TELEMETRY),
    not_moved("llm_usage", _TELEMETRY),
    not_moved("tool_execution_logs", _TELEMETRY),
    not_moved("notifications", _TELEMETRY),
    not_moved("notification_preferences", "Delivery settings bound to this install's people and channels."),
    not_moved("routing_decisions", _TELEMETRY),
    not_moved("unrouted_events", _TELEMETRY),
    not_moved("memory_access_log", _TELEMETRY),
    not_moved("substrate_metric_events", _TELEMETRY),
    not_moved("voice_calls", _TELEMETRY),
    not_moved("voice_turns", _TELEMETRY),
    not_moved("digest_feedback", _TELEMETRY),
    not_moved("rag_feedback", "Feedback on search answers, naming chunks the target rebuilds."),
    not_moved("proactive_runs", _TELEMETRY),
    not_moved("proactive_suggestions", "Suggestions raised on this install; they expire and are raised again."),
    not_moved("harness_runs", _TELEMETRY),
    not_moved("harness_task_ledger", _TELEMETRY),
    not_moved("harness_prescriptions", _TELEMETRY),
    not_moved("learning_outcomes", _TELEMETRY),
    not_moved("tool_routing_affinities", "Learned tool-routing weights; relearned on the target."),
    not_moved("tool_routing_edges", "Learned tool-routing edges; relearned on the target."),
    not_moved("orchestration_runs", "Mission run history (§3)."),
    not_moved("orchestration_archive", "Archived Mission runs (§3)."),
    not_moved("recipe_executions", "Playbook run history (§3)."),
    not_moved("workflow_executions", "Workflow run history (§3)."),
    not_moved("social_post_stats", "Network numbers read on this install."),
    not_moved("document_chunks", _REBUILT),
    not_moved("document_entities", _REBUILT),
    not_moved("document_relationships", _REBUILT),
    not_moved("codegraph_projects", "Code graphs are rebuilt from the repo (§3)."),
    not_moved("codegraph_files", "Code graphs are rebuilt from the repo (§3)."),
    not_moved("codegraph_symbols", "Code graphs are rebuilt from the repo (§3)."),
    not_moved("codegraph_relationships", "Code graphs are rebuilt from the repo (§3)."),
    not_moved("codegraph_query_logs", _TELEMETRY),
    not_moved("theme_installations", _THEMES),
    not_moved("theme_packages", _THEMES),
    not_moved("theme_choices", _THEMES),
    not_moved("theme_looks", _THEMES),
    not_moved("theme_opt_ins", _THEMES),
    not_moved("theme_workspace_settings", _THEMES),
    not_moved("widget_installations", "Widgets installed for this install's people."),
    not_moved("workspace_move_jobs", "This install's own export and import jobs."),
    not_moved("v_workspace_outputs", "A view over blog_posts, agent_reports and deliverables, which move themselves."),
    # Children of tables that stay behind, or history of ones that move.
    child("composio_connections", "composio_entities", "entity_id", None, _CONNECTION),
    child("trigger_subscriptions", "composio_entities", "entity_id", None, _CONNECTION),
    child("credential_audit_logs", "credentials", "credential_id", None, _TELEMETRY, secret_columns=_CREDENTIAL_LINK),
    child("tenant_tool_config", "credentials", "credential_id", None, _CONNECTION, secret_columns=_CREDENTIAL_LINK),
    child("database_query_audit", "database_knowledge_sources", "source_id", None, _TELEMETRY),
    child("database_query_templates", "database_knowledge_sources", "source_id", None, _CONNECTION),
    child("semantic_dimensions", "database_knowledge_sources", "source_id", None, _CONNECTION),
    child("semantic_metrics", "database_knowledge_sources", "source_id", None, _CONNECTION),
    child("nl2sql_benchmark_results", "nl2sql_benchmark_runs", "benchmark_run_id", None, _TELEMETRY),
    child("document_usage", "documents", "document_id", None, _TELEMETRY),
    child("marketplace_cloud_items", "agents", "agent_id", None, "Which rows a Sync with Automatos Cloud wrote."),
    child("marketplace_installs", "agents", "cloned_agent_id", None, "Marketplace install records of this install."),
    child("orchestration_events", "orchestration_runs", "run_id", None, "Mission run history (§3)."),
    child("orchestration_tasks", "orchestration_runs", "run_id", None, "Mission run history (§3)."),
    child("social_post_targets", "social_posts", "post_id", None, "Publishing state on this install's networks."),
    child("skill_audit_log", "skills", "skill_id", None, _TELEMETRY),
    child("theme_review_events", "theme_packages", "package_id", None, _THEMES),
    child("tool_usage_logs", "agents", "agent_id", None, _TELEMETRY),
    child("watch_events", "watches", "watch_id", None, _TELEMETRY),
)


def _index(rules: Iterable[TableRule]) -> Mapping[str, TableRule]:
    found: Dict[str, TableRule] = {}
    for rule in rules:
        if rule.table in found:
            raise ValueError(f"{rule.table} is classified twice")
        found[rule.table] = rule
    return MappingProxyType(found)


TABLE_RULES: Mapping[str, TableRule] = _index((*_MOVED, *_NOT_MOVED))
# Tables that are nothing but secrets and connections: no export carries them (§5.6).
SECRET_TABLES = frozenset(table for table, rule in TABLE_RULES.items() if rule.secret_table)


def rule_for(table: str) -> Optional[TableRule]:
    """The rule of ``table``; None when no rule classifies it."""
    return TABLE_RULES.get(table)


def secret_columns(table: str) -> tuple:
    """The columns of ``table`` no export carries (none for an unclassified table)."""
    rule = TABLE_RULES.get(table)
    return rule.secret_columns if rule else ()


__all__ = ["SECRET_TABLES", "TABLE_RULES", "rule_for", "secret_columns"]
