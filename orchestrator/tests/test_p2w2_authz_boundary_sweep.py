"""PRD-195 S2 (P2-14) — the authZ boundary sweep: every mutating route classified.

Source-of-truth-driven, the PRD-143 S16 idiom: the committed route manifest
(``reports/route-manifest.json``) is the contract, the LIVE app (probed in a
clean subprocess by ``tests/authz_sweep_probe.py``, DB-free — the
``test_route_manifest.py`` precedent) is the ground truth, and every mutating
route (POST/PUT/PATCH/DELETE) must be classified **exactly one** way:

(a) carries ``require_workspace_permission`` (the S2 gate — swept across the
    families by S3–S6);
(b) super-admin-locked router-wide (the PRD-143 obs tier — classified, never
    relaxed);
(c) ``require_workspace_admin``-gated (PRD-185 S12 — classified, never relaxed);
(d) own non-hybrid auth (widget plane key auth, HMAC/secret webhooks, the
    Shopify machine lane) — enumerated explicitly, and structurally proven to
    NOT ride the shared hybrid dependency;
(e) public by design (``accept-invitation`` — verifies a Clerk JWT directly);
(f) admin-gated in the handler body via the shared ``caller_is_admin``
    helpers (the 8 admin-flavoured routers) — enumerated, and structurally
    proven by the endpoint source;
(g) its own explicit in-handler gate (credentials ``/resolve`` — admin/API-key
    only, S7 leaves its gate in place);
(h) TEMPORARY: a ``PENDING_FAMILY_*`` block, the visible debt each family
    story (S3–S6) shrinks to zero — a pending route that gains the gate must
    leave its block (asserted), and deleting an entry without gating the
    route fails the exactly-one classification.

An unclassified mutating route fails CI forever after — new routers can never
ship decorative roles again (dossier G.1 becomes a gate, not a metric).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ORCH_ROOT / "reports" / "route-manifest.json"

MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# ---------------------------------------------------------------------------
# Classification sets — every entry commented with WHY it bypasses the gate
# ---------------------------------------------------------------------------

# Stale manifest rows: routes DELETED from the tree (PRD-172 F006 removed the
# unscoped workflow-execute lane) whose manifest regeneration never landed.
# They stay in the committed manifest because the frontend route-contract
# baseline still references the dead api-client callers; cleaning both sides
# belongs to the frontend workflow-surface cleanup. The sweep proves the app
# does NOT serve them.
STALE_MANIFEST_ENTRIES = {
    ("POST", "/api/workflows/execute"),
    ("POST", "/api/workflows/executions/"),
    ("POST", "/api/workflows/{workflow_id}/execute"),
}

# (e) Public by design — no workspace context exists yet for the caller.
PUBLIC_BY_DESIGN = {
    # Verifies the Clerk JWT directly; the invitee may not be a member of any
    # workspace yet (api/team.py public_router, dossier §B).
    ("POST", "/api/team/accept-invitation"),
}

# (d) Own non-hybrid auth. Structurally enforced below: none of these may
# carry the shared hybrid dependency.
OWN_AUTH_ROUTES = {
    # Bearer-token machine ingest (core/monitoring/automatos_alerts.py).
    ("POST", "/api/alerts/ingest"),
    # HMAC-verified webhooks (GITHUB_WEBHOOK_SECRET / Composio signature /
    # URL-as-secret workspace key / recipe webhook) — hardening posture is
    # PRD-194 (P2-13) territory; classified here, not re-gated.
    ("POST", "/api/code-graph/webhook/github"),
    ("POST", "/api/composio/webhook"),
    ("POST", "/api/github/webhook"),
    ("POST", "/api/webhooks/recipe/{webhook_id}"),
    ("POST", "/api/webhooks/ws/{workspace_key}"),
    # Shopify machine lane — _verify_internal_key shared-secret auth.
    ("POST", "/api/shopify/connect"),
    ("POST", "/api/shopify/deactivate"),
    ("POST", "/api/shopify/events"),
    ("POST", "/api/shopify/provision"),
    ("POST", "/api/shopify/sync"),
    # Widget SDK plane — its own key auth (api/widgets/auth.py widget_auth +
    # require_permission, empty=deny since S1).
    ("POST", "/api/widgets/api/widget/documents/search"),
    ("POST", "/api/widgets/auth"),
    ("POST", "/api/widgets/callback"),
    ("POST", "/api/widgets/chat"),
    ("POST", "/api/widgets/data/execute"),
    ("POST", "/api/widgets/data/query"),
    ("POST", "/api/widgets/docs/search"),
}

# (f) Admin-gated in the handler body — the 8 admin-flavoured routers keep
# their in-handler `caller_is_admin` gates (super-admin-inclusive since S1).
# Structurally enforced below via the endpoint source.
ADMIN_GATED_IN_HANDLER = {
    ("DELETE", "/api/admin/plugins/{plugin_id}"),
    ("DELETE", "/api/admin/prompts/{prompt_id}/versions/{version_id}"),
    ("DELETE", "/api/admin/workspaces/{workspace_id}"),
    ("DELETE", "/api/marketplace/items/{item_id}"),
    ("PATCH", "/api/admin/prompts/{prompt_id}/futureagi-toggle"),
    ("POST", "/api/admin/plugins/backfill-categories"),
    ("POST", "/api/admin/plugins/import-github"),
    ("POST", "/api/admin/plugins/upload"),
    ("POST", "/api/admin/plugins/{plugin_id}/approve"),
    ("POST", "/api/admin/plugins/{plugin_id}/deactivate"),
    ("POST", "/api/admin/plugins/{plugin_id}/reject"),
    ("POST", "/api/admin/prompts/{prompt_id}/assess"),
    ("POST", "/api/admin/prompts/{prompt_id}/rollback"),
    ("POST", "/api/admin/prompts/{prompt_id}/versions"),
    ("POST", "/api/admin/prompts/{prompt_id}/versions/{version_id}/activate"),
    ("POST", "/api/admin/workspaces/{workspace_id}/pause"),
    ("POST", "/api/admin/workspaces/{workspace_id}/purge"),
    ("POST", "/api/admin/workspaces/{workspace_id}/restore"),
    ("POST", "/api/admin/workspaces/{workspace_id}/resume"),
    ("POST", "/api/marketplace/items/{item_id}/approve"),
    ("POST", "/api/marketplace/items/{item_id}/toggle-featured"),
    # Skills git import is admin-only until a safe user-facing flow exists
    # (PRD-70 FIX-01).
    ("POST", "/api/v1/skills/sources/git"),
    ("PUT", "/api/widget-marketplace/widgets/{widget_id}/approve"),
    ("PUT", "/api/widget-marketplace/widgets/{widget_id}/suspend"),
}

# (g) Own explicit in-handler gate.
OWN_GATE_IN_HANDLER = {
    # Returns DECRYPTED secrets: admin / env-API-key only, gated in the
    # handler; the NULL-workspace BOLA on its lookups is S7's fix.
    ("POST", "/api/credentials/resolve"),
}

# The agent-tool RBAC fossil (api/permissions.py, vocabulary #5) — unmounted
# and deleted by S8 together with this set and its manifest rows.
FOSSIL_PENDING_DELETE = {
    ("DELETE", "/permissions/revoke"),
    ("POST", "/permissions/assign"),
    ("POST", "/permissions/bulk-assign"),
}

# ---------------------------------------------------------------------------
# (h) PENDING blocks — the visible debt. Each family story (S3–S6) gates its
# routes and shrinks its block to ZERO. Green at every merge; a gated route
# still listed here fails; an entry deleted without gating fails.
# ---------------------------------------------------------------------------

PENDING_FAMILY_AGENTS = {
    ("DELETE", "/api/agents/{agent_id}"),
    ("DELETE", "/api/agents/{agent_id}/skills/{skill_id}"),
    ("DELETE", "/api/chat/{chat_id}"),
    ("DELETE", "/api/composio/connections/{app_name}"),
    ("DELETE", "/api/composio/triggers/{subscription_id}"),
    ("DELETE", "/api/routing/rules/{rule_id}"),
    ("DELETE", "/api/tools/remove-from-workspace/{app_name}"),
    ("DELETE", "/api/v1/skills/agents/{agent_id}/skills"),
    ("DELETE", "/api/v1/skills/sources/{source_id}"),
    ("DELETE", "/api/v1/skills/{skill_id}"),
    ("DELETE", "/api/workspaces/{workspace_id}/personas/{persona_id}"),
    ("DELETE", "/api/workspaces/{workspace_id}/plugins/{plugin_id}"),
    ("DELETE", "/api/workspaces/{workspace_id}/skills/{skill_id}"),
    ("DELETE", "/api/workspaces/{workspace_id}/skills/{skill_id}/owned"),
    ("PATCH", "/api/chat/vote"),
    ("PATCH", "/api/chat/{chat_id}"),
    ("PATCH", "/api/workspaces/{workspace_id}/skills/{skill_id}"),
    ("POST", "/api/agents/"),
    ("POST", "/api/agents/batch-create"),
    ("POST", "/api/agents/bulk"),
    ("POST", "/api/agents/create-specialized"),
    ("POST", "/api/agents/reindex-embeddings"),
    ("POST", "/api/agents/{agent_id}/add-skills"),
    ("POST", "/api/agents/{agent_id}/execute"),
    ("POST", "/api/agents/{agent_id}/skills"),
    ("POST", "/api/agents/{agent_id}/switch-model"),
    ("POST", "/api/agents/{agent_id}/test-capabilities"),
    ("POST", "/api/chat"),
    ("POST", "/api/chat/voice"),
    ("POST", "/api/chat/{chat_id}/switch-agent"),
    ("POST", "/api/composio/agents/{agent_id}/apps/{app_name}/disable-all"),
    ("POST", "/api/composio/agents/{agent_id}/apps/{app_name}/enable-all"),
    ("POST", "/api/composio/connect/{app_name}"),
    ("POST", "/api/composio/connect/{app_name}/callback"),
    ("POST", "/api/composio/triggers/subscribe"),
    ("POST", "/api/learning/feedback"),
    ("POST", "/api/learning/feedback/process"),
    ("POST", "/api/models/estimate-cost"),
    ("POST", "/api/models/recommend"),
    ("POST", "/api/query/platform-help"),
    ("POST", "/api/rag/feedback"),
    ("POST", "/api/recommendations/generate"),
    ("POST", "/api/routing/corrections"),
    ("POST", "/api/routing/rules"),
    ("POST", "/api/routing/semantic/reindex"),
    ("POST", "/api/routing/triggers/setup"),
    ("POST", "/api/tools/add-to-workspace"),
    ("POST", "/api/tools/connect"),
    ("POST", "/api/tools/refresh-connections"),
    ("POST", "/api/tools/sync"),
    ("POST", "/api/tools/sync/backfill-params"),
    ("POST", "/api/tools/{app_name}/actions"),
    ("POST", "/api/v1/skills/admin/cleanup-old-mappings"),
    ("POST", "/api/v1/skills/agents/{agent_id}/skills"),
    ("POST", "/api/v1/skills/recommend"),
    ("POST", "/api/v1/skills/sources/{source_id}/rollback"),
    ("POST", "/api/v1/skills/sources/{source_id}/update"),
    ("POST", "/api/workspaces/{workspace_id}/personas"),
    ("POST", "/api/workspaces/{workspace_id}/plugins"),
    ("POST", "/api/workspaces/{workspace_id}/skills"),
    ("POST", "/api/workspaces/{workspace_id}/skills/create"),
    ("PUT", "/api/agents/{agent_id}"),
    ("PUT", "/api/agents/{agent_id}/model-config"),
    ("PUT", "/api/agents/{agent_id}/persona"),
    ("PUT", "/api/agents/{agent_id}/plugins"),
    ("PUT", "/api/composio/agents/{agent_id}/apps/{app_name}/features"),
    ("PUT", "/api/workspaces/{workspace_id}/personas/{persona_id}"),
}

PENDING_FAMILY_EXECUTION = {
    ("DELETE", "/api/missions/{mission_id}"),
    ("DELETE", "/api/v1/tasks/{task_id}"),
    ("DELETE", "/api/workflow-recipes/{recipe_id}"),
    ("DELETE", "/api/workflow-templates/{template_id}"),
    ("DELETE", "/api/workflows/cleanup/old"),
    ("DELETE", "/api/workflows/{workflow_id}"),
    ("PATCH", "/api/missions/{mission_id}/plan"),
    ("PATCH", "/api/v1/scheduled-tasks/{task_id}/status"),
    ("PATCH", "/api/v1/tasks/{task_id}"),
    ("PATCH", "/api/v1/tasks/{task_id}/status"),
    ("POST", "/api/harness/prescriptions/{rx_id}/approve"),
    ("POST", "/api/harness/prescriptions/{rx_id}/reject"),
    ("POST", "/api/missions"),
    ("POST", "/api/missions/import-plan"),
    ("POST", "/api/missions/upload"),
    ("POST", "/api/missions/{mission_id}/approve"),
    ("POST", "/api/missions/{mission_id}/cancel"),
    ("POST", "/api/missions/{mission_id}/field/query"),
    ("POST", "/api/missions/{mission_id}/pause"),
    ("POST", "/api/missions/{mission_id}/reject"),
    ("POST", "/api/missions/{mission_id}/replan"),
    ("POST", "/api/missions/{mission_id}/resume"),
    ("POST", "/api/missions/{mission_id}/save-as-routine"),
    ("POST", "/api/playbooks/mine"),
    ("POST", "/api/v1/tasks"),
    ("POST", "/api/v1/tasks/plan"),
    ("POST", "/api/v1/tasks/plan/refine"),
    ("POST", "/api/v1/tasks/{task_id}/approve"),
    ("POST", "/api/v1/tasks/{task_id}/reject"),
    ("POST", "/api/v1/tasks/{task_id}/run-now"),
    ("POST", "/api/workflow-recipes"),
    ("POST", "/api/workflow-recipes/install/{recipe_id}"),
    ("POST", "/api/workflow-recipes/submit"),
    ("POST", "/api/workflow-recipes/{recipe_id}/assess-quality"),
    ("POST", "/api/workflow-recipes/{recipe_id}/execute"),
    ("POST", "/api/workflow-recipes/{recipe_id}/executions/{execution_id}/cancel"),
    ("POST", "/api/workflow-recipes/{recipe_id}/learn"),
    ("POST", "/api/workflow-recipes/{recipe_id}/use"),
    ("POST", "/api/workflow-templates"),
    ("POST", "/api/workflow-templates/{template_id}/use"),
    ("POST", "/api/workflows"),
    ("POST", "/api/workflows/executions/{execution_id}/cancel"),
    ("POST", "/api/workflows/stream"),
    ("POST", "/api/workflows/{workflow_id}/cancel"),
    ("POST", "/api/workflows/{workflow_id}/duplicate"),
    ("POST", "/api/workflows/{workflow_id}/execute-advanced"),
    ("POST", "/api/workflows/{workflow_id}/pause"),
    ("POST", "/api/workflows/{workflow_id}/resume"),
    ("POST", "/v1/workflows/{workflow_id}/store-orchestration-event"),
    ("PUT", "/api/missions/approval-policy"),
    ("PUT", "/api/workflow-recipes/{recipe_id}"),
    ("PUT", "/api/workflow-templates/{template_id}"),
    ("PUT", "/api/workflows/{workflow_id}"),
}

PENDING_FAMILY_CONTENT = {
    ("DELETE", "/api/attachments/{attachment_id}"),
    ("DELETE", "/api/blog/posts/{post_id}"),
    ("DELETE", "/api/cloud-documents/connections/{connection_id}"),
    ("DELETE", "/api/code-graph/projects/{project_id}"),
    ("DELETE", "/api/context/rag/config/{config_id}"),
    ("DELETE", "/api/deliverables/{deliverable_id}"),
    ("DELETE", "/api/documents/templates/{template_id}"),
    ("DELETE", "/api/documents/{document_id}"),
    ("DELETE", "/api/documents/{document_id}/pin"),
    ("DELETE", "/api/knowledge/graph"),
    ("DELETE", "/api/knowledge/sources/database/{source_id}"),
    ("DELETE", "/api/knowledge/sources/database/{source_id}/examples/{example_id}"),
    ("DELETE", "/api/memory/{memory_id}"),
    ("DELETE", "/api/patterns/{pattern_id}"),
    ("DELETE", "/api/voice/profiles/{profile_id}"),
    ("PATCH", "/api/documents/{document_id}/team-access"),
    ("PATCH", "/api/knowledge/graph/community/{community_id}/label"),
    ("POST", "/api/attachments"),
    ("POST", "/api/blog/cover-image/upload"),
    ("POST", "/api/blog/missions"),
    ("POST", "/api/blog/posts"),
    ("POST", "/api/blog/posts/{post_id}/publish"),
    ("POST", "/api/blog/posts/{post_id}/unpublish"),
    ("POST", "/api/cloud-documents/connections/{connection_id}/select-folder"),
    ("POST", "/api/cloud-documents/connections/{connection_id}/sync"),
    ("POST", "/api/cloud-documents/rag/query"),
    ("POST", "/api/code-graph/index/github"),
    ("POST", "/api/code-graph/projects/{project_id}/ask"),
    ("POST", "/api/code-graph/projects/{project_id}/reindex"),
    ("POST", "/api/context/add"),
    ("POST", "/api/context/initialize"),
    ("POST", "/api/context/rag/config"),
    ("POST", "/api/context/rag/{config_id}/test"),
    ("POST", "/api/context/summarize"),
    ("POST", "/api/deliverables/retention"),
    ("POST", "/api/documents/analytics/track"),
    ("POST", "/api/documents/bulk-team-access"),
    ("POST", "/api/documents/generate"),
    ("POST", "/api/documents/preview-blocks"),
    ("POST", "/api/documents/rag/retrieve"),
    ("POST", "/api/documents/reprocess-all"),
    ("POST", "/api/documents/search"),
    ("POST", "/api/documents/templates"),
    ("POST", "/api/documents/templates/upload"),
    ("POST", "/api/documents/templates/{template_id}/preview"),
    ("POST", "/api/documents/upload"),
    ("POST", "/api/documents/{document_id}/pin"),
    ("POST", "/api/documents/{document_id}/reprocess"),
    ("POST", "/api/knowledge/graph/build"),
    ("POST", "/api/knowledge/graph/import"),
    ("POST", "/api/knowledge/items"),
    ("POST", "/api/knowledge/search"),
    ("POST", "/api/knowledge/share"),
    ("POST", "/api/knowledge/sources/database/"),
    ("POST", "/api/knowledge/sources/database/templates/{template_id}/execute"),
    ("POST", "/api/knowledge/sources/database/{source_id}/benchmark/run"),
    ("POST", "/api/knowledge/sources/database/{source_id}/examples"),
    ("POST", "/api/knowledge/sources/database/{source_id}/examples/import"),
    ("POST", "/api/knowledge/sources/database/{source_id}/introspect"),
    ("POST", "/api/knowledge/sources/database/{source_id}/query"),
    ("POST", "/api/knowledge/sources/database/{source_id}/query/sql"),
    ("POST", "/api/knowledge/sources/database/{source_id}/schema/refresh"),
    ("POST", "/api/knowledge/sources/database/{source_id}/semantic"),
    ("POST", "/api/knowledge/upload"),
    ("POST", "/api/memory"),
    ("POST", "/api/patterns/"),
    ("POST", "/api/policy/abtest/set"),
    ("POST", "/api/policy/{policy_id}/assemble"),
    ("POST", "/api/shopify/sync/orders/start"),
    ("POST", "/api/shopify/sync/products/start"),
    ("POST", "/api/voice/profiles"),
    ("POST", "/api/voice/profiles/clone"),
    ("POST", "/api/voice/profiles/{profile_id}/preview"),
    ("PUT", "/api/blog/posts/{post_id}"),
    ("PUT", "/api/context/rag/config/{config_id}"),
    ("PUT", "/api/documents/brand-kit"),
    ("PUT", "/api/documents/templates/{template_id}"),
    ("PUT", "/api/knowledge/sources/database/{source_id}/examples/{example_id}"),
    ("PUT", "/api/knowledge/sources/database/{source_id}/examples/{example_id}/verify"),
    ("PUT", "/api/policy/{policy_id}"),
    ("PUT", "/api/voice/profiles/{profile_id}"),
}

PENDING_FAMILY_WORKSPACE = {
    ("DELETE", "/api/api-keys/{key_id}"),
    ("DELETE", "/api/channels/{channel_id}"),
    ("DELETE", "/api/credentials/{credential_id}"),
    ("DELETE", "/api/keys/platform/{provider}"),
    ("DELETE", "/api/keys/{key_id}"),
    ("DELETE", "/api/system-settings/{setting_id}"),
    ("DELETE", "/api/widget-marketplace/reviews/{review_id}"),
    ("DELETE", "/api/widget-marketplace/widgets/{widget_id}/install"),
    ("DELETE", "/api/workspaces/{workspace_id}/canvas/sessions"),
    ("PATCH", "/api/sites/{site_id}"),
    ("PATCH", "/api/sites/{site_id}/settings"),
    ("PATCH", "/api/wizard/profile/{profile_id}"),
    ("POST", "/api/api-keys"),
    ("POST", "/api/bug-reports/"),
    ("POST", "/api/cache/clear/{namespace}"),
    ("POST", "/api/cache/invalidate/cloud/{connection_id}"),
    ("POST", "/api/channels"),
    ("POST", "/api/channels/{channel_id}/start"),
    ("POST", "/api/channels/{channel_id}/stop"),
    ("POST", "/api/channels/{channel_id}/test"),
    ("POST", "/api/credentials/"),
    ("POST", "/api/credentials/cache/clear"),
    ("POST", "/api/credentials/{credential_id}/test"),
    ("POST", "/api/emails"),
    ("POST", "/api/emails/{email_id}/reply"),
    ("POST", "/api/keys"),
    ("POST", "/api/keys/{key_id}/test"),
    ("POST", "/api/marketplace/items/{item_id}/install"),
    ("POST", "/api/marketplace/llm/models/{model_id:path}/install"),
    ("POST", "/api/marketplace/llm/models/{model_id:path}/uninstall"),
    ("POST", "/api/marketplace/submit"),
    ("POST", "/api/notifications/read-all"),
    ("POST", "/api/notifications/{notification_id}/dismiss"),
    ("POST", "/api/notifications/{notification_id}/read"),
    ("POST", "/api/openrouter/sync"),
    ("POST", "/api/sites"),
    ("POST", "/api/sites/{site_id}/callback/test"),
    ("POST", "/api/system-settings/"),
    ("POST", "/api/system-settings/bulk-update"),
    ("POST", "/api/system-settings/reset-to-defaults"),
    ("POST", "/api/system/agent/{agent_id}/execute"),
    ("POST", "/api/system/config"),
    ("POST", "/api/system/learning-state/update"),
    ("POST", "/api/system/performance-test"),
    ("POST", "/api/system/rag"),
    ("POST", "/api/system/rag/{config_id}/test"),
    ("POST", "/api/teams"),
    ("POST", "/api/widget-marketplace/widgets"),
    ("POST", "/api/widget-marketplace/widgets/{widget_id}/install"),
    ("POST", "/api/widget-marketplace/widgets/{widget_id}/reviews"),
    ("POST", "/api/widget-marketplace/widgets/{widget_id}/submit"),
    ("POST", "/api/wizard/plan/{profile_id}"),
    ("POST", "/api/wizard/scan/{profile_id}"),
    ("POST", "/api/wizard/scrape/{profile_id}"),
    ("POST", "/api/wizard/start"),
    ("POST", "/api/workspaces/{workspace_id}/canvas/commit"),
    ("POST", "/api/workspaces/{workspace_id}/canvas/sessions"),
    ("POST", "/api/workspaces/{workspace_id}/exec"),
    ("POST", "/api/workspaces/{workspace_id}/github/clone"),
    ("PUT", "/api/channels/{channel_id}"),
    ("PUT", "/api/credentials/{credential_id}"),
    ("PUT", "/api/keys/platform"),
    ("PUT", "/api/notification-preferences"),
    ("PUT", "/api/notification-preferences/"),
    ("PUT", "/api/settings/onboarding-agents/{slug}"),
    ("PUT", "/api/system-settings/{setting_id}"),
    ("PUT", "/api/system/config/{config_key}"),
    ("PUT", "/api/widget-marketplace/reviews/{review_id}"),
    ("PUT", "/api/widget-marketplace/widgets/{widget_id}"),
    ("PUT", "/api/workspaces/current/byok-preferences"),
    ("PUT", "/api/workspaces/current/integrations"),
    ("PUT", "/api/workspaces/current/orchestrator"),
    ("PUT", "/api/workspaces/{workspace_id}/files/content"),
}

PENDING_BLOCKS = {
    "PENDING_FAMILY_AGENTS": PENDING_FAMILY_AGENTS,
    "PENDING_FAMILY_EXECUTION": PENDING_FAMILY_EXECUTION,
    "PENDING_FAMILY_CONTENT": PENDING_FAMILY_CONTENT,
    "PENDING_FAMILY_WORKSPACE": PENDING_FAMILY_WORKSPACE,
}

# The PRD-143 obs tier modules that serve mutating routes — their su lock is
# classified and additionally asserted per-route (never relaxed).
SU_LOCKED_ENDPOINT_MODULES = {
    "api.analytics_api",
    "api.analytics_charts",
    "api.llm_analytics",
    "api.memory_stats",
    "api.heartbeat",
    "api.reports",
}


# ---------------------------------------------------------------------------
# The probe — one clean subprocess, module-scoped
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_records():
    env = dict(os.environ)
    env.update(
        {
            # Unreachable Postgres: proves the probe (like dump_routes) is
            # DB-free; CI's real POSTGRES_* would work too, but the contract
            # is stronger this way.
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "59432",
            "POSTGRES_DB": "test",
            "DATABASE_URL": "postgresql://test:test@127.0.0.1:59432/test",
        }
    )
    proc = subprocess.run(
        [sys.executable, str(Path("tests") / "authz_sweep_probe.py")],
        cwd=str(ORCH_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, (
        f"authz probe failed (rc={proc.returncode})\nSTDERR:\n{proc.stderr[-3000:]}"
    )
    return json.loads(proc.stdout)


def _app_mutating(records):
    return {
        (r["method"], r["path"]): r
        for r in records
        if r["method"] in MUTATING_METHODS
    }


def _manifest_mutating():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        (r["method"], r["path"])
        for r in manifest["routes"]
        if r["method"] in MUTATING_METHODS
    }


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def test_manifest_and_app_agree(app_records):
    app_mut = set(_app_mutating(app_records))
    man_mut = _manifest_mutating()

    assert len(app_mut) >= 300, "mutating surface collapsed — probe broken?"

    # Stale rows are stale precisely because the app no longer serves them.
    served_stale = STALE_MANIFEST_ENTRIES & app_mut
    assert not served_stale, (
        f"'stale' manifest rows are actually served — reclassify: {sorted(served_stale)}"
    )

    missing_from_app = (man_mut - STALE_MANIFEST_ENTRIES) - app_mut
    assert not missing_from_app, (
        f"manifest mutating routes the app does not serve: {sorted(missing_from_app)}"
    )
    unmanifested = app_mut - man_mut
    assert not unmanifested, (
        "mutating routes served but missing from the committed manifest "
        f"(regenerate reports/route-manifest.json): {sorted(unmanifested)}"
    )


def test_manifest_sweep_no_unclassified_mutating_routes(app_records):
    app_mut = _app_mutating(app_records)

    pending_all = {}
    for block_name, block in PENDING_BLOCKS.items():
        for key in block:
            assert key not in pending_all, f"{key} in two PENDING blocks"
            pending_all[key] = block_name

    failures = []
    for key, rec in sorted(app_mut.items()):
        buckets = []
        if rec["perm"]:
            buckets.append(f"gated[{rec['perm']}]")
        if rec["su"]:
            buckets.append("su-locked")
        if rec["wsadmin"]:
            buckets.append("workspace-admin")
        if key in OWN_AUTH_ROUTES:
            buckets.append("own-auth")
        if key in PUBLIC_BY_DESIGN:
            buckets.append("public")
        if key in ADMIN_GATED_IN_HANDLER:
            buckets.append("admin-in-handler")
        if key in OWN_GATE_IN_HANDLER:
            buckets.append("own-gate")
        if key in FOSSIL_PENDING_DELETE:
            buckets.append("fossil")
        if key in pending_all:
            buckets.append(pending_all[key])

        if len(buckets) != 1:
            failures.append(f"{key[0]} {key[1]} -> {buckets or 'UNCLASSIFIED'}")
            continue

        bucket = buckets[0]
        if bucket == "own-auth" or bucket == "public":
            if rec["hybrid"]:
                failures.append(
                    f"{key[0]} {key[1]} classified {bucket} but rides the "
                    "shared hybrid dependency"
                )
        elif bucket == "admin-in-handler":
            if not rec["admin_in_handler"]:
                failures.append(
                    f"{key[0]} {key[1]} classified admin-in-handler but the "
                    "endpoint source carries no shared admin assertion"
                )
        elif bucket == "own-gate":
            if not rec["own_gate_in_handler"]:
                failures.append(
                    f"{key[0]} {key[1]} classified own-gate but the endpoint "
                    "source carries no explicit gate"
                )
        elif bucket.startswith("PENDING_"):
            if rec["perm"]:
                failures.append(
                    f"{key[0]} {key[1]} is gated ({rec['perm']}) — delete it "
                    f"from {bucket}"
                )

    assert not failures, (
        "authZ boundary sweep failures:\n  " + "\n  ".join(failures)
    )


def test_classification_sets_have_no_zombie_entries(app_records):
    """Every explicit classification entry must name a route the app actually
    serves — a deleted route must leave its set in the same PR."""
    app_mut = set(_app_mutating(app_records))
    for name, entries in {
        "PUBLIC_BY_DESIGN": PUBLIC_BY_DESIGN,
        "OWN_AUTH_ROUTES": OWN_AUTH_ROUTES,
        "ADMIN_GATED_IN_HANDLER": ADMIN_GATED_IN_HANDLER,
        "OWN_GATE_IN_HANDLER": OWN_GATE_IN_HANDLER,
        "FOSSIL_PENDING_DELETE": FOSSIL_PENDING_DELETE,
        **PENDING_BLOCKS,
    }.items():
        zombies = set(entries) - app_mut
        assert not zombies, f"{name} lists routes the app does not serve: {sorted(zombies)}"


def test_obs_tier_mutating_routes_stay_super_admin_locked(app_records):
    """PRD-143 lock honoured: every mutating route on an obs-tier module keeps
    require_super_admin — classified, never relaxed, never re-gated."""
    checked = 0
    for rec in app_records:
        if rec["method"] not in MUTATING_METHODS:
            continue
        if rec["module"] in SU_LOCKED_ENDPOINT_MODULES:
            checked += 1
            assert rec["su"], (
                f"{rec['method']} {rec['path']} on {rec['module']} lost its "
                "require_super_admin lock"
            )
    assert checked >= 10, f"su sweep went vacuous (checked={checked})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
