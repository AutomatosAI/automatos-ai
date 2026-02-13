# PRD-53: Webhook & Trigger System

**Version:** 1.0
**Status:** 🟢 Implementation Complete — Testing Required
**Date:** February 13, 2026
**Author:** Claude Code
**Prerequisites:** PRD-37 (SaaS Foundation — Workspaces), PRD-36 (Composio Integration)
**Branch:** `trigger-testing`

---

## Executive Summary

Two clean inbound webhook paths for connecting external services (Jira, GitHub, Slack, WhatsApp, Telegram, etc.) to Automatos without requiring Composio:

1. **Recipe Webhooks** — Each recipe with trigger type gets a unique URL. External services POST to it, triggering that specific recipe.
2. **General Workspace Webhook** — Single URL per workspace. Any incoming request is routed through UniversalRouter to the right agent (chatbot-style).

Both use the **URL-as-secret** pattern — the URL itself contains a 128-bit random key that acts as the credential. No additional auth headers required.

### Problem

- The recipe trigger UI offered "Composio App" as the primary option (users don't have Composio access)
- The "Custom Webhook" option generated a fake client-side URL that wasn't persisted
- No general workspace webhook for chatbot-style routing from external sources
- Composio triggers are useful for outbound tool execution but shouldn't be required for inbound webhooks

### Solution

| Path | Purpose | Auth |
|------|---------|------|
| `POST /api/webhooks/recipe/{webhook_id}` | Trigger a specific recipe | URL-as-secret (`webhook_id` = `uuid4().hex`) |
| `POST /api/webhooks/ws/{workspace_key}` | Route to any agent via UniversalRouter | URL-as-secret (`workspace_key` = `uuid4().hex`) |

---

## Architecture

### Request Flow — Recipe Webhook

```
External Service (GitHub, Jira, etc.)
    │
    ▼ POST /api/webhooks/recipe/{webhook_id}
    │
    ├── Look up recipe by schedule_config.webhook_id
    ├── Create RecipeExecution record
    ├── Dispatch execute_recipe_direct() as async task
    └── Return { execution_id } immediately
```

### Request Flow — Workspace Webhook

```
External Service / cURL / Integration
    │
    ▼ POST /api/webhooks/ws/{workspace_key}
    │
    ├── Look up workspace by webhook_key
    ├── WebhookIngestor → RequestEnvelope
    ├── UniversalRouter.route(envelope)
    │   ├── Tier 0: agent_id override in body
    │   ├── Tier 1: Routing cache
    │   ├── Tier 2: Rule-based matching
    │   ├── Tier 3: Trigger subscriptions
    │   └── Tier 4: LLM fallback
    │
    ├── Route → Agent: Execute synchronously, return result
    ├── Route → Workflow: Dispatch async, return execution_id
    └── No route: Return { routed: false }
```

### Data Model Changes

```
workspaces table
├── webhook_key: String(64), UNIQUE, nullable
│   └── Auto-generated uuid4().hex on workspace creation
│   └── Backfilled for existing workspaces via migration

workflow_templates table (recipes)
├── schedule_config (JSON)
│   └── webhook_id: String — auto-generated when type = "trigger"
│   └── Already existed, now always populated for trigger type
```

### New Enum Value

```python
class ChannelSource(str, Enum):
    CHATBOT = "chatbot"
    JIRA_TRIGGER = "jira_trigger"
    WEBHOOK = "webhook"        # ← NEW
```

---

## Implementation Status

### Part 1: Recipe Webhook UI — DONE

| File | Change | Status |
|------|--------|--------|
| `frontend/components/workflows/recipe-schedule-config.tsx` | Removed Composio trigger UI, shows real webhook URL or "save to generate" placeholder | ✅ |
| `frontend/components/workflows/create-recipe-modal.tsx` | Passes `webhookId` prop to RecipeScheduleConfig; added `webhook_id?: string` to schedule_config type | ✅ |
| `frontend/hooks/use-recipe-form.ts` | Extracts `webhook_id` from API response after create/update; exposes `lastSavedWebhookId` | ✅ |
| `orchestrator/api/workflow_recipes.py` | Guarded `_auto_register_trigger()` — skips Composio SDK when `source != 'composio'` | ✅ |

**Key changes:**
- Removed `TRIGGER_SOURCE_OPTIONS` (Composio App / Custom Webhook distinction)
- Removed Composio app dropdown, trigger dropdown, "Configure in Composio" button
- Trigger mode now shows webhook URL prominently with copy button
- Before save: shows placeholder "Save the recipe to generate a webhook URL"
- After save: shows real `{API_URL}/api/webhooks/recipe/{webhook_id}` with copy + usage example
- `trigger_config` is always sent (even empty `{}`) for trigger type — fixes backend validation

### Part 2: General Workspace Webhook — DONE

| File | Change | Status |
|------|--------|--------|
| `orchestrator/core/models/routing.py` | Added `WEBHOOK = "webhook"` to ChannelSource enum | ✅ |
| `orchestrator/core/models/workspaces.py` | Added `webhook_key` column (String(64), unique, nullable) | ✅ |
| `orchestrator/alembic/versions/20260213_add_workspace_webhook_key.py` | Migration: add column, backfill existing workspaces, create unique index | ✅ |
| `orchestrator/core/auth/hybrid.py` | Generate `webhook_key = uuid4().hex` during workspace provisioning | ✅ |
| `orchestrator/core/routing/ingestors/webhook.py` | **NEW** — WebhookIngestor: normalizes POST body → RequestEnvelope | ✅ |
| `orchestrator/api/webhooks.py` | **NEW** — `POST /api/webhooks/ws/{workspace_key}` endpoint | ✅ |
| `orchestrator/main.py` | Registered `general_webhooks_router` | ✅ |
| `orchestrator/api/workspaces.py` | Returns `webhook_url` and `webhook_key` in workspace response; auto-generates key if missing | ✅ |

**WebhookIngestor behavior:**
- Extracts content from `body.message`, `body.text`, or `body.content`
- Falls back to JSON stringify of full body
- `body.agent_id` → Tier-0 override (routes directly to specific agent)
- `body.source`, `body.channel`, `body.event_type`, `body.service` → metadata for routing rules

### Part 3: Settings UI — DONE

| File | Change | Status |
|------|--------|--------|
| `frontend/components/settings/WebhooksSettingsTab.tsx` | **NEW** — Shows workspace webhook URL with copy button, example POST, field docs | ✅ |
| `frontend/components/settings/SettingsPanel.tsx` | Added Webhooks tab (5-column layout) | ✅ |
| `frontend/components/workspace-provider.tsx` | Added `webhookUrl` and `webhookKey` to Workspace interface + state mapping | ✅ |

---

## Files Modified (Complete List)

### Modified (11 files)
| File | Lines Changed |
|------|---------------|
| `frontend/components/settings/SettingsPanel.tsx` | +8 -6 |
| `frontend/components/workflows/create-recipe-modal.tsx` | +2 -1 |
| `frontend/components/workflows/recipe-schedule-config.tsx` | +130 -200 (rewrite) |
| `frontend/components/workspace-provider.tsx` | +4 |
| `frontend/hooks/use-recipe-form.ts` | +15 -8 |
| `orchestrator/api/workflow_recipes.py` | +10 |
| `orchestrator/api/workspaces.py` | +14 -2 |
| `orchestrator/core/auth/hybrid.py` | +5 -2 |
| `orchestrator/core/models/routing.py` | +1 |
| `orchestrator/core/models/workspaces.py` | +3 |
| `orchestrator/main.py` | +2 |

### New (4 files)
| File | Purpose |
|------|---------|
| `frontend/components/settings/WebhooksSettingsTab.tsx` | Webhooks settings tab UI |
| `orchestrator/alembic/versions/20260213_add_workspace_webhook_key.py` | Database migration |
| `orchestrator/api/webhooks.py` | General workspace webhook endpoint |
| `orchestrator/core/routing/ingestors/webhook.py` | Webhook ingestor (BaseIngestor subclass) |

---

## API Reference

### Recipe Webhook

```
POST /api/webhooks/recipe/{webhook_id}
Content-Type: application/json

{
  "any": "json payload"
}

Response:
{
  "status": "dispatched",
  "execution_id": "wh-abc123def456",
  "recipe_id": 42,
  "recipe_name": "Process GitHub PR"
}
```

### Workspace Webhook

```
POST /api/webhooks/ws/{workspace_key}
Content-Type: application/json

{
  "message": "Check my latest emails",
  "agent_id": 5,           // optional — force route to specific agent
  "source": "slack",        // optional — metadata for routing rules
  "channel": "#support"     // optional — metadata for routing rules
}

Response (agent route):
{
  "status": "completed",
  "routed": true,
  "route_type": "agent",
  "agent_id": 5,
  "confidence": 0.95,
  "result": { ... }
}

Response (workflow route):
{
  "status": "dispatched",
  "routed": true,
  "route_type": "workflow",
  "workflow_id": 12,
  "execution_id": "ws-webhook-abc123"
}

Response (no route):
{
  "status": "received",
  "routed": false,
  "reason": "No route found — configure routing rules or add agents to your workspace."
}
```

---

## Security Considerations

| Concern | Mitigation |
|---------|-----------|
| Webhook key entropy | 128-bit random hex (`uuid4().hex` = 32 hex chars) — brute force infeasible |
| Key in URL | Standard webhook pattern (GitHub, Stripe, Slack all do this). HTTPS encrypts the URL in transit. |
| No auth headers | By design — URL-as-secret is simpler for webhook integrations |
| Key rotation | **Not yet implemented** — would need a rotation endpoint + grace period |
| Rate limiting | **Not yet implemented** — unauthenticated endpoints should have per-key throttling |
| Key exposure | Only returned to authenticated workspace members via `/api/workspaces/current` |

---

## What's Left To Do

### Pre-Release (Required)

- [ ] **Run alembic migration** — `alembic upgrade head` to add `webhook_key` column
- [ ] **End-to-end testing** — See test plan below
- [ ] **Verify no TypeScript build errors** — `npm run build` in frontend

### Post-Release (Follow-up)

- [ ] **Rate limiting** — Add per-key rate limits on unauthenticated webhook endpoints (suggested: 60 req/min per key)
- [ ] **Key rotation** — Endpoint to regenerate `webhook_key` / `webhook_id` with optional grace period for old key
- [ ] **Webhook logs** — UI to show recent webhook invocations per recipe / workspace (timestamp, status, payload preview)
- [ ] **Retry / dead letter** — If recipe execution fails, allow configurable retry or dead-letter queue
- [ ] **Signature verification** — Optional HMAC signature validation for webhook payloads (for services that support it)
- [ ] **Composio re-integration** — Once users have Composio access, add it back as an optional trigger source alongside webhooks (not instead of)

---

## Test Plan

### 1. Recipe Webhook (Create + Trigger)

```bash
# 1. Create a recipe with trigger type via UI
# 2. Verify webhook URL is shown after save
# 3. Copy the URL and trigger it:

curl -X POST {recipe_webhook_url} \
  -H 'Content-Type: application/json' \
  -d '{"test": true, "source": "manual_test"}'

# Expected: 200 with execution_id
# Verify: RecipeExecution record created, steps begin executing
```

### 2. General Workspace Webhook (Route to Agent)

```bash
# 1. Get workspace webhook URL from Settings > Webhooks tab
# 2. Send a message:

curl -X POST {workspace_webhook_url} \
  -H 'Content-Type: application/json' \
  -d '{"message": "check my latest emails"}'

# Expected: Routes to an agent, returns result synchronously

# 3. Test explicit agent override:

curl -X POST {workspace_webhook_url} \
  -H 'Content-Type: application/json' \
  -d '{"message": "hello", "agent_id": 1}'

# Expected: Routes directly to agent 1 (Tier-0 override)
```

### 3. No Composio Errors

```bash
# 1. Create a recipe with trigger type (no Composio configured)
# 2. Save recipe
# Expected: No errors in backend logs about Composio SDK
# Verify: _auto_register_trigger skips Composio, logs "Non-Composio trigger"
```

### 4. Edge Cases

- Empty body: `curl -X POST {url} -d '{}'` — should route with content `"{}"`
- Invalid workspace key: `curl -X POST /api/webhooks/ws/nonexistent` — should return 404
- Invalid recipe webhook_id: `curl -X POST /api/webhooks/recipe/nonexistent` — should return 404
- Recipe with no steps: should return gracefully (no crash)
- Workspace with no agents/routes: should return `{ routed: false }`

### 5. Migration

```bash
cd orchestrator
alembic upgrade head
# Verify: webhook_key column exists, existing workspaces have keys, unique index created
```

---

## Relationship to Other PRDs

| PRD | Relationship |
|-----|-------------|
| PRD-36 (Composio) | Composio triggers still work when `source = "composio"`. This PRD makes Composio optional, not removed. |
| PRD-37 (SaaS Foundation) | Extends workspace model with `webhook_key`. Uses existing auth/provisioning flow. |
| PRD-39 (Mem0) | No direct relationship. Webhook-triggered agent executions will use memory if agent has Mem0 configured. |
