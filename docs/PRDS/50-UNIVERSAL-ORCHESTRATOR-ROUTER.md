# PRD: Universal Orchestrator Router (PRD-50)

## Introduction

Transform the Automatos Orchestrator from a workflow-only executor into a **universal request router** that receives input from any channel — Chatbot, Jira triggers, Slack, WhatsApp, external APIs — normalizes it into a standard envelope, and intelligently routes it to the right pre-configured agent or recipe/workflow.

Today, the three consumers (Workflows, Chatbot, APIs) operate as separate systems with separate routing logic. The chatbot requires the user to select an agent. Webhooks are received but not dispatched (TODO at `api/composio.py:509`). This PRD unifies all input into a single routing pipeline, where the orchestrator makes a lightweight LLM classification, caches the decision, and hands off to a purpose-built agent that already has its tools assigned — minimizing cost by avoiding redundant reasoning at every step.

**Phase 1 scope:** Chatbot + Jira triggers. Slack, WhatsApp, and other external channels follow in Phase 2.

### Problem Statement

1. **Chatbot requires manual agent selection** — users must know which agent handles what, or it falls back to a generic default agent
2. **Webhook events are received but discarded** — `POST /api/composio/webhook` logs and returns "received" without triggering any agent or workflow
3. **No event-driven agent dispatch** — agents only execute via explicit workflow runs or chatbot conversations
4. **Three separate routing paths** — chatbot, workflows, and APIs each have independent logic, leading to duplicated intent classification

### Current Infrastructure (already built)

| Component | Location | Status |
|---|---|---|
| Composio webhook endpoint | `api/composio.py:467-516` | Receives + validates, no dispatch |
| TriggerSubscription model | `core/models/composio.py:88-108` | Schema exists, unused |
| Trigger subscribe/unsubscribe API | `api/composio.py:523-631` | Endpoints exist, no downstream effect |
| IntentClassifier (rule-based) | `core/services/intent_classifier.py` | 11 categories, regex patterns |
| ToolRouterService (LLM-based) | `services/tool_router_service.py` | Category-to-app mapping, caching |
| ActionClassifier (heuristic + LLM) | `modules/tools/capabilities/classifier.py` | Heuristic first, LLM fallback |
| LLMAgentSelector | `modules/orchestrator/llm/llm_agent_selector.py` | Function-calling agent selection |
| Chat streaming endpoint | `api/chat.py:150-293` | Agent-based streaming, uses `agentId` or default |
| Workflow execution pipeline | `api/workflows.py:923+` | 9-stage async execution |
| ComposioToolExecutor | `core/composio/tool_executor.py` | Execute Composio actions with validation |
| CodeGraph (repo indexing) | `modules/codegraph/` | Clone, parse, embed, search code |
| GitHub webhook handler | `api/github_webhooks.py` | PR events → workflow trigger (working pattern) |

---

## Goals

- Unify all input channels through a single routing pipeline with a standard request envelope
- Replace manual agent selection in chatbot with intelligent auto-routing (with manual override)
- Complete the webhook → agent/workflow dispatch (fill the TODO at `composio.py:509`)
- Enable event-driven agent execution via Composio triggers (starting with Jira)
- Reduce LLM cost per request by routing to pre-configured agents (one classification call, not full orchestration)
- Cache routing decisions so repeated patterns (e.g., "check my emails") resolve instantly
- Build the Jira → Coder Agent autonomous pipeline: ticket created → agent reads ticket → clones code → opens PR → moves ticket to "In Review"

---

## Architecture

```
Input Channels              Universal Router                    Execution Layer
──────────────              ────────────────                    ───────────────

Chatbot UI ────┐                                               ┌─ Agent (direct)
               │            ┌────────────────┐                 │   Pre-configured with
Jira Trigger ──┤──► Ingest ─┤  Normalize to  ├─► Route ──────┤   tools & permissions
               │            │  RequestEnvelope│    │           │
Slack* ────────┤            └────────────────┘    │           ├─ Recipe/Workflow
               │                                   │           │   Predefined pipeline
WhatsApp* ─────┤            ┌────────────────┐    │           │
               │            │ Routing Engine  │◄───┘           └─ Full Orchestration
API Call ──────┘            │                 │                    Decompose → Select
                            │ 1. Cache hit?   │                    → Execute (9-stage)
                            │ 2. Rule match?  │
                            │ 3. LLM classify │    ▲
                            └────────────────┘    │
                                                   │
                            User Override ─────────┘
                            (select agent / pin route)

* Phase 2
```

### Request Envelope (Standard Format)

Every input, regardless of source, is normalized to:

```python
class RequestEnvelope:
    id: UUID                          # Unique request ID
    source: ChannelSource             # "chatbot" | "jira_trigger" | "slack" | "whatsapp" | "api"
    content: str                      # The actual message or event summary
    raw_payload: Dict[str, Any]       # Original unmodified payload (for agents that need it)
    user: RequestUser                 # Who sent it
    workspace_id: UUID                # Tenant context
    metadata: Dict[str, Any]          # Channel-specific data (Jira issue fields, Slack channel, etc.)
    override_agent_id: Optional[int]  # User pinned a specific agent (skip routing)
    override_workflow_id: Optional[int] # Route to specific recipe
    conversation_id: Optional[UUID]   # For conversational channels (chatbot, Slack threads)
    timestamp: datetime
```

### Routing Decision Output

```python
class RoutingDecision:
    route_type: str             # "agent" | "workflow" | "orchestrate"
    agent_id: Optional[int]     # Direct agent dispatch
    workflow_id: Optional[int]  # Recipe/workflow dispatch
    confidence: float           # 0-1, how confident the router is
    reasoning: str              # Why this route was chosen (for audit/debugging)
    cached: bool                # Whether this came from cache
    intent_category: str        # EMAIL, CODE, PROJECT, etc.
```

### Routing Tiers (evaluated in order)

| Tier | Method | Latency | Cost | When Used |
|---|---|---|---|---|
| 0 | **User override** | 0ms | Free | User selected an agent or recipe explicitly |
| 1 | **Cache hit (Redis)** | <5ms | Free | Same intent pattern seen before, cached in Redis |
| 2 | **Rule-based** | <10ms | Free | Trigger source rules (jira_trigger → Bug Triage), keyword patterns |
| 3 | **LLM classification** | 200-500ms | ~$0.001 | Ambiguous requests, first-time patterns |

After Tier 3 resolves, the decision is cached in Redis (keyed by `routing:decision:{workspace_id}:{content_hash}`) so subsequent identical requests hit Tier 1. Cache persists across process restarts and is shared across workers.

---

## User Stories

### US-001: Create RequestEnvelope and RoutingDecision models

**Description:** As a developer, I need standard data models for the universal request envelope and routing decision so that all channels produce consistent input and the router produces consistent output.

**Acceptance Criteria:**
- [ ] `RequestEnvelope` Pydantic model with all fields defined above
- [ ] `RoutingDecision` Pydantic model with all fields defined above
- [ ] `ChannelSource` enum: `chatbot`, `jira_trigger`, `slack`, `whatsapp`, `api`, `workflow`
- [ ] `RequestUser` model: `id`, `email`, `name`, `auth_type`, `clerk_user_id`
- [ ] Models in new file `orchestrator/core/models/routing.py`
- [ ] Typecheck passes

---

### US-002: Build the Routing Engine

**Description:** As the orchestrator, I need a routing engine that takes a `RequestEnvelope` and returns a `RoutingDecision` using tiered evaluation (override → cache → rules → LLM).

**Acceptance Criteria:**
- [ ] New class `UniversalRouter` in `orchestrator/core/routing/engine.py`
- [ ] Tier 0: If `override_agent_id` or `override_workflow_id` set, return immediately
- [ ] Tier 1: Check `RoutingCache` (in-memory dict or Redis) keyed by `hash(normalized_content + source)`
- [ ] Tier 2: Evaluate source-based rules (e.g., `source=jira_trigger` → match TriggerSubscription for `agent_id`/`workflow_id`)
- [ ] Tier 2b: Evaluate keyword rules using existing `IntentClassifier` patterns from `core/services/intent_classifier.py`
- [ ] Tier 3: Call LLM with lightweight prompt: "Classify this request into one of: {agent_list_with_descriptions}. Return agent_id and confidence."
- [ ] Cache the LLM result for future Tier 1 hits
- [ ] Return `RoutingDecision` with `reasoning` field populated at every tier
- [ ] Routing decision logged to database for audit (`routing_decisions` table)
- [ ] Total routing latency < 50ms for Tiers 0-2, < 500ms for Tier 3

---

### US-003: Create routing cache with feedback learning

**Description:** As the system, I need to cache routing decisions and incorporate user corrections so routing accuracy improves over time.

**Acceptance Criteria:**
- [ ] `RoutingCache` class in `orchestrator/core/routing/cache.py`
- [ ] **Redis-backed** via the existing `RedisClient` infrastructure (`core.redis.client.get_redis_client()`), following `DatabaseCacheService` patterns (key prefixes, `setex()` for TTL, `incr()` for stats)
- [ ] Key prefixes: `routing:decision:` for cached decisions, `routing:corr:` for correction counters, `routing:stats:` for hit/miss counters
- [ ] Cache key: `routing:decision:{workspace_id}:{sha256(normalized_content + source)}`
- [ ] Cache TTL: 24 hours default, configurable via `ROUTING_CACHE_TTL_HOURS` env var
- [ ] Graceful degradation: all methods return None / no-op when Redis is unavailable (router skips Tier 1)
- [ ] Singleton `get_routing_cache()` function — all consumers (chat.py, composio.py, routing.py) share one instance
- [ ] `record_correction(workspace_id, content, source, correct_agent_id)` method — uses Redis `incr()` to count corrections; when same correction made 2+ times, updates the cached decision
- [ ] Corrections update cache: if same intent pattern gets corrected 2+ times to the same agent, update the cached route
- [ ] `RoutingDecision` table in database: `id`, `request_id`, `envelope_hash`, `route_type`, `agent_id`, `workflow_id`, `confidence`, `cached`, `was_corrected`, `corrected_agent_id`, `created_at`
- [ ] Migration for new table

---

### US-004: Build channel ingestors — Chatbot

**Description:** As a chatbot user, when I send a message, the orchestrator should auto-route it to the best agent instead of requiring me to select one from a dropdown.

**Acceptance Criteria:**
- [ ] New `ChatbotIngestor` class in `orchestrator/core/routing/ingestors/chatbot.py`
- [ ] Converts `ChatRequest` → `RequestEnvelope` with `source="chatbot"`
- [ ] Populates `user` from Clerk JWT (`RequestContext`)
- [ ] If `request.agentId` is set, maps to `override_agent_id` (manual override preserved)
- [ ] If `request.agentId` is NOT set, `override_agent_id=None` → router decides
- [ ] Modify `api/chat.py` stream_chat endpoint: before streaming, run `UniversalRouter.route(envelope)` to determine `effective_agent_id`
- [ ] Replace `get_default_agent_id()` fallback with router decision
- [ ] Agent dropdown in chatbot UI still works as manual override
- [ ] Conversation history maintained regardless of which agent responds

---

### US-005: Build channel ingestors — Jira Trigger

**Description:** As the system, when a Jira trigger webhook fires (new issue created), I need to normalize the event into a `RequestEnvelope` and dispatch it through the router.

**Acceptance Criteria:**
- [ ] New `JiraTriggerIngestor` class in `orchestrator/core/routing/ingestors/jira_trigger.py`
- [ ] Converts Composio webhook payload → `RequestEnvelope` with `source="jira_trigger"`
- [ ] Extracts from Jira payload: issue key, summary, description, issue type, priority, reporter, project
- [ ] Sets `content` to: `"[{issue_key}] {summary}\n\n{description}"`
- [ ] Populates `metadata` with full Jira fields
- [ ] Sets `user` from Jira reporter field (or system user if unavailable)
- [ ] Resolves `workspace_id` from Composio `entity_id` → `composio_entities` table → `workspace_id`

---

### US-006: Complete webhook → router dispatch

**Description:** As a developer, I need to replace the TODO at `api/composio.py:509` with actual routing logic that dispatches webhook events through the universal router.

**Acceptance Criteria:**
- [ ] Remove TODO comment at line 509
- [ ] After parsing webhook payload, create `RequestEnvelope` via appropriate ingestor (Jira, Slack, etc.)
- [ ] Call `UniversalRouter.route(envelope)` to get `RoutingDecision`
- [ ] If `route_type == "agent"`: execute agent directly with the envelope content as the task
- [ ] If `route_type == "workflow"`: call `execute_workflow(workflow_id, execution_data)` with envelope as context
- [ ] If `route_type == "orchestrate"`: trigger full 9-stage orchestration pipeline
- [ ] If no route found (no matching subscription, no cache, no rule, LLM low confidence): log warning, store event in `unrouted_events` table for manual review
- [ ] Return webhook response with routing decision summary
- [ ] Webhook processing is async (return 200 immediately, dispatch in background)

---

### US-007: Set up Jira trigger subscription

**Description:** As an admin, I need to register the `JIRA_NEW_ISSUE_TRIGGER` for the PILOT project so that new Jira tickets automatically fire webhook events to the orchestrator.

**Acceptance Criteria:**
- [ ] Configure Composio project webhook URL to point to orchestrator's `/api/composio/webhook` endpoint
- [ ] Set `COMPOSIO_WEBHOOK_SECRET` env var for signature verification
- [ ] Register `JIRA_NEW_ISSUE_TRIGGER` via Composio SDK with `trigger_config: { project_key: "PILOT" }`
- [ ] Create `TriggerSubscription` record in database linking trigger to the coder agent or Bug Triage workflow
- [ ] Add `COMPOSIO_WEBHOOK_SECRET` to `orchestrator/.env.example`
- [ ] Add management command or API endpoint to register/deregister triggers
- [ ] Verify: create a Jira ticket in PILOT → webhook fires → orchestrator receives and logs

---

### US-008: Build Jira Bug Triage autonomous workflow

**Description:** As the system, when a Jira bug ticket is created in PILOT, I need to autonomously: read the ticket, analyze the codebase, plan a fix, clone the repo, apply changes, open a PR, and move the Jira ticket to "In Review".

**Acceptance Criteria:**
- [ ] New workflow/recipe "Jira Bug Triage" created in database (or as a system workflow)
- [ ] Step 1 — **Read Ticket**: Use `JIRA_GET_ISSUE` to fetch full issue details (description, comments, attachments)
- [ ] Step 2 — **Analyze**: Agent analyzes bug description, identifies affected area of codebase using CodeGraph search
- [ ] Step 3 — **Plan**: Agent produces a fix plan (which files to change, what the fix looks like) and posts it as a Jira comment via `JIRA_ADD_COMMENT`
- [ ] Step 4 — **Clone & Fix**: Agent clones the repo (via CodeGraph or GitHub tools), creates a branch (`fix/{issue_key}`), applies code changes
- [ ] Step 5 — **Open PR**: Agent opens a draft pull request via `GITHUB_CREATE_PULL_REQUEST` with description referencing the Jira ticket
- [ ] Step 6 — **Update Jira**: Agent transitions the Jira ticket to "In Review" via `JIRA_UPDATE_ISSUE` and adds PR link as comment
- [ ] Each step logs progress for observability
- [ ] If any step fails, agent posts failure summary as Jira comment and stops (no partial PRs)
- [ ] Workflow linked to `JIRA_NEW_ISSUE_TRIGGER` via TriggerSubscription

---

### US-009: Add routing configuration API

**Description:** As an admin, I need API endpoints to manage routing rules, view routing decisions, and configure agent-to-intent mappings.

**Acceptance Criteria:**
- [ ] `GET /api/routing/decisions` — list recent routing decisions with filters (source, agent, confidence, was_corrected)
- [ ] `POST /api/routing/rules` — create a manual routing rule (source pattern + intent keywords → agent_id or workflow_id)
- [ ] `GET /api/routing/rules` — list all routing rules
- [ ] `DELETE /api/routing/rules/{rule_id}` — delete a routing rule
- [ ] `POST /api/routing/corrections` — record a user correction (request_id + correct_agent_id)
- [ ] `GET /api/routing/cache/stats` — cache hit rate, size, top cached routes
- [ ] All endpoints require `get_request_context_hybrid` auth
- [ ] Router in `api/routing.py`, registered in `main.py`

---

### US-010: Update chatbot UI for auto-routing

**Description:** As a chatbot user, I want to see which agent the orchestrator selected for my message, with the option to override.

**Acceptance Criteria:**
- [ ] When `agentId` is not selected in chatbot UI, display "Auto" or "Orchestrator" as the selected agent
- [ ] After message is sent and routed, show a subtle indicator: "Routed to: Communication Agent (confidence: 0.92)"
- [ ] If user disagrees, they can select a different agent from the dropdown for the next message — this triggers a correction via `POST /api/routing/corrections`
- [ ] Agent dropdown still works as before (pin to specific agent)
- [ ] "Auto" option added to agent dropdown as first/default option
- [ ] Verify in browser using dev-browser skill

---

## Functional Requirements

- **FR-1:** All input channels MUST normalize to `RequestEnvelope` before routing
- **FR-2:** Router MUST evaluate tiers in order: override → cache → rules → LLM. Stop at first match.
- **FR-3:** LLM routing decisions MUST be cached in Redis via the existing `RedisClient` infrastructure (`core.redis.client`). Cache key: `routing:decision:{workspace_id}:{sha256(normalized_content + source)}`. Follows `DatabaseCacheService` patterns (key prefixes, `setex()`, `incr()` stats).
- **FR-4:** User override MUST always take precedence over auto-routing at any point in a conversation
- **FR-5:** Webhook dispatch MUST be async — return 200 to Composio immediately, process in background
- **FR-6:** Every routing decision MUST be logged to database with full audit trail
- **FR-7:** User corrections MUST feed back into cache to improve future routing
- **FR-8:** Jira Bug Triage workflow MUST be fully autonomous: read → analyze → plan → fix → PR → update ticket
- **FR-9:** If Bug Triage fails at any step, it MUST post a failure comment on the Jira ticket and halt cleanly
- **FR-10:** Chatbot conversation history MUST be maintained across agent switches (if router selects a different agent mid-conversation)

---

## Security Requirements

| Channel | Auth Method | Validation |
|---|---|---|
| Chatbot | Clerk JWT (`Authorization: Bearer <token>`) | `get_request_context_hybrid` extracts user + workspace |
| Jira Trigger | Composio webhook signature | HMAC SHA256 via `COMPOSIO_WEBHOOK_SECRET` |
| Slack (Phase 2) | Slack signing secret | HMAC SHA256 via `SLACK_SIGNING_SECRET` |
| WhatsApp (Phase 2) | Composio webhook signature | Same as Jira trigger path |
| External API | API key (`X-API-Key` header) | Existing `require_api_key` middleware |

- **SR-1:** Webhook endpoints MUST validate signatures before processing. Reject with 401 on mismatch.
- **SR-2:** All routing endpoints MUST require authentication via `get_request_context_hybrid`
- **SR-3:** Agents executing via trigger dispatch MUST operate within the workspace context of the trigger subscription (not a global/admin context)
- **SR-4:** Rate limiting per channel: Chatbot — 60 req/min per user. Webhooks — 120 req/min per workspace. API — configurable per API key.
- **SR-5:** Routing decisions MUST NOT leak data across workspaces. Cache is workspace-scoped.
- **SR-6:** `RequestEnvelope.raw_payload` MUST be sanitized — strip any auth tokens or secrets from stored payloads
- **SR-7:** GitHub operations (clone, branch, PR) MUST use workspace-scoped credentials, not global tokens

---

## Non-Goals (Out of Scope)

- **Phase 2 channels** — Slack, WhatsApp, Telegram ingestors are not in this PRD. Only the ingestor interface is defined; implementations come later.
- **UI for routing rule management** — Phase 1 is API-only. A visual rule builder in the frontend is a future story.
- **Multi-workspace routing** — Each workspace has its own routing context. Cross-workspace routing is not supported.
- **Real-time streaming from webhook-triggered agents** — Webhook-triggered executions run async. Results are stored, not streamed. (Chatbot channel retains streaming.)
- **Custom LLM model for routing** — Uses the workspace's configured LLM. No fine-tuned routing model.
- **Automatic agent creation** — Router only selects from existing agents. It does not create new agents on-the-fly.

---

## Technical Considerations

### New Files

| File | Purpose |
|---|---|
| `orchestrator/core/models/routing.py` | RequestEnvelope, RoutingDecision, RoutingRule Pydantic + ORM models |
| `orchestrator/core/routing/engine.py` | `UniversalRouter` class — tiered routing logic |
| `orchestrator/core/routing/cache.py` | `RoutingCache` — Redis-backed caching via existing `RedisClient` infrastructure, with `get_routing_cache()` singleton |
| `orchestrator/core/routing/ingestors/base.py` | `BaseIngestor` abstract class |
| `orchestrator/core/routing/ingestors/chatbot.py` | `ChatbotIngestor` — ChatRequest → RequestEnvelope |
| `orchestrator/core/routing/ingestors/jira_trigger.py` | `JiraTriggerIngestor` — Composio webhook → RequestEnvelope |
| `orchestrator/api/routing.py` | Routing management API endpoints |

### Modified Files

| File | Change |
|---|---|
| `orchestrator/api/composio.py` | Replace TODO at line 509 with router dispatch |
| `orchestrator/api/chat.py` | Integrate router before agent selection (line 253) |
| `orchestrator/main.py` | Register routing router |
| `orchestrator/config.py` | Add `COMPOSIO_WEBHOOK_SECRET`, routing config vars |
| `orchestrator/.env.example` | Document new env vars |

### Database Migrations

- `routing_decisions` table — audit log of all routing decisions
- `routing_rules` table — user-defined routing rules (source pattern, intent keywords, target agent/workflow)
- `unrouted_events` table — events that couldn't be routed (for manual review)

### Dependencies

- Existing `RedisClient` (`core/redis/client.py`) + `get_redis_client()` is used for routing cache — same infrastructure as `DatabaseCacheService`
- Existing `IntentClassifier` (rule-based) is reused in Tier 2
- Existing `LLMAgentSelector` logic is adapted for Tier 3 (lighter-weight prompt)
- Existing `ComposioToolExecutor` handles all Composio action execution
- Existing `TriggerSubscription` model is used for trigger → agent/workflow mapping

### Performance

- Tier 0-2 routing: < 50ms (no LLM call)
- Tier 3 routing: < 500ms (single LLM call with small prompt)
- Cache hit rate target: > 70% after 1 week of usage
- Webhook processing: return 200 within 100ms, dispatch async

### Cost Model

| Scenario | LLM Calls | Estimated Cost |
|---|---|---|
| Cached route (Tier 1) | 0 | $0.00 |
| Rule-matched route (Tier 2) | 0 | $0.00 |
| LLM classification (Tier 3) | 1 (small prompt ~500 tokens) | ~$0.001 |
| Agent execution | 1-3 (agent's own reasoning) | $0.01-0.05 |
| Full orchestration | 5-9 (decompose, select, execute) | $0.05-0.20 |

Compared to routing everything through full orchestration: **~90% cost reduction** for routine requests.

---

## Dependency Order

```
US-001 (models)
  → US-002 (routing engine) + US-003 (cache)
    → US-004 (chatbot ingestor) + US-005 (jira ingestor)
      → US-006 (webhook dispatch) + US-007 (trigger setup)
        → US-008 (Bug Triage workflow)
US-009 (routing API) — can start after US-002
US-010 (chatbot UI) — can start after US-004
```

---

## Success Metrics

- Chatbot users send messages without selecting an agent and get correct routing > 85% of the time
- Routing latency < 50ms for 70%+ of requests (cache/rule hits)
- Jira ticket created in PILOT → PR opened autonomously within 5 minutes
- Zero webhook events lost — all received events either routed or stored in `unrouted_events`
- Routing cache hit rate > 70% after 1 week of production usage
- User corrections decrease over time as cache learns (week-over-week improvement)

---

## Open Questions

1. **Coder Agent scope** — Should the Bug Triage workflow be limited to the Automatos repo, or should it support multiple repos configured per workspace?
2. **Conflict resolution** — If the LLM routes to Agent A but the user corrects to Agent B, and later the same pattern appears, should it always use Agent B or re-evaluate?
3. **Conversation handoff** — If the router switches agents mid-conversation (e.g., user asks about emails then asks about code), should the new agent see the full conversation history from the previous agent?
4. **Webhook retry** — If async dispatch fails (agent error, DB down), should the webhook handler store the event for retry? What's the retry policy?
5. **Trigger management UI** — Should there be a frontend page to manage trigger subscriptions (enable/disable Jira trigger, set up Slack triggers), or is API-only sufficient for Phase 1?
