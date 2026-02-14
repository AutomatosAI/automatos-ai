# PRD-54: Multi-Provider LLM Marketplace & Cost Optimization

## Executive Summary

Transform Automatos AI from a single-provider (OpenAI) platform into a multi-tier LLM marketplace where users browse, compare, and deploy 200+ models from direct providers and aggregators. Users select models per-agent, per-workspace, and per-task — enabling cost optimization where cheap models handle 80% of work and premium models handle the 20% that matters.

**Three-tier architecture:**
- **Tier 1**: Direct provider integrations (OpenAI, Anthropic, Google) — best pricing, highest reliability
- **Tier 2**: OpenRouter aggregator — 200+ open-source and commercial models via single API
- **Tier 3**: BYOK (Bring Your Own Key) — users plug in their own provider keys

## Current State

### What Exists
- 7 LLM client implementations: OpenAI, Anthropic, Google Gemini, Azure OpenAI, HuggingFace, AWS Bedrock, Grok/xAI
- `ModelRegistry` with rich metadata (context_window, costs, capabilities, recommended_for)
- `ModelInfo` dataclass with cost tracking fields
- `/api/models/` endpoints: list, recommendations, compare, cost-estimate
- Per-agent `model_config` JSON field with provider/model/temperature/max_tokens
- Per-service system settings (orchestrator_llm, chatbot, codegraph, rag, etc.)
- Marketplace with `llm` item type defined but **not yet built**
- `marketplace-llms-tab.tsx` exists as placeholder
- Analytics with token usage and cost tracking in `WorkflowExecution.models_used`
- Credential resolution: settings → name patterns → type-based → env vars

### What's Missing
- Only OpenAI actively used; Anthropic connected but underutilized
- No OpenRouter or aggregator integration
- No BYOK key management for end users
- No LLM marketplace UI (browse, compare, install models to workspace)
- No model-level analytics (cost per model, tokens per model, success rate per model)
- No workspace-level model access control (plan-based model gating)
- No usage quotas or billing integration
- Model costs in registry are hardcoded, not dynamically updated
- No model health monitoring or automatic fallback

## Target State

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                    AUTOMATOS UI                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────────┐│
│  │ Agent   │  │ Settings │  │ Marketplace > LLMs   ││
│  │ Config  │  │ Default  │  │ Browse, Compare,     ││
│  │ Model ▼ │  │ Models   │  │ Install to Workspace ││
│  └────┬────┘  └────┬─────┘  └──────────┬───────────┘│
└───────┼────────────┼───────────────────┼────────────┘
        │            │                   │
┌───────▼────────────▼───────────────────▼────────────┐
│              MODEL ROUTER (Backend)                  │
│  ┌─────────────────────────────────────────────────┐│
│  │  Request → Resolve Model → Check Quota →        ││
│  │  Select Provider → Execute → Track Usage        ││
│  └─────────────────────────────────────────────────┘│
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Tier 1   │  │ Tier 2   │  │ Tier 3           │  │
│  │ Direct   │  │ OpenRouter│  │ BYOK             │  │
│  │ ─────── │  │ ──────── │  │ ──────────────── │  │
│  │ OpenAI   │  │ Llama 3  │  │ User's OpenAI key│  │
│  │ Anthropic│  │ Mistral  │  │ User's Anthropic │  │
│  │ Google   │  │ DeepSeek │  │ User's Google key │  │
│  │          │  │ Qwen     │  │ User's OpenRouter │  │
│  │          │  │ Cohere   │  │                    │  │
│  │          │  │ 200+ more│  │                    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Implementation Details

---

### Part 1: OpenRouter Integration (Tier 2)

#### 1A. New LLM Client — `openrouter_client.py`

**File**: `orchestrator/core/llm/openrouter_client.py`

OpenRouter uses the OpenAI-compatible API format. The client extends the existing OpenAI client pattern with:

```python
class OpenRouterClient(BaseLLMProvider):
    BASE_URL = "https://openrouter.ai/api/v1"

    # Config
    api_key: str  # OPENROUTER_API_KEY

    # Methods (same interface as OpenAI)
    async def generate(prompt, system_prompt, config) -> LLMResponse
    async def generate_with_tools(prompt, tools, config) -> LLMResponse
    async def stream(prompt, config) -> AsyncIterator[str]

    # OpenRouter-specific headers
    headers = {
        "HTTP-Referer": "https://automatos.app",
        "X-Title": "Automatos AI",
    }
```

Key differences from OpenAI client:
- Different `base_url`
- Extra headers (`HTTP-Referer`, `X-Title`) for OpenRouter's leaderboard
- Model IDs use provider prefix format: `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-405b`
- Pricing returned in response headers (`x-ratelimit-*`)

#### 1B. Register Provider in LLM Manager

**File**: `orchestrator/core/llm/manager.py`

Add `openrouter` to the provider resolution chain. When system settings or agent config specifies `provider: openrouter`, route to the new client.

**File**: `orchestrator/core/llm/base.py`

Add `OPENROUTER = "openrouter"` to `LLMProvider` enum.

#### 1C. Seed OpenRouter Models in Registry

**File**: `orchestrator/core/llm/model_registry.py`

Add 30-50 most popular OpenRouter models with full metadata. Group by category:

**Fast & Cheap (for simple agent tasks):**
- `meta-llama/llama-3.1-8b-instruct` — $0.055/1M, 128K context
- `mistralai/mistral-7b-instruct` — $0.055/1M, 32K context
- `google/gemini-flash-1.5` — $0.075/1M, 1M context
- `qwen/qwen-2.5-7b-instruct` — $0.054/1M, 128K context

**Balanced (for most agent work):**
- `meta-llama/llama-3.1-70b-instruct` — $0.52/1M, 128K context
- `mistralai/mixtral-8x22b-instruct` — $0.90/1M, 65K context
- `deepseek/deepseek-chat` — $0.14/1M, 128K context
- `anthropic/claude-3.5-sonnet` (via OpenRouter) — $3.00/1M, 200K context

**Premium (for complex reasoning):**
- `anthropic/claude-3-opus` — $15.00/1M, 200K context
- `openai/gpt-4o` — $2.50/1M, 128K context
- `google/gemini-pro-1.5` — $1.25/1M, 2M context

**Coding Specialists:**
- `deepseek/deepseek-coder` — $0.14/1M, 128K context
- `qwen/qwen-2.5-coder-32b` — $0.18/1M, 128K context

Each model entry includes full `ModelInfo`:
```python
ModelInfo(
    id="openrouter/meta-llama/llama-3.1-70b-instruct",
    provider="openrouter",
    model_id="meta-llama/llama-3.1-70b-instruct",
    display_name="Llama 3.1 70B Instruct",
    model_family="llama",
    context_window=131072,
    max_output_tokens=4096,
    input_cost_per_1k=0.00052,
    output_cost_per_1k=0.00052,
    capabilities={
        "reasoning": "good",
        "coding": "good",
        "analysis": "good",
        "creativity": "good",
        "function_calling": True,
    },
    recommended_for=["general_tasks", "content_generation", "analysis"],
    supports_functions=True,
    supports_vision=False,
    supports_streaming=True,
    status="active",
    description="Meta's flagship open-source model. Strong all-round performance at fraction of GPT-4 cost.",
    tier="balanced",
    tags=["open-source", "meta", "popular"],
)
```

#### 1D. Dynamic Model Sync from OpenRouter API

**File**: New `orchestrator/core/llm/openrouter_sync.py`

OpenRouter provides `GET https://openrouter.ai/api/v1/models` — returns all available models with pricing. Build a sync service that:

1. Fetches model list periodically (daily cron or on-demand)
2. Updates `ModelInfo` entries with current pricing
3. Flags deprecated models
4. Adds new models automatically
5. Stores in database table (not just in-memory registry)

---

### Part 2: Activate Anthropic & Google (Tier 1)

#### 2A. Verify Anthropic Client

**File**: `orchestrator/core/llm/anthropic_client.py`

Already implemented. Verify:
- Tool/function calling works (Anthropic uses different format)
- Streaming works
- All Claude 3.x and 3.5 models registered in ModelRegistry
- Cost tracking accurate

Models to ensure are registered:
| Model | Input Cost/1M | Output Cost/1M | Context | Strengths |
|-------|-------------|----------------|---------|-----------|
| claude-3-5-sonnet-20241022 | $3.00 | $15.00 | 200K | Best balance of capability and cost |
| claude-3-5-haiku-20241022 | $0.80 | $4.00 | 200K | Fast, cheap, good for simple tasks |
| claude-3-opus-20240229 | $15.00 | $75.00 | 200K | Most capable, complex reasoning |
| claude-sonnet-4-5-20250929 | $3.00 | $15.00 | 200K | Latest Sonnet |
| claude-haiku-4-5-20251001 | $0.80 | $4.00 | 200K | Latest Haiku |

#### 2B. Activate Google Gemini

**File**: `orchestrator/core/llm/google_client.py`

Already implemented. Verify and add to registry:
| Model | Input Cost/1M | Output Cost/1M | Context | Strengths |
|-------|-------------|----------------|---------|-----------|
| gemini-2.0-flash | $0.075 | $0.30 | 1M | Extremely cheap, fast, huge context |
| gemini-2.0-flash-lite | $0.0375 | $0.15 | 1M | Even cheaper |
| gemini-1.5-pro | $1.25 | $5.00 | 2M | Largest context window available |

#### 2C. System Settings — Default Model per Service

**File**: `orchestrator/api/system_settings.py`

Ensure each service category has configurable defaults:

| Service | Recommended Default | Reason |
|---------|-------------------|---------|
| orchestrator_llm | gpt-4o-mini | Routing decisions are simple |
| chatbot | gpt-4o or claude-3-5-sonnet | User-facing, needs quality |
| codegraph | deepseek-coder or gpt-4o | Code-specific |
| document_processing | gemini-2.0-flash | Cheap, huge context for docs |
| rag | gpt-4o-mini | Retrieval augmentation is simple |
| embeddings | BAAI/bge-large-en-v1.5 | Keep existing (good quality, local) |

---

### Part 3: BYOK — Bring Your Own Key (Tier 3)

#### 3A. User API Key Storage

**File**: `orchestrator/core/models/core.py` — new `UserApiKey` model

```python
class UserApiKey(Base):
    __tablename__ = "user_api_keys"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID, ForeignKey("workspaces.id"), nullable=False)
    provider = Column(String(50), nullable=False)  # openai, anthropic, google, openrouter
    encrypted_key = Column(Text, nullable=False)     # Encrypted with CREDENTIAL_ENCRYPTION_KEY
    display_name = Column(String(255))               # "My OpenAI Key"
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
```

Uses existing `CREDENTIAL_ENCRYPTION_KEY` for encryption at rest.

#### 3B. API Keys CRUD Endpoint

**File**: New `orchestrator/api/user_api_keys.py`

```
POST   /api/keys                — Add a new API key
GET    /api/keys                — List keys (masked)
DELETE /api/keys/{id}           — Remove a key
POST   /api/keys/{id}/test      — Test key validity (make a small API call)
```

#### 3C. Key Resolution in LLM Manager

**File**: `orchestrator/core/llm/manager.py`

Update credential resolution order:
1. Agent-level model_config (explicit key — admin only)
2. **User API key** for the provider (BYOK — new)
3. System settings credential mapping (platform key)
4. Environment variable fallback

When a BYOK key exists and the workspace is on a BYOK-eligible plan, use their key. Track usage separately for billing (platform fee only, no token markup).

#### 3D. Frontend — API Keys Management

**File**: `frontend/components/settings/ApiKeysSettingsTab.tsx` (new)

Add "API Keys" tab in Settings:
- List connected providers with status (active/inactive)
- Add key with provider dropdown and paste field
- Test key button (validates with a small API call)
- Usage stats per key
- Remove key

**File**: `frontend/components/settings/SettingsPanel.tsx`

Add new tab (6-column grid):
```
System Settings | Webhooks | API Keys | Credentials | Credential Types | Audit
```

---

### Part 4: LLM Marketplace UI

#### 4A. Database — Model Catalog Table

**File**: New alembic migration

Create `llm_models` table for persistent model catalog (currently in-memory registry):

```sql
CREATE TABLE llm_models (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,       -- e.g., "openai/gpt-4o"
    provider VARCHAR(100) NOT NULL,               -- openai, anthropic, google, openrouter
    tier VARCHAR(50) NOT NULL,                    -- direct, aggregator, byok_only
    display_name VARCHAR(255) NOT NULL,
    model_family VARCHAR(100),                    -- gpt-4, claude-3, llama-3, etc.
    description TEXT,

    -- Capabilities
    context_window INTEGER NOT NULL,
    max_output_tokens INTEGER,
    supports_functions BOOLEAN DEFAULT false,
    supports_vision BOOLEAN DEFAULT false,
    supports_streaming BOOLEAN DEFAULT true,
    capabilities JSONB DEFAULT '{}',              -- {reasoning: "excellent", coding: "good", ...}
    recommended_for JSONB DEFAULT '[]',           -- ["code_review", "analysis", ...]

    -- Pricing (per 1M tokens for display, per 1K for calculation)
    input_cost_per_1k FLOAT,
    output_cost_per_1k FLOAT,

    -- Marketplace metadata
    tags JSONB DEFAULT '[]',
    category VARCHAR(100),                        -- general, coding, creative, fast, premium
    popularity_score INTEGER DEFAULT 0,
    install_count INTEGER DEFAULT 0,
    avg_rating FLOAT,

    -- Status
    status VARCHAR(50) DEFAULT 'active',          -- active, beta, deprecated, coming_soon
    is_featured BOOLEAN DEFAULT false,
    is_default BOOLEAN DEFAULT false,             -- included in all workspaces
    requires_plan VARCHAR(50),                    -- NULL (all plans), pro, enterprise

    -- Sync metadata
    external_id VARCHAR(255),                     -- OpenRouter model ID
    pricing_updated_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

-- Workspace model access (which models a workspace has installed)
CREATE TABLE workspace_models (
    id SERIAL PRIMARY KEY,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    model_id INTEGER REFERENCES llm_models(id) ON DELETE CASCADE,
    installed_at TIMESTAMP DEFAULT now(),
    is_active BOOLEAN DEFAULT true,
    source VARCHAR(50) DEFAULT 'marketplace',     -- default, marketplace, admin
    UNIQUE(workspace_id, model_id)
);
```

#### 4B. Seed Default Models

**File**: New seed script or migration

Pre-populate `llm_models` with all Tier 1 and key Tier 2 models. Mark Tier 1 models as `is_default = true` (auto-installed in all workspaces).

Default models (included in all plans):
- gpt-4o-mini (OpenAI)
- gemini-2.0-flash (Google)
- claude-3-5-haiku (Anthropic)

Pro plan models (available on Pro+):
- gpt-4o (OpenAI)
- claude-3-5-sonnet (Anthropic)
- gemini-1.5-pro (Google)
- All OpenRouter models

Enterprise models:
- gpt-4-turbo (OpenAI)
- claude-3-opus (Anthropic)
- All BYOK models

#### 4C. Marketplace API — LLM Endpoints

**File**: `orchestrator/api/marketplace.py` (extend existing)

```
GET  /api/marketplace/items?type=llm     — List all available LLM models
     Query params: category, provider, tier, min_context, max_cost,
                   capabilities, search, sort_by (cost|context|popularity)

GET  /api/marketplace/items/{model_id}   — Model detail page data
     Returns: Full ModelInfo + reviews + usage stats

POST /api/marketplace/items/{model_id}/install  — Add model to workspace
DELETE /api/marketplace/items/{model_id}/uninstall — Remove from workspace

GET  /api/marketplace/llm/compare?ids=   — Side-by-side comparison
GET  /api/marketplace/llm/recommend       — Smart recommendations based on workspace usage
```

#### 4D. Frontend — LLM Marketplace Tab

**File**: `frontend/components/marketplace/marketplace-llms-tab.tsx` (replace placeholder)

Full marketplace experience:

**Browse View:**
- Filter bar: Provider | Category | Price Range | Context Size | Capabilities
- Sort: Most Popular | Cheapest | Largest Context | Newest
- Grid of model cards (see 4E below)
- Category sections: "Fast & Cheap", "Balanced", "Premium", "Coding", "Creative"

**Model Card Component:**
```
┌─────────────────────────────────────┐
│ [Provider Logo]  GPT-4o             │
│ gpt-4o by OpenAI                    │
│                                     │
│ ● openai    Context: 128K tokens    │
│              Max Out: 16K tokens    │
│              Cost: $2.50/1M input   │
│                                     │
│ Capabilities:                       │
│  coding: ████████░░ excellent       │
│  analysis: ████████░░ excellent     │
│  reasoning: ████████░░ excellent    │
│  creativity: ██████░░░░ good        │
│                                     │
│ ✅ Function Calling  ✅ Streaming   │
│ ✅ Vision                           │
│                                     │
│ Recommended for:                    │
│  code review • analysis • complex   │
│                                     │
│ 📊 1,234 workspaces using this     │
│                                     │
│ [Install to Workspace]  [Compare]   │
└─────────────────────────────────────┘
```

**Detail Modal** (click on card):
- Full description and changelog
- Performance benchmarks
- Cost calculator (estimate monthly cost based on usage)
- Usage in your workspace (if installed)
- Similar models recommendations

**Compare View:**
- Side-by-side table of 2-4 models
- Capability radar chart
- Cost comparison for sample workloads
- Context window comparison

#### 4E. Frontend — Model Card Component

**File**: `frontend/components/marketplace/llm-model-card.tsx` (new)

Props:
```typescript
interface LLMModelCard {
  id: string
  provider: string
  displayName: string
  modelId: string
  modelFamily: string
  description: string
  contextWindow: number
  maxOutputTokens: number
  inputCostPer1M: number
  outputCostPer1M: number
  capabilities: Record<string, string>  // {coding: "excellent", reasoning: "good"}
  recommendedFor: string[]
  supportsFunctions: boolean
  supportsVision: boolean
  supportsStreaming: boolean
  status: string
  tier: string
  tags: string[]
  installCount: number
  isInstalled: boolean  // in current workspace
  isDefault: boolean
  requiresPlan?: string
}
```

#### 4F. Frontend — Model Detail Modal

**File**: `frontend/components/marketplace/llm-model-detail-modal.tsx` (new)

Sections:
1. **Overview**: Name, provider, description, tier badge
2. **Specifications**: Context window, max output, cost per 1K/1M tokens
3. **Capabilities**: Visual bars (reasoning, coding, analysis, creativity, speed)
4. **Features**: Function calling, streaming, vision, JSON mode
5. **Recommended For**: Task type badges
6. **Cost Calculator**: Input estimated tokens/month → monthly cost
7. **Usage in Workspace**: Token usage, cost to date (if installed)
8. **Similar Models**: Other models in same category/price range

---

### Part 5: Agent Model Selection UI

#### 5A. Agent Create/Edit — Model Picker

**File**: `frontend/components/agents/` (modify existing agent form)

Replace the current simple model dropdown with a rich model picker:

1. **Quick Select**: Dropdown grouped by tier (Recommended → Direct → OpenRouter → BYOK)
2. **Browse Models**: Opens marketplace LLM tab filtered to workspace-installed models
3. **Model Info Inline**: Show cost, context, capabilities for selected model
4. **Fallback Model**: Secondary model if primary fails or is rate-limited

Agent model config structure (extends existing):
```json
{
  "provider": "openai",
  "model_id": "gpt-4o-mini",
  "temperature": 0.7,
  "max_tokens": 4096,
  "top_p": 1.0,
  "fallback_model_id": "openrouter/meta-llama/llama-3.1-70b-instruct",
  "cost_limit_per_execution": 0.50,
  "prefer_byok": true
}
```

#### 5B. Smart Model Recommendation

When creating an agent, suggest models based on:
- Agent type (code_architect → coding models, communication → general models)
- Task complexity (simple routing → cheap models, complex reasoning → premium)
- Workspace plan (starter → default models only)
- Budget preference (cost-optimized vs quality-optimized)

---

### Part 6: Analytics & Usage Tracking

#### 6A. Database — Usage Tracking Table

**File**: New alembic migration

```sql
CREATE TABLE llm_usage (
    id BIGSERIAL PRIMARY KEY,
    workspace_id UUID REFERENCES workspaces(id) NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    tier VARCHAR(50) NOT NULL,                    -- direct, aggregator, byok

    -- Request details
    agent_id INTEGER,
    execution_id VARCHAR(255),                    -- recipe execution or chat session
    request_type VARCHAR(50),                     -- chat, agent, recipe, routing, embedding

    -- Token usage
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,

    -- Cost
    input_cost FLOAT NOT NULL,
    output_cost FLOAT NOT NULL,
    total_cost FLOAT NOT NULL,
    is_byok BOOLEAN DEFAULT false,                -- true if user's own key was used

    -- Performance
    latency_ms INTEGER,
    status VARCHAR(50),                           -- success, error, timeout, rate_limited
    error_message TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT now()
);

-- Indexes for analytics queries
CREATE INDEX idx_llm_usage_workspace_created ON llm_usage(workspace_id, created_at);
CREATE INDEX idx_llm_usage_model_created ON llm_usage(model_id, created_at);
CREATE INDEX idx_llm_usage_agent ON llm_usage(agent_id, created_at);
```

#### 6B. Usage Tracking Middleware

**File**: `orchestrator/core/llm/usage_tracker.py` (new)

Wrap every LLM call to automatically record usage:

```python
class UsageTracker:
    async def track(
        workspace_id: UUID,
        model_id: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: Optional[int],
        execution_id: Optional[str],
        request_type: str,
        latency_ms: int,
        status: str,
        is_byok: bool,
    ) -> None:
        # Calculate cost from model registry
        # Insert into llm_usage table
        # Update workspace quota counters (Redis)
        # Emit event for real-time dashboard
```

Integrate into each LLM client's `generate()` method via decorator or post-hook.

#### 6C. Analytics API — Model Usage Endpoints

**File**: `orchestrator/api/analytics.py` (extend)

```
GET /api/analytics/llm/usage
    Query: period (1h|24h|7d|30d), group_by (model|provider|agent|tier)
    Returns: Token counts, costs, request counts grouped by dimension

GET /api/analytics/llm/costs
    Query: period, breakdown (model|provider|agent|daily)
    Returns: Cost breakdown with trends

GET /api/analytics/llm/models/{model_id}
    Returns: Model-specific usage, avg latency, error rate, cost

GET /api/analytics/llm/summary
    Returns: Dashboard summary — total spend, tokens, top models, cost trend

GET /api/analytics/llm/recommendations
    Returns: Cost optimization suggestions
    e.g., "Agent 19 used GPT-4o for 80% simple tasks — switch to GPT-4o-mini to save $X/month"
```

#### 6D. Frontend — Analytics Dashboard Updates

**File**: `frontend/components/analytics/` (extend existing)

Add "LLM Usage" section to analytics dashboard:

1. **Cost Overview Card**: Total spend this period, trend %, top 3 models by cost
2. **Usage Chart**: Stacked area chart — tokens over time, colored by model/provider
3. **Model Breakdown Table**: Model | Requests | Tokens | Cost | Avg Latency | Error Rate
4. **Cost Optimization Tips**: AI-generated suggestions based on usage patterns
5. **Per-Agent Cost**: Which agents cost the most, model recommendations

#### 6E. Quota & Billing Integration

**File**: `orchestrator/core/services/quota_service.py` (new)

Plan-based quotas:

| Plan | Monthly Token Quota | Models Available | BYOK |
|------|-------------------|-----------------|------|
| Starter (Free) | 100K tokens | Default models only | No |
| Pro | 5M tokens | All models | Yes |
| Enterprise | Unlimited | All models | Yes |

Quota enforcement:
1. Before each LLM call, check remaining quota (Redis counter)
2. If exceeded, return 429 with upgrade prompt
3. BYOK calls don't count against quota (user pays directly)
4. Track overage for billing (Pro plan: $X per 1M tokens over quota)

---

### Part 7: Model Health & Fallback

#### 7A. Model Health Monitor

**File**: `orchestrator/core/llm/health_monitor.py` (new)

Track per-model health metrics in Redis:
- Success rate (last 100 requests)
- Average latency (last 100 requests)
- Error rate and error types
- Rate limit status

If a model's error rate exceeds 20% in a 5-minute window:
1. Mark as `degraded`
2. Auto-fallback to configured fallback model
3. Alert in dashboard
4. Retry original model after cooldown

#### 7B. Automatic Fallback Chain

Agent model config supports fallback:
```json
{
  "model_id": "gpt-4o",
  "fallback_chain": ["claude-3-5-sonnet", "gemini-1.5-pro", "openrouter/llama-3.1-70b"]
}
```

LLM Manager tries each model in order until one succeeds.

---

## Database Changes Summary

| Table | Change |
|-------|--------|
| `llm_models` | **NEW** — Model catalog with pricing, capabilities, marketplace metadata |
| `workspace_models` | **NEW** — Which models a workspace has installed |
| `user_api_keys` | **NEW** — BYOK encrypted API keys per workspace |
| `llm_usage` | **NEW** — Per-request usage tracking for analytics and billing |
| `system_settings` | Add default model settings for each service category |

## API Changes Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/marketplace/items?type=llm` | GET | Browse LLM models |
| `/api/marketplace/items/{id}/install` | POST | Install model to workspace |
| `/api/marketplace/llm/compare` | GET | Compare models side-by-side |
| `/api/marketplace/llm/recommend` | GET | Smart recommendations |
| `/api/keys` | CRUD | Manage BYOK API keys |
| `/api/keys/{id}/test` | POST | Validate API key |
| `/api/analytics/llm/usage` | GET | Usage analytics |
| `/api/analytics/llm/costs` | GET | Cost analytics |
| `/api/analytics/llm/summary` | GET | Dashboard summary |
| `/api/analytics/llm/recommendations` | GET | Cost optimization tips |

## Frontend Changes Summary

| Component | Change |
|-----------|--------|
| `marketplace-llms-tab.tsx` | **REWRITE** — Full LLM marketplace browse/install |
| `llm-model-card.tsx` | **NEW** — Rich model card component |
| `llm-model-detail-modal.tsx` | **NEW** — Model detail view |
| `llm-compare-view.tsx` | **NEW** — Side-by-side model comparison |
| `ApiKeysSettingsTab.tsx` | **NEW** — BYOK key management |
| `SettingsPanel.tsx` | Add API Keys tab |
| Agent create/edit form | Add rich model picker |
| Analytics dashboard | Add LLM usage section |

## Implementation Phases

### Phase 1: Foundation (Tier 1 Activation)
1. Verify Anthropic client works with function calling and streaming
2. Verify Google Gemini client works
3. Seed all Tier 1 models in registry with accurate pricing
4. Update system settings defaults (cheap models for simple services)
5. Ensure analytics tracks all LLM calls with token/cost data

### Phase 2: Model Catalog & Marketplace UI
6. Create `llm_models` and `workspace_models` tables
7. Seed model catalog
8. Build LLM marketplace tab (browse, filter, compare, install)
9. Build model card and detail modal components
10. Update agent create/edit with rich model picker

### Phase 3: OpenRouter Integration (Tier 2)
11. Build OpenRouter client
12. Add OpenRouter models to catalog
13. Build model sync service (daily pricing updates)
14. Test function calling across OpenRouter models

### Phase 4: BYOK & Billing (Tier 3)
15. Create `user_api_keys` table
16. Build BYOK key management API and UI
17. Update LLM Manager key resolution for BYOK
18. Build `llm_usage` tracking table and middleware
19. Implement quota service and plan limits

### Phase 5: Analytics & Optimization
20. Build LLM analytics endpoints
21. Update analytics dashboard with cost tracking
22. Build cost optimization recommendations engine
23. Build model health monitor and auto-fallback

## Verification

1. **Tier 1**: Create agent with Claude Sonnet → chat → verify response streams correctly
2. **Tier 1**: Create agent with Gemini Flash → chat → verify function calling works
3. **Marketplace**: Browse LLMs → filter by "coding" → install DeepSeek Coder → assign to agent → test
4. **Tier 2**: Create agent with Llama 3.1 70B (via OpenRouter) → complex prompt → verify quality
5. **BYOK**: Add personal OpenAI key → create agent → verify uses personal key → verify no platform quota deducted
6. **Analytics**: Run 10 chat messages across 3 models → verify dashboard shows per-model costs
7. **Fallback**: Block a model (rate limit) → verify auto-fallback to secondary model
8. **Quota**: Set low quota on starter plan → exceed it → verify 429 response with upgrade message
