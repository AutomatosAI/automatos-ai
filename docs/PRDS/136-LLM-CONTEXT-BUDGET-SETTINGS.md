# PRD-136 — Collapse 12 LLM Silos to 3 Tiers (Auto / System / Embeddings)

**Status:** Active
**Type:** Refactor / Consolidation
**Owner:** Platform
**Last updated:** 2026-04-28

---

## Problem

The platform has **12 LLM configuration silos** spread across enums, env vars, hardcoded defaults, seed files, and 5 separate UI panels. Same fields (`provider`, `model`, `temperature`, `max_tokens`) are duplicated under different names per silo (`extraction_max_tokens`, `chat_max_tokens`, `orchestrator_max_tokens`, etc.). Admin must configure the same dial 12 times to change platform behavior.

Symptom: Auto cannot read RAG'd documents because the result formatter caps at 4500 chars, while the LLM has 128K context. Auto says "I found 4 documents" instead of synthesizing them.

**Root cause:** No single concept of "LLM tier" — every internal service invented its own settings namespace.

## Goal

**Three tiers. One schema. One source of truth.**

| Tier | Purpose | Default model | Cost profile |
|---|---|---|---|
| **Auto** (`orchestrator_llm`) | The brain. Chat, user-facing reasoning, planning. | GPT-5.5 / Opus 4.7 (premium) | High per-call, low volume |
| **System** (`system_llm`) | Everything internal: codegraph, coordination, complexity routing, memory, document processing, RAG synthesis, NL2SQL, knowledge graph extraction, chatbot scaffolding. | google/gemini-2.5-flash (cheap-fast) | Low per-call, high volume |
| **Embeddings** (`embeddings`) | Vectorization only. Different model family. | qwen/qwen3-embedding-8b | Bulk, cached |

Each tier shares the **same canonical schema** (provider, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, timeout, retries). Per-agent overrides reuse the same schema via `Agent.model_config`.

## Non-goals

- Per-service granular dials (e.g. separate `nl2sql_temperature`). If a service needs custom behavior, override `model_config` on the agent that runs it.
- Cost-attribution rework. `llm_cost_audit` already tags every call with service+model — that stays.
- Workspace-level billing. Pilot continues admin-paid; multi-tenant billing is a separate PRD.

---

## Implementation

### 1. Backend — collapse `SettingCategory`

**File:** `orchestrator/core/models/system_settings.py`

Remove these enum values: `CODEGRAPH`, `CHATBOT`, `COMPLEXITY_ASSESSOR`, `COORDINATION`, `KNOWLEDGE_GRAPH`, `MEMORY_MANAGEMENT`.

Add: `SYSTEM_LLM = "system_llm"`, `EMBEDDINGS = "embeddings"`.

Final LLM-tier enum surface:
- `ORCHESTRATOR_LLM` — Auto (kept)
- `SYSTEM_LLM` — new, replaces 6 collapsed silos
- `EMBEDDINGS` — new, lifted out of `GENERAL`

Non-LLM categories untouched: `GENERAL`, `SYSTEM_LOGGING`, `API_RATE_LIMITING`, `BACKEND_API_KEYS`, `PERFORMANCE`, `SECURITY`, `NOTIFICATIONS`, `BACKUPS`, `MONITORING`, `LLM_COST_AUDIT`.

### 2. Backend — collapse `SERVICE_CATEGORY_MAP`

**File:** `orchestrator/core/llm/manager.py`

```python
SERVICE_CATEGORY_MAP = {
    # Auto tier
    "orchestrator": "orchestrator_llm",
    "heartbeat": "orchestrator_llm",

    # System tier (the 9)
    "chatbot": "system_llm",
    "codegraph": "system_llm",
    "coordination": "system_llm",
    "complexity_assessor": "system_llm",
    "memory_integration": "system_llm",
    "document_processing": "system_llm",
    "rag": "system_llm",
    "nl2sql": "system_llm",
    "graph_extraction": "system_llm",

    # Embeddings tier
    "embeddings": "embeddings",
}
```

Anything not in the map falls back to `orchestrator_llm` (existing behavior in `create_llm_manager`).

### 3. Backend — canonical schema per tier

Each tier seeds these keys (and only these):

| Key | Type | Auto default | System default | Embeddings default |
|---|---|---|---|---|
| `provider` | string | openrouter | openrouter | openrouter |
| `model` | string | openai/gpt-5.5 | google/gemini-2.5-flash | qwen/qwen3-embedding-8b |
| `temperature` | number | 0.7 | 0.3 | n/a |
| `max_tokens` | number | 8000 | 8000 | n/a |
| `top_p` | number | 1.0 | 1.0 | n/a |
| `frequency_penalty` | number | 0.0 | 0.0 | n/a |
| `presence_penalty` | number | 0.0 | 0.0 | n/a |
| `timeout_seconds` | number | 120 | 60 | 60 |
| `max_retries` | number | 3 | 3 | 3 |

Embeddings adds: `dimensions`, `batch_size`, `cache_dir`, `max_seq_length`.

**Tooltip copy** lives on each setting's `description` field — plain language, no jargon. Examples:
- `max_tokens`: "The longest reply this LLM is allowed to write. Higher = more detail but slower and more expensive. 8000 is comfortable for most tasks."
- `temperature`: "How creative the LLM is. 0 = deterministic and factual. 1 = playful and varied. 0.7 is the sweet spot for chat."
- `model`: "Which model handles requests for this tier. Auto runs your premium model (the brain). System runs a cheap-fast model for the dozens of internal calls per request."

### 4. Backend — agent `model_config` reuses canonical schema

`ModelConfiguration` dataclass in `agent_factory.py:61` already matches the schema. Confirm field parity, then **delete the per-tier hardcoded fallbacks** and have `_create_llm_manager()` always read from `model_config` first, then fall back to the agent's tier (`orchestrator_llm` for Auto, `system_llm` for everything else) via `create_llm_manager`.

### 5. Migration — `alembic/versions/<new>_collapse_llm_tiers.py`

One-shot data migration:

1. For every row in `system_settings` where `category` IN (`codegraph`, `chatbot`, `complexity_assessor`, `coordination`, `knowledge_graph`, `memory_management`):
   - If the user customized a value (DB value ≠ default), preserve it on the new `system_llm` row using a "first non-default wins" rule, logged.
   - Otherwise drop.
2. Move embedding keys (`embedding_model`, `embedding_provider`, `embedding_*`) from `general` → `embeddings`, stripping the `embedding_` prefix.
3. Insert canonical seed keys for `orchestrator_llm`, `system_llm`, `embeddings` if missing.
4. Delete the now-orphaned rows.

Migration is idempotent and safe to re-run.

### 6. Legacy deletions (same PR)

| File | Line | Delete |
|---|---|---|
| `orchestrator/api/chatbot_llm.py` | 315 | `default_model = "gpt-4"` — read from `system_llm.model` |
| `orchestrator/api/chat.py` | 200 | `selectedChatModel: Optional[str] = "gpt-4"` — drop default, resolve via tier |
| `orchestrator/api/agent_endpoints.py` | 622, 761 | Hardcoded `"model_id": "gpt-4"` and `"claude-3-sonnet-20240229"` — pull tier defaults |
| `orchestrator/core/seeds/seed_system_settings.py` | various | Remove all collapsed-category seed blocks (codegraph, chatbot, complexity_assessor, coordination, knowledge_graph, memory_management) |

### 7. Result-formatter truncation fix

**File:** `orchestrator/modules/tools/formatting/result_formatter.py`

Three concrete edits:
- Line 354: `max_chars=500` → `max_chars=4000` (per-doc excerpt usable for synthesis).
- Line 738: `format_for_llm(..., max_chars: int = 4500)` → `max_chars: int = 20000` (overall envelope to fit 4 docs × 4000).
- Line 789: stop reading `excerpt`; read `content` (the `# Full content for LLM` field at line 374). Excerpt was a UI preview, not synthesis input.
- Lines 783-787: rewrite the lying system-prompt prefix. Replace "up to 800 chars each" with "Full document content below, ready to synthesize."

These numbers stay literals — they are output-shape constants, not user-tunable LLM behavior.

### 8. Frontend — collapse 5 panels → 3

**Settings page (`automatos-ai/frontend/app/admin/settings/...`):**

Old panels (delete):
- Orchestrator LLM (rename → keep as Auto)
- System Settings > System LLMs (renamed → System LLM)
- CodeGraph LLM
- Knowledge Graph LLM
- General > ML/AI Model Configuration (move → Embeddings)

New layout: three cards, identical shape (provider / model / temperature / max_tokens / advanced collapsible). Each field has a tooltip from the seed `description`. Match the existing LLM Configuration card styling already on the Orchestrator panel.

Delete any orphaned components, hooks (`useCodegraphSettings`, `useKnowledgeGraphSettings`, etc.), and route handlers.

---

## Verification

1. **Inbuild UK chat (`28a228aa-dd63-46c7-baac-d29a0eb67283`):** Ask a question that requires reading 3+ ingested docs. Auto synthesizes, doesn't say "I found N documents."
2. **Settings page:** Three LLM cards visible, all five legacy panels gone.
3. **DB:** `SELECT DISTINCT category FROM system_settings WHERE category LIKE '%llm%' OR category IN ('codegraph','chatbot','complexity_assessor','coordination','knowledge_graph','memory_management','embeddings');` returns only `orchestrator_llm`, `system_llm`, `embeddings`.
4. **Hardcoded defaults gone:** `rg "gpt-4|claude-3-sonnet" orchestrator/api/` returns nothing.
5. **Cost audit:** `llm_cost_audit` rows still tag service correctly (chatbot vs codegraph vs graph_extraction) — service identity is preserved even though all three resolve to `system_llm` tier.
6. **Per-agent overrides:** Comms agent with custom `model_config` still uses its overridden model, not the system tier default.

## Critical files

| Path | Change |
|---|---|
| `orchestrator/core/models/system_settings.py` | Collapse `SettingCategory` enum |
| `orchestrator/core/llm/manager.py` | Rewrite `SERVICE_CATEGORY_MAP` to 3 tiers |
| `orchestrator/core/seeds/seed_system_settings.py` | Replace 6 silo blocks with 3 tier blocks (canonical schema + tooltips) |
| `orchestrator/alembic/versions/<new>.py` | Data migration |
| `orchestrator/modules/agents/factory/agent_factory.py` | `_create_llm_manager` falls back to tier, not silo |
| `orchestrator/modules/tools/formatting/result_formatter.py` | 3 edits — fix truncation + lying prompt |
| `orchestrator/api/chatbot_llm.py:315` | Delete `gpt-4` default |
| `orchestrator/api/chat.py:200` | Delete `gpt-4` default |
| `orchestrator/api/agent_endpoints.py:622,761` | Delete hardcoded model fallbacks |
| `automatos-ai/frontend/app/admin/settings/*` | Three cards, tooltips, delete dead panels |

## Reused (do not rebuild)

- `LLMManager`, `create_llm_manager()` factory — `core/llm/manager.py`
- `ModelConfiguration` dataclass — `agent_factory.py:61` (already canonical)
- `Agent.model_config` JSON column — `core/models/core.py:253`
- DB-overrides-env pattern via `get_system_setting()` — `core/llm/manager.py:50`
- `llm_cost_audit` service-tagging — leave alone

## Out of scope (explicitly)

- New per-service dials of any kind.
- Renaming any non-LLM category.
- Workspace-level billing.
- Backwards-compat shims for old category names — clean delete per CLAUDE.md §5.
