# System Prompt Management

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/settings/SystemPromptsTab.tsx](frontend/components/settings/SystemPromptsTab.tsx)
- [orchestrator/api/admin_prompts.py](orchestrator/api/admin_prompts.py)
- [orchestrator/core/models/system_prompts.py](orchestrator/core/models/system_prompts.py)
- [orchestrator/core/seeds/seed_system_prompts.py](orchestrator/core/seeds/seed_system_prompts.py)
- [orchestrator/core/services/futureagi_service.py](orchestrator/core/services/futureagi_service.py)
- [orchestrator/core/services/prompt_registry.py](orchestrator/core/services/prompt_registry.py)
- [orchestrator/tests/test_httpx_request_urls_are_never_logged.py](orchestrator/tests/test_httpx_request_urls_are_never_logged.py)
- [services/agent-opt-worker/automatos_logging.py](services/agent-opt-worker/automatos_logging.py)
- [services/agent-opt-worker/automatos_metrics.py](services/agent-opt-worker/automatos_metrics.py)
- [services/agent-opt-worker/main.py](services/agent-opt-worker/main.py)
- [services/media-render/.dockerignore](services/media-render/.dockerignore)
- [services/media-render/automatos_logging.py](services/media-render/automatos_logging.py)
- [services/media-render/kokoro/SHA256SUMS](services/media-render/kokoro/SHA256SUMS)
- [services/shared/automatos_logging.py](services/shared/automatos_logging.py)
- [services/shared/automatos_metrics.py](services/shared/automatos_metrics.py)
- [services/workspace-worker/automatos_logging.py](services/workspace-worker/automatos_logging.py)
- [services/workspace-worker/automatos_metrics.py](services/workspace-worker/automatos_metrics.py)

</details>



## Purpose and Scope

System Prompt Management provides database-backed prompt definitions and versions, runtime lookup through a registry, seed data for initial prompts, and an admin interface for browsing and editing prompts. It also includes the data models and service boundary used for FutureAGI evaluation and scoring. [orchestrator/core/models/system_prompts.py:32-138](), [orchestrator/core/services/prompt_registry.py:1-10](), [orchestrator/core/seeds/seed_system_prompts.py:1-7](), [frontend/components/settings/SystemPromptsTab.tsx:101-115](), [orchestrator/core/services/futureagi_service.py:1-14]()

Sources: [orchestrator/core/models/system_prompts.py:32-138](), [orchestrator/core/services/prompt_registry.py:1-10](), [orchestrator/core/seeds/seed_system_prompts.py:1-7](), [frontend/components/settings/SystemPromptsTab.tsx:101-115](), [orchestrator/core/services/futureagi_service.py:1-14]()

---

## Database Schema & Data Models

Three SQLAlchemy models represent prompt definitions, prompt versions, and evaluation runs: `SystemPrompt`, `SystemPromptVersion`, and `SystemPromptEvalRun`. The prompt’s `slug` is its stable code-facing identifier. Each version belongs to a prompt; the model declares a unique constraint on `(prompt_id, version_number)`. Version statuses are represented as strings, with `draft`, `active`, and `archived` documented by the model. [orchestrator/core/models/system_prompts.py:32-38](), [orchestrator/core/models/system_prompts.py:56-68](), [orchestrator/core/models/system_prompts.py:71-105]()

- `SystemPrompt` stores prompt metadata such as `display_name`, `category`, `description`, `variables`, `is_active`, and `futureagi_eval_enabled`. [orchestrator/core/models/system_prompts.py:32-54]()
- `SystemPromptVersion` stores versioned text in `content`, along with `change_note`, `status`, `created_by`, `created_at`, and optional `eval_scores`. [orchestrator/core/models/system_prompts.py:71-105]()
- `SystemPromptEvalRun` associates an evaluation run with a prompt and, optionally, a version. It stores `run_type`, `status`, `scores`, metadata, errors, and timestamps. [orchestrator/core/models/system_prompts.py:108-138]()

### Data Model Entity Space

The diagram connects the prompt-management concepts to their SQLAlchemy model names and relationships. [orchestrator/core/models/system_prompts.py:32-138]()

```mermaid
erDiagram
    SystemPrompt ||--o{ SystemPromptVersion : "has_versions"
    SystemPrompt ||--o{ SystemPromptEvalRun : "has_eval_runs"
    SystemPromptVersion ||--o{ SystemPromptEvalRun : "version_id"

    SystemPrompt {
        UUID id
        String slug
        String display_name
        String category
        JSONB variables
        Boolean is_active
        Boolean futureagi_eval_enabled
    }

    SystemPromptVersion {
        UUID id
        UUID prompt_id
        Integer version_number
        Text content
        String status
        JSONB eval_scores
    }

    SystemPromptEvalRun {
        UUID id
        UUID prompt_id
        UUID version_id
        String run_type
        String status
        JSONB scores
        JSONB metadata
    }
```

Sources: [orchestrator/core/models/system_prompts.py:32-138]()

---

## Prompt Registry & Resolution

`PromptRegistry` is a singleton service with an in-memory cache and database-backed lookup. `get_instance()` creates the singleton on first use and returns it on subsequent calls. [orchestrator/core/services/prompt_registry.py:35-53]()

- `get(slug, **variables)` resolves content and, when variables are supplied, interpolates them with `str.format_map()`. If interpolation encounters a missing key, the method logs a warning and returns the raw prompt string. [orchestrator/core/services/prompt_registry.py:59-76]()
- `get_raw(slug)` returns the resolved content without interpolation. [orchestrator/core/services/prompt_registry.py:78-81]()
- `clear_cache(slug)` removes one slug from the cache; without a slug it clears the cache. [orchestrator/core/services/prompt_registry.py:82-87]()

Resolution checks the cache first, then the database, then hardcoded defaults. Cached entries use a 60-second TTL. The database lookup selects an active `SystemPrompt` by slug, then selects its active version with the highest version number. If no database content is available, `_resolve()` checks `_HARDCODED_DEFAULTS`; if no fallback exists, it returns `None`. [orchestrator/core/services/prompt_registry.py:24-33](), [orchestrator/core/services/prompt_registry.py:93-115](), [orchestrator/core/services/prompt_registry.py:117-143]()

### Prompt Resolution Data Flow

This flow shows the registry lookup path and how resolved text reaches `get()` for optional interpolation. [orchestrator/core/services/prompt_registry.py:59-76](), [orchestrator/core/services/prompt_registry.py:93-143]()

```mermaid
flowchart TD
    A["PromptRegistry.get(slug, variables)"] --> B["PromptRegistry._resolve(slug)"]
    B --> C["CachedPrompt cache lookup"]
    C --> D{"Cache fresh?"}
    D -->|"yes"| E["CachedPrompt.content"]
    D -->|"no"| F["PromptRegistry._load_from_db(slug)"]
    F --> G["SystemPrompt active slug lookup"]
    G --> H["SystemPromptVersion active version lookup"]
    H --> I["Active version content"]
    F -->|"missing or exception"| J["_HARDCODED_DEFAULTS lookup"]
    E --> K["str.format_map(variables)"]
    I --> K
    J --> K
    K --> L["Resolved prompt string"]
```

Sources: [orchestrator/core/services/prompt_registry.py:59-76](), [orchestrator/core/services/prompt_registry.py:93-143]()

---

## Prompt Lifecycle & Versioning API

The admin prompts router is mounted at `/api/admin/prompts`. Its `_assert_admin(ctx)` helper allows API-key authentication for service-to-service calls; otherwise, it checks the authenticated user through `caller_is_admin` and returns HTTP 403 when the check fails. [orchestrator/api/admin_prompts.py:42-62]()

`create_version()` accepts a `PromptVersionCreate` body and an `activate` query parameter that defaults to `False`. The function checks that the prompt exists and begins determining the next version number from the maximum existing `version_number`. [orchestrator/api/admin_prompts.py:178-199]()

### Version State Machine

The model documents the `draft`, `active`, and `archived` statuses and states that only one version per prompt can be active at a time. The diagram summarizes those statuses; it does not imply additional transition behavior beyond the model and API code cited here. [orchestrator/core/models/system_prompts.py:71-105]()

```mermaid
stateDiagram-v2
    [*] --> Draft: "Create version"
    Draft --> Active: "Activate version"
    Active --> Archived: "Replace active version"
    Archived --> Active: "Rollback or reactivate"
    Draft --> [*]: "Delete draft"
```

Sources: [orchestrator/core/models/system_prompts.py:71-105]()

---

## Seed Data & System Prompt Manifest

`seed_system_prompts.py` defines an idempotent seed for LLM-facing system prompts. Its `PROMPT_MANIFEST` includes chatbot personality prompts such as `chatbot-friendly`, `chatbot-professional`, and `chatbot-technical`, as well as orchestrator prompts including `routing-classifier` and `task-decomposer`. The visible manifest entries define each prompt’s slug, display name, category, description, variables, and content. [orchestrator/core/seeds/seed_system_prompts.py:1-7](), [orchestrator/core/seeds/seed_system_prompts.py:19-123]()

### Natural Language Space to Code Entity Space

This diagram associates human-readable prompt names with manifest slugs, the `SystemPrompt` model, and the runtime `PromptRegistry.get()` lookup. [orchestrator/core/seeds/seed_system_prompts.py:23-123](), [orchestrator/core/models/system_prompts.py:32-42](), [orchestrator/core/services/prompt_registry.py:35-76]()

```mermaid
flowchart LR
    subgraph NaturalLanguageSpace["Natural Language Space"]
        A1["Chatbot Personality: Friendly"]
        A2["Routing Classifier"]
        A3["Task Decomposer"]
    end

    subgraph CodeEntitySpace["Code Entity Space"]
        B1["chatbot-friendly slug"]
        B2["routing-classifier slug"]
        B3["task-decomposer slug"]
        C1["SystemPrompt model"]
        D1["PromptRegistry.get(slug)"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    B1 --> C1
    B2 --> C1
    B3 --> C1
    C1 --> D1
```

Sources: [orchestrator/core/seeds/seed_system_prompts.py:23-123](), [orchestrator/core/models/system_prompts.py:32-42](), [orchestrator/core/services/prompt_registry.py:35-76]()

---

## Admin Prompts API Reference

The admin router defines list, category, prompt-detail, version-list, and version-creation endpoints in the inspected portion of `admin_prompts.py`. These endpoints use `_assert_admin(ctx)` for authorization. [orchestrator/api/admin_prompts.py:42-62](), [orchestrator/api/admin_prompts.py:106-199]()

| Method | Route | Behavior |
|---|---|---|
| `GET` | `/api/admin/prompts` | Lists prompts, optionally filtered by `category` and `search`. Search matches display name, slug, or description. [orchestrator/api/admin_prompts.py:106-128]() |
| `GET` | `/api/admin/prompts/categories` | Returns distinct categories with prompt counts. [orchestrator/api/admin_prompts.py:131-144]() |
| `GET` | `/api/admin/prompts/{prompt_id}` | Retrieves a prompt and returns 404 when it is not found. [orchestrator/api/admin_prompts.py:147-158]() |
| `GET` | `/api/admin/prompts/{prompt_id}/versions` | Lists a prompt’s versions in descending `version_number` order. [orchestrator/api/admin_prompts.py:161-175]() |
| `POST` | `/api/admin/prompts/{prompt_id}/versions` | Starts creating a new version; the `activate` query parameter defaults to `False`. [orchestrator/api/admin_prompts.py:178-199]() |

Sources: [orchestrator/api/admin_prompts.py:42-62](), [orchestrator/api/admin_prompts.py:106-199]()

---

## Frontend Management (`SystemPromptsTab`)

`SystemPromptsTab` is a client-side React component that tracks the prompt list, categories, selected prompt, versions, assessment runs, editor state, and active tab. It fetches prompts and categories from the admin API and requests versions and assessment runs when a prompt is selected. [frontend/components/settings/SystemPromptsTab.tsx:101-115](), [frontend/components/settings/SystemPromptsTab.tsx:121-161](), [frontend/components/settings/SystemPromptsTab.tsx:180-188]()

When assessment runs include `pending` or `running` statuses and a prompt is selected, the component refreshes the assessment-run list every three seconds; the interval is cleared when the effect is cleaned up. [frontend/components/settings/SystemPromptsTab.tsx:166-174]()

Sources: [frontend/components/settings/SystemPromptsTab.tsx:101-115](), [frontend/components/settings/SystemPromptsTab.tsx:121-188]()

---

## FutureAGI Integration & Worker Service

The orchestrator-side `futureagi_service.py` describes itself as a thin HTTP client that routes FutureAGI operations to the isolated worker. Its stated responsibilities include database reads and writes, fire-and-forget dispatch, and dataset collection; the worker handles assessment, safety, optimization, and live scoring. The client defines `X-Internal-Token` as the worker token header and uses `AGENT_OPT_WORKER_TOKEN` when configured. [orchestrator/core/services/futureagi_service.py:1-14](), [orchestrator/core/services/futureagi_service.py:32-45]()

The worker’s FastAPI application declares an authenticated router for internal operations. The application documentation lists `/assess`, `/safety`, `/optimize`, `/optimize/{job_id}`, and `/score`, with `/health` and `/metrics` also documented. The worker configures a dependency requiring the internal token on routes registered on that router, and disables API documentation endpoints. [services/agent-opt-worker/main.py:9-20](), [services/agent-opt-worker/main.py:44-51]()

### Natural Language Space to Worker Entity Space

This diagram maps evaluation and safety concepts to the worker’s documented HTTP routes and the orchestrator client boundary. [orchestrator/core/services/futureagi_service.py:1-45](), [services/agent-opt-worker/main.py:9-20](), [services/agent-opt-worker/main.py:44-51]()

```mermaid
flowchart LR
    subgraph NaturalLanguageSpace["Natural Language Space"]
        A1["Prompt assessment"]
        A2["Prompt safety check"]
        A3["Prompt optimization"]
        A4["Live chat scoring"]
    end

    subgraph CodeEntitySpace["Code Entity Space"]
        B1["futureagi_service.py"]
        B2["WORKER_URL"]
        C1["POST /assess"]
        C2["POST /safety"]
        C3["POST /optimize"]
        C4["POST /score"]
        D1["agent-opt-worker main.py"]
        E1["X-Internal-Token"]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    B1 --> B2
    B2 --> C1
    B2 --> C2
    B2 --> C3
    B2 --> C4
    C1 --> D1
    C2 --> D1
    C3 --> D1
    C4 --> D1
    D1 --> E1
```

Sources: [orchestrator/core/services/futureagi_service.py:1-45](), [services/agent-opt-worker/main.py:9-20](), [services/agent-opt-worker/main.py:44-51]()

---