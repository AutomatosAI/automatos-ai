# State Management

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/PRDS/52-UNIFIED-ANALYTICS.md](docs/PRDS/52-UNIFIED-ANALYTICS.md)
- [frontend/app/analytics/page.tsx](frontend/app/analytics/page.tsx)
- [frontend/components/agents/agent-configuration-modal.tsx](frontend/components/agents/agent-configuration-modal.tsx)
- [frontend/components/agents/agent-configuration.tsx](frontend/components/agents/agent-configuration.tsx)
- [frontend/components/agents/agent-details-modal.tsx](frontend/components/agents/agent-details-modal.tsx)
- [frontend/components/agents/agent-performance.tsx](frontend/components/agents/agent-performance.tsx)
- [frontend/components/agents/agent-roster.tsx](frontend/components/agents/agent-roster.tsx)
- [frontend/components/agents/agent-skills.tsx](frontend/components/agents/agent-skills.tsx)
- [frontend/components/agents/agent-status-control-modal.tsx](frontend/components/agents/agent-status-control-modal.tsx)
- [frontend/components/agents/create-agent-modal.tsx](frontend/components/agents/create-agent-modal.tsx)
- [frontend/components/agents/create-skill-modal.tsx](frontend/components/agents/create-skill-modal.tsx)
- [frontend/components/agents/model-selector.tsx](frontend/components/agents/model-selector.tsx)
- [frontend/components/agents/skill-configuration-modal.tsx](frontend/components/agents/skill-configuration-modal.tsx)
- [frontend/components/analytics/analytics-admin.tsx](frontend/components/analytics/analytics-admin.tsx)
- [frontend/components/analytics/analytics-agents.tsx](frontend/components/analytics/analytics-agents.tsx)
- [frontend/components/analytics/analytics-costs.tsx](frontend/components/analytics/analytics-costs.tsx)
- [frontend/components/analytics/analytics-documents.tsx](frontend/components/analytics/analytics-documents.tsx)
- [frontend/components/analytics/analytics-memory.tsx](frontend/components/analytics/analytics-memory.tsx)
- [frontend/components/analytics/analytics-openrouter-credits.tsx](frontend/components/analytics/analytics-openrouter-credits.tsx)
- [frontend/components/analytics/analytics-overview.tsx](frontend/components/analytics/analytics-overview.tsx)
- [frontend/components/analytics/analytics-page.tsx](frontend/components/analytics/analytics-page.tsx)
- [frontend/components/analytics/analytics-pandas-chart.tsx](frontend/components/analytics/analytics-pandas-chart.tsx)
- [frontend/components/analytics/analytics-plan-usage.tsx](frontend/components/analytics/analytics-plan-usage.tsx)
- [frontend/components/analytics/analytics-recommendations.tsx](frontend/components/analytics/analytics-recommendations.tsx)
- [frontend/components/analytics/analytics-workflows.tsx](frontend/components/analytics/analytics-workflows.tsx)
- [frontend/components/system/rag-configuration.tsx](frontend/components/system/rag-configuration.tsx)
- [frontend/hooks/use-agent-api.ts](frontend/hooks/use-agent-api.ts)
- [frontend/hooks/use-model-api.ts](frontend/hooks/use-model-api.ts)
- [frontend/hooks/use-unified-analytics.ts](frontend/hooks/use-unified-analytics.ts)
- [frontend/lib/agent-constants.ts](frontend/lib/agent-constants.ts)
- [orchestrator/alembic/versions/add_job_title_to_agents.py](orchestrator/alembic/versions/add_job_title_to_agents.py)
- [orchestrator/api/agent_endpoints.py](orchestrator/api/agent_endpoints.py)
- [orchestrator/api/agents.py](orchestrator/api/agents.py)
- [orchestrator/core/models/__init__.py](orchestrator/core/models/__init__.py)
- [orchestrator/core/models/core.py](orchestrator/core/models/core.py)

</details>



## Purpose and Scope

This document describes the frontend state management architecture in Automatos AI. The system utilizes a hybrid approach: **React Query** (TanStack Query) for server-side state synchronization, **Zustand** for global application state, and standard React state for transient UI logic. Central to this architecture is the **wsScope** pattern, which ensures strict multi-tenancy by scoping all cached data to the active workspace. The architecture also includes a sophisticated theme management system (Classic, Dark, Matte, and Studio) that leverages CSS variables and Tailwind variants.

Sources: [frontend/components/providers.tsx:77-85](), [frontend/hooks/use-studio-theme.ts:1-15]()

---

## State Management Architecture

The frontend state is organized into three primary layers that handle authentication, global application context, and asynchronous server data.

### Architecture Diagram: State Management Layers

```mermaid
graph TB
    subgraph "Global Context Layer (Providers.tsx)"
        Auth["AuthBoundary<br/>(Clerk or Local)"]
        Workspaces["WorkspaceProvider<br/>(Active Workspace)"]
        QueryProv["QueryClientProvider<br/>(TanStack Query)"]
        ZustandStores["Zustand Stores<br/>(Global App State)"]
        Theme["ThemeProvider<br/>(next-themes)"]
    end

    subgraph "Server State Layer (React Query Hooks)"
        AgentHooks["useAgents / useAgent<br/>(Agent Registry)"]
        DocHooks["useDocuments / useDocument<br/>(RAG Knowledge)"]
        AnalyticsHooks["useUnifiedAnalytics<br/>(Usage & Cost)"]
        Cache["Query Cache<br/>(staleTime: 60s)"]
    end

    subgraph "UI Component Layer"
        PageHeader["PageHeader<br/>(Editorial Lede)"]
        Dialog["Dialog / Modal<br/>(Radix UI)"]
        Glossary["GlossaryTooltip<br/>(Term Persistence)"]
    end

    subgraph "API Client Layer"
        API["apiClient<br/>(lib/api-client)"]
    end

    Auth --> ZustandStores
    ZustandStores --> Theme
    Theme --> QueryProv
    QueryProv --> Workspaces
    Workspaces --> Cache

    Cache --> AgentHooks
    Cache --> DocHooks
    Cache --> AnalyticsHooks

    AgentHooks --> PageHeader
    DocHooks --> Dialog
    Glossary -.-> "Local Storage"

    AgentHooks -.-> API
    DocHooks -.-> API
```

Sources: [frontend/components/providers.tsx:77-113](), [frontend/hooks/use-page-api.ts:1-34](), [frontend/components/ui/glossary-tooltip.tsx:7-31]()

---

## Server State Management (React Query)

Automatos AI uses `@tanstack/react-query` to manage the lifecycle of data fetched from the FastAPI backend. The `QueryClient` is initialized in `Providers` with a default `staleTime` of 60 seconds and a retry limit of 1 to balance data freshness with network overhead [frontend/components/providers.tsx:78-85]().

### Workspace-Scoped Query Keys

To maintain data isolation in a multi-tenant environment, query keys are structured as arrays where the workspace context (often implicitly handled by the `apiClient`) or specific IDs are used to partition the cache. This prevents data from Workspace A appearing when a user switches to Workspace B. The `wsScope()` function in `frontend/hooks/use-unified-analytics.ts` is a prime example, returning the current workspace ID or 'own' for the user's default workspace [frontend/hooks/use-unified-analytics.ts:12-14]().

For example, the `unifiedAnalyticsKeys` object defines various query keys, all of which include `wsScope()` to ensure workspace isolation [frontend/hooks/use-unified-analytics.ts:18-43]().

```typescript
export const unifiedAnalyticsKeys = {
  overview: (days: number) => ['unified-analytics', wsScope(), 'overview', days] as const,
  agents: (days: number) => ['unified-analytics', wsScope(), 'agents', days] as const,
  // ... other keys
}
```
Sources: [frontend/hooks/use-unified-analytics.ts:12-43]()

### Implementation Patterns

The application follows a "Hook-per-Resource" pattern. These hooks encapsulate fetching logic, normalization, and integration with the `apiClient`.

1.  **Page Attribution**: The `usePageAPI` hook tags outgoing requests with the `pageName` (e.g., 'dashboard', 'agents') to assist in observability and request tracking [frontend/hooks/use-page-api.ts:25-34]().
2.  **Optimistic UI**: Mutations use the query cache to provide immediate feedback, later reconciling with the server response. For instance, when updating an agent's configuration, `useUpdateAgentConfig` invalidates relevant queries to refetch fresh data after a successful mutation [frontend/hooks/use-agent-api.ts:300-317](). Similarly, `useAddSkillToAgent` and `useRemoveSkillFromAgent` optimistically update the agent's skills list before the server confirms the change [frontend/hooks/use-agent-api.ts:440-467](), [frontend/hooks/use-agent-api.ts:480-507]().
3.  **Refetching**: Hooks for long-running processes (like mission execution or RAG ingestion) utilize polling intervals to keep the UI updated without manual refreshes. For example, `useAgent` refetches every 10 seconds to keep agent status up-to-date [frontend/hooks/use-agent-api.ts:155]().

### React Query Hooks under `frontend/hooks`

The `frontend/hooks` directory contains a collection of custom React Query hooks that abstract away data fetching and mutation logic for various API resources.

#### `use-agent-api.ts`
This file provides hooks for managing agents, their configurations, skills, and performance metrics.
- `useAgents()`: Fetches all agents, optionally including system agents, and injects icon mappings [frontend/hooks/use-agent-api.ts:100-125]().
- `useAgent(agentId)`: Fetches a single agent by ID, with a refetch interval of 10 seconds [frontend/hooks/use-agent-api.ts:143-157]().
- `useCreateAgent()`: Handles the creation of new agents, invalidating the `agents` query key on success [frontend/hooks/use-agent-api.ts:200-215]().
- `useUpdateAgentConfig()`: Updates an agent's configuration, invalidating `agent` and `agentConfig` query keys [frontend/hooks/use-agent-api.ts:300-317]().
- `useAddSkillToAgent()` and `useRemoveSkillFromAgent()`: Optimistically update an agent's skills list [frontend/hooks/use-agent-api.ts:440-467](), [frontend/hooks/use-agent-api.ts:480-507]().

#### `use-unified-analytics.ts`
This file contains hooks for fetching various analytics data, all scoped by `wsScope()`.
- `useAnalyticsOverview(days)`: Fetches a high-level overview of agents, workflows, documents, and costs [frontend/hooks/use-unified-analytics.ts:46-106]().
- `useAgentAnalytics(days)`: Retrieves detailed agent statistics, including memory usage [frontend/hooks/use-unified-analytics.ts:120-137]().
- `useWorkflowAnalytics(days)`: Fetches workflow and recipe performance metrics [frontend/hooks/use-unified-analytics.ts:200-216]().
- `useCostsAnalytics(days)`: Provides cost breakdown by model, agent, and provider [frontend/hooks/use-unified-analytics.ts:280-296]().

#### `use-model-api.ts`
Hooks for managing LLM models and their configurations.
- `useModels()`: Fetches all available LLM models [frontend/hooks/use-model-api.ts:10-16]().
- `useAgentModelConfig(agentId)`: Retrieves the model configuration for a specific agent [frontend/hooks/use-model-api.ts:26-32]().
- `useUpdateAgentModelConfig()`: Updates an agent's model configuration, invalidating relevant queries [frontend/hooks/use-model-api.ts:42-57]().

#### `use-skills-api.ts`
Hooks for managing skills.
- `useSkillsApi()`: Fetches all skills [frontend/hooks/use-skills-api.ts:10-16]().
- `useCreateSkill()`: Creates a new skill, invalidating the `skills` query [frontend/hooks/use-skills-api.ts:26-36]().

### React Query Flow Diagram (Agent Configuration Example)

```mermaid
graph TD
    A[AgentConfigurationModal] --> B{useAgent(agentId)}
    A --> C{useAgentConfig(agentId)}
    A --> D{useAgentModelConfig(agentId)}
    A --> E{useAgentSkills(agentId)}
    A --> F{useTools()}
    A --> G{useSystemIcons()}

    B -- "Agent Data" --> H[Form State]
    C -- "Config Data" --> H
    D -- "Model Config Data" --> H
    E -- "Assigned Skills" --> H
    F -- "Available Tools" --> H
    G -- "Icon Mappings" --> H

    H -- "User Edits" --> I[handleSave]

    I --> J{useUpdateAgentConfig.mutate}
    I --> K{useUpdateAgentModelConfig.mutate}
    I --> L{useAddSkillToAgent.mutate / useRemoveSkillFromAgent.mutate}

    J -- "Success" --> M[QueryClient.invalidateQueries(['agents', agentId])]
    K -- "Success" --> N[QueryClient.invalidateQueries(['agent-models', agentId])]
    L -- "Success" --> O[QueryClient.invalidateQueries(['agents', agentId, 'skills'])]

    M --> B
    N --> D
    O --> E

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    style H fill:#ffc,stroke:#333,stroke-width:2px
    style I fill:#fcf,stroke:#333,stroke-width:2px
    style J fill:#cfc,stroke:#333,stroke-width:2px
    style K fill:#cfc,stroke:#333,stroke-width:2px
    style L fill:#cfc,stroke:#333,stroke-width:2px
    style M fill:#fcc,stroke:#333,stroke-width:2px
    style N fill:#fcc,stroke:#333,stroke-width:2px
    style O fill:#fcc,stroke:#333,stroke-width:2px
```
This diagram illustrates the data flow for agent configuration. When the `AgentConfigurationModal` [frontend/components/agents/agent-configuration-modal.tsx]() opens, it uses several React Query hooks to fetch initial data. User edits update local form state. Upon saving, mutation hooks are triggered, which then invalidate relevant query keys, causing React Query to refetch the updated data and keep the UI synchronized.

Sources: [frontend/hooks/use-page-api.ts:10-34](), [frontend/components/providers.tsx:78-85](), [frontend/hooks/use-agent-api.ts:100-507](), [frontend/hooks/use-unified-analytics.ts:46-296](), [frontend/hooks/use-model-api.ts:10-57](), [frontend/hooks/use-skills-api.ts:10-36]()

---

## Zustand Stores

For global, non-server-state management, Automatos AI utilizes Zustand. Zustand is a lightweight, flexible state management solution that avoids the boilerplate of Redux while offering similar capabilities. It's particularly useful for UI-related global states that don't directly map to server data.

### Example: Workspace Provider

The `useWorkspace` hook, defined in `frontend/components/workspace-provider.tsx`, is a Zustand store that manages the currently active workspace. This is critical for multi-tenancy, as many API calls and UI components depend on the `workspace.id`.

```typescript
// frontend/components/workspace-provider.tsx
import { create } from 'zustand'

interface WorkspaceState {
  workspace: {
    id: string
    name: string
    // ... other workspace properties
  } | null
  setWorkspace: (workspace: WorkspaceState['workspace']) => void
  // ... other actions
}

export const useWorkspace = create<WorkspaceState>((set) => ({
  workspace: null,
  setWorkspace: (workspace) => set({ workspace }),
  // ...
}))
```
This store provides a `workspace` object and a `setWorkspace` function to update it. Components can then subscribe to this store to react to changes in the active workspace.

Sources: [frontend/components/workspace-provider.tsx:1-15]()

---

## Theme and Visual State

Automatos AI implements a multi-theme system that goes beyond simple light/dark modes. The system supports `light`, `dark`, `matte`, and the editorial-first `studio` theme [frontend/components/providers.tsx:90-97]().

### Theme Detection and Persistence
The `useStudioThemeFlag` hook detects a specific URL parameter (`?theme=studio-preview`) to activate the Studio theme, which is then persisted via `next-themes` [frontend/hooks/use-studio-theme.ts:16-26](). The `ThemeToggle` component allows users to switch between these modes manually, with icons representing each state (e.g., `BookOpen` for Studio, `Square` for Matte) [frontend/components/ui/theme-toggle.tsx:30-43]().

### Visual State Integration Diagram

```mermaid
graph LR
    subgraph "Theme Configuration"
        Config["tailwind.config.ts"]
        Globals["globals.css"]
    end

    subgraph "State Logic"
        useTheme["useTheme (next-themes)"]
        useStudio["useStudioThemeFlag"]
    end

    subgraph "Styled Components"
        PH["PageHeader"]
        Input["Input"]
        Dialog["DialogContent"]
    end

    Config -->|".matte / .dark variants"| Globals
    useStudio -->|setTheme('studio')| useTheme
    useTheme -->|"--background / --primary"| Globals
    Globals --> PH
    Globals --> Input
    Globals --> Dialog
```

Sources: [frontend/tailwind.config.ts:3-6](), [frontend/hooks/use-studio-theme.ts:16-26](), [frontend/components/ui/theme-toggle.tsx:46-80]()

---

## Client-Side Persistence (Local Storage)

Beyond server state, the frontend manages persistent UI preferences using `localStorage`.

### Glossary Term Learning
The `GlossaryTooltip` component tracks how many times a user has viewed a specific technical term (e.g., "mission", "handoff"). After `SUPPRESS_AFTER` (default 3) sightings, the tooltip is suppressed, assuming the user has learned the concept [frontend/components/ui/glossary-tooltip.tsx:7-12]().

*   **Key**: `automatos-glossary-seen` [frontend/components/ui/glossary-tooltip.tsx:7]()
*   **Functions**: `readCounts()` and `writeCounts()` handle the serialization of sighting frequencies [frontend/components/ui/glossary-tooltip.tsx:14-31]().

### Theme Storage
The `ThemeProvider` persists the user's selected theme under the key `automatos-theme` [frontend/components/providers.tsx:95]().

---

## UI State Primitives

The design system enforces consistent state representation through shared components:

| Component | State Managed | Role |
| :--- | :--- | :--- |
| `Dialog` | `data-[state=open]` | Handles animations (zoom-in, fade-in) and backdrop blur via Radix UI [frontend/components/ui/dialog.tsx:51-65]() |
| `Tabs` | `data-[state=active]` | Manages selection state and font-weight transitions [frontend/components/ui/tabs.tsx:29-37]() |
| `Input` | `focus-visible` | Manages the primary focus ring and shadow glow (`0_0_12px_hsla(var(--primary)/0.15)`) [frontend/components/ui/input.tsx:14-14]() |
| `PageHeader` | `motion.div` | Manages entry animations (opacity/y-axis) using Framer Motion [frontend/components/shared/page-header.tsx:48-56]() |

Sources: [frontend/components/ui/dialog.tsx:51-65](), [frontend/components/ui/tabs.tsx:29-37](), [frontend/components/ui/input.tsx:14-14](), [frontend/components/shared/page-header.tsx:48-56]()

---

## Summary of State Entities

| Code Entity | Location | Role |
| :--- | :--- | :--- |
| `QueryClient` | [frontend/components/providers.tsx:78]() | Central coordinator for all server-state caching and retries. |
| `useWorkspace` | [frontend/components/workspace-provider.tsx:15]() | Zustand store for managing the active workspace. |
| `usePageAPI` | [frontend/hooks/use-page-api.ts:25]() | Tracks the active page for API request attribution. |
| `useStudioThemeFlag` | [frontend/hooks/use-studio-theme.ts:16]() | Intercepts URL flags to enable the Studio preview mode. |
| `GlossaryTooltip` | [frontend/components/ui/glossary-tooltip.tsx:57]() | Manages term learning state and tooltip suppression. |
| `AuthBoundary` | [frontend/components/providers.tsx:29]() | Switches between Clerk (SaaS) and LocalAuth (Self-hosted) token providers. |

Sources: [frontend/components/providers.tsx:29-78](), [frontend/hooks/use-page-api.ts:25-34](), [frontend/hooks/use-studio-theme.ts:16-26](), [frontend/components/ui/glossary-tooltip.tsx:57-126](), [frontend/components/workspace-provider.tsx:15]()

---