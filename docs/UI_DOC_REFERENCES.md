# UI Documentation References — Implementation Guide

How to add contextual help links throughout the Automatos UI, connecting users to the GitBook documentation at the point of need.

## Existing Infrastructure

The frontend already has production-ready components for contextual help. No new libraries or components needed — just content.

### Components (`/frontend/components/ui/help-tooltip.tsx`)

| Component | Use case | Example |
| --- | --- | --- |
| `HelpTooltip` | Icon-triggered tooltip next to labels | Settings field explanations |
| `InlineHelp` | Small inline hint text with optional doc link | Form field guidance |
| `SectionHelp` | Section-level help block with description | Top of configuration panels |
| `FieldHelp` | Per-field tooltip with doc link | Agent creation form fields |

### Tooltip Registry (`/frontend/lib/tooltips.json`)

Centralised content store. Components look up text via dot-notation paths:

```json
{
  "agents": {
    "roster": {
      "create_button": {
        "text": "Create a new AI agent with a specific role, model, and tools",
        "docLink": "/agents/creating"
      }
    }
  }
}
```

### Base URL

All `docLink` values are relative to:

```
https://automatos.gitbook.io/automatos-ai
```

This is already configured in the `HelpTooltip` component.

### Shepherd Tours (`/frontend/lib/shepherd/tours/`)

Page-specific onboarding walkthroughs. Tours use `data-tour` attributes on DOM elements for step targeting.

---

## Priority 1 — Agent Creation Modal

**File:** `/frontend/components/agents/agent-configuration.tsx` (and related creation dialog)

Add `FieldHelp` to each step of the agent creation form:

```tsx
// Step 2: Basic information
<div className="flex items-center gap-2">
  <Label>Name</Label>
  <FieldHelp
    text="A display name for the agent. Used in chat headers and the roster."
    docLink="/agents/creating#step-2-basic-information"
  />
</div>

<div className="flex items-center gap-2">
  <Label>Description</Label>
  <FieldHelp
    text="Describe what this agent does. The Universal Router uses this to match incoming messages."
    docLink="/agents/creating#step-2-basic-information"
  />
</div>

// Step 3: Model selection
<div className="flex items-center gap-2">
  <Label>Provider</Label>
  <FieldHelp
    text="Choose the LLM provider. OpenRouter gives access to 100+ models via a single key."
    docLink="/settings/models"
  />
</div>

<div className="flex items-center gap-2">
  <Label>Temperature</Label>
  <FieldHelp
    text="Controls creativity. 0 = deterministic, 1 = creative. Use 0.1-0.3 for code tasks, 0.7+ for writing."
    docLink="/agents/creating#step-3-model-selection"
  />
</div>

// Step 5: Tools and skills
<div className="flex items-center gap-2">
  <Label>Tools</Label>
  <FieldHelp
    text="External apps this agent can use (GitHub, Slack, Jira). Connect apps first in Tools & Integrations."
    docLink="/tools/assigning"
  />
</div>
```

**Add to `tooltips.json`:**

```json
{
  "agents": {
    "create": {
      "name": {
        "text": "A display name for the agent. Used in chat headers and the roster.",
        "docLink": "/agents/creating#step-2-basic-information"
      },
      "description": {
        "text": "Describe what this agent does. The Universal Router uses this to match incoming messages — be specific.",
        "docLink": "/agents/creating#step-2-basic-information"
      },
      "category": {
        "text": "Agent type — helps with roster organisation, routing, and analytics grouping.",
        "docLink": "/agents/creating#step-2-basic-information"
      },
      "provider": {
        "text": "Choose the LLM provider. OpenRouter gives access to 100+ models via a single key.",
        "docLink": "/settings/models"
      },
      "model": {
        "text": "The specific model this agent uses. Different models have different strengths and costs.",
        "docLink": "/settings/models"
      },
      "temperature": {
        "text": "Controls creativity. 0 = deterministic, 1 = creative. Use 0.1–0.3 for code, 0.7+ for writing.",
        "docLink": "/agents/creating#step-3-model-selection"
      },
      "persona": {
        "text": "Defines the agent's communication style — professional, casual, technical, or custom.",
        "docLink": "/agents/creating#step-4-persona"
      },
      "system_prompt": {
        "text": "The core instructions that shape this agent's behaviour. See prompt engineering tips.",
        "docLink": "/agents/creating#prompt-engineering-tips"
      },
      "tools": {
        "text": "External apps this agent can call. Connect apps first in Tools & Integrations.",
        "docLink": "/tools/assigning"
      },
      "skills": {
        "text": "Git-based capability packages that extend what the agent can do.",
        "docLink": "/marketplace/capabilities"
      }
    }
  }
}
```

---

## Priority 2 — Document Upload Form

**File:** `/frontend/components/documents/document-management.tsx` (and upload dialog)

```tsx
// Upload dialog
<SectionHelp
  title="Upload Documents"
  description="Upload files for your agents to search and reference. Documents are automatically chunked and embedded for semantic search."
  docLink="/knowledge/documents"
/>

// File type hint
<InlineHelp
  text="Supported: PDF, DOCX, TXT, MD, CSV, and images. Max 20 MB per file."
  docLink="/knowledge/documents#supported-file-types"
/>

// RAG configuration section
<FieldHelp
  text="Similarity threshold for search results. Lower = more results, higher = more relevant."
  docLink="/knowledge/documents#rag-configuration"
/>
```

**Add to `tooltips.json`:**

```json
{
  "documents": {
    "upload": {
      "formats": {
        "text": "Supported: PDF, DOCX, TXT, MD, CSV, images. Max 20 MB per file.",
        "docLink": "/knowledge/documents"
      },
      "processing": {
        "text": "Documents are automatically split into chunks and embedded for vector search. Processing takes 10–60 seconds.",
        "docLink": "/knowledge/documents#processing-pipeline"
      }
    },
    "search": {
      "similarity": {
        "text": "Minimum similarity score (0–1). Default 0.7. Lower values return more results.",
        "docLink": "/knowledge/documents#rag-configuration"
      },
      "top_k": {
        "text": "Maximum number of chunks to return. Default 5.",
        "docLink": "/knowledge/documents#rag-configuration"
      }
    },
    "cloud_sync": {
      "setup": {
        "text": "Sync documents automatically from cloud storage providers.",
        "docLink": "/knowledge/cloud-sync"
      }
    }
  }
}
```

---

## Priority 3 — Settings Tabs

**File:** `/frontend/components/settings/SettingsPanel.tsx` and individual tab components

Each settings tab should have a `SectionHelp` at the top:

```tsx
// API Keys tab
<SectionHelp
  title="API Keys & Credentials"
  description="Manage API keys for LLM providers and external services. Keys are encrypted at rest."
  docLink="/settings/credentials"
/>

// Models tab
<SectionHelp
  title="Model Configuration"
  description="Set default models, configure fallback chains, and manage embedding settings."
  docLink="/settings/models"
/>

// General tab
<SectionHelp
  title="General Settings"
  description="Workspace name, feature flags, notification preferences, and system defaults."
  docLink="/settings/general"
/>

// Audit Logs tab
<SectionHelp
  title="Audit Logs"
  description="Complete record of user actions, agent operations, and configuration changes."
  docLink="/settings/audit-logs"
/>
```

**Per-provider API key help** (`SystemLLMSettingsTab.tsx`):

```tsx
// Each provider section
<FieldHelp
  text="Get your OpenAI key at platform.openai.com/api-keys"
  docLink="/settings/credentials#adding-a-credential"
/>

<FieldHelp
  text="Get your Anthropic key at console.anthropic.com/settings/keys"
  docLink="/settings/credentials#adding-a-credential"
/>

<FieldHelp
  text="Get your OpenRouter key at openrouter.ai/keys — gives access to 100+ models"
  docLink="/settings/credentials#adding-a-credential"
/>
```

**Add to `tooltips.json`:**

```json
{
  "settings": {
    "api_keys": {
      "openai": {
        "text": "Get your key at platform.openai.com/api-keys",
        "docLink": "/settings/credentials"
      },
      "anthropic": {
        "text": "Get your key at console.anthropic.com/settings/keys",
        "docLink": "/settings/credentials"
      },
      "openrouter": {
        "text": "Get your key at openrouter.ai/keys — access 100+ models with one key",
        "docLink": "/settings/credentials"
      },
      "deepseek": {
        "text": "Get your key at platform.deepseek.com",
        "docLink": "/settings/credentials"
      }
    },
    "models": {
      "default_model": {
        "text": "Used when agents don't have a specific model set, and for system tasks like routing.",
        "docLink": "/settings/models#default-model"
      },
      "fallback": {
        "text": "Backup models used when the primary is unavailable or rate-limited.",
        "docLink": "/settings/models#model-fallback"
      },
      "embedding": {
        "text": "Used for document chunking and semantic search. Changing this requires re-processing all documents.",
        "docLink": "/settings/models#embedding-model"
      }
    },
    "general": {
      "feature_flags": {
        "text": "Toggle optional features: voice chat, advanced analytics, knowledge graph, mission mode.",
        "docLink": "/settings/general#feature-flags"
      },
      "notifications": {
        "text": "Configure in-app, email, and channel notification preferences.",
        "docLink": "/settings/general#notification-preferences"
      }
    }
  }
}
```

---

## Priority 4 — Empty States

**Pattern:** Create a reusable `EmptyStateWithDocs` component or add doc links to existing empty states.

```tsx
// Suggested component
function EmptyStateWithDocs({
  icon,
  title,
  description,
  docLink,
  actionLabel,
  onAction,
}: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {icon}
      <h3 className="mt-4 text-lg font-semibold">{title}</h3>
      <p className="mt-2 text-sm text-muted-foreground max-w-md">{description}</p>
      <div className="mt-6 flex gap-3">
        <Button onClick={onAction}>{actionLabel}</Button>
        <Button variant="outline" asChild>
          <a href={`https://automatos.gitbook.io/automatos-ai${docLink}`}
             target="_blank" rel="noopener noreferrer">
            Read the Guide
          </a>
        </Button>
      </div>
    </div>
  );
}
```

**Where to use it:**

| Page | Empty state trigger | `docLink` | Action button |
| --- | --- | --- | --- |
| Agent Roster | No agents created | `/agents/creating` | + Create Agent |
| Documents | No documents uploaded | `/knowledge/documents` | Upload Documents |
| Playbooks | No recipes created | `/agents/recipes` | + Create Recipe |
| Knowledge Graph | No entities extracted | `/knowledge/knowledge-graph` | Upload Documents |
| Missions | No missions created | `/activity/missions` | + New Mission |
| Channels | No channels connected | `/tools/channels` | Connect a Channel |
| Marketplace | First visit | `/marketplace` | Browse Marketplace |

---

## Priority 5 — Sidebar Documentation Link

**File:** `/frontend/components/layout/sidebar.tsx`

Add a persistent "Docs" link in the sidebar footer:

```tsx
// In the sidebar footer section
<SidebarFooter>
  <SidebarMenuItem>
    <SidebarMenuButton asChild>
      <a
        href="https://automatos.gitbook.io/automatos-ai"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2"
      >
        <BookOpen className="h-4 w-4" />
        <span>Documentation</span>
        <ExternalLink className="h-3 w-3 ml-auto opacity-50" />
      </a>
    </SidebarMenuButton>
  </SidebarMenuItem>
</SidebarFooter>
```

Also consider adding a help dropdown in the header (`/frontend/components/layout/header.tsx`):

```tsx
<DropdownMenu>
  <DropdownMenuTrigger asChild>
    <Button variant="ghost" size="icon">
      <HelpCircle className="h-4 w-4" />
    </Button>
  </DropdownMenuTrigger>
  <DropdownMenuContent align="end">
    <DropdownMenuItem asChild>
      <a href="https://automatos.gitbook.io/automatos-ai" target="_blank">
        Documentation
      </a>
    </DropdownMenuItem>
    <DropdownMenuItem asChild>
      <a href="https://automatos.gitbook.io/automatos-ai/api-reference" target="_blank">
        API Reference
      </a>
    </DropdownMenuItem>
    <DropdownMenuSeparator />
    <DropdownMenuItem asChild>
      <a href="https://github.com/AutomatosAI/automatos-ai/issues" target="_blank">
        Report a Bug
      </a>
    </DropdownMenuItem>
  </DropdownMenuContent>
</DropdownMenu>
```

---

## Full `tooltips.json` Expansion

Merge the entries from Priorities 1–3 above into the existing `/frontend/lib/tooltips.json`. The file already has partial coverage — extend it with all the new entries while preserving existing content.

### Checklist

- [ ] **Priority 1:** Agent creation modal — 10 field tooltips
- [ ] **Priority 2:** Document upload form — 5 field tooltips + section help
- [ ] **Priority 3:** Settings tabs — 4 section helps + 10 field tooltips
- [ ] **Priority 4:** Empty states — 7 pages with doc-linked empty states
- [ ] **Priority 5:** Sidebar docs link + header help dropdown
- [ ] **Merge** all new entries into `tooltips.json`
- [ ] **Test** doc links resolve correctly on the GitBook site

### URL Mapping (GitBook path → UI location)

| GitBook Path | UI Location |
| --- | --- |
| `/chat` | Chat page |
| `/chat/voice` | Voice chat input |
| `/chat/routing` | Auto mode selector |
| `/agents/creating` | Agent creation modal |
| `/agents/creating#prompt-engineering-tips` | System prompt editor |
| `/agents/details` | Agent detail panel |
| `/agents/configuration` | Global agent config tab |
| `/agents/coordination` | Coordination tab |
| `/agents/recipes` | Recipes tab |
| `/activity/missions` | Missions tab |
| `/activity/memory` | Memory browser |
| `/tools/connecting-apps` | Tool connection flow |
| `/tools/channels` | Channel setup |
| `/tools/security` | Security tab |
| `/knowledge/documents` | Document upload/search |
| `/knowledge/database` | SQL Explorer |
| `/knowledge/knowledge-graph` | Knowledge Graph tab |
| `/knowledge/codegraph` | CodeGraph tab |
| `/marketplace` | Marketplace homepage |
| `/marketplace/capabilities` | Plugin/skill browser |
| `/analytics/llm-costs` | Cost dashboard |
| `/settings/credentials` | API key management |
| `/settings/models` | Model configuration |
| `/settings/general` | General settings |
| `/settings/audit-logs` | Audit log viewer |
| `/team/inviting` | Team invite flow |
| `/team/roles` | Roles configuration |
| `/api-reference` | Developer integrations |
