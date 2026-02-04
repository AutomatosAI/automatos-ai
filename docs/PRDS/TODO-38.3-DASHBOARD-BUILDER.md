# PRD-38.3: Dashboard Builder

**Version:** 1.0
**Status:** 🟡 Planned
**Priority:** MEDIUM
**Author:** Automatos AI Platform Team
**Last Updated:** 2026-01-27
**Dependencies:** PRD-38.1, PRD-38.2 (Extended Widgets)
**Timeline:** Weeks 6-7

---

## Executive Summary

Enable users to **save, customize, and share** their workspace layouts. Transform the workspace from a session-based artifact viewer into a **persistent, customizable dashboard system** that users can tailor to their workflows.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Save Layouts** | Persist widget arrangements as named workspaces |
| **Templates** | Pre-built workspace templates for common use cases |
| **Sharing** | Share workspaces with team members |
| **Customization** | Widget sizing, theming, and configuration |
| **Quick Layouts** | One-click layout presets (grid, split, focus) |

---

## 1) Goals & Success Metrics

### Goals

| ID | Goal | Description |
|----|------|-------------|
| G1 | **Persistence** | Workspaces persist across sessions and devices |
| G2 | **Templates** | 5+ pre-built templates for common workflows |
| G3 | **Sharing** | Users can share workspaces within their organization |
| G4 | **Customization** | Full control over widget layout and appearance |
| G5 | **Productivity** | Users can switch between workspaces instantly |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Workspace save/load time | < 1s | Performance test |
| Template adoption | 30% of users | Analytics |
| Average workspaces per user | 3+ | Database query |
| Workspace sharing rate | 10% of workspaces | Analytics |

---

## 2) Feature Specifications

### 2.1 Workspace Persistence

**Data Model:**
```typescript
interface SavedWorkspace {
  id: string
  name: string
  description?: string
  ownerId: string
  organizationId?: string

  // Layout configuration
  layout: WorkspaceLayout
  layoutMode: 'grid' | 'freeform' | 'split' | 'focus'

  // Widget configurations
  widgets: SavedWidget[]

  // Metadata
  createdAt: string
  updatedAt: string
  lastOpenedAt?: string

  // Sharing
  visibility: 'private' | 'team' | 'organization' | 'public'
  sharedWith?: string[]  // User IDs

  // Template info
  isTemplate: boolean
  templateCategory?: string
  templateIcon?: string
}

interface SavedWidget {
  type: WidgetType
  title: string
  position: WidgetPosition
  size: WidgetSize
  config?: WidgetConfig  // Widget-specific settings
  dataSource?: DataSourceConfig  // How to refresh data
}

interface WorkspaceLayout {
  columns: number
  rowHeight: number
  margin: [number, number]
  containerPadding: [number, number]
}

interface WidgetConfig {
  // Common settings (validated by WIDGET_CONFIG_SCHEMA)
  refreshInterval?: number  // Auto-refresh in seconds (min: 5, max: 86400)
  theme?: 'default' | 'minimal' | 'compact'
  showHeader?: boolean
  showBorder?: boolean

  // Widget-specific typed fields
  rowsPerPage?: number       // DataWidget: 10-100
  showCharts?: boolean       // DataWidget
  chartType?: 'bar' | 'line' | 'pie'
  fontSize?: number          // CodeWidget: 10-24
  showLineNumbers?: boolean  // CodeWidget
  wordWrap?: boolean         // CodeWidget

  // Extensible custom fields (validated, no arbitrary keys)
  customFields?: Record<string, string | number | boolean>
}

// ⚠️ SECURITY: Strict Zod validation schema for WidgetConfig
// All widget configs MUST be validated through this schema before use
import { z } from 'zod'

const WIDGET_CONFIG_SCHEMA = z.object({
  // Common settings with constraints
  refreshInterval: z.number().min(5).max(86400).optional(),
  theme: z.enum(['default', 'minimal', 'compact']).optional(),
  showHeader: z.boolean().optional(),
  showBorder: z.boolean().optional(),

  // Widget-specific typed fields
  rowsPerPage: z.number().min(10).max(100).optional(),
  showCharts: z.boolean().optional(),
  chartType: z.enum(['bar', 'line', 'pie']).optional(),
  fontSize: z.number().min(10).max(24).optional(),
  showLineNumbers: z.boolean().optional(),
  wordWrap: z.boolean().optional(),

  // Custom fields with size and type constraints
  customFields: z.record(
    z.string().max(50),  // Key max length
    z.union([
      z.string().max(1000),  // Value max length for strings
      z.number(),
      z.boolean()
    ])
  ).refine(
    (obj) => Object.keys(obj).length <= 20,  // Max 20 custom fields
    { message: 'Maximum 20 custom fields allowed' }
  ).optional(),
}).strict()  // Reject any unknown keys

// Usage in Week 7, Day 2 implementation:
function validateWidgetConfig(config: unknown): WidgetConfig {
  const result = WIDGET_CONFIG_SCHEMA.safeParse(config)
  if (!result.success) {
    console.error('Invalid widget config:', result.error.flatten())
    throw new Error('Widget configuration validation failed')
  }
  return result.data
}

interface DataSourceConfig {
  type: 'static' | 'query' | 'tool' | 'api'
  source: string  // Tool name, query, or API endpoint
  params?: Record<string, any>  // TODO: Add schema validation for params
  cacheTime?: number
}
```

### 2.2 Workspace Management UI

**Components:**
```
frontend/components/workspace/
├── WorkspaceManager.tsx        # Workspace list/selector
├── WorkspaceSaveDialog.tsx     # Save workspace modal
├── WorkspaceSettings.tsx       # Workspace configuration
├── WorkspaceShareDialog.tsx    # Sharing controls
├── TemplateGallery.tsx         # Browse templates
└── LayoutPresets.tsx           # Quick layout buttons
```

**WorkspaceManager UI:**
```
┌─────────────────────────────────────────────────────────────┐
│ Workspaces                                    [+ New] [⚙️]  │
├─────────────────────────────────────────────────────────────┤
│ RECENT                                                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 📊 Analytics Dashboard          Updated 2h ago    [⋮]  │ │
│ │ Data queries, charts, metrics                          │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 📧 Email Manager                Updated yesterday [⋮]  │ │
│ │ Inbox, compose, templates                              │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ TEMPLATES                                                   │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│ │ 📈      │ │ 💼      │ │ 🔧      │ │ 📝      │       │
│ │ Data    │ │ CRM     │ │ DevOps  │ │ Content │       │
│ │ Analysis│ │ Dashboard│ │ Monitor │ │ Creator │       │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Pre-built Templates

| Template | Widgets | Use Case |
|----------|---------|----------|
| **Data Analysis** | 2x DataWidget, 1x CodeWidget, 1x DocumentWidget | SQL queries, data exploration |
| **CRM Dashboard** | EmailWidget, DataWidget, WorkflowWidget | Customer management |
| **DevOps Monitor** | TerminalWidget, WorkflowWidget, DataWidget | System monitoring |
| **Content Creator** | DocumentWidget, ImageWidget, EmailWidget | Content production |
| **Research Assistant** | DocumentWidget, MemoryWidget, CodeWidget | Research and learning |
| **Project Manager** | WorkflowWidget, DataWidget, EmailWidget | Project tracking |

**Template Configuration:**
```typescript
const DATA_ANALYSIS_TEMPLATE: SavedWorkspace = {
  id: 'template-data-analysis',
  name: 'Data Analysis',
  description: 'Explore and analyze data with SQL queries and visualizations',
  isTemplate: true,
  templateCategory: 'analytics',
  templateIcon: '📈',
  layoutMode: 'grid',
  layout: {
    columns: 12,
    rowHeight: 100,
    margin: [12, 12],
    containerPadding: [12, 12],
  },
  widgets: [
    {
      type: 'data',
      title: 'Query Results',
      position: { x: 0, y: 0 },
      size: { width: 8, height: 4 },
      config: { theme: 'default' },
    },
    {
      type: 'data',
      title: 'Charts',
      position: { x: 8, y: 0 },
      size: { width: 4, height: 4 },
      config: { theme: 'compact', showCharts: true },
    },
    {
      type: 'code',
      title: 'SQL Editor',
      position: { x: 0, y: 4 },
      size: { width: 6, height: 3 },
      config: { language: 'sql' },
    },
    {
      type: 'document',
      title: 'Documentation',
      position: { x: 6, y: 4 },
      size: { width: 6, height: 3 },
    },
  ],
}
```

### 2.4 Layout Presets

Quick-switch layout options:

| Preset | Description | Layout |
|--------|-------------|--------|
| **Grid** | Equal-sized widgets in a grid | Default |
| **Split** | Two large widgets side by side | 50/50 split |
| **Focus** | One main widget, others minimized | 70/30 split |
| **Stack** | Widgets stacked vertically | Single column |
| **Tiled** | Many small widgets | 4+ columns |

**LayoutPresets Component:**
```typescript
// frontend/components/workspace/LayoutPresets.tsx

const LAYOUT_PRESETS = [
  {
    id: 'grid',
    name: 'Grid',
    icon: <Grid className="h-4 w-4" />,
    apply: (widgets: string[]) => {
      const cols = Math.ceil(Math.sqrt(widgets.length))
      return widgets.map((id, i) => ({
        id,
        position: { x: (i % cols) * (12 / cols), y: Math.floor(i / cols) * 4 },
        size: { width: 12 / cols, height: 4 },
      }))
    },
  },
  {
    id: 'split',
    name: 'Split',
    icon: <SplitSquareHorizontal className="h-4 w-4" />,
    apply: (widgets: string[]) => {
      if (widgets.length === 0) return []
      if (widgets.length === 1) {
        return [{ id: widgets[0], position: { x: 0, y: 0 }, size: { width: 12, height: 6 } }]
      }
      return [
        { id: widgets[0], position: { x: 0, y: 0 }, size: { width: 6, height: 6 } },
        { id: widgets[1], position: { x: 6, y: 0 }, size: { width: 6, height: 6 } },
        ...widgets.slice(2).map((id, i) => ({
          id,
          position: { x: (i % 2) * 6, y: 6 + Math.floor(i / 2) * 3 },
          size: { width: 6, height: 3 },
        })),
      ]
    },
  },
  {
    id: 'focus',
    name: 'Focus',
    icon: <Maximize2 className="h-4 w-4" />,
    apply: (widgets: string[], activeId?: string) => {
      const mainId = activeId || widgets[0]
      const others = widgets.filter((id) => id !== mainId)
      return [
        { id: mainId, position: { x: 0, y: 0 }, size: { width: 9, height: 6 } },
        ...others.map((id, i) => ({
          id,
          position: { x: 9, y: i * 2 },
          size: { width: 3, height: 2 },
        })),
      ]
    },
  },
]
```

### 2.5 Workspace Sharing

**Sharing Levels:**
- **Private**: Only owner can access
- **Team**: Shared with specific users
- **Organization**: All org members can view
- **Public**: Accessible via link (read-only)

**Share Dialog:**
```
┌─────────────────────────────────────────────────────────────┐
│ Share "Analytics Dashboard"                          [×]    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Visibility: [Private ▾]                                     │
│                                                             │
│ Share with specific people:                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 🔍 Search by name or email...                          │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Shared with:                                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 👤 John Doe         john@example.com      Can edit [×] │ │
│ │ 👤 Jane Smith       jane@example.com      Can view [×] │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Or share via link:                                          │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ https://app.automatos.ai/workspace/abc123   [📋 Copy]  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│                              [Cancel]  [Save Changes]       │
└─────────────────────────────────────────────────────────────┘
```

### 2.6 Widget Configuration

Allow per-widget customization:

```typescript
interface WidgetConfigOptions {
  // Appearance
  theme: 'default' | 'minimal' | 'compact'
  showHeader: boolean
  showBorder: boolean
  backgroundColor?: string

  // Behavior
  autoRefresh: boolean
  refreshInterval: number  // seconds
  showLastUpdated: boolean

  // Widget-specific
  // DataWidget
  rowsPerPage?: number
  showCharts?: boolean
  chartType?: 'bar' | 'line' | 'pie'

  // CodeWidget
  fontSize?: number
  showLineNumbers?: boolean
  wordWrap?: boolean

  // etc.
}
```

**Config Dialog:**
```
┌─────────────────────────────────────────────────────────────┐
│ Widget Settings: "Sales Query"                       [×]    │
├─────────────────────────────────────────────────────────────┤
│ APPEARANCE                                                  │
│ Theme:         [Default ▾]                                  │
│ Show header:   [✓]                                          │
│ Show border:   [✓]                                          │
│                                                             │
│ BEHAVIOR                                                    │
│ Auto-refresh:  [✓]                                          │
│ Interval:      [30] seconds                                 │
│                                                             │
│ DATA WIDGET OPTIONS                                         │
│ Rows per page: [25 ▾]                                       │
│ Show charts:   [✓]                                          │
│ Chart type:    [Bar ▾]                                      │
│                                                             │
│                              [Cancel]  [Save]               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3) Implementation Plan

### Week 6: Core Persistence + Templates

| Day | Task |
|-----|------|
| 1 | Database schema for workspaces |
| 2 | API endpoints (CRUD) |
| 3 | WorkspaceManager UI |
| 4 | Save/Load workspace logic |
| 5 | Template gallery |

### Week 7: Customization + Sharing

| Day | Task |
|-----|------|
| 1 | Layout presets |
| 2 | Widget configuration system |
| 3 | Sharing backend |
| 4 | Share dialog UI |
| 5 | Testing and polish |

---

## 4) API Endpoints

```
# Workspace CRUD
GET    /api/workspaces                    # List user's workspaces
GET    /api/workspaces/:id                # Get single workspace
POST   /api/workspaces                    # Create workspace
PUT    /api/workspaces/:id                # Update workspace
DELETE /api/workspaces/:id                # Delete workspace

# Templates
GET    /api/workspaces/templates          # List available templates
POST   /api/workspaces/from-template/:id  # Create from template

# Sharing
POST   /api/workspaces/:id/share          # Update sharing settings
GET    /api/workspaces/shared-with-me     # Workspaces shared with user
```

**Request/Response Examples:**

```typescript
// POST /api/workspaces
// Request:
{
  "name": "My Analytics Dashboard",
  "description": "Daily metrics and reports",
  "layout": { "columns": 12, "rowHeight": 100 },
  "layoutMode": "grid",
  "widgets": [
    {
      "type": "data",
      "title": "Sales Query",
      "position": { "x": 0, "y": 0 },
      "size": { "width": 6, "height": 4 }
    }
  ]
}

// Response:
{
  "id": "ws_abc123",
  "name": "My Analytics Dashboard",
  "createdAt": "2026-01-27T10:00:00Z",
  // ... full workspace object
}
```

---

## 5) Database Schema

```sql
-- Workspaces table
CREATE TABLE workspaces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  description TEXT,
  owner_id UUID NOT NULL REFERENCES users(id),
  organization_id UUID REFERENCES organizations(id),

  -- Layout
  layout JSONB NOT NULL DEFAULT '{"columns": 12, "rowHeight": 100}',
  layout_mode VARCHAR(20) DEFAULT 'grid',
  widgets JSONB NOT NULL DEFAULT '[]',

  -- Template
  is_template BOOLEAN DEFAULT FALSE,
  template_category VARCHAR(50),
  template_icon VARCHAR(10),

  -- Sharing
  visibility VARCHAR(20) DEFAULT 'private',

  -- ⚠️ SECURITY: Public workspace settings (required for visibility='public')
  -- Controls discovery, embedding, and analytics visibility for public workspaces
  public_workspace_settings JSONB NOT NULL DEFAULT '{
    "discovery": "direct_link_only",
    "analytics_visible": false,
    "embedding_allowed": false,
    "require_attribution": true
  }',

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  last_opened_at TIMESTAMP WITH TIME ZONE,

  -- Indexes
  CONSTRAINT valid_visibility CHECK (visibility IN ('private', 'team', 'organization', 'public')),
  CONSTRAINT valid_layout_mode CHECK (layout_mode IN ('grid', 'freeform', 'split', 'focus')),

  -- Ensure public_workspace_settings is valid when visibility is 'public'
  CONSTRAINT valid_public_settings CHECK (
    visibility != 'public' OR (
      public_workspace_settings ? 'discovery' AND
      public_workspace_settings ? 'analytics_visible' AND
      public_workspace_settings ? 'embedding_allowed'
    )
  )
);

CREATE INDEX idx_workspaces_owner ON workspaces(owner_id);
CREATE INDEX idx_workspaces_org ON workspaces(organization_id);
CREATE INDEX idx_workspaces_template ON workspaces(is_template) WHERE is_template = TRUE;
CREATE INDEX idx_workspaces_public ON workspaces(visibility) WHERE visibility = 'public';

-- Workspace sharing table
CREATE TABLE workspace_shares (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES users(id),
  permission VARCHAR(20) NOT NULL DEFAULT 'view',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  UNIQUE(workspace_id, user_id),
  CONSTRAINT valid_permission CHECK (permission IN ('view', 'edit', 'admin'))
);

CREATE INDEX idx_workspace_shares_user ON workspace_shares(user_id);

-- ⚠️ SECURITY: Workspace moderation table (for public workspaces)
CREATE TABLE workspace_moderation (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

  -- Moderation status
  moderation_status VARCHAR(20) NOT NULL DEFAULT 'pending',
  abuse_reported BOOLEAN DEFAULT FALSE,
  report_count INTEGER DEFAULT 0,

  -- Review tracking
  reviewed_by UUID REFERENCES users(id),
  reviewed_at TIMESTAMP WITH TIME ZONE,
  review_notes TEXT,

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  UNIQUE(workspace_id),
  CONSTRAINT valid_moderation_status CHECK (
    moderation_status IN ('pending', 'approved', 'flagged', 'suspended', 'removed')
  )
);

CREATE INDEX idx_workspace_moderation_status ON workspace_moderation(moderation_status);
CREATE INDEX idx_workspace_moderation_abuse ON workspace_moderation(abuse_reported) WHERE abuse_reported = TRUE;
```

### Public Workspace Settings Schema

```typescript
// Type definition for public_workspace_settings JSONB column
interface PublicWorkspaceSettings {
  // Discovery mode
  discovery: 'discoverable' | 'direct_link_only'  // Gallery vs link-only

  // Visibility controls
  analytics_visible: boolean   // Show view count, etc. to visitors
  embedding_allowed: boolean   // Allow embedding via iframe

  // Attribution
  require_attribution: boolean  // Require "Powered by Automatos" badge
}

// API handler must enforce these rules before listing in gallery
async function listPublicWorkspaces(req: Request) {
  // Only return workspaces that are:
  // 1. visibility = 'public'
  // 2. public_workspace_settings.discovery = 'discoverable'
  // 3. workspace_moderation.moderation_status = 'approved'

  const workspaces = await db.workspace.findMany({
    where: {
      visibility: 'public',
      public_workspace_settings: {
        path: ['discovery'],
        equals: 'discoverable',
      },
      moderation: {
        moderation_status: 'approved',
      },
    },
  })

  return workspaces
}

// Rate limiting for public workspace access
const PUBLIC_WORKSPACE_RATE_LIMIT = {
  windowMs: 60 * 1000,  // 1 minute
  max: 30,              // 30 requests per minute (stricter than authenticated)
}
```

### UX Flow: Public vs Direct Link

```
User sets visibility = 'public'
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ How should this workspace be shared?                      │
│                                                           │
│ ○ Direct link only                                        │
│   Anyone with the link can view, but it won't appear      │
│   in the public gallery or search results.                │
│                                                           │
│ ○ Discoverable (requires review)                          │
│   Appears in public gallery after moderation review.      │
│   Expected review time: 24-48 hours.                      │
│                                                           │
│ Additional settings:                                      │
│ ☐ Show view count and analytics to visitors               │
│ ☐ Allow embedding on external websites                    │
│ ☑ Require "Powered by Automatos" attribution              │
│                                                           │
│                    [Cancel]  [Make Public]                │
└───────────────────────────────────────────────────────────┘
```

**⚠️ Product Decision Required:** Public workspaces with `discovery: 'discoverable'` should NOT be enabled by default. This feature requires:
1. Moderation team/process in place
2. Abuse reporting mechanism
3. Terms of service update
4. Privacy policy review

---

## 6) Files to Create

```
frontend/
├── components/workspace/
│   ├── WorkspaceManager.tsx
│   ├── WorkspaceSaveDialog.tsx
│   ├── WorkspaceSettings.tsx
│   ├── WorkspaceShareDialog.tsx
│   ├── TemplateGallery.tsx
│   ├── LayoutPresets.tsx
│   └── WidgetConfigDialog.tsx
│
├── lib/workspace/
│   ├── api.ts              # API client
│   ├── templates.ts        # Template definitions
│   └── presets.ts          # Layout presets
│
└── stores/
    └── workspace-store.ts  # Update with persistence

backend/
├── api/workspaces.py       # Workspace endpoints
├── models/workspace.py     # SQLAlchemy model
└── services/workspace.py   # Business logic
```

---

## 7) Store Updates

```typescript
// Updated workspace-store.ts

interface WorkspaceState {
  // ... existing state ...

  // Persistence
  currentWorkspaceId: string | null
  savedWorkspaces: SavedWorkspace[]
  isLoading: boolean
  isSaving: boolean
  lastSaved: Date | null
  hasUnsavedChanges: boolean
}

interface WorkspaceActions {
  // ... existing actions ...

  // Persistence
  loadWorkspaces: () => Promise<void>
  loadWorkspace: (id: string) => Promise<void>
  saveWorkspace: (name?: string) => Promise<void>
  saveAsWorkspace: (name: string, description?: string) => Promise<void>
  deleteWorkspace: (id: string) => Promise<void>

  // Templates
  loadTemplates: () => Promise<void>
  createFromTemplate: (templateId: string) => Promise<void>

  // Sharing
  shareWorkspace: (id: string, settings: ShareSettings) => Promise<void>

  // Presets
  applyLayoutPreset: (presetId: string) => void
}
```

---

## 8) Testing Checklist

### Persistence
- [ ] Create new workspace
- [ ] Save workspace with widgets
- [ ] Load workspace restores layout
- [ ] Update workspace name/description
- [ ] Delete workspace
- [ ] Auto-save on changes

### Templates
- [ ] Load template gallery
- [ ] Create workspace from template
- [ ] Template widgets are configured correctly

### Sharing
- [ ] Share with specific users
- [ ] View shared workspaces
- [ ] Permissions enforced (view vs edit)
- [ ] Public link sharing

### Layout Presets
- [ ] Grid preset works
- [ ] Split preset works
- [ ] Focus preset works
- [ ] Preset preserves widget content

### Widget Configuration
- [ ] Theme changes apply
- [ ] Auto-refresh works
- [ ] Settings persist in saved workspace

---

## 9) Success Criteria

Phase 3 is complete when:

1. [ ] Workspaces can be saved and loaded
2. [ ] 5+ templates available
3. [ ] Layout presets work correctly
4. [ ] Widget configuration dialogs functional
5. [ ] Sharing with team members works
6. [ ] Workspace list UI shows all user workspaces
7. [ ] Auto-save indicates unsaved changes
8. [ ] Performance: save/load < 1s

---

## 10) Open Questions

1. **Versioning**: Should workspaces have version history?
2. **Duplication**: Can users duplicate a workspace?
3. **Export/Import**: JSON export for backup?
4. **Keyboard shortcuts**: Ctrl+S to save, etc.?
5. **Mobile**: How do saved layouts translate to mobile?

---

## 11) References

- [React Grid Layout](https://github.com/react-grid-layout/react-grid-layout)
- [Notion Templates](https://www.notion.so/templates) - Inspiration
- [Figma Files](https://www.figma.com/) - Sharing model
- PRD-38.1, PRD-38.2

---

*Document Version: 1.0*
*Created: 2026-01-27*
*Estimated Implementation: 2 weeks*
