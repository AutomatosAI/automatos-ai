# Multi-Purpose Workspace UI Plan

## Executive Summary

Transform the current chatbot interface into a comprehensive **AI-Powered Workspace** that mirrors the productivity of tools like Claude Code, Cursor, and Notion AI - but for your entire platform's capabilities: RAG, NL2SQL, CodeGraph, Composio integrations, and agent orchestration.

---

## Current State Analysis

### What We Have

```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar (chat history)  │  Chat Area        │  Artifact   │
│                          │                   │  Viewer     │
│  - History list          │  - Messages       │  (overlay)  │
│  - Search                │  - Scroll         │             │
│  - New chat              │  - Input          │  - Code     │
│                          │                   │  - Text     │
│                          │                   │  - Sheet    │
│                          │                   │  - Image    │
└─────────────────────────────────────────────────────────────┘
```

**Strengths:**
- 4 artifact types already supported (code, text, sheet, image)
- Good animation system (Framer Motion)
- Tool call transparency
- RAG chunk inspector
- Database result visualization

**Limitations:**
- Artifact viewer is modal/overlay - not persistent workspace
- Single artifact at a time
- No artifact history/management
- No side-by-side artifact comparison
- Read-only artifacts (no editing)
- No artifact persistence independent of chat

---

## Proposed Architecture: "Canvas Workspace"

### Design Philosophy

> **"Chat is the command line, Canvas is the workspace"**

The chat becomes a natural language interface to manipulate a persistent workspace where artifacts live, evolve, and can be arranged freely.

### Layout Concept

```
┌──────────────────────────────────────────────────────────────────────────┐
│  HEADER BAR                                                              │
│  [Workspace: Project Alpha ▾]  [Tools: 12 active]  [Agent: Research ▾]  │
├────────────┬─────────────────────────────────────────────────────────────┤
│            │                                                             │
│  CHAT      │                    CANVAS / WORKSPACE                       │
│  PANEL     │                                                             │
│  (320px)   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│            │   │  Document   │  │   Chart     │  │   Code      │       │
│ [Messages] │   │  Artifact   │  │   Artifact  │  │   Artifact  │       │
│            │   │             │  │             │  │             │       │
│            │   │  📄 Report  │  │  📊 Sales   │  │  🐍 main.py │       │
│            │   │             │  │             │  │             │       │
│            │   └─────────────┘  └─────────────┘  └─────────────┘       │
│            │                                                             │
│            │   ┌─────────────────────────────────┐                      │
│            │   │       Database Results          │                      │
│            │   │  📋 composio_apps_cache         │                      │
│            │   │  ┌──────────────────────────┐   │                      │
│            │   │  │ app_name │ actions │ ... │   │                      │
│            │   │  ├──────────────────────────┤   │                      │
│            │   │  │ GMAIL    │ 45      │     │   │                      │
│            │   │  │ SLACK    │ 32      │     │   │                      │
│            │   │  └──────────────────────────┘   │                      │
│            │   └─────────────────────────────────┘                      │
│            │                                                             │
│ ┌────────┐ │   [+ Add Artifact]                    [Artifact Tray ▾]   │
│ │ Input  │ │                                                             │
│ │ Box    │ ├─────────────────────────────────────────────────────────────┤
│ └────────┘ │  STATUS BAR: 3 artifacts • Last tool: search_knowledge     │
└────────────┴─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Chat Panel (Slimmed Down)

**Purpose:** Command interface for the workspace

**Changes from Current:**
- Narrower (320px vs current 400px when artifact open)
- Always visible (not conditional)
- Collapsible to icon-only mode (48px)
- Shows tool execution inline but moves results to canvas

**New Features:**
- Quick commands: `/search`, `/sql`, `/code`, `/email`, `/image`
- Artifact references: `@doc-1`, `@chart-sales`
- Context indicator: Shows what artifacts are "in context"

```typescript
interface ChatPanelProps {
  isCollapsed: boolean
  onToggleCollapse: () => void
  activeArtifacts: Artifact[]  // Artifacts currently in context
  onArtifactMention: (artifactId: string) => void
}
```

### 2. Canvas (New - Central Workspace)

**Purpose:** Visual workspace for all artifacts

**Layout Modes:**
1. **Grid View** - Auto-arranged cards (default)
2. **Freeform View** - Drag-and-drop positioning
3. **Split View** - 2-3 artifacts side by side
4. **Focus View** - Single artifact maximized

**Features:**
- Drag to reorder/resize
- Snap-to-grid alignment
- Artifact grouping (folders/sections)
- Zoom in/out for overview

```typescript
interface CanvasProps {
  artifacts: Artifact[]
  layout: 'grid' | 'freeform' | 'split' | 'focus'
  onArtifactSelect: (artifact: Artifact) => void
  onArtifactMove: (id: string, position: Position) => void
  onArtifactResize: (id: string, size: Size) => void
  onArtifactClose: (id: string) => void
}
```

### 3. Artifact Cards (Enhanced)

**Universal Card Structure:**
```
┌────────────────────────────────────────┐
│ 📊 Sales Report Q4        [⋮] [×]      │  ← Header with type icon, menu
├────────────────────────────────────────┤
│                                        │
│         [ARTIFACT CONTENT]             │  ← Type-specific renderer
│                                        │
├────────────────────────────────────────┤
│ Source: NL2SQL • 2 mins ago • @chart-1 │  ← Footer with metadata
└────────────────────────────────────────┘
```

**Artifact Types to Support:**

| Type | Icon | Source | Capabilities |
|------|------|--------|--------------|
| **Document** | 📄 | RAG, Write | View, Edit, Export PDF |
| **Code** | 🐍 | CodeGraph, Write | View, Edit, Run, Copy |
| **Chart** | 📊 | NL2SQL, PandasAI | View, Refresh, Export PNG |
| **Table** | 📋 | NL2SQL, Query | View, Filter, Export CSV |
| **Image** | 🖼️ | Image Agent | View, Download, Regenerate |
| **Email** | ✉️ | Gmail/Composio | View, Reply, Forward |
| **File** | 📁 | File Tools | View, Download, Edit |
| **Terminal** | 💻 | Shell Tools | View output, Re-run |
| **Workflow** | 🔄 | Orchestrator | View status, Edit, Run |
| **Memory** | 🧠 | Memory System | View, Search, Manage |

```typescript
interface ArtifactCard {
  id: string
  type: ArtifactType
  title: string
  content: any  // Type-specific content
  source: {
    tool: string
    timestamp: Date
    toolCallId?: string
  }
  position?: { x: number, y: number }
  size?: { width: number, height: number }
  state: 'loading' | 'ready' | 'error' | 'stale'
  actions: ArtifactAction[]  // Available actions for this type
}
```

### 4. Artifact Tray (New)

**Purpose:** Persistent storage of all artifacts from the session

```
┌─────────────────────────────────────────────────────────┐
│ ARTIFACT TRAY                               [↑ Expand] │
├─────────────────────────────────────────────────────────┤
│ 📄 Doc-1  📊 Chart-1  📋 Table-1  🐍 Code-1  ✉️ Email-1 │
│ (click to open on canvas, drag to position)            │
└─────────────────────────────────────────────────────────┘
```

**Features:**
- Horizontal scrollable list
- Drag artifacts to canvas
- Quick preview on hover
- Badge for new/updated artifacts
- Filter by type

### 5. Context Panel (New - Optional Sidebar)

**Purpose:** Deep inspection of selected artifact

```
┌──────────────────────────────┐
│ CONTEXT: Sales Chart         │
├──────────────────────────────┤
│ Type: Chart (Bar)            │
│ Source: smart_query_database │
│ Created: 2 mins ago          │
│ SQL Query: [View]            │
├──────────────────────────────┤
│ ACTIONS                      │
│ [Refresh Data]               │
│ [Export as PNG]              │
│ [Edit SQL Query]             │
│ [Add to Report]              │
├──────────────────────────────┤
│ RELATED                      │
│ 📋 Raw data table            │
│ 📄 Q4 Report draft           │
└──────────────────────────────┘
```

---

## Interaction Flows

### Flow 1: "Write me a document/report"

```
User: "Write me a report on our Q4 sales performance"

1. Chat shows: "Writing report..." with tool indicator
2. Tool: search_knowledge → finds relevant docs (RAG)
3. Tool: smart_query_database → gets sales data
4. Tool: write_file → creates report.md

Canvas Updates:
┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐
│ 📊 Source   │  │ 📋 Sales    │  │ 📄 Q4 Sales Report   │
│ Documents   │  │ Data Table  │  │                      │
│ (3 chunks)  │  │             │  │ ## Executive Summary │
└─────────────┘  └─────────────┘  │ ...                  │
                                  └──────────────────────┘

User can: Edit the report, export as PDF, reference in future prompts
```

### Flow 2: "Show me charts from my DB"

```
User: "Show me a chart of daily active users this month"

1. Tool: smart_query_database → generates SQL
2. PandasAI: Generates chart from data

Canvas Updates:
┌──────────────────────┐  ┌──────────────────────┐
│ 📋 DAU Raw Data      │  │ 📊 DAU Chart         │
│ ┌──────────────────┐ │  │                      │
│ │ date    │ users  │ │  │  [Line Chart Image]  │
│ ├──────────────────┤ │  │                      │
│ │ Jan 1   │ 1,234  │ │  │  Peak: Jan 15        │
│ │ Jan 2   │ 1,456  │ │  │  Avg: 1,345          │
│ └──────────────────┘ │  │                      │
└──────────────────────┘  └──────────────────────┘

User can: Filter table, export CSV, regenerate chart with different type
```

### Flow 3: "Help me debug AgentFactory"

```
User: "I need help debugging an issue with AgentFactory"

1. Tool: search_codebase → finds AgentFactory code
2. Shows code with context

Canvas Updates:
┌────────────────────────────────────────────────┐
│ 🐍 agent_factory.py                            │
├────────────────────────────────────────────────┤
│ class AgentFactory:                            │
│     """Factory for creating agents..."""       │
│                                                │
│     def __init__(self, db_session):           │
│         self.db = db_session                   │
│         ...                                    │
├────────────────────────────────────────────────┤
│ CONTEXT: AgentFactory class (890 lines)        │
│ Methods: activate_agent, create_agent, ...     │
│ Imports: 15 dependencies                       │
└────────────────────────────────────────────────┘

User can: Ask follow-up questions with code in context,
          jump to specific methods, view call graph
```

### Flow 4: "Review today's emails"

```
User: "Show me my emails from today"

1. Tool: composio_execute (GMAIL_LIST_EMAILS)
2. Results displayed as email cards

Canvas Updates:
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ ✉️ From: John   │ │ ✉️ From: Sarah  │ │ ✉️ From: GitHub │
│ Re: Project     │ │ Meeting notes   │ │ PR merged       │
│ 10:30 AM        │ │ 11:45 AM        │ │ 2:00 PM         │
│                 │ │                 │ │                 │
│ [Reply] [Arch.] │ │ [Reply] [Arch.] │ │ [View PR]       │
└─────────────────┘ └─────────────────┘ └─────────────────┘

User can: Click to expand, reply inline, archive, forward
```

### Flow 5: "Create an image"

```
User: "Create an image of a futuristic city at sunset"

1. Tool: image_generation (via assigned image agent)
2. Image appears on canvas

Canvas Updates:
┌────────────────────────────────────────┐
│ 🖼️ Generated Image                    │
│                                        │
│  [   Futuristic City Image   ]         │
│                                        │
│ Prompt: "futuristic city at sunset"    │
│ Model: DALL-E 3 • 1024x1024            │
│                                        │
│ [Download] [Regenerate] [Edit Prompt]  │
└────────────────────────────────────────┘
```

---

## State Management

### Proposed: Zustand Store

```typescript
interface WorkspaceStore {
  // Artifacts
  artifacts: Map<string, Artifact>
  activeArtifactId: string | null
  artifactPositions: Map<string, Position>

  // Layout
  layout: 'grid' | 'freeform' | 'split' | 'focus'
  chatPanelWidth: number
  isChatCollapsed: boolean

  // Context
  artifactsInContext: string[]  // IDs of artifacts mentioned in chat

  // Actions
  addArtifact: (artifact: Artifact) => void
  updateArtifact: (id: string, updates: Partial<Artifact>) => void
  removeArtifact: (id: string) => void
  setActiveArtifact: (id: string | null) => void
  setLayout: (layout: LayoutType) => void
  addToContext: (artifactId: string) => void
}
```

### Persistence

```typescript
// Workspace saved to backend
interface WorkspaceSession {
  id: string
  name: string
  artifacts: Artifact[]
  layout: LayoutConfig
  chatHistory: ChatMessage[]
  createdAt: Date
  updatedAt: Date
}

// API endpoints
POST /api/workspace - Create workspace
GET /api/workspace/:id - Load workspace
PUT /api/workspace/:id - Save workspace
DELETE /api/workspace/:id - Delete workspace
```

---

## Component Breakdown

### New Components to Create

| Component | Priority | Complexity | Description |
|-----------|----------|------------|-------------|
| `Canvas.tsx` | P0 | High | Main workspace area |
| `ArtifactCard.tsx` | P0 | Medium | Universal artifact wrapper |
| `ArtifactTray.tsx` | P0 | Low | Bottom artifact list |
| `CanvasGrid.tsx` | P0 | Medium | Grid layout manager |
| `CanvasFreeform.tsx` | P1 | High | Drag-and-drop layout |
| `ContextPanel.tsx` | P1 | Medium | Right sidebar inspector |
| `EmailArtifact.tsx` | P1 | Medium | Email-specific renderer |
| `TerminalArtifact.tsx` | P2 | Medium | Command output renderer |
| `WorkflowArtifact.tsx` | P2 | High | Workflow status/editor |
| `MemoryArtifact.tsx` | P2 | Medium | Memory inspection |
| `ImageEditor.tsx` | P3 | High | Image editing capabilities |
| `CodeEditor.tsx` | P3 | High | Monaco-based code editing |

### Modified Components

| Component | Changes |
|-----------|---------|
| `chat.tsx` | Slim down, extract artifact logic |
| `message.tsx` | Remove inline artifact rendering, emit to canvas |
| `artifact-viewer.tsx` | Transform into ArtifactCard |
| `sheet-artifact.tsx` | Enhance with more actions |
| `code-artifact.tsx` | Add editing capability |

---

## Technical Considerations

### Libraries to Add

```json
{
  "dependencies": {
    "@dnd-kit/core": "^6.0.0",      // Drag and drop
    "@dnd-kit/sortable": "^7.0.0",  // Sortable lists
    "zustand": "^4.4.0",            // State management
    "@monaco-editor/react": "^4.6", // Code editing
    "react-resizable-panels": "^1.0", // Panel resizing
    "react-virtuoso": "^4.6.0",     // Virtual scrolling for large lists
    "html2canvas": "^1.4.0"         // Export screenshots
  }
}
```

### Performance Considerations

1. **Virtualization**: For large artifact lists
2. **Lazy Loading**: Load artifact content on demand
3. **Memoization**: Prevent unnecessary re-renders
4. **Web Workers**: Offload heavy computations (code parsing, data processing)
5. **Intersection Observer**: Only render visible artifacts

### Responsive Design

```
Desktop (>1200px): Full layout with chat + canvas
Tablet (768-1200px): Collapsible chat, smaller canvas
Mobile (<768px): Tab-based switching between chat and canvas
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Create `Canvas.tsx` with grid layout
- [ ] Create `ArtifactCard.tsx` wrapper
- [ ] Create `ArtifactTray.tsx`
- [ ] Modify `chat.tsx` to emit artifacts to canvas
- [ ] Set up Zustand store
- [ ] Basic artifact lifecycle (add, remove, select)

### Phase 2: Artifact Types (Week 3-4)
- [ ] Enhance existing: code, text, sheet, image
- [ ] Add: email, terminal, file
- [ ] Add artifact actions (export, edit, refresh)
- [ ] Implement artifact references in chat (`@artifact-1`)

### Phase 3: Advanced Layout (Week 5-6)
- [ ] Freeform drag-and-drop layout
- [ ] Resizable panels
- [ ] Split view for comparisons
- [ ] Artifact grouping/folders

### Phase 4: Persistence & Polish (Week 7-8)
- [ ] Workspace save/load API
- [ ] Workspace templates
- [ ] Keyboard shortcuts
- [ ] Mobile responsive
- [ ] Performance optimization

---

## User Experience Principles

1. **Chat is the Controller**: Natural language drives workspace changes
2. **Artifacts are First-Class**: Persist beyond the message that created them
3. **Context is Key**: System knows which artifacts are relevant to current conversation
4. **Non-Destructive**: Easy undo, versioning for important artifacts
5. **Progressive Disclosure**: Simple by default, powerful when needed

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Artifact creation time | < 2s from tool completion |
| Layout switch time | < 100ms |
| Time to first meaningful artifact | < 5s from user prompt |
| Workspace load time | < 1s for 20 artifacts |
| User satisfaction | > 4.5/5 rating |

---

## Open Questions

1. **Collaboration**: Should multiple users be able to view/edit the same workspace?
2. **Versioning**: Should artifacts have version history?
3. **Templates**: Should we offer pre-built workspace templates (Data Analysis, Code Review, Research)?
4. **Sharing**: Can users share individual artifacts or whole workspaces?
5. **Offline**: Should workspace work offline with sync?

---

## Appendix: Inspiration Sources

- **Claude Code**: Chat + file panel design
- **Cursor**: AI-integrated code editing
- **Notion AI**: Block-based document creation
- **Figma**: Freeform canvas with collaboration
- **VS Code**: Panel-based workspace with extensions
- **Jupyter**: Cell-based execution with outputs
- **Miro**: Infinite canvas with cards
- **Linear**: Clean task/artifact management

---

*Document Version: 1.0*
*Created: 2026-01-26*
*Author: Claude (AI Assistant)*
