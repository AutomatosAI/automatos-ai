# PRD-66: Automatos Code Canvas

**Version:** 1.0
**Status:** Draft
**Priority:** P0 — Critical (Differentiating Feature)
**Created:** 2026-02-25
**Author:** Automatos AI Platform Team
**Dependencies:** PRD-56 (Physical Workspaces), PRD-38.1 (Widget Architecture), PRD-38.2 (Extended Widgets)
**Estimated Effort:** Phase 1: 30h | Phase 2: 20h | Phase 3: 40h

---

## Executive Summary

Add a **Claude Code-style coding canvas** to Automatos Chat — a compound widget where agents clone repos, edit files, run tests, and show diffs in real-time. The user watches their agent work in a Monaco editor + file tree + terminal panel, then accepts or rejects changes.

PRD-56 Phase 2 solved the hard problem (physical workspaces with sandboxed execution). This PRD is the **frontend experience** that makes that infrastructure visible and interactive.

### Before & After

```
CURRENT STATE                              TARGET STATE
────────────────────────────               ────────────────────────────────
┌────────┐ ┌──────────────────┐            ┌────────┐ ┌────────────────────────┐
│        │ │                  │            │        │ │  File Tree │ Monaco    │
│  Chat  │ │  Canvas (docs,   │            │  Chat  │ │  src/      │ ┌──────┐  │
│  Panel │ │  tables, code    │            │  Panel │ │  ├─ auth.ts│ │- old │  │
│        │ │  as READ-ONLY    │            │        │ │  ├─ test.ts│ │+ new │  │
│        │ │  snippets)       │            │        │ │  └─ ...    │ └──────┘  │
│        │ │                  │            │        │ │            │           │
│        │ │                  │            │        │ │  ┌─ Terminal ────────┐ │
│        │ │                  │            │        │ │  │ $ pytest -v      │ │
│        │ │                  │            │        │ │  │ 4 passed, 0 fail │ │
│        │ │                  │            │        │ │  └──────────────────┘ │
└────────┘ └──────────────────┘            └────────┘ └────────────────────────┘

- Read-only code snippets                 - Live Monaco editor with diff view
- Static terminal output                  - Streaming terminal (xterm.js)
- No file navigation                      - Full workspace file tree
- No accept/reject                        - Accept/Reject change controls
```

---

## 1) Problem Statement

Automatos agents can now clone repos, run tests, and fix bugs in physical workspaces (PRD-56). But users have **zero visibility** into what's happening:

1. **File edits are invisible** — `file_write` tool results show as a chat message, not a diff
2. **Terminal output is static** — command results arrive as a blob after execution, no streaming
3. **No file browsing** — users can't see the workspace filesystem their agent is operating on
4. **No change review** — no way to accept/reject edits before the agent commits

Competitors (Claude Code, Cursor, Windsurf) all have integrated coding canvases. Without one, Automatos agents feel like chatbots that happen to have tools, not coding assistants.

---

## 2) Goals & Success Metrics

### Goals

| ID | Goal | Description |
|----|------|-------------|
| G1 | **Code Visibility** | Agent file operations displayed in Monaco editor in real-time |
| G2 | **Live Terminal** | Streaming command output via xterm.js during execution |
| G3 | **File Navigation** | Workspace file tree browsable from the canvas |
| G4 | **Change Review** | Accept/reject diff view for agent edits (like Claude Code) |
| G5 | **Widget Integration** | Registers as a standard widget — no special canvas plumbing |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Monaco editor load time | < 500ms | Performance profiling |
| Terminal streaming latency | < 200ms end-to-end | Redis pub/sub to SSE to xterm |
| File tree renders 500+ files | 60fps | Chrome DevTools |
| User can accept/reject edits | Working | E2E test |
| Compound widget renders in canvas | Working | Existing canvas tabs |

---

## 3) What Already Exists

### Infrastructure (PRD-56 — Done)

| Component | File | Status |
|-----------|------|--------|
| Physical workspaces | `services/workspace-worker/workspace_manager.py` | Shipped |
| Sandboxed shell execution | `services/workspace-worker/executor.py` | Shipped |
| File read/write in workspace | `services/workspace-worker/executor.py` | Shipped |
| Git clone/pull | `services/workspace-worker/executor.py` | Shipped |
| Task queue (Redis) | `orchestrator/core/task_runner/queued.py` | Shipped |
| Task events pub/sub | `QueuedTaskRunner.stream_updates()` | Shipped |
| Task REST API | `orchestrator/api/tasks.py` | Shipped |
| SSE event stream | `GET /api/tasks/{id}/events` | Shipped |

### Frontend Widget System (PRD-38.1/38.2 — Done)

| Component | File | Status |
|-----------|------|--------|
| Widget registry | `frontend/components/widgets/registry.ts` | Shipped |
| Tool to widget router | `frontend/components/widgets/router.ts` | Shipped |
| Canvas with tabs | `frontend/components/workspace/Canvas.tsx` | Shipped |
| Zustand workspace store | `frontend/stores/workspace-store.ts` | Shipped |
| CodeWidget (Prism, read-only) | `frontend/components/widgets/CodeWidget/` | Shipped |
| TerminalWidget (ANSI, static) | `frontend/components/widgets/TerminalWidget/` | Shipped |
| FileWidget (list/preview) | `frontend/components/widgets/FileWidget/` | Shipped |
| WidgetBase chrome | `frontend/components/widgets/WidgetBase.tsx` | Shipped |

### The Gap

| Missing Piece | Category | Solution |
|---------------|----------|----------|
| Monaco editor | npm package | `@monaco-editor/react` |
| Monaco diff editor | Built into Monaco | `DiffEditor` component |
| Interactive terminal | npm package | `xterm` + `@xterm/addon-fit` |
| Workspace file tree API | Backend endpoint | `GET /api/workspaces/{id}/files` |
| Real-time file change events | SSE bridge | Worker to Redis pub/sub to SSE to widget |
| Accept/reject flow | Frontend UX | New component in CodingCanvas |

---

## 4) Technical Architecture

### 4.1 New Widget: `coding_canvas`

A **compound widget** that registers as a single type in the existing widget system. It contains three sub-panels: file tree, code editor, and terminal.

```
┌──────────────────────────────────────────────────────────────────┐
│  Coding Canvas                             [Repo v] [gear] [x]  │
├──────────────┬───────────────────────────────────────────────────┤
│              │  auth.ts  x  │  auth.test.ts  x  │               │
│  File Tree   ├──────────────────────────────────────────────────┤
│              │  1  import { verify } from 'jsonwebtoken'        │
│  src/        │  2  import { db } from './db'                    │
│  ├─ auth.ts  │  3                                               │
│  ├─ db.ts    │  4  export async function authenticate(          │
│  └─ test/    │  5-   token: string                              │
│     └─ ...   │  5+   token: string, opts?: AuthOpts             │
│              │  6  ) {                                           │
│  package.json│  7    ...                                        │
│  tsconfig    │                                                  │
│              ├──────────────────────────────────────────────────┤
│              │ [Accept] [Reject] [Toggle Diff]                  │
├──────────────┴──────────────────────────────────────────────────┤
│  $ pytest tests/ -v                                              │
│  tests/test_auth.py::test_login PASSED                          │
│  tests/test_auth.py::test_refresh PASSED                        │
│  4 passed in 0.23s                                              │
│                                                                  │
│  agent@workspace:~/repos/automatos-ai $                          │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Hierarchy

```
CodingCanvasWidget/
├── index.tsx                 <- Compound widget entry, registers in widget registry
├── CodingCanvasLayout.tsx    <- 3-panel resizable layout (react-resizable-panels)
├── FileExplorer/
│   ├── index.tsx             <- Tree view with icons, open/collapse
│   ├── FileTreeNode.tsx      <- Recursive tree node (file/dir)
│   └── useWorkspaceFiles.ts  <- Hook: GET /api/workspaces/{id}/files/{path}
├── CodeEditor/
│   ├── index.tsx             <- Monaco wrapper with tab bar
│   ├── EditorTabs.tsx        <- Open file tabs
│   ├── DiffView.tsx          <- Monaco DiffEditor for accept/reject
│   └── ChangeControls.tsx    <- Accept / Reject / Toggle Diff buttons
├── Terminal/
│   ├── index.tsx             <- xterm.js wrapper
│   └── useTaskStream.ts      <- Hook: SSE /api/tasks/{id}/events for stdout streaming
└── types.ts                  <- CodingCanvasWidgetData interface
```

### 4.3 Data Flow

```
Agent runs task in workspace
      │
      ├── file_read("src/auth.ts")
      │     │
      │     v
      │   Redis pub/sub event: { type: "file_read", path: "src/auth.ts", content: "..." }
      │     │
      │     v
      │   SSE /api/tasks/{id}/events -> frontend
      │     │
      │     v
      │   CodingCanvas: Monaco.setValue(content), FileTree highlights file
      │
      ├── file_write("src/auth.ts", newContent)
      │     │
      │     v
      │   Redis pub/sub event: { type: "file_write", path: "src/auth.ts", old: "...", new: "..." }
      │     │
      │     v
      │   CodingCanvas: DiffEditor shows old vs new, Accept/Reject buttons appear
      │
      ├── shell("pytest tests/ -v")
      │     │
      │     v
      │   Redis pub/sub: { type: "stdout_chunk", chunk: "tests/test_auth.py PASSED\n" }
      │     │                                                              (streaming)
      │     v
      │   CodingCanvas: xterm.write(chunk) in real-time
      │
      └── git_clone("https://github.com/org/repo.git")
            │
            v
          Redis pub/sub: { type: "git_operation", repo: "org/repo", status: "complete" }
            │
            v
          CodingCanvas: FileTree refreshes with cloned repo structure
```

### 4.4 Widget Type Registration

```typescript
// frontend/components/widgets/types.ts — ADD:

export type WidgetType =
  // ... existing types ...
  | 'coding_canvas'   // PRD-66: Code Canvas

export interface CodingCanvasWidgetData {
  // Workspace context
  workspaceId: string
  taskId?: string                  // Active task being watched

  // Repository info
  repo?: {
    name: string
    url: string
    branch: string
    clonedAt?: string
  }

  // File tree state
  fileTree?: FileTreeNode[]
  expandedPaths?: string[]

  // Open files in editor
  openFiles: OpenFile[]
  activeFilePath?: string

  // Terminal state
  terminalHistory: TerminalEntry[]
  isTerminalStreaming?: boolean

  // Change tracking
  pendingChanges: FileChange[]
  changeReviewMode?: boolean
}

interface FileTreeNode {
  name: string
  path: string                    // Relative to workspace root
  type: 'file' | 'directory'
  children?: FileTreeNode[]
  size?: number
  modified?: boolean              // Has pending changes
  language?: string               // Inferred from extension
}

interface OpenFile {
  path: string
  content: string
  language: string
  isDirty: boolean                // Has unsaved changes
  originalContent?: string        // For diff view
}

interface FileChange {
  path: string
  type: 'create' | 'modify' | 'delete'
  oldContent?: string
  newContent?: string
  accepted?: boolean              // null = pending review
}

interface TerminalEntry {
  command: string
  output: string
  exitCode?: number
  timestamp: string
  isStreaming?: boolean
}
```

### 4.5 Tool to Widget Routing

```typescript
// frontend/components/widgets/router.ts — ADD to TOOL_WIDGET_MAP:

// PRD-66: Coding Canvas tools
// When TASK_RUNNER_BACKEND=queued, workspace tools route to coding canvas
workspace_file_read: 'coding_canvas',
workspace_file_write: 'coding_canvas',
workspace_bash: 'coding_canvas',
workspace_git_clone: 'coding_canvas',
workspace_git_pull: 'coding_canvas',
workspace_git_status: 'coding_canvas',
workspace_git_commit: 'coding_canvas',
workspace_git_push: 'coding_canvas',
```

**Routing logic:** When a task is running in queued mode, tool results include a `task_id` and `workspace_id`. The router checks if a `coding_canvas` widget already exists for that `taskId` — if so, **updates** it instead of creating a new widget. This makes all tool results for a single task converge into one canvas.

```typescript
// router.ts addition:
function handleCodingCanvasResult(
  toolName: string,
  result: any,
  taskId: string,
  existingWidgets: Record<string, Widget>
): string | null {
  // Find existing canvas for this task
  const existing = Object.values(existingWidgets).find(
    w => w.type === 'coding_canvas' && w.data.taskId === taskId
  )

  if (existing) {
    // UPDATE existing canvas widget instead of creating new one
    return existing.id
  }

  // First tool result for this task — create new canvas
  return null  // triggers normal addWidget()
}
```

---

## 5) Backend API Additions

### 5.1 Workspace File Browser

```
GET /api/workspaces/{workspace_id}/files?path={relative_path}
```

Returns directory listing for the workspace filesystem. Used by the FileExplorer component.

**Response:**
```json
{
  "path": "repos/automatos-ai/src",
  "entries": [
    { "name": "auth.ts", "type": "file", "size": 2340, "modified_at": "2026-02-25T10:30:00Z" },
    { "name": "test", "type": "directory", "children_count": 5 },
    { "name": "utils.ts", "type": "file", "size": 890, "modified_at": "2026-02-25T09:15:00Z" }
  ]
}
```

**Implementation:** Reads from `/workspaces/{workspace_id}/{path}` via the workspace-worker (or directly if API has volume access). Restricted to workspace root — path traversal blocked by `WorkspaceManager.resolve_safe_path()`.

### 5.2 File Content Read

```
GET /api/workspaces/{workspace_id}/files/{path}?content=true
```

Returns file content for display in Monaco. Size-limited to 1MB.

**Response:**
```json
{
  "path": "repos/automatos-ai/src/auth.ts",
  "content": "import { verify } from 'jsonwebtoken'...",
  "size": 2340,
  "language": "typescript",
  "encoding": "utf-8"
}
```

### 5.3 File Write (User Edits)

```
PUT /api/workspaces/{workspace_id}/files/{path}
```

Allows users to edit files directly in Monaco and save back to workspace. Phase 3 only.

**Request:**
```json
{
  "content": "updated file content...",
  "create_if_missing": false
}
```

### 5.4 Enhanced Task Events

The existing `GET /api/tasks/{id}/events` SSE endpoint already streams `TaskEvent` objects. Add new event types for coding canvas:

```python
class TaskEventType(str, Enum):
    # ... existing ...
    FILE_READ = "file_read"         # Agent read a file
    FILE_WRITE = "file_write"       # Agent wrote/modified a file
    FILE_DELETE = "file_delete"     # Agent deleted a file
    STDOUT_CHUNK = "stdout_chunk"   # Streaming terminal output
    GIT_OPERATION = "git_operation" # Clone, pull, commit, push
```

**Event payloads:**
```json
// file_write event
{
  "event_type": "file_write",
  "task_id": "abc-123",
  "data": {
    "path": "src/auth.ts",
    "old_content": "const token = ...",
    "new_content": "const token = verify(...)",
    "diff_summary": "+1 -1 lines"
  }
}

// stdout_chunk event (streaming)
{
  "event_type": "stdout_chunk",
  "task_id": "abc-123",
  "data": {
    "chunk": "tests/test_auth.py::test_login PASSED\n",
    "stream": "stdout"
  }
}
```

### 5.5 Worker-Side Changes

The `WorkspaceToolExecutor` in `services/workspace-worker/executor.py` already handles file_read, file_write, and shell commands. It needs to **emit events** via Redis pub/sub as it runs:

```python
# executor.py additions:

async def execute_step(self, step, task_id):
    """Execute a step and emit real-time events."""

    if step["action"] == "file_read":
        result = await self._read_file(step["path"])
        await self._emit_event(task_id, "file_read", {
            "path": step["path"],
            "content": result["content"],
            "language": self._detect_language(step["path"]),
        })
        return result

    elif step["action"] == "file_write":
        old_content = await self._try_read_existing(step["path"])
        result = await self._write_file(step["path"], step["content"])
        await self._emit_event(task_id, "file_write", {
            "path": step["path"],
            "old_content": old_content,
            "new_content": step["content"],
        })
        return result

    elif step["action"] == "shell":
        # Stream stdout chunks as they arrive
        async for chunk in self._execute_streaming(step["command"]):
            await self._emit_event(task_id, "stdout_chunk", {
                "chunk": chunk,
                "stream": "stdout",
            })
        return result
```

---

## 6) NPM Dependencies

| Package | Version | Purpose | Bundle Size |
|---------|---------|---------|-------------|
| `@monaco-editor/react` | ^4.6 | Code editor + diff editor | ~2.5MB (lazy loaded) |
| `xterm` | ^5.5 | Terminal emulator | ~350KB |
| `@xterm/addon-fit` | ^0.10 | Auto-resize terminal to container | ~5KB |
| `@xterm/addon-web-links` | ^0.11 | Clickable URLs in terminal | ~10KB |

**Note:** Monaco is large but loads lazily via web workers. It only loads when the coding canvas widget is opened, not on page load. The existing Prism.js-based CodeWidget remains for simple code snippets.

---

## 7) Phased Implementation

### Phase 1: Code Viewer Widget (Week 1)

**Goal:** Replace Prism.js CodeWidget with Monaco for workspace files. Read-only.

**User Stories:**
- US-01: As a user, I can see my workspace file tree in the coding canvas
- US-02: As a user, I can click a file to view it in Monaco with syntax highlighting
- US-03: As a user, I can open multiple files in editor tabs
- US-04: When my agent reads a file, it opens automatically in the editor

**Implementation:**

1. **Install dependencies**
   ```bash
   cd frontend && npm install @monaco-editor/react
   ```

2. **Create `CodingCanvasWidget/` component directory**
   - `index.tsx` — compound widget with WidgetBase chrome
   - `FileExplorer/index.tsx` — tree view using recursive `FileTreeNode`
   - `CodeEditor/index.tsx` — Monaco wrapper with `@monaco-editor/react`
   - `CodeEditor/EditorTabs.tsx` — tab bar for open files
   - `types.ts` — `CodingCanvasWidgetData` interface

3. **Add `coding_canvas` to widget type union** (`types.ts`)

4. **Register widget** in registry (`CodingCanvasWidget/index.tsx`)

5. **Add backend endpoint** `GET /api/workspaces/{id}/files` in new `orchestrator/api/workspace_files.py`

6. **Wire `file_read` tool events** — when agent reads a file, canvas opens it in editor

**Deliverables:**
- [ ] `CodingCanvasWidget/` component tree
- [ ] `GET /api/workspaces/{id}/files` endpoint
- [ ] `GET /api/workspaces/{id}/files/{path}?content=true` endpoint
- [ ] Widget registered and routing `workspace_file_read` to `coding_canvas`
- [ ] File tree renders workspace directory structure
- [ ] Monaco editor displays file content with syntax highlighting

---

### Phase 2: Interactive Terminal (Week 2)

**Goal:** Stream command output to xterm.js in real-time as the agent runs commands.

**User Stories:**
- US-05: As a user, I see command output streaming in real-time as my agent runs tests
- US-06: Terminal shows ANSI colors, scrollback, and copy support
- US-07: Terminal panel is resizable within the coding canvas

**Implementation:**

1. **Install dependencies**
   ```bash
   cd frontend && npm install xterm @xterm/addon-fit @xterm/addon-web-links
   ```

2. **Create `Terminal/index.tsx`** — xterm.js integration with fit addon

3. **Create `Terminal/useTaskStream.ts`** — hook that:
   - Connects to `GET /api/tasks/{id}/events` SSE endpoint
   - Filters for `stdout_chunk` events
   - Writes chunks to xterm instance via `terminal.write(chunk)`

4. **Modify `WorkspaceToolExecutor.execute_command()`** — emit `stdout_chunk` events via Redis as subprocess produces output (line-buffered)

5. **Modify `services/workspace-worker/main.py`** — publish events to Redis channel `workspace:task:{task_id}:events`

6. **Add `TaskEventType.STDOUT_CHUNK`** to `orchestrator/core/task_runner/models.py`

**Deliverables:**
- [ ] xterm.js terminal in coding canvas
- [ ] Streaming stdout from worker to Redis to SSE to xterm
- [ ] ANSI color rendering
- [ ] Auto-scroll + manual scroll lock
- [ ] Resizable terminal panel (drag handle)

---

### Phase 3: Full Coding Experience (Weeks 3-4)

**Goal:** Agent file writes show as diffs. Users can accept/reject. Users can edit files. Git operations visible.

**User Stories:**
- US-08: When my agent writes a file, I see a diff view (old vs new)
- US-09: I can accept or reject each file change
- US-10: I can edit files directly in Monaco and save to workspace
- US-11: Git operations (status, commit, push) are visible in the canvas
- US-12: I see a summary of all pending changes across files

**Implementation:**

1. **DiffView component** — Monaco `DiffEditor` showing original vs modified
   ```typescript
   import { DiffEditor } from '@monaco-editor/react'
   <DiffEditor original={old} modified={newContent} language={lang} />
   ```

2. **ChangeControls component** — Accept/Reject buttons per file change
   - Accept: marks change as accepted, moves to next
   - Reject: reverts content in workspace via `PUT /api/workspaces/{id}/files/{path}`
   - Toggle: switch between inline edit and diff view

3. **User editing** — Monaco editor in read-write mode
   - Save hotkey (Cmd+S) triggers `PUT /api/workspaces/{id}/files/{path}`
   - Dirty indicator on tab

4. **Git panel** — shows git status, staged files, commit message input
   - `POST /api/workspaces/{id}/git/status`
   - `POST /api/workspaces/{id}/git/commit`
   - `POST /api/workspaces/{id}/git/push`

5. **Change summary sidebar** — list of all modified files with accept-all / reject-all

**Deliverables:**
- [ ] Monaco DiffEditor for file changes
- [ ] Accept/reject per-file with ChangeControls
- [ ] User file editing with save-to-workspace
- [ ] Git status/commit/push UI
- [ ] Change summary panel
- [ ] `PUT /api/workspaces/{id}/files/{path}` endpoint
- [ ] Git operation endpoints

---

## 8) Security Considerations

### File Access

- **Path traversal**: All file access goes through `WorkspaceManager.resolve_safe_path()` — already blocks `../`, symlink escape, absolute paths
- **File size limits**: Monaco content endpoint capped at 1MB. Binary files return metadata only.
- **Workspace isolation**: API verifies `workspace_id` belongs to authenticated user's workspace

### Terminal

- **Read-only terminal**: Phase 1-2 terminal is output-only (agent commands streamed). Users cannot type commands.
- **Phase 3 interactive shell**: If added later, subject to same command whitelist as `WorkspaceToolExecutor` (50+ allowed commands, blocked patterns like `sudo`, `rm -rf /`, `docker`)
- **Output sanitization**: xterm.js handles ANSI escapes natively. No raw HTML injection possible.

### File Editing (Phase 3)

- **Write-back validation**: `PUT /api/workspaces/{id}/files/{path}` validates:
  - Path within workspace bounds
  - File size under quota
  - User owns workspace
- **No arbitrary code execution**: Saving a file doesn't run it. Only the agent (via task queue) runs commands.

---

## 9) Performance Considerations

### Monaco Lazy Loading

Monaco is ~2.5MB. Load strategy:
- **Code splitting**: `React.lazy(() => import('./CodingCanvasWidget'))`
- **Monaco web workers**: Load via CDN or self-hosted `/monaco-editor/` path
- **Only loads when widget opens**: Existing CodeWidget (Prism.js) stays for inline code snippets

```typescript
// Dynamic import pattern
const MonacoEditor = dynamic(
  () => import('@monaco-editor/react').then(mod => mod.Editor),
  { ssr: false, loading: () => <CodeSkeleton /> }
)
```

### Terminal Streaming

- **Buffering**: Collect stdout chunks for 50ms before flushing to xterm (prevents per-character renders)
- **Scrollback limit**: Cap at 10,000 lines to prevent memory issues
- **Auto-dispose**: Disconnect SSE when terminal tab is inactive

### File Tree

- **Lazy loading**: Only fetch children when directory is expanded
- **Caching**: Cache directory listings in workspace store, invalidate on file events
- **Virtual scrolling**: Use virtualized list for repos with 1000+ files

---

## 10) Migration Path

### Existing Widgets Unaffected

- `CodeWidget` (Prism.js) — stays for simple code snippets from `search_codebase`, `analyze_code`, etc.
- `TerminalWidget` (ANSI render) — stays for one-off `execute_command` results in local mode
- `FileWidget` — stays for file list/preview from `list_files`, `read_file` in local mode

### When CodingCanvas Activates

The coding canvas only appears when:
1. `TASK_RUNNER_BACKEND=queued` (physical workspace is active)
2. A task with workspace tools is running
3. The tool router detects `workspace_*` tool names

When `TASK_RUNNER_BACKEND=local`, tool results continue routing to existing individual widgets.

---

## 11) Future Extensions

| Feature | Phase | Description |
|---------|-------|-------------|
| **Collaborative editing** | 4+ | Multiple users watching same workspace (CRDT/OT) |
| **Breakpoints/debugging** | 4+ | Step-through debugging in Monaco |
| **AI autocomplete** | 4+ | Copilot-style completions within Monaco |
| **Split editor** | 3+ | Side-by-side file comparison |
| **Search across files** | 3+ | Ctrl+Shift+F workspace-wide search |
| **Minimap** | 3 | Monaco minimap toggle |
| **Branch switching** | 3+ | Git branch selector in toolbar |

---

## 12) Competitive Landscape

| Product | Code Canvas | File Tree | Diff View | Terminal | Accept/Reject |
|---------|-------------|-----------|-----------|----------|---------------|
| **Claude Code (Web)** | Monaco | Yes | Yes | Yes | Yes |
| **Cursor** | VS Code fork | Yes | Yes | Yes | Yes |
| **Windsurf** | VS Code fork | Yes | Yes | Yes | Yes |
| **Bolt.new** | Monaco | Yes | No | Partial | No |
| **v0.dev** | Monaco | Limited | No | No | No |
| **Automatos (current)** | Prism (read-only) | FileWidget | No | Static output | No |
| **Automatos (PRD-66)** | Monaco + Diff | Yes | Yes | xterm.js streaming | Yes |

---

## 13) Implementation Checklist

### Phase 1 (Code Viewer)
- [ ] `npm install @monaco-editor/react`
- [ ] Create `frontend/components/widgets/CodingCanvasWidget/` directory
- [ ] Implement `CodingCanvasWidget/index.tsx` with WidgetBase
- [ ] Implement `FileExplorer/index.tsx` with recursive tree
- [ ] Implement `CodeEditor/index.tsx` with Monaco
- [ ] Implement `CodeEditor/EditorTabs.tsx`
- [ ] Add `coding_canvas` to WidgetType union in `types.ts`
- [ ] Add `CodingCanvasWidgetData` interface to `types.ts`
- [ ] Register widget in registry
- [ ] Add `workspace_*` tool mappings to `router.ts`
- [ ] Create `orchestrator/api/workspace_files.py` with GET endpoints
- [ ] Register workspace_files router in `main.py`
- [ ] Wire `file_read` events to open files in editor

### Phase 2 (Live Terminal)
- [ ] `npm install xterm @xterm/addon-fit @xterm/addon-web-links`
- [ ] Implement `Terminal/index.tsx` with xterm.js
- [ ] Implement `Terminal/useTaskStream.ts` SSE hook
- [ ] Add `STDOUT_CHUNK` to `TaskEventType` enum
- [ ] Modify `WorkspaceToolExecutor` to emit streaming stdout events
- [ ] Modify `workspace-worker/main.py` to publish stdout to Redis pub/sub
- [ ] Add resizable panel split between editor and terminal
- [ ] Test streaming latency end-to-end

### Phase 3 (Full Coding)
- [ ] Implement `CodeEditor/DiffView.tsx` with Monaco DiffEditor
- [ ] Implement `CodeEditor/ChangeControls.tsx` (Accept/Reject)
- [ ] Add `FILE_WRITE`, `FILE_DELETE`, `GIT_OPERATION` event types
- [ ] Modify executor to emit old+new content on file_write
- [ ] Implement `PUT /api/workspaces/{id}/files/{path}` endpoint
- [ ] Enable read-write Monaco mode with Cmd+S save
- [ ] Add dirty indicator on editor tabs
- [ ] Implement change summary panel
- [ ] Add git operation endpoints (status, commit, push)
- [ ] Implement git panel in canvas toolbar
- [ ] E2E test: agent clones repo, edits file, user reviews diff, accepts, agent commits

---

## 14) Open Questions

| # | Question | Options | Decision |
|---|----------|---------|----------|
| 1 | Monaco CDN vs self-hosted? | CDN (faster first load) vs bundled (offline capable) | TBD |
| 2 | Interactive terminal in Phase 2 or Phase 3? | Phase 2 (output only) vs Phase 2 (input too) | Phase 2 = output only |
| 3 | Per-file accept or batch accept-all? | Per-file (safer) vs batch (faster) | Both — per-file default, accept-all button |
| 4 | File tree from API or from task events? | API polling vs event-driven | Hybrid: initial load via API, updates via events |
