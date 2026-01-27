# PRD-38.2: Extended Widgets

**Version:** 1.0
**Status:** 🟡 Planned
**Priority:** HIGH
**Author:** Automatos AI Platform Team
**Last Updated:** 2026-01-27
**Dependencies:** PRD-38.1 (Widget-Ready Workspace)
**Timeline:** Weeks 4-5

---

## Executive Summary

Building on the widget foundation from Phase 1, this phase adds **5 new widget types** to support the full range of Automatos tools and capabilities:

| Widget | Source Tools | Key Capability |
|--------|-------------|----------------|
| **EmailWidget** | Composio Gmail/Outlook | View, compose, reply to emails |
| **TerminalWidget** | Shell execution tools | Command output display |
| **WorkflowWidget** | Workflow orchestrator | Status, steps, control |
| **MemoryWidget** | Memory system | Context inspection |
| **FileWidget** | File operations | Preview, download, manage |

---

## 1) Goals & Success Metrics

### Goals

| ID | Goal | Description |
|----|------|-------------|
| G1 | **Full Tool Coverage** | All major tool categories have a dedicated widget |
| G2 | **Email Integration** | Users can manage emails directly from the workspace |
| G3 | **Workflow Visibility** | Complete transparency into workflow execution |
| G4 | **Memory Transparency** | Users can see and manage AI memory/context |
| G5 | **File Management** | File operations are visual and intuitive |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| All 5 widgets implemented | 100% | Feature completion |
| Email widget handles 100+ emails | Smooth scrolling | Performance test |
| Terminal supports ANSI colors | Working | Visual QA |
| Workflow shows real-time updates | < 1s lag | Timing test |
| Memory widget loads in < 500ms | Working | Performance test |

---

## 2) Widget Specifications

### 2.1 EmailWidget

**Purpose:** Display, compose, and manage emails from Composio integrations (Gmail, Outlook, etc.)

**Source Tools:**
- `GMAIL_SEND_EMAIL`
- `GMAIL_LIST_EMAILS`
- `GMAIL_GET_EMAIL`
- `GMAIL_REPLY_EMAIL`
- `OUTLOOK_SEND_EMAIL`
- `OUTLOOK_LIST_EMAILS`

**Data Structure:**
```typescript
interface EmailWidgetData {
  mode: 'list' | 'view' | 'compose'

  // For list mode
  emails?: EmailSummary[]
  totalCount?: number
  unreadCount?: number

  // For view mode
  email?: EmailFull

  // For compose mode
  draft?: EmailDraft
  replyTo?: EmailFull
}

interface EmailSummary {
  id: string
  from: EmailAddress
  to: EmailAddress[]
  subject: string
  snippet: string
  date: string
  isRead: boolean
  hasAttachments: boolean
  labels?: string[]
}

interface EmailFull extends EmailSummary {
  body: string
  bodyHtml?: string  // ⚠️ SECURITY: Must be sanitized before rendering (see Security section)
  cc?: EmailAddress[]
  bcc?: EmailAddress[]
  attachments?: EmailAttachment[]
  threadId?: string
}

interface EmailDraft {
  to: string[]
  cc?: string[]
  bcc?: string[]
  subject: string
  body: string
  attachments?: File[]
}
```

**Component Structure:**
```
EmailWidget/
├── index.tsx           # Main component with mode switching
├── EmailList.tsx       # List view with filtering/sorting
├── EmailViewer.tsx     # Full email display (⚠️ sanitizes bodyHtml)
├── EmailComposer.tsx   # Compose/reply form
└── EmailActions.tsx    # Reply, forward, archive buttons
```

**⚠️ Security: Email HTML Rendering (XSS Prevention)**

The `EmailWidgetData.bodyHtml` field contains raw HTML from email providers which is a significant XSS risk. The following security measures are **required** in `EmailViewer.tsx`:

```typescript
// EmailWidget/EmailViewer.tsx

import DOMPurify from 'dompurify'

// 1. HTML Sanitization with DOMPurify
const DOMPURIFY_CONFIG: DOMPurify.Config = {
  ALLOWED_TAGS: [
    'p', 'br', 'div', 'span', 'a', 'b', 'i', 'u', 'strong', 'em',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'img',  // Images handled separately via proxy
  ],
  ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class', 'style'],
  ALLOW_DATA_ATTR: false,
  FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input'],
  FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover'],
}

function sanitizeEmailHtml(html: string): string {
  return DOMPurify.sanitize(html, DOMPURIFY_CONFIG)
}

// 2. Image Proxy for External Images (blocks tracking pixels)
const IMAGE_PROXY_URL = '/api/image-proxy'

function proxyExternalImages(html: string): string {
  return html.replace(
    /src=["']((https?:)?\/\/[^"']+)["']/gi,
    (match, url) => {
      // Skip already-proxied images
      if (url.startsWith(IMAGE_PROXY_URL)) return match
      // Proxy external images
      const proxiedUrl = `${IMAGE_PROXY_URL}?url=${encodeURIComponent(url)}`
      return `src="${proxiedUrl}"`
    }
  )
}

// 3. EmailViewer Component with Security
function EmailViewer({ email }: { email: EmailFull }) {
  const sanitizedHtml = useMemo(() => {
    if (!email.bodyHtml) return null
    const sanitized = sanitizeEmailHtml(email.bodyHtml)
    return proxyExternalImages(sanitized)
  }, [email.bodyHtml])

  return (
    <div className="email-viewer">
      {sanitizedHtml ? (
        <div
          className="email-body-html"
          dangerouslySetInnerHTML={{ __html: sanitizedHtml }}
        />
      ) : (
        <pre className="email-body-text">{email.body}</pre>
      )}
    </div>
  )
}
```

**Content-Security-Policy for EmailViewer:**
```typescript
// The EmailViewer iframe/container should enforce strict CSP:
const EMAIL_VIEWER_CSP = {
  'default-src': "'none'",
  'style-src': "'self' 'unsafe-inline'",  // Allow inline styles from emails
  'img-src': "'self' data: /api/image-proxy",  // Only proxied images
  'font-src': "'self'",
  // NO script-src - prevents any JS execution
}

// Apply via meta tag or iframe sandbox
<iframe
  srcDoc={sanitizedHtml}
  sandbox="allow-same-origin"  // Minimal sandbox, no scripts
  csp={Object.entries(EMAIL_VIEWER_CSP).map(([k, v]) => `${k} ${v}`).join('; ')}
/>
```

**Implementation Checklist for EmailViewer.tsx:**
- [ ] Install DOMPurify: `npm install dompurify @types/dompurify`
- [ ] Implement `sanitizeEmailHtml()` with whitelist config
- [ ] Implement `/api/image-proxy` backend endpoint
- [ ] Replace all external `<img src>` with proxied URLs
- [ ] Add CSP headers to email viewing context
- [ ] Test with malicious email samples (XSS payloads)
- [ ] Verify tracking pixels are blocked

**Features:**
- [x] List view with search and filter
- [x] Read/unread indicators
- [x] Full email viewer with HTML support (**must sanitize**)
- [x] Reply/forward actions
- [x] Compose new email
- [x] Attachment preview
- [ ] Thread view (Phase 3+)
- [ ] Label management (Phase 3+)

**UI Mockup:**
```
┌─────────────────────────────────────────────────────────────┐
│ ✉️ Gmail - Inbox                              [↻] [📝] [×]  │
├─────────────────────────────────────────────────────────────┤
│ [🔍 Search emails...]                                       │
├─────────────────────────────────────────────────────────────┤
│ ● john@example.com              10:30 AM                    │
│   Re: Project Update                                        │
│   Here are the latest changes...                            │
├─────────────────────────────────────────────────────────────┤
│   sarah@example.com             Yesterday                   │
│   Meeting Notes                                             │
│   Attached are the notes from...                            │
├─────────────────────────────────────────────────────────────┤
│   github@noreply.com            Jan 25                      │
│   [automatos-ai] PR merged                                  │
│   Your pull request has been...                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.2 TerminalWidget

**Purpose:** Display command execution output with proper formatting (ANSI colors, scrollback)

**Source Tools:**
- `execute_command`
- `run_script`
- `shell_execute`
- Any MCP tool that returns terminal output

**Data Structure:**
```typescript
interface TerminalWidgetData {
  command: string
  output: string
  exitCode?: number
  executionTime?: number
  workingDirectory?: string
  environment?: Record<string, string>
  isStreaming?: boolean
}
```

**Component Structure:**
```
TerminalWidget/
├── index.tsx           # Main component
├── TerminalOutput.tsx  # ANSI-aware output renderer
└── TerminalHeader.tsx  # Command info header
```

**Features:**
- [x] ANSI color code support
- [x] Exit code display (success/error styling)
- [x] Execution time
- [x] Copy output button
- [x] Re-run command action
- [x] Scrollback with search
- [ ] Multiple command history (Phase 3+)

**Libraries:**
- `ansi-to-html` or `ansi-to-react` for ANSI parsing

**UI Mockup:**
```
┌─────────────────────────────────────────────────────────────┐
│ 💻 Terminal                               [📋] [↻] [×]      │
├─────────────────────────────────────────────────────────────┤
│ $ npm run build                                             │
│ /home/user/automatos-ai                      ✓ Exit: 0      │
├─────────────────────────────────────────────────────────────┤
│ > automatos@1.0.0 build                                     │
│ > next build                                                │
│                                                             │
│ ✓ Creating optimized production build                       │
│ ✓ Compiled successfully                                     │
│ ✓ Linting and type checking                                 │
│ ✓ Collecting page data                                      │
│                                                             │
│ Route (app)                    Size    First Load JS        │
│ ┌ ○ /                         5.2 kB        89.2 kB         │
│ ├ ○ /chat                     12.1 kB       96.1 kB         │
│ └ ○ /api/chat                 0 B           0 B             │
├─────────────────────────────────────────────────────────────┤
│ Completed in 45.2s                                          │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.3 WorkflowWidget

**Purpose:** Display workflow execution status, steps, and provide control actions

**Source Tools:**
- `run_workflow`
- `get_workflow_status`
- `pause_workflow`
- `resume_workflow`
- `cancel_workflow`

**Data Structure:**
```typescript
interface WorkflowWidgetData {
  workflowId: string
  workflowName: string
  status: WorkflowStatus
  steps: WorkflowStep[]
  startedAt?: string
  completedAt?: string
  error?: string
  result?: any
  variables?: Record<string, any>
}

type WorkflowStatus =
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'

interface WorkflowStep {
  id: string
  name: string
  type: 'action' | 'condition' | 'loop' | 'parallel'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  startedAt?: string
  completedAt?: string
  duration?: number
  result?: any
  error?: string
  children?: WorkflowStep[]  // For nested/parallel steps
}
```

**Component Structure:**
```
WorkflowWidget/
├── index.tsx              # Main component
├── WorkflowStatus.tsx     # Status badge and controls
├── WorkflowSteps.tsx      # Step list/tree
├── WorkflowTimeline.tsx   # Visual timeline
└── StepDetail.tsx         # Expanded step view
```

**Features:**
- [x] Overall status with color coding
- [x] Step-by-step progress
- [x] Expandable step details
- [x] Pause/resume/cancel controls
- [x] Duration tracking
- [x] Error display
- [x] Result preview
- [ ] Visual flow diagram (Phase 3+)

**UI Mockup:**
```
┌─────────────────────────────────────────────────────────────┐
│ 🔄 Generate Report Workflow           ⏸️ ▶️ ⏹️  [×]          │
├─────────────────────────────────────────────────────────────┤
│ Status: 🟡 Running (Step 3/5)         Duration: 2m 34s      │
├─────────────────────────────────────────────────────────────┤
│ ✅ 1. Fetch Data Sources                           0.8s     │
│ ✅ 2. Query Database                               12.4s    │
│ 🔄 3. Generate Analysis        ← Currently running          │
│    └─ Processing 1,234 rows...                              │
│ ⏳ 4. Create Visualizations                                 │
│ ⏳ 5. Compile Report                                        │
├─────────────────────────────────────────────────────────────┤
│ Variables: { "reportType": "weekly", "format": "pdf" }      │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.4 MemoryWidget

**Purpose:** Display AI memory/context that was injected or stored during the conversation

**Source Tools:**
- Memory injection events
- `store_memory` tool
- `recall_memory` tool
- Memory system internals

**Data Structure:**
```typescript
interface MemoryWidgetData {
  mode: 'injected' | 'stored' | 'all'

  // Memories that were injected into this conversation
  injectedMemories?: Memory[]

  // Memories that were stored from this conversation
  storedMemories?: Memory[]

  // Search results
  searchResults?: Memory[]
  searchQuery?: string

  // Stats
  totalMemories?: number
  conversationMemories?: number
}

interface Memory {
  id: string
  type: 'fact' | 'preference' | 'context' | 'instruction'
  content: string
  source: {
    conversationId?: string
    timestamp: string
    trigger?: string
  }
  relevance?: number  // For injected memories
  metadata?: Record<string, any>
}
```

**Component Structure:**
```
MemoryWidget/
├── index.tsx           # Main component with tabs
├── MemoryList.tsx      # List of memories
├── MemoryItem.tsx      # Single memory display
├── MemorySearch.tsx    # Search interface
└── MemoryActions.tsx   # Add/delete actions
```

**Features:**
- [x] View injected memories (what AI "remembers")
- [x] View stored memories (what AI learned)
- [x] Search memories
- [x] Delete specific memories
- [x] Add explicit memory ("Remember this")
- [x] Memory type badges
- [ ] Memory graph visualization (Phase 3+)

**UI Mockup:**
```
┌─────────────────────────────────────────────────────────────┐
│ 🧠 Memory                                    [🔍] [+] [×]   │
├─────────────────────────────────────────────────────────────┤
│ [Injected] [Stored] [All]                                   │
├─────────────────────────────────────────────────────────────┤
│ 📌 fact                                         92% match   │
│ User prefers dark mode and concise responses                │
│ Learned: Jan 15, 2026                              [🗑️]     │
├─────────────────────────────────────────────────────────────┤
│ 📌 context                                      87% match   │
│ Current project: Automatos widget system                    │
│ Learned: Today                                     [🗑️]     │
├─────────────────────────────────────────────────────────────┤
│ 📌 instruction                                  85% match   │
│ Always include code examples in responses                   │
│ Learned: Jan 20, 2026                              [🗑️]     │
├─────────────────────────────────────────────────────────────┤
│ Showing 3 of 12 memories                                    │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.5 FileWidget

**Purpose:** Display file information, preview content, and provide file operations

**Source Tools:**
- `read_file`
- `write_file`
- `list_files`
- `delete_file`
- `move_file`
- `copy_file`
- File upload results

**Data Structure:**
```typescript
interface FileWidgetData {
  mode: 'single' | 'list' | 'preview'

  // Single file
  file?: FileInfo

  // File list
  files?: FileInfo[]
  currentPath?: string

  // Preview content
  previewContent?: string
  previewType?: 'text' | 'image' | 'pdf' | 'code' | 'binary'
}

interface FileInfo {
  name: string
  path: string
  type: 'file' | 'directory'
  size: number
  mimeType?: string
  createdAt?: string
  modifiedAt?: string
  permissions?: string
}
```

**Component Structure:**
```
FileWidget/
├── index.tsx           # Main component with mode switching
├── FileList.tsx        # Directory listing
├── FilePreview.tsx     # Content preview (text, image, etc.)
├── FileInfo.tsx        # File metadata display
└── FileActions.tsx     # Download, delete, rename actions
```

**Features:**
- [x] File metadata display
- [x] Text file preview
- [x] Image preview
- [x] Download action
- [x] Copy path action
- [x] Directory listing (for list_files results)
- [x] File type icons
- [ ] Inline editing (Phase 3+)
- [ ] File upload (Phase 3+)

**UI Mockup:**
```
┌─────────────────────────────────────────────────────────────┐
│ 📁 config.json                           [📋] [⬇️] [×]      │
├─────────────────────────────────────────────────────────────┤
│ Path: /home/user/automatos-ai/config.json                   │
│ Size: 2.4 KB  |  Modified: Jan 27, 2026 10:30 AM           │
├─────────────────────────────────────────────────────────────┤
│ {                                                           │
│   "version": "1.0.0",                                       │
│   "database": {                                             │
│     "host": "localhost",                                    │
│     "port": 5432,                                           │
│     "name": "automatos"                                     │
│   },                                                        │
│   "features": {                                             │
│     "widgets": true,                                        │
│     "memory": true                                          │
│   }                                                         │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3) Implementation Plan

### Week 4: EmailWidget + TerminalWidget

| Day | Task |
|-----|------|
| 1-2 | EmailWidget - List view and email viewer |
| 3 | EmailWidget - Compose and reply |
| 4 | TerminalWidget - Output with ANSI support |
| 5 | Integration testing |

### Week 5: WorkflowWidget + MemoryWidget + FileWidget

| Day | Task |
|-----|------|
| 1-2 | WorkflowWidget - Status and steps |
| 3 | MemoryWidget - List and search |
| 4 | FileWidget - Preview and actions |
| 5 | Integration testing and polish |

---

## 4) Files to Create

```
frontend/components/widgets/
├── EmailWidget/
│   ├── index.tsx
│   ├── EmailList.tsx
│   ├── EmailViewer.tsx
│   ├── EmailComposer.tsx
│   └── EmailActions.tsx
│
├── TerminalWidget/
│   ├── index.tsx
│   ├── TerminalOutput.tsx
│   └── TerminalHeader.tsx
│
├── WorkflowWidget/
│   ├── index.tsx
│   ├── WorkflowStatus.tsx
│   ├── WorkflowSteps.tsx
│   └── StepDetail.tsx
│
├── MemoryWidget/
│   ├── index.tsx
│   ├── MemoryList.tsx
│   ├── MemoryItem.tsx
│   └── MemorySearch.tsx
│
└── FileWidget/
    ├── index.tsx
    ├── FileList.tsx
    ├── FilePreview.tsx
    └── FileInfo.tsx
```

---

## 5) Tool Router Updates

Add new mappings to `router.ts`:

```typescript
// Add to TOOL_WIDGET_MAP
const TOOL_WIDGET_MAP: Record<string, WidgetType> = {
  // ... existing mappings ...

  // Email Tools
  'GMAIL_SEND_EMAIL': 'email',
  'GMAIL_LIST_EMAILS': 'email',
  'GMAIL_GET_EMAIL': 'email',
  'GMAIL_REPLY_EMAIL': 'email',
  'OUTLOOK_SEND_EMAIL': 'email',
  'OUTLOOK_LIST_EMAILS': 'email',

  // Terminal Tools
  'execute_command': 'terminal',
  'run_script': 'terminal',
  'shell_execute': 'terminal',

  // Workflow Tools
  'run_workflow': 'workflow',
  'get_workflow_status': 'workflow',
  'pause_workflow': 'workflow',
  'resume_workflow': 'workflow',

  // Memory Tools
  'store_memory': 'memory',
  'recall_memory': 'memory',
  'search_memory': 'memory',
  'delete_memory': 'memory',

  // File Tools
  'read_file': 'file',
  'write_file': 'file',
  'list_files': 'file',
  'delete_file': 'file',
}
```

---

## 6) Dependencies

```bash
# ANSI color support for TerminalWidget
npm install ansi-to-react

# Optional: Rich text editor for email compose
npm install @tiptap/react @tiptap/starter-kit
```

---

## 7) Testing Checklist

### EmailWidget
- [ ] List emails with pagination
- [ ] View full email with HTML rendering
- [ ] Compose new email
- [ ] Reply to email
- [ ] Handle attachments display

### TerminalWidget
- [ ] Display plain text output
- [ ] Render ANSI colors correctly
- [ ] Show exit code with appropriate styling
- [ ] Copy output to clipboard
- [ ] Re-run command action

### WorkflowWidget
- [ ] Display workflow status correctly
- [ ] Show step progress
- [ ] Pause/resume/cancel actions work
- [ ] Real-time status updates
- [ ] Error display

### MemoryWidget
- [ ] List injected memories
- [ ] List stored memories
- [ ] Search memories
- [ ] Delete memory
- [ ] Add explicit memory

### FileWidget
- [ ] Display file metadata
- [ ] Preview text files
- [ ] Preview images
- [ ] Download file
- [ ] List directory contents

---

## 8) Backend Requirements

### New Streaming Events

```typescript
// Memory injection event
interface MemoryInjectedEvent {
  type: 'memory-injected'
  data: {
    memories: Memory[]
    totalMatched: number
  }
}

// Memory stored event
interface MemoryStoredEvent {
  type: 'memory-stored'
  data: {
    memory: Memory
    reason: string
  }
}

// Workflow status update event
interface WorkflowUpdateEvent {
  type: 'workflow-update'
  data: {
    workflowId: string
    status: WorkflowStatus
    currentStep?: string
    progress?: number
  }
}
```

### API Endpoints Needed

```
# Email
GET  /api/emails              # List emails (via Composio)
GET  /api/emails/:id          # Get single email
POST /api/emails              # Send email
POST /api/emails/:id/reply    # Reply to email

# Memory
GET  /api/memory              # List memories
GET  /api/memory/search       # Search memories
POST /api/memory              # Store memory
DELETE /api/memory/:id        # Delete memory

# Workflow
GET  /api/workflows/:id       # Get workflow status
POST /api/workflows/:id/pause # Pause workflow
POST /api/workflows/:id/resume # Resume workflow
POST /api/workflows/:id/cancel # Cancel workflow
```

---

## 9) Success Criteria

Phase 2 is complete when:

1. [ ] All 5 widget types are implemented and registered
2. [ ] Tool results correctly route to new widgets
3. [ ] Email widget can display and compose emails
4. [ ] Terminal widget correctly renders ANSI output
5. [ ] Workflow widget shows real-time status updates
6. [ ] Memory widget displays and manages memories
7. [ ] File widget previews and downloads files
8. [ ] All widgets integrate with workspace store
9. [ ] Performance remains smooth with many widgets

---

## 10) References

- [ansi-to-react](https://github.com/nteract/ansi-to-react) - ANSI parsing
- [Tiptap](https://tiptap.dev/) - Rich text editor
- PRD-38.1: Widget-Ready Workspace
- PRD-20: MCP Integration (Composio tools)
- PRD-10: Workflow Orchestration

---

*Document Version: 1.0*
*Created: 2026-01-27*
*Estimated Implementation: 2 weeks*
