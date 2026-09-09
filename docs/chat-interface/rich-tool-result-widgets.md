# Rich Tool Result Widgets

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/widgets/CodingCanvasWidget/CodeEditor.tsx](frontend/components/widgets/CodingCanvasWidget/CodeEditor.tsx)
- [frontend/components/widgets/CodingCanvasWidget/EditorTabs.tsx](frontend/components/widgets/CodingCanvasWidget/EditorTabs.tsx)
- [frontend/components/widgets/CodingCanvasWidget/FileExplorer.tsx](frontend/components/widgets/CodingCanvasWidget/FileExplorer.tsx)
- [frontend/components/widgets/CodingCanvasWidget/index.tsx](frontend/components/widgets/CodingCanvasWidget/index.tsx)
- [frontend/components/widgets/CodingCanvasWidget/useWorkspaceFiles.ts](frontend/components/widgets/CodingCanvasWidget/useWorkspaceFiles.ts)
- [frontend/components/widgets/FileWidget/FilePreview.tsx](frontend/components/widgets/FileWidget/FilePreview.tsx)
- [frontend/components/widgets/FileWidget/index.tsx](frontend/components/widgets/FileWidget/index.tsx)
- [frontend/components/widgets/ToolApprovalWidget/index.tsx](frontend/components/widgets/ToolApprovalWidget/index.tsx)
- [frontend/components/widgets/__tests__/tool-approval-widget.test.tsx](frontend/components/widgets/__tests__/tool-approval-widget.test.tsx)
- [frontend/components/widgets/index.ts](frontend/components/widgets/index.ts)
- [frontend/components/widgets/router.ts](frontend/components/widgets/router.ts)
- [frontend/components/widgets/types.ts](frontend/components/widgets/types.ts)
- [frontend/components/workspace/WorkspaceExplorer.tsx](frontend/components/workspace/WorkspaceExplorer.tsx)
- [frontend/components/workspace/gallery-view/deliverable-preview.tsx](frontend/components/workspace/gallery-view/deliverable-preview.tsx)
- [orchestrator/modules/tools/formatting/result_formatter.py](orchestrator/modules/tools/formatting/result_formatter.py)
- [orchestrator/tests/test_p2w2_tool_approval_card.py](orchestrator/tests/test_p2w2_tool_approval_card.py)

</details>



This page documents the Rich Tool Result Widgets subsystem (PRD-38.1, PRD-66, PRD-170, PRD-193), which bridges raw tool outputs and agent events to interactive React UI widgets inside the Automatos AI chat interface and workspace canvas. It covers the widget routing architecture, registration manifest, specialized widget implementations (`CodingCanvasWidget`, `FileWidget`, `TerminalWidget`, `ToolApprovalWidget`), and the backend tool result formatting pipeline.

---

## 1. Widget Router & Registration Manifest

The widget subsystem dynamically translates tool executions and structured data payloads into rich visual components. The routing tier maps tool names and result schemas to concrete widget types via `TOOL_WIDGET_MAP` and type-inspection logic `[frontend/components/widgets/router.ts:33-171]`.

```mermaid
graph TD
    A["ToolExecutionResult"] --> B["routeToolToWidget(toolName, result)"]
    B --> C{"TOOL_WIDGET_MAP Match?"}
    C -- Yes --> D["Return WidgetType (e.g., 'coding_canvas', 'file', 'tool_approval')"]
    C -- No --> E["Inspect Schema & Array/Key Patterns"]
    E --> F["Fallback Inference (database_results, documents, code_snippets)"]
    F --> D
    D --> G["transformToolResultToWidget()"]
    G --> H["WidgetStore / Canvas Render"]

    subclassDef code fill:#f9f9f9,stroke:#333,stroke-width:1px;
    class A,B,C,D,E,F,G,H code;
```
*Sources: [frontend/components/widgets/router.ts:33-171]()*, [frontend/components/widgets/index.ts:1-73]()

Widgets are registered through a central registry pattern (`registry.ts`) that initializes default component definitions upon import `[frontend/components/widgets/index.ts:1-22]`. Supported widget types span standard Phase 1 code, data, document, and image viewers, as well as Phase 2+ advanced surfaces like `coding_canvas`, `terminal`, `file`, `mission_approval`, and `tool_approval` `[frontend/components/widgets/types.ts:15-36]`.

Sources: [frontend/components/widgets/router.ts:33-171](), [frontend/components/widgets/types.ts:15-36](), [frontend/components/widgets/index.ts:1-22]()

---

## 2. CodingCanvasWidget & Workspace Integration

The `CodingCanvasWidget` (PRD-66 and PRD-170) combines a Monaco-based workspace file explorer with a streamed Auto/SDK session panel `[frontend/components/widgets/CodingCanvasWidget/index.tsx:4-12]`. It handles live-refresh synchronization where agent file edits automatically trigger directory tree invalidation and file reloading `[frontend/components/widgets/CodingCanvasWidget/index.tsx:54-77]`.

```mermaid
graph TD
    subcode["Frontend Workspace Space"]
    A["CodingCanvasWidget"] --> B["WorkspaceExplorer"]
    A --> C["CanvasSessionPanel"]
    B --> D["FileExplorer"]
    B --> E["CodeEditor (Monaco)"]
    B --> F["InteractiveTerminal"]
    C --> G["useCanvasSession()"]
    G --> H["Streamed SDK Turns"]
    H -- "file_edit event" --> I["lastEvent trigger"]
    I -- "invalidateCache & fetchDirectory" --> B

    subclassDef code fill:#f9f9f9,stroke:#333,stroke-width:1px;
    class A,B,C,D,E,F,G,H,I code;
```
*Sources: [frontend/components/widgets/CodingCanvasWidget/index.tsx:33-106](), [frontend/components/workspace/WorkspaceExplorer.tsx:50-130]()*, [frontend/components/widgets/CodingCanvasWidget/CodeEditor.tsx:29-57]()

The underlying `WorkspaceExplorer` supports compounding layouts using `react-resizable-panels`, tab management with dirty-state tracking via `EditorTabs`, and keyboard shortcuts (`Ctrl+S` to save via workspace API; `Ctrl+\`` to toggle the terminal) `[frontend/components/workspace/WorkspaceExplorer.tsx:50-102]`, `[frontend/components/widgets/CodingCanvasWidget/CodeEditor.tsx:38-57]`, `[frontend/components/widgets/CodingCanvasWidget/EditorTabs.tsx:29-120]`.

Sources: [frontend/components/widgets/CodingCanvasWidget/index.tsx:4-123](), [frontend/components/workspace/WorkspaceExplorer.tsx:50-205](), [frontend/components/widgets/CodingCanvasWidget/CodeEditor.tsx:29-112](), [frontend/components/widgets/CodingCanvasWidget/EditorTabs.tsx:29-120]()

---

## 3. FileWidget & FilePreview Renderer

The `FileWidget` and its shared `FilePreview` component serve as the single source of truth for rendering files across chat widgets, the Workspace Outputs Hub (`DeliverablePreview`), and the Workspace Explorer `[frontend/components/widgets/FileWidget/FilePreview.tsx:4-30]`. 

File types are categorized into:
- **Text-based formats** (rendered via `content` with syntax highlighting or source/preview toggles): `html`, `markdown`, `code`, `json`, `csv`/`tsv`, and plain `text` `[frontend/components/widgets/FileWidget/FilePreview.tsx:12-19]`.
- **Binary formats** (rendered via authenticated URLs or object URLs): `pdf`, `image`, `video`, `audio`, `docx` (via `mammoth.js`), and `xlsx` (via SheetJS) `[frontend/components/widgets/FileWidget/FilePreview.tsx:20-27]`.

For binary or protected relative URLs, `useAuthenticatedBlobUrl` fetches bytes through the authenticated Automatos API client (`apiClient.getAuthHeaders()`) and constructs short-lived local object URLs for safe rendering inside `<img>`, `<video>`, `<audio>`, and `<iframe>` tags `[frontend/components/widgets/FileWidget/FilePreview.tsx:59-118]`.

Sources: [frontend/components/widgets/FileWidget/FilePreview.tsx:4-145](), [frontend/components/workspace/gallery-view/deliverable-preview.tsx:1-82]()

---

## 4. TerminalWidget & Workspace Explorer Integration

Terminal capabilities are exposed through `TerminalWidget` (routing to shell execution tools like `execute_command`, `run_bash`, and `exec`) and embedded inside `WorkspaceExplorer` via `InteractiveTerminal` `[frontend/components/widgets/router.ts:76-82]`, `[frontend/components/workspace/WorkspaceExplorer.tsx:28]`. 

Users can toggle terminal access via `Ctrl+\`` or programmatically invoke workspace shell actions. The terminal session interfaces directly with the backend `WorkspaceWorker` Redis queue and containerized execution subprocesses.

Sources: [frontend/components/widgets/router.ts:76-82](), [frontend/components/workspace/WorkspaceExplorer.tsx:28-102]()

---

## 5. ToolApprovalWidget

The `ToolApprovalWidget` (PRD-193 S3 / P2-12) renders an in-chat confirmation card for gated tool calls awaiting human approval `[frontend/components/widgets/ToolApprovalWidget/index.tsx:4-16]`. 

```mermaid
graph TD
    A["Gated Tool Call Triggered"] --> B["Create Approval Grant (PRD-181)"]
    B --> C["Render ToolApprovalWidget in Chat"]
    C --> D{"User Action"}
    D -- "Approve & Run" --> E["useGrantApproval() Mutation"]
    E --> F["Server Resumes Gated Execution"]
    F --> G["Report Honest Outcome (success / execution error)"]
    D -- "Deny" --> H["useDenyApproval() Mutation"]
    H --> I["Record Denial & Refuse Execution"]

    subclassDef code fill:#f9f9f9,stroke:#333,stroke-width:1px;
    class A,B,C,D,E,F,G,H,I code;
```
*Sources: [frontend/components/widgets/ToolApprovalWidget/index.tsx:65-112](), [frontend/components/widgets/__tests__/tool-approval-widget.test.tsx:60-148]()*, [frontend/components/widgets/types.ts:69-81]()

The card displays:
- The target action name and message `[frontend/components/widgets/ToolApprovalWidget/index.tsx:126-130]`
- AI-Act oversight metadata (`risk_tier`, `risk_class`, `oversight_rationale`) inside a dedicated warning banner `[frontend/components/widgets/ToolApprovalWidget/index.tsx:139-158]`
- A parameter digest (`paramEntries`) formatted as a key-value list rather than raw JSON `[frontend/components/widgets/ToolApprovalWidget/index.tsx:160-172]`
- Action buttons wired to `useGrantApproval` and `useDenyApproval` hooks, displaying honest execution outcomes (never masking backend failures as fake successes) `[frontend/components/widgets/ToolApprovalWidget/index.tsx:76-111]`, `[frontend/components/widgets/__tests__/tool-approval-widget.test.tsx:76-112]`.

Sources: [frontend/components/widgets/ToolApprovalWidget/index.tsx:4-184](), [frontend/components/widgets/__tests__/tool-approval-widget.test.tsx:1-148](), [frontend/components/widgets/types.ts:69-81]()

---

## 6. Tool Result Mapping & Formatting Pipeline

Before tool outputs reach the widget router or LLM context, they pass through the unified backend formatter (`ToolResultFormatter`) located in `orchestrator/modules/tools/formatting/result_formatter.py` `[orchestrator/modules/tools/formatting/result_formatter.py:1-22]`.

The formatter standardizes document names by stripping hash prefixes `[orchestrator/modules/tools/formatting/result_formatter.py:24-42]`, extracts smart text excerpts with sentence boundary truncation `[orchestrator/modules/tools/formatting/result_formatter.py:45-67]`, and resolves full document contents by attempting direct S3 text downloads or falling back to reassembling chunk rows from the `document_chunks` database table `[orchestrator/modules/tools/formatting/result_formatter.py:70-165]`.

Sources: [orchestrator/modules/tools/formatting/result_formatter.py:1-174]()

---