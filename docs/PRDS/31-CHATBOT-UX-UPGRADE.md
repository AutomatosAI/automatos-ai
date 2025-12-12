# PRD-31: Chatbot UX Upgrade (Vercel Chat SDK UX + Automatos Explorer)

**Version:** 0.2
**Status:** 🟢 Ready for Implementation (Phase 0/1 partially shipped)
**Priority:** CRITICAL
**Author:** Automatos AI Platform Team
**Last Updated:** 2025-12-12

---

## Executive Summary

Automatos already has powerful backend capabilities (RAG, NL2SQL, tools, memory, learning), but the chatbot UI does not *surface* those capabilities with a modern, trustworthy, “explore everything” UX. Users get summaries instead of **inspectable evidence** (RAG chunks, SQL rows, charts, tool traces), and there’s no cohesive “assistant as entrance” experience.

This PRD defines a **Vercel Chat SDK–quality chat experience**, cherry-picking the best UX patterns from:

- **Vercel Chat SDK** template UX and architecture: artifacts side panel, streaming, tool transparency, uploads, history, model selector.
  - Template: `https://vercel.com/templates/ai/nextjs-ai-chatbot`
  - Repo: `https://github.com/vercel/ai-chatbot`
  - Local reference: `vercel-ai-chatbot-reference/`
- **Incredible.one**: “what’s eating up your day?” prompting, templates/use-case discovery, app/tool browsing.
  - Site: `https://www.incredible.one/`

…and merges them into **Automatos visual language + layout** while upgrading backend/frontend contracts so the assistant can:
- show RAG sources and **chunk-level evidence**,
- show NL2SQL **SQL + full result tables + charts + insights**,
- show **tool calls and outputs** as first-class UI,
- create **reports/artifacts** (documents, dashboards, code) with citations and exports,
- expose **memory** and **learning** in a safe, user-controlled way,
- act as the **entrance to Automatos** (documents, databases, code, tools, workflows).

---

## 0) Implementation Update (what is already shipped)

The following items from **Phase 0/1** have been implemented in code and deployed to the test server.

### Shipped (backend)
- **Tool lifecycle streaming (AI SDK Data Stream)**: emits `tool-start` / `tool-end` and streams `tool-data` incrementally.
- **Tool UI payload completeness**:
  - database results now include richer metadata (status/clarifications/explanation/etc) for display.
  - document results now include `chunks[]` and `full_content` so the UI can render chunk evidence.
- **RAG semantic search embedding mismatch fix**: `/api/documents/search` now uses the centralized embedding manager (DB-configured provider/dimensions) instead of hard-coded 1536-dim embeddings.

### Shipped (frontend)
- **Tool cards in chat** (running/completed/error) with collapsible inputs + errors.
- **DB “Data Explorer” artifact (sheet)**: searchable table, pagination, CSV export, base64 chart rendering.
- **Document Viewer improvements**: chunk inspector, copy chunk, open artifact from streamed payload.
- **Uploads in chat input**: upload to `/api/documents/upload`, show chips, send file parts in chat request.

### Validation performed (test server)
- **UI reachable**: `ui.automatos.app` returns HTTP 200.
- **Backend healthy**: `/health` returns HTTP 200.
- **Documents search healthy**: `/api/documents/search` returns HTTP 200 (no vector dimension 1024 vs 1536 failures).
- **Chat streaming smoke test**: `/api/chat` returns streaming deltas (verified by server-side request reading streaming lines).

## 1) Problem Statement

### Current pain
- **Missing inspectability**: tool outputs (RAG chunks / SQL results) aren’t reliably displayed, so users can’t verify answers.
- **No “data explorer” UX**: database query results don’t open into an interactive table/dashboard; users get summaries.
- **Low tool transparency**: the UI doesn’t show which tools ran, with what inputs, and what outputs.
- **Disconnected platform**: chat does not feel like the entry point for exploring documents, databases, code, workflows.

### Impact
- Reduced trust, reduced repeat usage, and inability to sell the “Automatos platform” value.

---

## 2) Goals & Success Metrics

### Goals
- **G1 (Trust)**: Every answer that uses RAG/DB/tools is accompanied by inspectable artifacts (sources, chunks, tables, charts).
- **G2 (Exploration)**: Chat becomes the primary entrance to explore docs/databases/code/tools/workflows with fast drill-down.
- **G3 (Smoothness)**: Streaming, interactions, and navigation match Vercel Chat SDK “slick” UX.
- **G4 (Cost-aware)**: UI + backend encourage efficient tool usage and token budgeting.

### Success metrics (MVP)
- **M1**: ≥ 90% of tool runs render a corresponding UI card/artifact (no silent tool outputs).
- **M2**: Median “time to evidence” (first RAG source card / SQL result card visible) < 2.0s after tool completes.
- **M3**: Users can export DB results (CSV) and reports (MD/PDF) in < 2 clicks.
- **M4**: 0 critical regressions in streaming stability (no stuck streams, no broken chat IDs).

---

## 2.1) Personas & Core User Journeys

### Personas
- **P1: Data Analyst / Ops (DB-heavy)**: wants quick answers with verifiable tables + charts; expects drill-down and export.
- **P2: Engineer (code-heavy)**: wants code search, precise snippets, diffs, and the ability to generate implementation plans.
- **P3: Product / Founder (docs + reporting)**: wants reports that combine docs + metrics + recommendations with citations.
- **P4: Automation Builder (tools/workflows)**: wants to connect tools, run workflows, and see execution traces.

### Core journeys (must feel “slick”)

#### J1: Ask a question → RAG evidence → open doc
1. User asks about a feature/design (“How does memory injection work?”)
2. Assistant runs RAG tool(s)
3. UI shows **Sources** cards with ranked chunks
4. User clicks a source → **Document Viewer** opens
5. Document Viewer highlights retrieved chunks and shows full text
6. Assistant response includes citations that map to chunks

#### J2: Ask for metrics → NL2SQL → explore table/chart
1. User asks: “Show failed workflows by day last 14 days”
2. Assistant runs `smart_query_database`
3. UI shows **Database Result** card with SQL preview + row count
4. User clicks → **Data Explorer** opens (table pagination + filters)
5. Charts appear (PandasAI) and can be downloaded/exported

#### J3: Generate a report → share/export
1. User asks: “Generate a weekly ops report”
2. Assistant runs DB query + RAG retrieval + optional tools
3. Assistant creates a **Report Artifact** with sections + citations
4. User exports to MD/PDF and shares

## 3) Non-Goals (for this PRD)

- Building a full BI product (Superset/Metabase-level) in v1.
- Implementing write-access SQL or destructive database operations in chat.
- Replacing the entire Automatos frontend stack.

---

## 4) Current State (Automatos)

### Frontend
- Next.js frontend at `automatos-ai/frontend/`.
- Chat page: `automatos-ai/frontend/app/chat/`.
- Chat UI components: `automatos-ai/frontend/components/chatbot/`.
- Edge proxy route: `automatos-ai/frontend/app/api/chat/route.ts` streams the backend response and sets `x-vercel-ai-data-stream: v1`.

### Backend
- Streaming chat consumer: `automatos-ai/orchestrator/consumers/chatbot/`.
- Chat API: `automatos-ai/orchestrator/api/chat.py`.
- Tool execution via `modules.tools` and `consumers/chatbot/tool_router.py`.
- RAG: `automatos-ai/orchestrator/modules/rag/`.
- NL2SQL: `automatos-ai/orchestrator/modules/nl2sql/`.
- Memory: `automatos-ai/orchestrator/modules/memory/`.
- Learning: `automatos-ai/orchestrator/modules/learning/`.

### Key finding
Automatos already streams in **AI SDK Data Stream format** (text deltas + data events) and emits `tool-data`, but the system needs:
- stronger, versioned **event contracts**,
- richer **tool lifecycle events**,
- consistent **frontend artifact mapping** (documents, database results, charts, etc.).

---

## 5) Competitive / Inspiration Benchmark

### 5.1 Vercel Chat SDK (template)
Cherry-pick these behaviors:
- **Artifacts overlay**: chat on left, artifact on right, smooth animation, persistent artifact state.
- **Message UX**: streaming, regenerate, edit user message, copy, vote.
- **Tool visibility**: tool call UI parts with statuses (pending/running/completed/error).
- **File uploads**: attachment previews + upload progress; paste image support.
- **Suggested actions**: starter prompts, follow-up suggestions.
- **Model selector**: quick switch; show usage/context.
- **Resumable stream (optional)**: refresh-safe streaming with stream IDs.

### 5.2 Incredible.one
Cherry-pick these behaviors:
- **“What’s eating up your day?”**: prompt-first onboarding.
- **Templates / use-cases**: categorized starter tasks (marketing/sales/dev/research/productivity).
- **Browse apps/tools**: visually browse integrations; quick connect.

### 5.3 Cherry-pick matrix (Vercel → Automatos mapping)

| Vercel Chat SDK feature | Why it matters | Where it lives (reference) | Automatos adaptation (target) |
|---|---|---|---|
| Artifacts overlay (chat left, artifact right) | Deep inspection without leaving chat | `vercel-ai-chatbot-reference/components/artifact.tsx` | Keep Automatos overlay in `automatos-ai/frontend/components/chatbot/chat.tsx`, expand artifact kinds (`sheet`, `report`) |
| Tool call UI parts (pending/running/completed) | Trust + transparency | `vercel-ai-chatbot-reference/components/elements/tool.tsx` + `vercel-ai-chatbot-reference/components/message.tsx` | Add `tool-start`/`tool-end` streaming events + render collapsible tool cards in `automatos-ai/frontend/components/chatbot/message.tsx` |
| Suggested actions | Instant onboarding | `vercel-ai-chatbot-reference/components/suggested-actions.tsx` | Keep Automatos suggestions; add categories + Templates drawer (“Browse use-cases”) |
| Model selector | Control + debugging | `vercel-ai-chatbot-reference/components/multimodal-input.tsx` | Keep Automatos selector (`automatos-ai/frontend/components/chatbot/model-selector.tsx`), add context/cost indicator |
| Attachments (upload progress + paste images) | Multimodal + doc ingestion | `vercel-ai-chatbot-reference/components/multimodal-input.tsx` + `.../api/files/upload/route.ts` | Implement upload + ingest into RAG; show previews + progress in `automatos-ai/frontend/components/chatbot/multimodal-input.tsx` |
| Message edit/regenerate/copy/vote | Fast iteration loop | `vercel-ai-chatbot-reference/components/message-actions.tsx`, `message-editor.tsx` | Add edit UX and keep votes; ensure regenerate respects chat-id + tool state |
| Context usage indicator | Cost awareness | `vercel-ai-chatbot-reference/components/elements/context.tsx` | Add compact context widget to Automatos input (tokens + optional cost) |
| Resumable stream | Robustness on refresh | `vercel-ai-chatbot-reference/app/(chat)/api/chat/[id]/stream/route.ts` | Optional: implement stream IDs + resume endpoint in Python or Next.js proxy |

---

## 6) Product Scope (Phased)

### Phase 0 (Stabilize + Fix “invisible results”) — MUST
- Ensure DB tool output (including `smart_query_database`) always renders as `database_results` cards.
- Ensure RAG tool output always renders as `documents` cards.
- Ensure charts (PandasAI) render inline and in artifacts.

### Phase 1 (Vercel-parity UX for tool transparency + exploration) — SHOULD
- Tool lifecycle UI: show tool calls as collapsible “tool cards” (input/output/errors).
- Dedicated artifact types:
  - **Document Viewer** (chunk list + open full doc + citations)
  - **Data Explorer** (table pagination + CSV export + chart gallery)
  - **Code Viewer** (syntax highlight + copy + file path)
- Attachments: upload docs/images; ingest to RAG; attach to chat message.
- Context indicator: tokens/cost/context usage.

### Phase 2 (Reports + Orchestrator/Agents integration) — SHOULD
- “Generate report” flow that produces a **Report Artifact** (Markdown + citations + tables/charts/code snippets).
- Agent run / orchestration trace in chat (high-level stage timeline + expandable details).

### Phase 3 (Learning + “assistant as entrance”) — COULD
- Learning loop from feedback (votes, “save as template”, “promote to playbook”).
- Persistent “workspace context”: pinned docs, pinned database sources, pinned workflows.

---

## 7) UX Requirements

### 7.0 Design system alignment (Automatos look & feel)
This upgrade MUST follow Automatos styling tokens and existing layout primitives:
- Use the existing CSS variables from `automatos-ai/frontend/app/globals.css` (dark background + orange primary).
- Prefer existing shadcn primitives in `automatos-ai/frontend/components/ui/*`.
- Visual emphasis: **orange gradient accents**, glass panels (`.glass-panel`), subtle borders (`--border`), and restrained motion (respect `prefers-reduced-motion`).

We are copying Vercel UX *patterns*, not Vercel branding.

### 7.1 Core Chat Layout
- **Left**: chat history sidebar (search, new chat, groups by date).
- **Center**: conversation.
- **Right (Inspector/Artifacts)**:
  - Opens on card click (doc/db/code/report).
  - Supports split-mode overlay like Vercel.
  - Supports quick copy/export.

### 7.2 Message Composition (Prompt Input)
- Multi-line input, Enter to send, Shift+Enter newline.
- Attachments button + drag/drop + paste images.
- Model selector in the input bar.
- Stop button while streaming.

### 7.3 Tool Evidence Surfaces (required)

#### Documents (RAG)
- Show a “Sources” card that lists:
  - filename/title
  - relevance score
  - number of chunks
  - preview excerpt
  - click opens Document Viewer artifact
- Document Viewer artifact shows:
  - chunk list (ranked)
  - full content (or chunk-expanded content)
  - download link (when available)
  - citations mapping (chunk → response references)

#### Database (NL2SQL)
- Show a “Database result” card that includes:
  - database name
  - row count
  - execution time
  - SQL preview (expand to full SQL)
  - PandasAI insight (if any)
  - chart thumbnails (if any)
  - click opens Data Explorer artifact
- Data Explorer artifact shows:
  - full table with pagination, search/filter, column toggle
  - export CSV
  - charts gallery
  - “ask follow-up” prompt suggestions

#### Code (CodeGraph)
- Code snippets list in message.
- Code Viewer artifact shows:
  - syntax highlighting
  - file path + line
  - copy code

---

## 8) Functional Requirements (FR)

### FR-1 Chat sessions
- Create chat (server-generated ID supported).
- Rename, delete.
- List history with pagination.

### FR-2 Streaming
- Stream assistant text in real time.
- Stop generation.
- Robust error handling (recoverable vs terminal).

### FR-3 Tool execution transparency
- UI must display tool activity:
  - tool name
  - input parameters
  - running status
  - output (rendered and raw JSON)
  - errors

### FR-4 RAG display
- Surface sources and chunk evidence.
- Open and read full docs.
- “Pin to context” (optional): keep a doc/chunk available across turns.

### FR-5 NL2SQL display
- Always show SQL.
- Always show table rows (paginated).
- Render charts and insights.
- Clarification flow:
  - if backend returns `needs_clarification`, present clarifying questions as UI choices.

### FR-6 Reports
- Provide a “Generate report” action that produces a Report Artifact.
- Report must:
  - be Markdown
  - include citations to docs and db results
  - embed tables/charts when available
  - include code samples when relevant

### FR-7 Tools & MCP
- “Browse tools/apps” UX:
  - list available tools
  - show required credentials
  - allow enable/disable tools per chat

### FR-8 Memory
- Show what memory was injected (high-level summary).
- Allow user to add explicit memories (“Remember this”).
- Allow user to delete/forget memory items.

### FR-9 Learning
- Capture feedback signals:
  - votes
  - “this was correct/incorrect”
  - “save as template/playbook”
- Learning module consumes these events.

---

## 8.1) Module-to-UX Mapping (Automatos capabilities → UI)

### `modules/rag` (documents + chunks)
- **Backend capability**: retrieve relevant chunks + sources.
- **UI surfaces**:
  - Sources cards in the assistant message.
  - Document Viewer artifact with chunk list + full content.
- **Streaming/data**:
  - `tool-start`/`tool-end` for the RAG tool call.
  - `tool-data.documents[]` for UI payload.
### `modules/nl2sql` (SQL + result sets + clarifications)
- **Backend capability**: generate SQL safely, execute, optionally analyze via PandasAI.
- **UI surfaces**:
  - Database Result card (SQL preview, row count).
  - Data Explorer artifact (table + export + charts).
  - Clarification UI when status is `needs_clarification`.
- **Streaming/data**:
  - `tool-data.database_results[]` with optional `pandas_ai.charts`.
### `modules/tools` (tool registry + execution)
- **Backend capability**: unified tool registry + MCP execution + formatting.
- **UI surfaces**:
  - Tool catalog (“Browse apps/tools”).
  - Tool cards showing inputs/outputs/errors.
- **Streaming/data**:
  - `tool-start`/`tool-end` for every tool call.
### `modules/memory` (injection + storage)
- **Backend capability**: inject relevant memory + store conversation facts.
- **UI surfaces**:
  - Memory inspector panel showing injected memories (high-level).
  - “Remember this” and “Forget” actions.
- **Streaming/data**:
  - Optional `memory-injected` / `memory-stored` data events.
### `modules/learning` (feedback → improvement)
- **Backend capability**: mine patterns/playbooks and improve prompts/tools.
- **UI surfaces**:
  - “Save as template/playbook” action.
  - Learning activity feed (what was learned, from which interactions).
- **Streaming/data**:
  - Feedback events emitted on vote/save.
---

## 9) Streaming / Data Contracts

Automatos currently uses AI SDK Data Stream framing:
- `0:<json-string>` for text deltas
- `d:<json>` for data events
- `e:<json>` for errors

### 9.1 Required data events
- `chat-id` (existing): `{ type: "chat-id", chatId: string }`
- `tool-start` (new): `{ type: "tool-start", data: { toolCallId, toolName, input } }`
- `tool-end` (new): `{ type: "tool-end", data: { toolCallId, toolName, success, error? } }`
- `tool-data` (existing, versioned):
  - `{ type: "tool-data", version: 1, data: ToolDataPayload }`
- `usage` (existing): `{ type: "usage", data: { promptTokens, completionTokens, totalTokens, cost? } }`
- `finish` (existing)

### 9.2 ToolDataPayload v1

```json
{
  "documents": [
    {
      "filename": "...",
      "title": "...",
      "file_path": "...",
      "relevance": 87,
      "chunk_count": 4,
      "preview": "...",
      "download_url": "...",
      "full_content": "..."
    }
  ],
  "database_results": [
    {
      "database": "automatos_main",
      "sql": "SELECT ...",
      "row_count": 123,
      "execution_time_ms": 42,
      "columns": ["..."],
      "data": [{"col": "value"}],
      "pandas_ai": {
        "summary": "...",
        "charts": [{"filename": "...", "mime_type": "image/png", "base64": "..."}]
      }
    }
  ],
  "code_snippets": [
    {
      "symbol_name": "...",
      "file_path": "...",
      "line_number": 123,
      "language": "python",
      "code": "..."
    }
  ]
}
```

---

## 10) Technical Design (High Level)

### Frontend
- Build/extend components under `automatos-ai/frontend/components/chatbot/`:
  - `Message` should render:
    - tool activity cards (new)
    - sources cards
    - db results cards
  - Add new artifacts:
    - `sheet` artifact (Data Explorer)
    - `report` artifact (Markdown report with citations)
- Add attachment upload flow:
  - upload → backend ingestion → message includes file parts.

### Backend
- Update tool result formatting to ensure *all* database tools and RAG tools emit UI-friendly payloads.
- Emit tool lifecycle events during execution (start/end + tool-data increments).
- Add “report generation” tool that returns a report artifact payload.
- Integrate learning feedback events (vote/save template) into `modules/learning`.

---

## 11) Security & Privacy

- Enforce auth for document access and database sources.
- Ensure download endpoints validate paths (no path traversal).
- Ensure SQL is read-only and includes strict limits/timeouts.
- Log tool calls with redaction of secrets.

---

## 12) Performance & Cost Controls

- Token budgets enforced by orchestrator stage/token manager.
- RAG chunk count and max tokens configurable (system settings).
- UI virtualization for long conversations and large tables.
- Avoid sending full data twice (LLM context vs UI payload).

---

## 13) Acceptance Criteria

- **AC-1**: When RAG tools run, the message shows a Sources card with at least 1 item (when results exist).
- **AC-2**: When NL2SQL tools run, the message shows a Database Result card with SQL and row count.
- **AC-3**: PandasAI charts render as images in message and/or artifact.
- **AC-4**: Clicking a doc/db card opens the appropriate artifact viewer.
- **AC-5**: Tool execution visibility shows “running → completed/error” state transitions.

---

## 14) Implementation Plan (Concrete)

### P0 tasks (stabilize)
- Backend: ensure database tools emit `database_results` for **both** `query_database` and `smart_query_database`.
- Frontend: confirm `tool-data` event populates message model consistently.

### P1 tasks (parity)
- Add tool lifecycle events and UI tool cards.
- Add Data Explorer artifact (`sheet`), CSV export.
- Add Document Viewer artifact (chunk list + full content + download).
- Add attachment upload + ingestion pipeline hook.

### P2 tasks (reports)
- Add `create_report` tool.
- Add Report artifact + export.
- Optional: agent/orchestrator trace timeline embedded in chat.

---

## 15) File-level Implementation Map (Where Work Lands)

### Backend (Python)
#### Streaming protocol + lifecycle events
- Update `automatos-ai/orchestrator/consumers/chatbot/streaming.py` to add helpers for:
  - `tool-start` / `tool-end`
  - versioned `tool-data` (`version: 1`)
- Update `automatos-ai/orchestrator/consumers/chatbot/service.py` to emit lifecycle events inside `_handle_tool_calls`.

#### Tool payload normalization
- Extend `automatos-ai/orchestrator/modules/tools/formatting/result_formatter.py` to normalize:
  - docs (`search_knowledge`, `semantic_search`, etc.) → `documents[]`
  - db (`query_database`, `smart_query_database`) → `database_results[]`
  - code (`search_codebase`) → `code_snippets[]`
  - multimodal (`search_tables`, `search_images`, `search_multimodal`) → `tables[]`, `images[]`

#### New tool: report generation
- Register tool in `automatos-ai/orchestrator/modules/tools/registry/tool_registry.py`.
- Implement executor in `automatos-ai/orchestrator/modules/tools/execution/unified_executor.py`.
- Return payload shaped like a Report Artifact (Markdown + references).

#### APIs
- Keep `automatos-ai/orchestrator/api/chat.py` as the single streaming entry-point.
- Add/extend document endpoints (content/download) with strict path validation and auth.

### Frontend (Next.js)

#### Stream parsing
- Extend `automatos-ai/frontend/lib/chat/hooks.ts` to:
  - handle `tool-start`/`tool-end` events
  - handle `tool-data.version`
  - store tool activity per message turn

#### Message rendering
- Extend `automatos-ai/frontend/components/chatbot/message.tsx` to:
  - render collapsible tool cards
  - render “clarification needed” UI for `smart_query_database` status

#### Artifacts
- Add `sheet-artifact.tsx` (Data Explorer)
- Add `report-artifact.tsx` (Report Viewer)
- Extend `artifact-viewer.tsx` to support new kinds

#### Prompt input + uploads
- Upgrade `automatos-ai/frontend/components/chatbot/multimodal-input.tsx` to support:
  - file picker + drag/drop
  - upload queue + preview
  - paste images

---

## 16) Test Plan

### Backend
- **Unit**: tool payload formatting
  - DB tools → `database_results[]` includes `sql`, `columns`, `data`, optional `pandas_ai`.
  - RAG tools → `documents[]` includes preview + file reference.
- **Integration**: streaming event ordering
  - `tool-start` emitted before `tool-end`
  - `tool-data` emitted when UI payload changes
  - abort/stop returns clean `finish`/`error` semantics

### Frontend
- **Stream parsing** (unit/light integration)
  - ensure new events correctly update UI state
  - ensure tool-data merges without clobbering prior results
- **UI**
  - clicking doc card opens Document Viewer
  - clicking db card opens Data Explorer
  - charts render from base64 payload

### E2E (Playwright recommended)
- DB question → db card visible → open explorer → export CSV.
- Doc question → sources visible → open doc viewer.

---

## 17) Risks & Mitigations
- **R1: Payload size (DB rows, doc chunks)**
  - Mitigation: paginate; stream metadata; fetch full content on demand.
- **R2: Token cost explosion (injecting full tool outputs)**
  - Mitigation: UI gets full payload; LLM gets truncated/optimized context; enforce token budgets.
- **R3: Document download/content security**
  - Mitigation: authorize by document ID; validate paths; disallow arbitrary filesystem reads.
- **R4: Edge streaming reliability**
  - Mitigation: keep protocol simple; optionally add resumable streams.

---

## 18) References

- Vercel template: `https://vercel.com/templates/ai/nextjs-ai-chatbot`
- Vercel repo: `https://github.com/vercel/ai-chatbot`
- Local reference implementation: `vercel-ai-chatbot-reference/`
- Incredible: `https://www.incredible.one/`
- Existing Automatos PRDs:
  - `08-RAG-SEMANTIC-SEARCH.md`
  - `17-DYNAMIC-TOOL-ASSIGNMENT.md`
  - `21-DATABASE_KNOWLADGE.md`
  - `CHATBOT-INTELLIGENCE-ENHANCEMENT.md`
