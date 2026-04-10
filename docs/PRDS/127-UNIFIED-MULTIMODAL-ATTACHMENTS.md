# PRD-127 — Universal Ephemeral Attachments

**Version:** 4.0
**Type:** Implementation
**Status:** Draft
**Priority:** P0
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-04-10

---

## 1. Goal

One short-lived upload pipeline. Four consumers: **chat, missions, tasks, channels**. Any file type — images, PDFs, docs, code, screenshots. The agent **sees** it this turn. It lives for the lifetime of the chat session / mission run / task execution / channel message, then the S3 lifecycle rule deletes it.

**Not RAG. Not Knowledge Base. Not DocumentManager.** Users wanting long-term knowledge use the Documents page — that path is untouched. This PRD is strictly about "I'm showing the agent this right now."

Fixes:
- The 2026-04-09 brand mission failure (images in attached PDFs were silently dropped).
- Chat paperclip uploads silently going through `DocumentManager` and landing in the RAG index as a side effect.
- Task board having no upload UI at all.
- Channels having no attachment handling at all.

---

## 2. Current state

| Surface | Today | Problem |
|---------|-------|---------|
| **Chat** | `multimodal-input.tsx:188` → `apiClient.uploadDocument()` → `/api/documents/upload` → `DocumentManager`. Message parts carry `document://{id}`. `consumers/chatbot/service.py:440` `_resolve_file_parts` reads from `document_chunks`. | Calls the RAG ingester as a side effect. Text-only. Race: upload+send before chunks exist → "still processing" message. |
| **Missions** | `create-mission-modal.tsx` → `/api/missions/upload` → `DocumentManager.upload_document(tags=["mission-attachment"])`. `planner.py:67` `_fetch_attachment_contents` + second caller at `services/coordinator_service.py:820` read `Document.file_path` and extract with PyMuPDF. | Same accidental RAG side effect. Images rejected. PyMuPDF `get_text()` strips images from PDFs. |
| **Task board** | `create-task-dialog.tsx` has no upload UI. `board_tasks` has no attachment column. | Nothing. |
| **Channels** | `orchestrator/channels/` has zero attachment code. | Channel adapters drop inbound media silently. |

---

## 3. What ships

| Component | Description |
|-----------|-------------|
| `AttachmentStore` | `orchestrator/modules/attachments/store.py`. Thin wrapper over S3 under `workspaces/{ws}/ephemeral-attachments/{id}/`. 7-day S3 lifecycle rule handles GC. No DB table. No chunking. No embedding. No DocumentManager import. |
| `extract.py` | `orchestrator/modules/attachments/extract.py`. Pure text extraction (pdfplumber, python-docx, openpyxl) for document types. Does not import from `modules/rag/`. |
| `POST /api/attachments` | `orchestrator/api/attachments.py`. Upload endpoint. Returns `{attachment_id, filename, mime, media_type, size}`. Accepts images + documents + code + text. |
| `GET /api/attachments/{id}` | Metadata lookup. |
| `DELETE /api/attachments/{id}` | Explicit delete (user removes before sending). |
| `AttachmentResolver` | `orchestrator/modules/attachments/resolver.py`. Single code path that converts `attachment_ids → list[ContentPart]`. Called from `ContextService.build_context()` as a post-processing step after text sections render. **Only place that builds multimodal parts.** |
| `ContextService.build_context(attachment_ids=...)` | New kwarg. Service resolves via `AttachmentResolver` and appends parts to the **last user message** in `result.messages` before returning. |
| Vision check | `AttachmentResolver` reads `model_registry.get_model(model_id).supports_vision` (already exists at `core/llm/model_registry.py:39`). Raises `VisionNotSupportedError` if images are attached to a text-only model. **No new capability field. No new registry.** |
| Chat wiring | Frontend: `multimodal-input.tsx` switches to `/api/attachments`, sends `attachment_ids: string[]` in message payload (no more `document://`). Backend: chat handler forwards `attachment_ids` into `ContextService.build_context()`. `_resolve_file_parts` deleted. |
| Mission wiring | Frontend: `create-mission-modal.tsx` switches to `/api/attachments`, accepts images. Backend: `mission_runs.attachment_ids` + `mission_tasks.attachment_ids` JSONB columns. Planner and mission_runner forward ids into `ContextService.build_context()`. `_fetch_attachment_contents` deleted. `services/coordinator_service.py:820` caller refactored. `/api/missions/upload` deleted. |
| Task wiring | Frontend: `create-task-dialog.tsx` gets attachment UI (reuse chat component). Backend: `board_tasks.attachment_ids` JSONB column. Task executor forwards ids into `ContextService.build_context()`. |
| Channel wiring | Channel adapters (`orchestrator/channels/`) call `AttachmentStore.put()` on inbound media bytes, attach ids to the resulting chat message. Same single path. |
| Migration | Adds `attachment_ids JSONB` to `board_tasks`, `mission_runs`, `mission_tasks`. Chat messages already store payload as JSONB — no schema change. **No `attachments` table.** |

### 3.1 What "ephemeral" means concretely

- **S3 prefix:** `s3://automatos-ai/workspaces/{workspace_id}/ephemeral-attachments/{attachment_id}/{filename}`
- **Lifecycle rule:** objects under `ephemeral-attachments/` expire after 7 days. S3 handles GC.
- **No attachments DB table.** IDs live on the owning row (chat message / mission / task) as inline JSON: `[{attachment_id, filename, mime, media_type}]`. Filename/mime cached inline so history renders as chips even after the blob expires (shown as `[expired: filename]`).
- **Metadata lookup:** `AttachmentStore.get()` calls `HeadObject` on S3. No DB round-trip.
- **Workspace isolation:** enforced by the S3 prefix. `AttachmentStore.get(id, workspace_id)` checks the prefix matches. Cross-workspace access is impossible.

---

## 4. Not in scope

- RAG / Knowledge Base ingestion. Users wanting RAG use `/api/documents/upload` (Documents page), untouched.
- `DocumentManager` and the entire `modules/rag/` tree — **not touched**.
- `multimodal/processors.py` — live RAG OCR code, broken or not, out of scope for this PRD.
- Video / audio.
- In-chat image generation.
- Per-agent vision model auto-routing (follow-up PRD).
- Thumbnails.
- Custom retention (7-day TTL only).

---

## 5. Architecture

### 5.1 `AttachmentStore`

```python
# orchestrator/modules/attachments/store.py

class MediaType(str, Enum):
    IMAGE = "image"
    DOCUMENT = "document"

@dataclass(frozen=True)
class AttachmentRef:
    attachment_id: UUID
    workspace_id: UUID
    media_type: MediaType
    mime: str
    filename: str
    size_bytes: int
    s3_key: str

class AttachmentStore:
    def __init__(self, s3: S3Client, bucket: str): ...

    async def put(
        self,
        *,
        workspace_id: UUID,
        uploaded_by: str,
        filename: str,
        content: bytes,
        declared_mime: Optional[str] = None,
    ) -> AttachmentRef:
        """Validate (MIME, size, magic bytes) → S3 put → return ref.
        No DB writes. No RAG. Subject to 7-day lifecycle rule."""

    async def get(self, attachment_id: UUID, workspace_id: UUID) -> AttachmentRef:
        """HeadObject. Raises NotFound if expired or missing."""

    async def open(self, attachment_id: UUID, workspace_id: UUID) -> bytes:
        """GetObject — used by extract.py for documents."""

    async def sign_url(self, attachment_id: UUID, workspace_id: UUID, ttl_seconds: int = 900) -> str:
        """Presigned GET URL for image_url parts."""

    async def delete(self, attachment_id: UUID, workspace_id: UUID) -> None: ...
```

### 5.2 `AttachmentResolver` — the single conversion point

Sections return `str`. Multimodal parts need `list[ContentPart]` on the user message. So attachments are **not** a `BaseSection` — they're a post-processing step inside `ContextService.build_context()` that injects parts into the last user message after text sections have rendered.

```python
# orchestrator/modules/attachments/resolver.py

class VisionNotSupportedError(Exception): ...

class AttachmentResolver:
    """The only code path that converts attachment_ids into LLM content parts.
    No other code in the codebase constructs image_url parts or calls extract_text."""

    def __init__(self, store: AttachmentStore):
        self._store = store

    async def resolve(
        self,
        attachment_ids: list[UUID],
        workspace_id: UUID,
        model_id: str,
        text_budget_tokens: int = 20_000,
    ) -> list[dict]:
        """Return a list of ContentPart dicts ready to append to a user message."""
        if not attachment_ids:
            return []

        refs = [await self._store.get(aid, workspace_id) for aid in attachment_ids]

        # Vision check — reuse existing model_registry, no new field
        if any(r.media_type == MediaType.IMAGE for r in refs):
            model = model_registry.get_model(model_id)
            if not model or not model.supports_vision:
                raise VisionNotSupportedError(model_id)

        parts: list[dict] = []
        budget_chars = text_budget_tokens * 4  # ~4 chars/token

        for ref in refs:
            if ref.media_type == MediaType.IMAGE:
                url = await self._store.sign_url(ref.attachment_id, workspace_id)
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                blob = await self._store.open(ref.attachment_id, workspace_id)
                text = extract_text(blob, ref.mime, ref.filename)  # pdfplumber/docx/xlsx
                text = text[:budget_chars]
                budget_chars -= len(text)
                parts.append({"type": "text", "text": f"### {ref.filename}\n{text}"})

        return parts
```

### 5.3 `ContextService.build_context()` — one new kwarg

```python
# orchestrator/modules/context/service.py (extension)

class ContextService:
    async def build_context(
        self,
        mode: ContextMode,
        agent: Any,
        workspace_id: str,
        messages: Optional[list[dict]] = None,
        task_description: Optional[str] = None,
        attachment_ids: Optional[list[UUID]] = None,   # NEW
        model_id: Optional[str] = None,                # NEW (for vision check)
        **kwargs: Any,
    ) -> ContextResult:
        # ... existing section rendering stays unchanged ...
        result = self._assemble_result(system_prompt, messages, tools, ...)

        # NEW — post-processing: inject attachment parts into last user message
        if attachment_ids:
            parts = await self._attachment_resolver.resolve(
                attachment_ids=attachment_ids,
                workspace_id=UUID(workspace_id),
                model_id=model_id or agent.llm_config.get("model"),
            )
            result = _inject_parts_into_last_user_message(result, parts)

        return result
```

`_inject_parts_into_last_user_message()` walks `result.messages` backwards, finds the last `role=user` entry, converts its `content: str` to `content: [{type: "text", text: ...}, *parts]`. Single helper, ~10 lines.

**Rule:** nothing outside `AttachmentResolver` constructs `image_url` parts or calls `extract_text`. Chat handler, planner, task executor, channel adapters only pass `attachment_ids` into `build_context()`. Enforced by CI grep (§9).

### 5.4 Database

```sql
ALTER TABLE board_tasks   ADD COLUMN attachment_ids JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE mission_runs  ADD COLUMN attachment_ids JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE mission_tasks ADD COLUMN attachment_ids JSONB NOT NULL DEFAULT '[]'::jsonb;
-- Chat messages already persist payload as JSONB; attachment_ids ride inline.
```

Each JSONB value:
```json
[{"attachment_id": "uuid", "filename": "brand.png", "mime": "image/png", "media_type": "image"}]
```

Filename/mime cached inline so history chips render after blob expiry.

### 5.5 Migration for existing data

- **Chat `document://` references in old messages:** resolver keeps a **read-only** compatibility branch for 30 days — if a message part has a `document://{id}` URL, it reads `document_chunks` and inlines text (current behavior). No new `document://` parts are created after ship. Branch deleted in a follow-up PR after 30 days. CI grep at §9 allows `document://` references in the compat branch only.
- **Existing `mission_runs.attachments` with `document_id` values:** old missions keep the old path (read via `Document.file_path`) until they age out naturally. No backfill. New missions use `attachment_ids`.
- **No data migration required.** Both old and new rows coexist. Old path is on a sunset timer.

---

## 6. Changes by file

### 6.1 New files

```
orchestrator/modules/attachments/__init__.py
orchestrator/modules/attachments/store.py
orchestrator/modules/attachments/validation.py
orchestrator/modules/attachments/extract.py
orchestrator/modules/attachments/resolver.py
orchestrator/api/attachments.py
alembic/versions/{ts}_attachment_id_columns.py
tests/attachments/test_store.py
tests/attachments/test_extract.py
tests/attachments/test_resolver.py
tests/attachments/test_api.py
tests/context/test_attachment_injection.py
tests/integration/test_multimodal_chat.py
tests/integration/test_multimodal_mission.py
tests/integration/test_multimodal_task.py
tests/integration/test_multimodal_channel.py
tests/integration/test_s3_lifecycle_isolation.py
```

### 6.2 Modified files

| File | Change |
|------|--------|
| `orchestrator/modules/context/service.py` | `build_context()` gains `attachment_ids` + `model_id` kwargs. Injects `AttachmentResolver` in `__init__`. Post-processing step appends parts to last user message. |
| `orchestrator/consumers/chatbot/service.py` | `_resolve_file_parts` at line 440 **deleted**. Call sites at lines 1794, 1986 removed. Chat handler forwards `attachment_ids` to `build_context()`. Compat branch reads `document://` from old messages only. |
| `orchestrator/consumers/chatbot/prompt_analyzer.py` | Stale comment at line 291 referencing `_resolve_file_parts` removed. |
| `orchestrator/modules/coordination/planner.py` | `_fetch_attachment_contents` at line 67 **deleted**. `decompose()` / `replan()` forward `attachment_ids` to `build_context()`. Planner output schema allows per-task `attachment_ids` (default = inherit from mission). |
| `orchestrator/services/coordinator_service.py` | Line 820 caller of `_fetch_attachment_contents` **deleted**. Coordinator forwards `attachment_ids` to `build_context()`. |
| `orchestrator/modules/coordination/mission_runner.py` | Reads `mission_tasks.attachment_ids`, forwards to each dispatched agent's `build_context()` call. |
| `orchestrator/api/missions.py` | `/api/missions/upload` endpoint at line 768 **deleted**. |
| `orchestrator/api/board_tasks.py` | POST payload accepts `attachment_ids`. Task executor forwards into `build_context()`. |
| `orchestrator/modules/agents/factory/agent_factory.py` | `execute_with_prompt()` forwards optional `attachment_ids` kwarg into `build_context()`. Never constructs parts directly. |
| `orchestrator/channels/` (all adapters) | On inbound media, call `AttachmentStore.put()` with the bytes, attach ids to the resulting chat message payload. |
| `frontend/components/chatbot/multimodal-input.tsx` | Switch upload call from `apiClient.uploadDocument` to `apiClient.uploadAttachment`. Delete `document://` URL construction at lines 136-144. Send `attachment_ids: string[]` in message payload. |
| `frontend/components/missions/create-mission-modal.tsx` | `ALLOWED_TYPES` extended with image MIMEs. Upload target switched to `apiClient.uploadAttachment`. |
| `frontend/components/activity/board/create-task-dialog.tsx` | Add attachment UI (reuse chat's component). Pass `attachment_ids` in task payload. |
| `frontend/lib/api-client.ts` | New `uploadAttachment(file)` method posting to `/api/attachments`. Existing `uploadDocument()` is left alone — Documents page still uses it. |

### 6.3 Deletions

- `orchestrator/consumers/chatbot/service.py:_resolve_file_parts` (and its 2 call sites)
- `orchestrator/modules/coordination/planner.py:_fetch_attachment_contents`
- `orchestrator/services/coordinator_service.py:820` import + caller
- `orchestrator/api/missions.py:768` `/api/missions/upload` endpoint
- `document://` URL construction in `multimodal-input.tsx:136-144`

**Not deleted:** `modules/rag/ingestion/multimodal/processors.py` — live RAG OCR code, separate ticket.

---

## 7. Implementation phases

### Phase 1 — Foundation (ships standalone)
1. Migration: `attachment_ids` columns on `board_tasks`, `mission_runs`, `mission_tasks`.
2. Configure S3 lifecycle rule on `workspaces/*/ephemeral-attachments/*` → 7-day expiry.
3. Build `AttachmentStore`, `validation.py`, `extract.py`, `AttachmentResolver`.
4. Extend `ContextService.build_context()` with `attachment_ids` + `model_id` kwargs and post-processing.
5. Build `/api/attachments` POST/GET/DELETE.
6. Unit tests: store, extract, resolver, injection, vision check, validation.
7. Ship. Nothing else changes.

### Phase 2 — Mission planner multimodal
1. Refactor `planner.decompose()` / `replan()` to forward `attachment_ids` to `build_context()`. Delete `_fetch_attachment_contents`.
2. Refactor `coordinator_service.py:820` caller.
3. Planner output schema allows per-task `attachment_ids` (default inherit).
4. `mission_runner` passes `mission_tasks.attachment_ids` to each agent's `build_context()`.
5. Extend `create-mission-modal.tsx`: accept images, switch upload target to `/api/attachments`.
6. Delete `/api/missions/upload`.
7. Integration test: mission with image → planner LLM call has `image_url` part → dispatched `mission_tasks` inherit ids.
8. Manual regression: re-run the 2026-04-09 brand mission.

### Phase 3 — Chat multimodal
1. Delete `_resolve_file_parts` and its call sites. Chat handler forwards `attachment_ids`.
2. Keep read-only `document://` compat branch for 30 days (sunset marked in code comment).
3. `multimodal-input.tsx`: switch to `apiClient.uploadAttachment`, delete `document://` construction.
4. Integration test: paste screenshot → LLM receives image.

### Phase 4 — Task board attachments
1. Add attachment UI to `create-task-dialog.tsx`.
2. Wire `board_tasks.attachment_ids` through POST → executor → `build_context()`.
3. Integration test: create task with image → executing agent's LLM call has image part.

### Phase 5 — Channel attachments
1. For each channel adapter in `orchestrator/channels/`: on inbound media, call `AttachmentStore.put()` with the bytes, attach ids to the chat message payload.
2. Integration test per adapter: inbound image → chat message carries attachment_id → agent LLM call has image part.

### Phase 6 — Sunset compat
1. After 30 days of Phase 3: delete the `document://` compat branch.
2. CI grep check: `rg "document://" frontend/ orchestrator/` returns zero.

---

## 8. Testing

### 8.1 Unit (≥90% on new modules)
- `AttachmentStore.put/get/open/sign_url/delete` — happy path, oversized, bad MIME, bad magic bytes, expired, wrong workspace.
- `extract.py` — pdf, docx, xlsx, csv, txt, code files, unknown type.
- `AttachmentResolver.resolve` — empty, all-image, all-doc, mixed, non-vision model raises, text budget enforced, expired attachment handled.
- `_inject_parts_into_last_user_message` — no user message (no-op), user message with string content, user message already parts-list.
- Vision check — known vision model, known text-only, unknown (conservative false).
- Validation — empty, filename traversal, NULL bytes, oversized.

### 8.2 Integration
- **Chat:** upload image → send → assert OpenRouter request has `image_url` part.
- **Mission:** create with mixed attachments → `decompose()` → planner LLM call has multimodal content → `mission_tasks.attachment_ids` populated → executing agent's LLM call has multimodal content.
- **Task:** create task with image → executor invokes agent → LLM call has multimodal content.
- **Channel:** inbound image via adapter → `AttachmentStore.put()` called → chat message has attachment_id → agent LLM call has image part. One test per adapter.
- **S3 isolation:** upload via `/api/attachments` → object under `ephemeral-attachments/`. Upload via `/api/documents/upload` → object under `documents/` prefix. Lifecycle rule exists on ephemeral prefix, not documents prefix.

### 8.3 E2E (manual)
1. **Regression:** 2026-04-09 brand mission — upload PDF + brand screenshots → mission plans successfully → research task output references visual elements.
2. Chat: paste Figma screenshot → "describe the layout" → agent responds with specifics.
3. Task: attach a diagram → agent's report references it.
4. Channel: send image via one channel adapter → agent responds using the image.
5. Expiry: upload, fast-forward lifecycle, assert chat history shows `[expired]` chip and does not 500.

### 8.4 Must not break
- `DocumentManager` / `/api/documents/upload` / RAG / Knowledge Base — **identical behavior**.
- Documents page upload flow — **identical**.
- `modules/rag/ingestion/multimodal/processors.py` — **not touched**.
- Existing missions with document-id attachments continue to work via compat branch.
- Existing chat messages with `document://` parts continue to render via compat branch.
- Workspace isolation: user A cannot reference user B's `attachment_id`.

---

## 9. Risks & mitigations (enforced via CI greps)

| Risk | CI check |
|------|----------|
| Some caller builds `ContentPart` / `image_url` outside the resolver | `rg -l "image_url.*url\|ContentPart\(" orchestrator/ \| grep -v "modules/attachments/resolver.py\|modules/context/service.py" && exit 1` |
| Attachments code imports RAG / DocumentManager | `rg -l "DocumentManager\|modules.rag.ingestion\|document_chunks" orchestrator/modules/attachments/ && exit 1` |
| A new `supports_vision` registry gets introduced | `rg "supports_vision" orchestrator/core/llm/ \| grep -v model_registry.py && exit 1` |
| `document://` scheme lingers past Phase 6 | Phase 6 check: `rg "document://" frontend/ orchestrator/ && exit 1` |
| Someone re-creates `/api/missions/upload` | `rg "/api/missions/upload" frontend/ orchestrator/ && exit 1` (post-Phase 2) |

Additional risks:
- **Signed URL expiry mid-request:** 15-minute expiry; LLM calls rarely exceed this. Fallback: inline base64 for images < 500KB.
- **Non-vision model receives images:** `VisionNotSupportedError` raised before the LLM call. Frontend shows model-switch CTA.
- **Large docs blow context:** `text_budget_tokens` cap in resolver. Image count cap per request (default 20).
- **History rendering after TTL:** chips show `[expired: filename]` from cached inline metadata. No 500.
- **Workspace cross-contamination:** S3 prefix includes `workspace_id`; `AttachmentStore.get()` enforces prefix match.

---

## 10. Success criteria

1. Brand mission runs with PDF + images; planner output references visual elements.
2. Chat screenshot-paste → LLM describes it accurately.
3. Task board Create Task has working attachment UI.
4. Channel adapter inbound image → agent sees it.
5. Mission tasks dispatched from planner carry `attachment_ids` — executing agents see what the planner saw.
6. **Zero attachments in the RAG index.** Verified: upload image via `/api/attachments` → query vector store for that workspace → not found.
7. **DocumentManager / Documents page / `modules/rag/` unchanged.** Regression-tested.
8. Zero call sites of `_fetch_attachment_contents`, `_resolve_file_parts`, `/api/missions/upload`.
9. `AttachmentResolver` is the only code constructing `image_url` parts (CI grep passes).
10. `supports_vision` referenced only in `model_registry.py` (CI grep passes).
11. S3 lifecycle rule active and scoped to `ephemeral-attachments/` only.
12. Unit coverage ≥90% on new modules.
13. `document://` compat branch deleted 30 days after Phase 3.

---

## 11. Open questions

1. **Inline base64 vs signed URL for small images.** Default: signed URLs > 500KB, base64 below. Revisit after measuring latency.
2. **Per-task attachment inheritance.** Planner output with empty `attachment_ids` on a task = inherit mission-level. Non-empty = replace. Confirm before Phase 2 ships.
3. **TTL length.** 7 days is a guess. Configurable via env var.
4. **Channel adapter media sources.** Each adapter (Slack, LINE, Email, Discord, etc.) has different media delivery semantics. Phase 5 needs one integration test per adapter — confirm adapter list before Phase 5.

---

## 12. Out of scope follow-ups

- **PRD-128:** Auto-route to vision-capable model when user attaches images.
- **PRD-129:** Thumbnail generation + gallery view.
- **PRD-130:** Configurable per-workspace retention.
- **PRD-131:** Video + audio attachments.
- **PRD-132:** Fix broken RAG multimodal OCR (`modules/rag/ingestion/multimodal/processors.py`).

---

## 13. Cross-references

| System | Touched? |
|--------|----------|
| `modules/context/service.py` (`ContextService.build_context`) | Extended — two new kwargs + post-processing step |
| `core/llm/model_registry.py` (`supports_vision`) | Read only — reuses existing field at line 39 |
| `modules/coordination/planner.py` + `services/coordinator_service.py` | Refactored — deletes legacy path |
| `orchestrator/channels/` | Extended — adapters gain attachment handling |
| `modules/rag/` (DocumentManager, ingestion, multimodal processors) | **Not touched** |
| `/api/documents/upload` + Documents page | **Not touched** |

---

## 14. Appendix — verified file references

- `orchestrator/api/missions.py:768-831` — `/api/missions/upload` (delete target)
- `orchestrator/modules/coordination/planner.py:67-211` — `_fetch_attachment_contents` (delete target)
- `orchestrator/services/coordinator_service.py:820-821` — second caller of `_fetch_attachment_contents` (delete target)
- `orchestrator/consumers/chatbot/service.py:440-490` — `_resolve_file_parts` (delete target)
- `orchestrator/consumers/chatbot/service.py:1794, 1986` — call sites of `_resolve_file_parts`
- `orchestrator/consumers/chatbot/prompt_analyzer.py:291` — stale comment reference
- `orchestrator/modules/context/service.py:44` — `ContextService.build_context` (extend here)
- `orchestrator/modules/context/sections/base.py` — `BaseSection`, `SectionContext` (reference only, not modified)
- `orchestrator/modules/context/result.py` — `ContextResult` (reference only)
- `orchestrator/core/llm/model_registry.py:39` — existing `supports_vision` field (reuse)
- `orchestrator/channels/` — channel adapters (extend per Phase 5)
- `frontend/components/chatbot/multimodal-input.tsx:136-144, 188` — upload call + `document://` construction (modify)
- `frontend/lib/api-client.ts:1621` — existing `uploadDocument()` (leave alone)
- `frontend/components/missions/create-mission-modal.tsx:42-57` — mission modal (modify)
- `frontend/components/activity/board/create-task-dialog.tsx` — task dialog (add upload UI)
