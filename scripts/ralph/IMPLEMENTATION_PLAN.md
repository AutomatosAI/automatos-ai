# PRD-129: Workspace Outputs Hub — Implementation Plan

## Overview

Transform the VS Code-style Workspace tab into a consumer-friendly **Gallery** of agent deliverables: reports, images, documents, code, slides. New `deliverables` table + `DeliverableService` + `/api/deliverables` endpoints + Gallery/Explorer/Activity view toggle. Auto-register deliverables when agents write files or submit reports.

## Architecture

```
agents write files (exec_workspace) ──┐
report_service.create_report() ───────┼──► DeliverableService.register() ──► deliverables table
backfill script (agent_reports) ──────┘                                              │
                                                                                      ▼
                                                                /api/deliverables (list/get/stats/delete)
                                                                                      │
                                                                                      ▼
                                                                  frontend GalleryView (cards + preview)
```

## Key Files

| File | Purpose |
|------|---------|
| `orchestrator/alembic/versions/prd129_deliverables.py` | Migration: deliverables table + indices |
| `orchestrator/services/deliverable_service.py` | DeliverableService with register/list/get/stats/soft_delete |
| `orchestrator/api/deliverables.py` | REST endpoints for list/get/stats/delete |
| `orchestrator/modules/tools/execution/exec_workspace.py` | Auto-register on workspace_write_file |
| `orchestrator/services/report_service.py` | Wire create_report → DeliverableService |
| `orchestrator/scripts/backfill_prd129_deliverables.py` | Backfill existing agent_reports |
| `orchestrator/main.py` | Register deliverables router |
| `frontend/hooks/use-deliverables-api.ts` | React Query hooks |
| `frontend/components/workspace/gallery-view/*` | Gallery UI (card, filter-bar, preview, index) |
| `frontend/components/workspace/workspace-view-toggle.tsx` | Gallery/Explorer/Activity toggle |
| `frontend/app/workspace/page.tsx` | Wire toggle + default to Gallery |

## Tasks

### Phase 0: Backend Foundation

- [x] **US-001**: Create deliverables table migration (alembic) — `prd129_deliverables.py`, standalone (down_revision=None), BIGINT size, unique partial index for idempotent register, import-test OK. DB not running locally — actual `alembic upgrade` will run on deploy.
- [x] **US-002**: Implement DeliverableService (register/list/get/stats/soft_delete + 26 unit tests, all passing). Uses `ON CONFLICT (workspace_id, file_path) WHERE deleted_at IS NULL` for idempotent register. Extra JSONB merged via `||` on conflict. Images skip file read (returns content_url). Never calls WorkspaceClient during register.
- [x] **US-003**: Add /api/deliverables endpoints + integration tests — `api/deliverables.py` (list/stats/get/delete), registered in `main.py`, 12 integration tests in `tests/api/test_deliverables_api.py` all passing. Stats route declared before `/{id}` to avoid path shadowing. Workspace isolation covered by asserting service is always constructed with `ctx.workspace_id`, ignoring `X-Workspace-ID` header.
- [x] **US-004**: Auto-register on workspace_write_file + report_service.create_report — `exec_workspace.py` now calls `_auto_register_deliverable()` after a successful write, plumbed `agent_id`/`caller_context` through `unified_executor.py`. `report_service.create_report()` mirrors into deliverables after DB insert. Skips non-registerable types (archive/audio/video). Failures are caught+logged and never break the write. 11 unit tests in `tests/modules/tools/execution/test_exec_workspace_deliverable.py` (all pass). Tests use sys.modules stubs for core.* / services.* to avoid pulling pgvector + DB config.
- [x] **US-005**: Backfill script for existing agent_reports — `orchestrator/scripts/backfill_prd129_deliverables.py`. Iterates `agent_reports` (LEFT JOIN agents for name), calls `DeliverableService.register(artifact_type='report')` per row. `source_type` = 'heartbeat' if `heartbeat_result_id` set or `report_type='standup'`, else 'task'. Idempotent via the US-001 unique partial index (ON CONFLICT updates in place). Flags: `--dry-run`, `--workspace-id <uuid>`, `--verbose`. Prints summary `Processed/Inserted/Updated/Skipped/Errors`. Exit code 1 if any errors. `--help` verified locally; full dry-run needs DB (runs on deploy). Config import is `from config import config` (matches repo convention, NOT `core.config`).

### Phase 1: Frontend

- [x] **US-006**: Create useDeliverables React Query hooks (infinite query) — `frontend/hooks/use-deliverables-api.ts`. Exports `Deliverable`, `FilterState`, `DEFAULT_FILTERS`, `useDeliverables` (useInfiniteQuery, PAGE_SIZE=24, offset-based `getNextPageParam`), `useDeliverable(id, includeContent)`, `useDeliverableStats()`, `useDeleteDeliverable()` mutation (invalidates `['deliverables', workspaceId]`). `date_range` translated to ISO `date_from` client-side ('today'/'week'/'month'). Workspace ID pulled from existing `useWorkspace()` hook — no separate `useWorkspaceId` exists in this repo; query keys still scoped by workspaceId so switching clears cache. React Query v4 API: no `initialPageParam` (v5-only) — `pageParam = 0` default via destructuring.
- [x] **US-007**: Build DeliverableCard component — `frontend/components/workspace/gallery-view/deliverable-card.tsx`. Memoized; ARTIFACT_ICONS/ARTIFACT_COLORS/SOURCE_ICONS maps (Lucide only, no emojis); image preview via `<img loading="lazy" object-cover>` when artifact_type='image' + preview_url, otherwise centered Lucide icon in colored chip; source badge (small icon in rounded pill) pinned top-right on preview; title line-clamp-2, agent · time-ago, file size below; `formatFileSize` returns '0 B' for 0 bytes (and '' for null/undefined to avoid empty line); uses `formatDistanceToNow` from date-fns; cursor-pointer + hover:border-primary/50 + hover:shadow-md; role="button" + keyboard Enter/Space support. Typecheck clean.
- [x] **US-008**: Build FilterBar component — `frontend/components/workspace/gallery-view/filter-bar.tsx`. Debounced search (local 300ms `useDebounce` hook, no shared one exists in repo), Type/Source/Date Select dropdowns using shadcn `Select`, Clear button shown only when `hasActiveFilters(filters)` is true, right-aligned total count with pluralization (`output`/`outputs`). Dropdowns use `'all'` sentinel value (Radix Select rejects empty-string values) and translate to `null` on change. Local `searchInput` state keeps typing responsive; `useEffect([debouncedSearch, filters.search, onFiltersChange])` pushes up only when different (no stale closure, no loop). External resets re-sync via a second effect on `filters.search`. Icons: Search/Calendar/FileType/Zap/X from lucide — no emojis. Typecheck not runnable locally (no node_modules) — imports verified manually against `@/components/ui/{button,input,select}` and `@/hooks/use-deliverables-api`.
- [x] **US-009**: Build DeliverablePreview slide-over — `frontend/components/workspace/gallery-view/deliverable-preview.tsx`. Uses shadcn `Sheet` (side=right, max-w-3xl, scrollable). Fetches via `useDeliverable(id, true)` with `enabled` gated on `open` so switching cards doesn't thrash. Header: title (line-clamp via leading-tight), agent · time-ago · artifact_type, Download button (asChild `<a>` with native `download` attr, points at `content_url || preview_url`), Open in Canvas button (router.push `/workspace?view=explorer&path=<encoded>`). Body dispatcher `PreviewBody`: image → `<img>` via `content_url`; code → Prism via `getLanguageFromPath()` (reuses same extensions as `code-block.tsx` — no `react-syntax-highlighter` dep because repo already standardises on Prism); report/markdown (artifact_type='report' OR file_type md/markdown) → `react-markdown` in `prose prose-sm dark:prose-invert`; other text → `<pre>` whitespace-pre-wrap; unsupported or `content_error` → `UnavailableContent` card with Download fallback. Summary (if present) shown in bordered card below content. Escape close handled by Radix + explicit `keydown` listener as belt-and-braces. `getLanguageFromPath` is exported for future reuse; recognises bare `Dockerfile`. Icons: Download/ExternalLink/FileWarning/Loader2 — no emojis. Typecheck not runnable (no node_modules) — imports verified against `@/components/ui/{sheet,button}`, `@/hooks/use-deliverables-api`, and Prism grammars already used by `components/chatbot/code-block.tsx`.
- [ ] **US-010**: Build GalleryView container with infinite scroll
- [ ] **US-011**: Build WorkspaceViewToggle and wire into /workspace page

## Constraints

- Column is `extra` not `metadata` (SQLAlchemy Base.metadata conflict)
- `file_size_bytes` is BIGINT (support >2GB)
- Soft delete via `deleted_at`
- UNIQUE `(workspace_id, file_path) WHERE deleted_at IS NULL` for idempotent re-registration
- `register()` does NOT hit WorkspaceClient during registration — size passed in from caller
- Registration failure MUST NOT break file write (try/except + log)
- All SQL uses `sqlalchemy.text()` with bound params
- Frontend uses Lucide icons only (no emojis)
- Default view is `gallery` (consumer-friendly)

## Quality Bar

- All new SQL parameterized (no string interpolation of user input)
- All method envelopes have `success` key
- Exceptions logged with `exc_info=True`
- Typecheck + tests pass for each story
- Backward compatible: existing Explorer behaviour preserved exactly
