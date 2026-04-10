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
- [ ] **US-004**: Auto-register on workspace_write_file + report_service.create_report
- [ ] **US-005**: Backfill script for existing agent_reports

### Phase 1: Frontend

- [ ] **US-006**: Create useDeliverables React Query hooks (infinite query)
- [ ] **US-007**: Build DeliverableCard component
- [ ] **US-008**: Build FilterBar component
- [ ] **US-009**: Build DeliverablePreview slide-over
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
