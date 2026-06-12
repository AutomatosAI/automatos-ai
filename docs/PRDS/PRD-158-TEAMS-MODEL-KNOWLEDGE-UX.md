# PRD-158 — Teams Model & Knowledge UX (WS-3)

**Chain:** Block B, branch `ralph/prd-158-teams-knowledge-ux` from main after PRD-157 merges (uses its filter builder). Size **M**.
**Source:** report §2.1 team findings; D1 BINDING.

## Overview

Finishes the locked Teams decision end-to-end: a real (small) teams entity, dropdowns instead of free-text, server-side team filtering on the knowledge page, and team management on documents after upload.

## Binding amendments

D1, Q2 default: `teams` table backfilled from existing `agents.team` strings; agents single-team for now (schema leaves room for multi-team later); documents stay multi-team via `team_access`, Q3: team selector is a pure filter for humans (no human RBAC here), Q5: cloud-sync connections get a per-connection default team (default empty/public), Q7: widget/SDK docs surface rebuilt on the real schema + vector search IF prod-schema check confirms drift (S5).

## User Stories

### S1: Teams table + API
Migration: `teams(id, workspace_id, name, normalized_name unique-per-workspace)`; backfill DISTINCT from `agents.team` + `documents.team_access` (lowercased, Q-default normalization); `GET/POST /api/teams`; org-chart reads join the table; ALL writes normalize through one helper (`core/team_access.py` extended — no second normalizer).
**Acceptance:**
- [ ] Backfill migration test on seeded mixed-case data ('Support'/'support' → one team)
- [ ] Reversible migration; alembic single head
- [ ] org-chart returns the same teams as `/api/teams` (consistency test)

### S2: Upload + cloud-sync team dropdowns
Replace free-text chips in `document-upload.tsx:448-482` and `upload-provider-modal.tsx:204-232` with a multi-select dropdown from `/api/teams` (+ inline "create team" for admins); cloud-storage connection settings gain a default-team picker applied to synced docs.
**Acceptance:**
- [ ] Upload with 2 teams persists normalized `team_access` (API test)
- [ ] No free-text team path remains (grep gate — phantom-domain bug dead)
- [ ] Cloud-synced doc inherits connection default (test)
- [ ] dev-browser verify both dialogs

### S3: Knowledge page team filtering
Server-side `team=` param on `GET /api/documents` (through the PRD-157 filter builder); team selector promoted from LocalStorageBrowser to the page header as a primary filter; per-team document counts; agent-eye-view toggle ("view as team X") for debugging scope.
**Acceptance:**
- [ ] `?team=` filters server-side with counts (API test, >100-doc fixture proving it's not the client-side ≤100 hack)
- [ ] Selector visible without entering Library cards — dev-browser verify

### S4: Team management after upload
Team chips + editor in document-details modal; bulk team assignment over the (PRD-154-fixed) PATCH/bulk endpoints; team badges on document rows.
**Acceptance:**
- [ ] Edit teams on an existing doc round-trips (API + dev-browser)
- [ ] Bulk assign 10 docs (test)

### S5: Widget/SDK docs surface — schema truth
Verify prod schema for `documents.title/content/updated_at` drift (Q7). If drifted: reconcile with a migration. Rebuild `api/widgets/docs.py` queries on the real schema + the PRD-157 retrieval path (vector search, not ILIKE).
**Acceptance:**
- [ ] Drift check documented in PR; queries match `core/models/core.py`
- [ ] Widget docs search returns results on seeded data (test, was impossible before)

## Non-Goals

Human RBAC by team, multi-team agents, SDK-key team locks (future PRD if needed — schema leaves room).

## Success Metrics

- Zero free-text team writes anywhere; one normalizer.
- Support-team agent demo: sees its docs, blind to engineering's — and the human can verify via the eye-view toggle.

## Testing

Migration tests, tenancy-matrix extension for `team=`, vitest for the selector, dev-browser verifications. Full suite + contract green.
