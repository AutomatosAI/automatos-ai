# PRD-167 — Document Templates Block Editor (WS-12)

**Chain:** Net-new track, branch `ralph/prd-167-template-editor` from main after PRD-156 (rides its sandboxing). Size **L**.
**Source:** report §2.10; D2 BINDING.

## Overview

Templates a non-technical user can build: Notion-style blocks, variable chips resolved from profiles/workspace/brand, a brand kit, live preview — JSON stays the storage format underneath, rendering to PDF (WeasyPrint) and DOCX (python-docx).

## Binding amendments

D2, D11 (LLM-assisted generation optional, not required path), Q71 default: DOCX via python-docx compiled from the same block tree, Q-defaults: global shared starter templates with copy-on-customize; generated documents register as deliverables; agent tools get list/get_schema (+ create behind a flag).

## User Stories

### S1: Editor-library evaluation (time-boxed)
The OSS research pass never ran (session limit). Time-boxed spike: **BlockNote vs Plate vs Puck** against requirements (custom variable-chip inline node, controlled JSON output, React 18, MIT/Apache license, table + image blocks). Decision memo in the PR; loser deps never enter package.json.
**Acceptance:**
- [ ] Memo with scored matrix + chosen library; spike branch artifacts deleted

### S2: Block schema + storage
`blocks` JSONB on the existing template tables (no new table — extend; repo rule); block types: heading, text, table, image/logo, variable, page-break, section; bidirectional mapping legacy-JSON → blocks for the seed templates (migrate seeds, delete the legacy-only render path after parity).
**Acceptance:**
- [ ] Schema validation suite (malformed blocks rejected with field-level errors — no silent swallow)
- [ ] All seed templates migrated + render byte-comparable PDFs (golden files)
- [ ] Reversible migration; single alembic head

### S3: Variable resolution service
`{{user.*}} / {{company.*}} / {{brand.*}} / {{date.*}}` resolver + `GET /api/documents/variables` (drives editor chips); unresolved-variable policy: render-time error list, not blanks; resolution respects the requesting user's profile (auth-provider-agnostic — PRD-150).
**Acceptance:**
- [ ] Resolver unit suite incl. missing-field policy
- [ ] Endpoint returns the full catalog with sample values (test)

### S4: Brand kit
Workspace brand kit: logo upload (S3 path per platform storage), colors, fonts → exposed as `{{brand.*}}` + applied to PDF/DOCX styles; replaces ALL hardcoded Automatos branding in render paths (grep gate).
**Acceptance:**
- [ ] Branded PDF golden test (logo embedded, palette applied)
- [ ] No hardcoded brand strings/colors remain in renderers

### S5: Editor + preview UX
Block editor page (chosen library) with chip insertion, image/logo block wired to brand kit; live preview pane (debounced server render or client approximation + render-on-demand); template gallery: starters, copy-on-customize, thumbnails.
**Acceptance:**
- [ ] Create → insert variables → add logo → preview → save → render PDF+DOCX, as a non-technical flow — dev-browser verify end-to-end
- [ ] Strict validation surfaces inline errors — dev-browser verify
- [ ] vitest for block-tree transforms (≥80% on the new module)

### S6: Agent + deliverable integration
`platform_list_templates` / `platform_get_template_schema` tools (3-file pattern); `generate_document(template_id, data)` produces a deliverable with source attribution (hooks PRD-164 S3 flywheel when merged — feature-flag otherwise); rendered docs registered as deliverables.
**Acceptance:**
- [ ] Auto fills a template from chat and the deliverable appears (integration test + dev-browser)
- [ ] Reachability gate green

## Non-Goals

Collaborative editing, template marketplace, arbitrary HTML blocks (SSTI surface stays closed — PRD-156 S4 sandbox is load-bearing), email templates.

## Success Metrics

- Non-technical flow (S5) completable in < 5 minutes without docs.
- 100% of seed templates on the block format; legacy render path deleted.
- At least one agent-generated branded document in pilot week.

## Testing

Golden render suite (PDF/DOCX), schema validation suite, resolver suite, vitest transforms, dev-browser e2e. SSTI tests from PRD-156 re-run against the editor path. Full suite + contract green.
