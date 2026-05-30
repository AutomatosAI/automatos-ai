# Ralph Build Prompt — PRD-142 Wave 0 (Measurement First)

You are an autonomous build agent. Each invocation, you implement **ONE** unchecked user story from the plan, then exit. The loop runs you again on the next story.

## Hard lock

Your working directory is **`/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai`** on branch **`ralph/prd-142-wave0-measurement`**.

- NEVER `cd` to another worktree or clone.
- NEVER check out a different branch.
- NEVER `git push`, NEVER open or merge a PR. You commit to this local branch ONLY. A human reviews before anything merges.
- All file edits, all reads, all commits happen inside this checkout.
- If you accidentally drift, abort with `RALPH_ABORT: drifted out of checkout`.

## The PRD

- `scripts/ralph/prd-142-wave0.json` — the user stories. **The `passes` field is the single source of truth for progress.** There is no separate checkbox file.
- `docs/PRDS/PRD-142-WAVE0-MEASUREMENT.md` — full Wave 0 context: 5-metric source table, reuse map (file:line cited), out-of-scope list.
- `CLAUDE.md` at repo root — reuse-first mandate, canonical terms, clean-coding rules.

## What this PRD is

**PRD-142 Wave 0 is EXTENSION + INSTRUMENTATION, not new product.** It surfaces five real numbers answering "is the platform working?": error rate by subsystem, mission success rate, widget engagement, activation, per-primitive health. Every metric reuses an existing sink/table/endpoint. Read the reuse map in the PRD before writing anything — the burden is on you to justify why an existing surface is not enough (CLAUDE.md §2).

## Scope of THIS run — backend stories only

You MAY execute, in priority order:

- **US-001** — persist record_error to a queryable `error_events` sink (Alembic migration + ORM model + best-effort write path)
- **US-002** — `GET /api/analytics/errors/by-subsystem` aggregation endpoint (BLOCKED until US-001 lands the table)
- **US-004** — `GET /api/analytics/widget-engagement` read-only aggregation endpoint
- **US-005** — `GET /api/analytics/activation` endpoint
- **US-006** — `GET /api/analytics/primitive-health` rollup endpoint

OUT OF SCOPE — do NOT start these; a human drives them:

- **US-003** — already DONE (`passes: true`). Skip it.
- **US-007** — frontend dashboard assembly. Needs browser verification; a human supervises it. Do NOT touch `frontend/`.
- **US-008** — operational live-verification gate. Human-only.

**Completion condition:** when US-001, US-002, US-004, US-005, US-006 all have `passes: true`, write the completion commit and emit `RALPH_COMPLETE`. Do NOT attempt US-007 or US-008.

## Canonical terminology (CLAUDE.md §10)

- **Mission** (never "Workflow"/"Job") in all user-facing labels and metric names.
- **Playbook** (never "Recipe"). **Deliverable** (never "Output"/"Artifact"). **Knowledge Graph** (never "Business Graph").

## TDD is mandatory (testing.md)

For every story: **write the test FIRST (RED), watch it fail, then implement (GREEN).** The story's acceptance criteria name the exact test file and test functions — create that file, make those tests real, and do not weaken an assertion to make it pass.

## 4-phase loop

### Phase 1 — Orient

1. Read `scripts/ralph/prd-142-wave0.json`. Among {US-001, US-002, US-004, US-005, US-006}, find the **first** (lowest `priority`) story with `passes: false`.
2. If all five are `passes: true`, write the completion commit (Phase 4) and emit `RALPH_COMPLETE`.
3. Read that story's `acceptanceCriteria` AND `notes`. The notes carry the reuse decision and the "do NOT touch X" boundaries — obey them literally.
4. Run `git status` and `git log --oneline -10` to confirm a clean tree and recent history. If the tree is dirty with work that is not yours, STOP and emit `RALPH_ABORT: dirty tree`.

### Phase 2 — Implement ONE story

- Read existing code first. Use Grep/Glob aggressively. The PRD reuse map cites exact files — `core/utils/exception_telemetry.py` (record_error), `core/models/widget_event_log.py` + `modules/widgets/telemetry.py` (widget pattern), `api/analytics_real.py` (the OrchestrationRun union + RunState enum), `api/heartbeat.py` (heartbeat_results). Reuse them.
- Make the smallest change that satisfies the acceptance criteria. Do not add fields, endpoints, or tables the story does not ask for.
- **Every new endpoint is workspace-scoped** via `ctx = Depends(get_request_context_hybrid)` and filtered by `ctx.workspace_id`, EXCEPT US-005 activation, which is an intentional platform-level cross-workspace rate. Each endpoint story must add a tenant-isolation test.
- **US-001 migration must be online-safe:** new table only, no backfill, no `NOT NULL` added to an existing large table, no data migration. Follow the `widget_event_log` / single-table-JSONB ORM pattern with `extend_existing=True`. Do NOT change `record_error`'s keyword-only signature or its never-raises contract — the sink write is best-effort (try/except, rollback, swallow); the `automatos.errors` logger emit stays unchanged.
- Add new endpoints to an EXISTING analytics router (`api/analytics_real.py` or `api/analytics.py`). Do NOT create a new router file.
- If you delete a surface the story replaces, delete it cleanly — no `_legacy_` shims (CLAUDE.md §4).

### Phase 3 — Validate

```bash
# 1. Syntax + import health
python3 -m py_compile $(git diff --name-only --diff-filter=AM HEAD | grep '\.py$')
cd orchestrator && python -c "from main import app; print('import OK')"

# 2. The story's own tests (named in its acceptanceCriteria) must pass
cd orchestrator && python -m pytest tests/<the story's test file> -v

# 3. The grep gates in the acceptanceCriteria must return what the AC says (usually ZERO)
```

All must pass. For US-001 also confirm the Alembic migration is syntactically valid and has a non-empty `downgrade()` (`python3 -m py_compile orchestrator/alembic/versions/<new>.py`). Do NOT run the migration against a live DB — a human applies migrations.

If validation fails:
- If a quick honest fix is obvious, apply it.
- Otherwise revert your changes (`git checkout -- .`) and exit with a commit message starting `BLOCKED:`. Don't half-ship, and never weaken a test to go green.

### Phase 4 — Flip passes + Commit + Exit

1. In `scripts/ralph/prd-142-wave0.json`, set the finished story's `"passes": false` to `"passes": true` and append a one-line completion note to its `notes` (what landed, commit-worthy facts). Keep the JSON valid.
2. Stage the relevant files **by name** (never `git add .`). Skip `.env`, anything in `archive/`, anything you did not touch.
3. Commit with this format:

   ```
   feat(prd-142): US-XXX — <one-line description>

   <2-4 line body: what was added/reused and why; cite the reused surface>

   Story: scripts/ralph/prd-142-wave0.json US-XXX
   PRD: docs/PRDS/PRD-142-WAVE0-MEASUREMENT.md
   ```

4. If this was the **last** of the five backend stories (all now `passes: true`), instead end the body with:

   ```
   PRD-142 Wave 0 backend complete (US-001/002/004/005/006). US-007 frontend + US-008 live gate remain for human.

   RALPH_COMPLETE
   ```

5. Exit. Do not loop into the next story yourself — the outer loop re-invokes you.

## Project conventions (do not violate)

- NO `os.getenv()` outside `orchestrator/config.py`.
- NO hardcoded URLs / API keys / tokens / magic metric values.
- SQLAlchemy: use `text()` with bind params, never f-string SQL. No divide-by-zero (rate is 0 when denominator is 0).
- Pydantic response schemas live in `orchestrator/api/schemas/`.
- Use the canonical `RunState.COMPLETED.value` enum (core/models/orchestration_enums.py), not the literal string, where it exists.
- Reuse the `ApiResponse`/envelope pattern the existing analytics endpoints already use.

## Anti-patterns (will be reverted on review)

- Faking a metric value (a hardcoded number with no DB source) — Wave 0 exists to DELETE fakes, not add them. Omit a field rather than fake it.
- Creating a new router file, a new frontend page, or touching `frontend/` in this run.
- Adding a second event-writer or changing `modules/widgets/telemetry.py` / the heartbeat writer (US-004/US-006 are READ-only).
- Changing `record_error`'s signature or making it raise (US-001).
- `# type: ignore` / weakening an assertion to make checks pass.
- Adding emoji to source files; writing a feature README unless the story requires it.
- `git push`, opening a PR, or merging. Branch-local commits ONLY.

## When in doubt

- Re-read the story's `notes` field and the PRD reuse map.
- Read `CLAUDE.md` at repo root.
- Search before you build. Smaller diff > bigger diff. Reuse > build.
- If a metric has no real source, report the honest gap (`unknown`/omit) — never invent coverage.

Begin Phase 1.
