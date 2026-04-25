# AUTOMATOS 0.2 — Success Metrics

**Purpose:** Machine-checkable and human-checkable exit criteria. Each wave has quantitative targets and one human test.

---

## 1. Scorecard (top-line)

| # | Metric | Baseline (2026-04-24) | Target (0.2 close) | How measured |
|---|---|---|---|---|
| 1 | API router files (`orchestrator/api/*.py`) | 103 | ≤25 | `ls orchestrator/api/*.py \| wc -l` |
| 2 | `include_router` calls in main.py | 103 | ≤30 | `grep -c include_router orchestrator/main.py` |
| 3 | SQLAlchemy `__tablename__` declarations | 109 | ≤75 | `grep -rc __tablename__ orchestrator/ --include=*.py` |
| 4 | Orphan tables (migration, no model) | 16 | 0 | diff from `/graphify db-report dead-tables` |
| 5 | `b_*_<date>` backup tables | 11 | 0 | DB query |
| 6 | Dead-routes report | unknown | 0 | `/graphify db-report dead-routes` |
| 7 | Dead-tables report | unknown | 0 | `/graphify db-report dead-tables` |
| 8 | Frontend `use-*-api*.ts` hook files | ~60 | ≤15 | `ls frontend/hooks/use-*-api*.ts \| wc -l` |
| 9 | Frontend duplicate-suffix hook files (`*-v1`, `*-v2`, `*-verified`, `*-simple`, `*-explorer`) | ~8 | 0 | file name scan |
| 10 | Frontend `app/` top-level routes | 40+ | ≤20 | `ls -d frontend/app/*/ \| wc -l` |
| 11 | React component files | 429 | ≤300 | `find frontend/components -name "*.tsx" \| wc -l` |
| 12 | `knip` unused exports | unknown | <10 | CI job |
| 13 | LOC Python in orchestrator | 228,297 | ≤200,000 | `find orchestrator -name "*.py" -exec wc -l {} + \| tail -1` |
| 14 | Alembic migration count (cumulative) | 94 | 103–110 | the 0.2 waves ADD migrations; not a reduction target |
| 15 | "Dead weight" LOC (unmounted + orphan) | ~2.3K | 0 | cross-audit |
| 16 | Canonical deliverables path writes per week | 0 | 100% of new outputs | telemetry |
| 17 | Legacy-path traffic (308 redirects) | — | <1% after 14 days per cut-over | log-relay / route metrics |

---

## 2. Per-wave exit criteria

### Wave 0 — Instrumentation
- [ ] `/graphify db-scan` runs < 60s against production.
- [ ] Code-to-DB walker: ≥80% of `text("...")` call sites produce edges on sample audit of 20 files.
- [ ] Three reports (dead-tables, dead-routes, consolidation-candidates) generate clean markdown.
- [ ] Pre-PRD-133b retrospective: `consolidation-candidates` surfaces `deliverables ↔ agent_reports` in top 5.
- [ ] Reports run in CI on PR that touches `orchestrator/alembic/` or `orchestrator/api/`.
- [ ] **Human test:** run `/graphify db-report dead-tables` on current main; verify 0% false positives on 20 flagged items.

### Wave 1 — Data-model collapse
- [ ] Orphan-table count (metric 4): 0.
- [ ] `b_*_<date>` count (metric 5): 0.
- [ ] Rename aliases (`orchestration_runs` → `runs`) dropped after one release.
- [ ] No migration reverted post-ship.
- [ ] Zero production errors attributable to schema changes.
- [ ] **Human test:** run `SELECT COUNT(*) FROM pg_stat_user_tables` on prod, confirm ≤75.

### Wave 2 — API-surface collapse
- [ ] Router count (metric 1): ≤25.
- [ ] Mount count (metric 2): ≤30.
- [ ] Dead-routes report (metric 6): 0.
- [ ] All 103 original routers accounted for (keep/merge/rename/delete) per [04-API-SURFACE.md §2](./04-API-SURFACE.md).
- [ ] 308-redirect telemetry shows <1% legacy-path hits after 14 days per cut-over.
- [ ] `main.py` mount block is declarative (auto-mount from `api/__init__.py`) — no manual ordering comments required.
- [ ] **Human test:** new engineer opens `orchestrator/api/` and lists the 10 domains without consulting docs.

### Wave 3 — Frontend surface collapse
- [ ] Hook count (metric 8): ≤15.
- [ ] Duplicate-suffix hooks (metric 9): 0.
- [ ] App top-level routes (metric 10): ≤20.
- [ ] Component files (metric 11): ≤300.
- [ ] Unused exports via `knip` (metric 12): <10.
- [ ] Four-tab shell: every route reachable from Goals / Deliverables / Knowledge / Agents in ≤2 clicks (for non-admin).
- [ ] Pixel-diff on key flows: ≤2% regression.
- [ ] **Human test:** user on `/` finds the four tabs and can navigate to "Deliverables → today's outputs" in ≤5 seconds.

### Wave 4 — Deliverables unification
- [ ] Every new agent output writes to `deliverables` (metric 16): 100%.
- [ ] Legacy writes (`artifacts`, `agent_reports.file_path`) stopped after 90-day dual-write window.
- [ ] `<DeliverableView />` renders all 8 MIME types ([07 §3](./07-DELIVERABLES-FLOW.md)).
- [ ] Grade → skill promotion flow shipped.
- [ ] **Human test:** an agent produces output in chat, in a mission, in a recipe run, in a plan run; all four land in the Deliverables tab within 5 seconds of creation.

### Wave 5 — Autonomous flow
- [ ] Unified `run` object serves all four kinds.
- [ ] Compose box auto-routes to correct mode.
- [ ] Run event stream is the single source of truth for activity displays.
- [ ] Human-gate pattern works end-to-end for mission approval.
- [ ] **Human test:** four-journey suite in [07 §10](./07-DELIVERABLES-FLOW.md) passes (chat→deliverable ≤60s, mission→deliverable ≤10min, recipe→recurring, plan→workspace configured ≤15min).

### Wave 6 — Skills & marketplace
- [ ] One-tab marketplace across five kinds.
- [ ] Workspace-template install path creates a `kind=plan` run.
- [ ] New vertical (Shopify, used as reference) installable from template with zero orchestrator code changes.
- [ ] **Human test:** install Shopify template in clean workspace; within 15 min of install, first Shopify-domain mission emits a deliverable.

### Wave 7 — Instrumentation-as-CI
- [ ] Nightly `/graphify db-scan` succeeds ≥95% of runs.
- [ ] PR gate fires on "new router/model without retired equivalent."
- [ ] Weekly close report posted to channel of choice.
- [ ] **Human test:** open a PR that adds a new router without removing one; CI fails with a clear message pointing to 04-API-SURFACE.md.

---

## 3. Running totals view (updated weekly by weekly report)

| Week | Dead tables remaining | Dead routes remaining | Routers | Hooks | LOC deleted cumulative |
|---|---|---|---|---|---|
| W0 (baseline) | 16 | unknown | 103 | 60 | 0 |
| W1 | … | … | … | … | … |
| W2 | … | … | … | … | … |
| …through close | 0 | 0 | ≤25 | ≤15 | ~10K |

Report to be generated by Phase 7.3 and appended here weekly.

---

## 4. Anti-metrics (things NOT to optimize)

- **Do NOT optimize LOC deleted as a vanity metric.** Some PRs are reorganization with zero net LOC change; they still count as progress.
- **Do NOT optimize test count.** Don't add tests just to hit coverage; Wave 1-3 don't change behaviour and existing tests should still pass.
- **Do NOT optimize for "zero redirects".** The 308 period is a feature, not debt.

---

## 5. Stop conditions (when to pause 0.2)

Halt the wave (not cancel — pause) if any of:

- A Wave 1 DROP causes a production 500.
- A Wave 2 consolidation causes a frontend regression unresolved within 1 day.
- The 14-day redirect telemetry shows >5% legacy-path hits — means Wave 3 didn't fully migrate, fix that before next Wave 2 phase.
- PRD-135 scanner produces >1 false-positive per 20 flagged items — means report is untrusted, re-tune before using it for Wave 1 decisions.

Resume after root-cause fix.

---

## 6. Final "0.2 closed" criteria

All of the following true for one full week:

1. Top-line scorecard (§1) all green.
2. Wave exit criteria (§2) all checked.
3. No open P0/P1 bug attributable to 0.2 changes.
4. Production post-mortem review: zero incidents traced to consolidation.
5. User sentiment (if measured): positive or neutral — no "where did X go?" support tickets for ≥7 days.
6. **Human sign-off:** Gerard reviews this doc + current state, confirms.

Then: tag release `v0.2.0`, cut release notes, close this plan.

---

## 7. Post-0.2 open questions (for 0.3)

Noted here so 0.2 doesn't sprawl:

- Parallel coordinator (PRD-82B)
- Budget hard-stop & governance (PRD-105)
- Learning loop v2 (skill examples auto-selected in context)
- Vector-store lineage in graphify
- Multi-workspace analytics roll-up
- SDK v1 (typed client for the canonical API)
- Billing & seats

---

**Cross-references:**
- [00-README.md](./00-README.md) — plan navigation
- [08-MIGRATION-PHASES.md](./08-MIGRATION-PHASES.md) — phase-by-phase PRs
- all other AUTOMATOS-0.2 docs for per-wave details
