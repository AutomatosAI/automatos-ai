# PRD-212: Phase 2 · Wave 4 — HARNESS rebuild · notifications digest · observability consolidation — P2-26

> **Status:** DRAFT — spec only, no build yet. Grounded @ `origin/main 9dd4c848a`. Three "second-order but protects every module" items from the deep review's Wave 4: the self-optimization loop is rebuilt on the eval substrate, the notification plane stops burying the signal, and the ten-router analytics sprawl collapses to one honest surface. **Builds ON prior waves — PRD-143 (obs su-lock), PRD-185 S4/S9/S10/S12, PRD-197 S4 — it does not undo them.**

**Phase:** Phase 2 — Module Deep-Review remediation · **Wave 4**
**Branch:** `feat/p2-w4-harness-notifications-observability`
**Dependencies:** Wave 0 (PRD-185) merged — esp. **S4** (`playbook_failed` event), **S9/S10** (eval substrate + first baselines), **S12** (`ws_router` health split); Wave 3 (PRD-197) **S4** (substrate-health tile). HARNESS (S1–S3) is **gated on real eval baselines existing** — see Sequencing.
**Framing (CLAUDE.md §3):** a mix — **Refactor/Consolidation** (obs sprawl → one), **Extension** (digest on top of events that now exist), **Rebuild** (HARNESS brain replaced on its existing skeleton).
**Build size:** M–L (HARNESS is the one L) · **Risk:** Medium — HARNESS touches a live loop; the obs stories delete routers. Recall/obs behaviour is held constant by construction (canonical surface already carries the tiles).
**Source:** `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 row **P2-26**; dossiers `planning-scheduling.md` (J7/E2), `notifications.md` (P1-digest), `observability-slos.md` (J5/C.4), `evals-learning.md` (T3).

---

## Overview

Three loops that guard *every* module are each half-built and quietly failing:

1. **The self-optimization loop is dead.** HARNESS is fully built (PRD-121/142 — a 5-phase weekly sweep with risk-tiered apply, a ledger, and rollback) yet has **produced 0 prescriptions in 15 months** across 21 workspaces. Its brain is three threshold heuristics with a literal placeholder proposed-value (`review_needed`) and **no eval anywhere in the loop** — it cannot know whether a change helped. And it writes baselines to volume *files*, so on a non-persistent deploy every week is "first run" and PRESCRIBE is never reached.
2. **The notification plane buries the signal.** It is honest about successes (~3,800 "ok" rows over two months) and was silent about the ~2.5-week 06-16 playbook outage. PRD-185 S4 gave failures an event type; nothing yet *aggregates* so failures surface and chatter collapses.
3. **The observability layer is a decade of duplication.** **Ten** analytics/statistics routers are mounted, two colliding on the same prefix, one dead — while the genuinely-good W10 primitive/SLO tiles already live on one canonical, correctly-scoped module.

**North Star** — *does this make Auto more autonomously capable and the agents' output higher-quality for clients?* Rebuilding HARNESS on evals is the point at which "self-optimization" stops being a dark switch and becomes a measured propose→measure→keep cycle; the digest is what makes a client-facing outage visible same-day; the consolidation is what makes the operator's cockpit one honest surface instead of ten. **No moat framing; no new capability** — arming, wiring, and a number.

**PILOT lens (locked):** 0 prescriptions is a **dead loop, not cold-start** — the loop never reaches its working phase on any deploy, so this is wiring, not "wait for traffic." Empty digest windows and un-fired SLOs during pilot are **not** failures. See `feedback-pilot-usage-not-quality-signal`.

---

## Current reality (grounded @ `9dd4c848a`; confirm by grep before editing — line numbers drift)

- **HARNESS is present and firing, but never prescribes.** `services/harness_service.py` runs a Sunday 02:00 sweep: `_phase_collect` → `_phase_diagnose` (rule-based ±10% deltas) → `_phase_prescribe` (3 heuristics, placeholder `review_needed`) → `_phase_apply` (risk-tiered) → `_phase_baseline`. Baselines persist as **files under `WORKSPACE_VOLUME_PATH`** (`config.py:554`) with a best-effort dual-write to the `harness_prescriptions` table (created by `alembic/versions/prd142_wave4_harness_store.py`). `HARNESS_ENABLED=true`, `HARNESS_SELF_MANAGEMENT_ENABLED=false` (`config.py:730,734`). `harness_prescriptions`: **0 rows ever**.
- **The eval substrate HARNESS should measure on already exists:** `orchestrator/evals/{retrieval_recall,memory_recall,operating_graph_uplift,graphiti_vs_baseline}.py` + frozen baselines `orchestrator/evals/baseline/kg_retrieval_2026-07.json` (PRD-185 S9/S10, PRD-188/198). `operating_graph_uplift.py` is the platform's one genuine decision-harness (train/test, exit-0, "the number is the deliverable", explicit do-not-flip) — the shape to reuse.
- **`playbook_failed` exists** in `core/services/notification_dispatcher.py:61` (`VALID_EVENT_TYPES`) and is dispatched from `api/recipe_executor.py:2102` (PRD-185 S4). The digest is the **consumer** on top. `auto_reporting.digest_frequency`/`quiet_hours` are already in the settings schema but **unread by the dispatcher**.
- **Ten analytics/statistics routers mounted** (`main.py:962-1029`): `analytics.py` (bare `/analytics`), `analytics_api.py` + `analytics_real.py` (**both** `/api/analytics` — first-include wins, the other shadowed), `analytics_charts.py`, `statistics.py`, `kpi_api.py`, `execution_history.py`, `database_analytics.py`, `llm_analytics.py`, `composio_analytics.py`. *(Dead `api/anthropic_client.py` — zero importers — is removed by **PRD-184 S2** in the llm-core dead-scaffolding batch, not here.)*
- **Prior waves already reshaped this surface — preserve, don't churn:**
  - **PRD-143 obs-lock:** `analytics_real.py`'s `router` is `dependencies=[Depends(require_super_admin)]` (`:44`) — platform/cross-tenant obs is **intentionally** super-admin-only. **Preserved unchanged.**
  - **PRD-185 S12:** `analytics_real.py` also exposes `ws_router` (`:51`) — workspace-admin own-workspace tiles (`/dashboard/success-rate`, `/slos`, `/errors/by-subsystem`, `/primitive-health`, `/activation/workspace`, `/deliverable-freshness`, `/commerce-integrity`). Consolidation builds on this split.
  - **PRD-197 S4:** the `/substrate-health` SLO tile already sits on `ws_router` (`:160`). Folded in, **not** duplicated.

---

## Findings → fix → story

| # | Finding (grounded) | Fix | Story |
|---|---|---|---|
| **HARNESS-0rx** | Baselines write to `WORKSPACE_VOLUME_PATH` files; PRESCRIBE needs a *previous* baseline → non-persistent volume ⇒ perpetual "first run" ⇒ **0 prescriptions / 15 months**. | Persist baselines to the DB (the `harness_prescriptions` store already exists); read last baseline from the DB, retire the file path. | **S1** |
| **HARNESS-blind** | Diagnose/prescribe is 3 threshold heuristics with placeholder `review_needed` and **no eval** — cannot verify its own changes. | Replace the brain with an eval-grounded reflective proposer (GEPA-style): propose → shadow-evaluate against `orchestrator/evals/` + frozen baselines → keep only if the number moves, else roll back. Keep the skeleton/apply/ledger. | **S2** |
| **HARNESS-tile-lie** | `/self-learning` tile reports "completed" over an empty store; actuation dark (`HARNESS_SELF_MANAGEMENT_ENABLED=false`). | Honest tile states; run the loop as a non-required exit-0 CI lane; self-management stays OFF until a soak with real prescriptions. | **S3** |
| **digest-buries** | ~3,800 "ok" rows bury signal; 06-16 outage was silent; `auto_reporting.digest_frequency`/`quiet_hours` unused by dispatcher. | A digest tick: roll "ok" events into one row/workspace/interval (honouring the existing config); elevate `playbook_failed`/`*critical` into a never-buried "Needs attention" surface same-day; add a terminal-event coverage check. | **S4** |
| **router-sprawl** | 10 analytics/statistics routers; two collide on `/api/analytics`; `analytics.py` bare mount. | Canonical = `analytics_real.py`; migrate live callers, grep-prove zero, **delete** the dups + bare mount; resolve the collision. Preserve su-lock + ws split + substrate tile. | **S5** |
| **ten-dashboards** | Sprawl backs overlapping dashboards; some domain routers may be genuinely distinct. | One Command Center **Health** tab wiring the *existing* canonical tiles; adjudicate which domain routers survive (§12). | **S6** |

---

## Stories (test-first — write the failing test, make it green; PURE tests mock the eval/DB/dispatch boundary)

### HARNESS rebuild (E2 — replace the brain, keep the skeleton) · _planning-scheduling J7/E2 · evals-learning T3_

**S1 · Baselines → DB (the 0-prescriptions root cause) — M**
**Files:** `services/harness_service.py` (`_phase_baseline:1072`, `_read_baseline:1222`); the `harness_prescriptions`/baseline DB store (prd142 migration exists — add a `harness_baselines` row if a distinct shape is needed, one migration).
**Test:** `test_baseline_roundtrips_through_db` writes a baseline on tick N and asserts tick N+1 reads it back from the DB (no `WORKSPACE_VOLUME_PATH` access); `test_second_tick_reaches_prescribe` asserts a non-first run with a DB baseline enters PRESCRIBE. Pure (mock the session).
**Notes:** This is the W4-S12 cutover the code itself documents (`harness_service.py:1054-1062`). Delete the volume-file baseline path in the same PR (no dual-write shim). One line un-sticks "first run forever."

**S2 · Eval-grounded proposer (GEPA-style propose→measure→keep) — L · _the self-optimization loop becomes real_**
**Files:** `services/harness_service.py` (`_phase_diagnose:518`, `_phase_prescribe:725`); a thin adapter to the `orchestrator/evals/` runners + frozen baselines. Adopt **DSPy/GEPA** (MIT) as the optimizer library where the target is prompt/description-shaped.
**Test:** `test_proposer_keeps_only_measured_wins` feeds a mocked eval delta and asserts a change that does **not** move the metric is rejected/rolled back and a change that does is kept; `test_no_placeholder_prescribed_value` asserts `review_needed` is gone; `test_proposal_measured_against_baseline` asserts the proposer scores against a frozen baseline, not a weekly aggregate. Pure (mock the eval boundary — no live LLM/Langfuse/DB).
**Notes:** Reuse the `operating_graph_uplift.py` honest-gate shape (shadow-evaluate → confirm the number moved → keep/rollback). Keep the existing risk-tiered apply/queue/ledger/rollback as the actuator — only the diagnose/prescribe brain is replaced. Measures on the offline `evals/` harnesses now; the PRD-185 S9 Langfuse score plane enriches later — **do not** stand up a parallel eval stack.

**S3 · Honest HARNESS tile + non-required eval lane — S**
**Files:** `api/harness.py` (`/self-learning` tile states, `:99-141`); a non-required CI lane matching `.github/workflows/nl2sql-eval-scheduled.yml` / PRD-185 S10.
**Test:** `test_harness_tile_reports_empty_honestly` asserts the tile shows "never run / no baseline / produced N (M awaiting review)" — never "completed" over 0 rows.
**Notes:** The propose-measure-keep cycle runs as an **exit-0** lane (the number is the deliverable). `HARNESS_SELF_MANAGEMENT_ENABLED` **stays false** — flipping it is a post-soak Gerard call (§12), not this PR.

### Notifications digest (P1 — the aggregation layer on events that now exist) · _notifications P1-digest_

**S4 · Notifications digest tick + failure elevation — M**
**Files:** `core/services/notification_dispatcher.py` (dispatch path + the digest batcher); the existing leader scheduler (the one hosting heartbeat/HARNESS jobs); the settings reader for `auto_reporting.digest_frequency`/`quiet_hours` (already in schema — wire them, add no new settings).
**Test:** `test_ok_events_roll_into_one_digest_row` (heartbeat/report "ok" events flush as one row/workspace/interval); `test_failures_bypass_digest_and_surface_same_day` (`playbook_failed`/`task_failed`/`*critical` never roll into the digest — they hit a "Needs attention" surface immediately); `test_digest_honours_existing_frequency_config`; `test_terminal_event_coverage_check` (an execution `done/failed` with no matching notification emits a critical). Pure (mock DB/dispatch).
**Notes:** Consumer of PRD-185 S4's `playbook_failed`. **No new table** — reads the existing `notifications` table; **no new config** — the frequency/quiet-hours fields exist. The coverage check is the safety net that catches the *next* silent break. Whether to time-box a Novu spike before a bespoke digest engine is a §12 call (dossier J-P2).

### Observability consolidation (J5 — ten routers → one canonical surface) · _observability-slos J5/C.4_

**S5 · Kill the analytics-router sprawl (canonical = `analytics_real.py`) — M**
**Files:** `main.py:962-1029` (mounts); losers `analytics_api.py`, `analytics.py` (+ bare `/analytics` mount), `statistics.py`, `kpi_api.py`, `execution_history.py`; `orchestrator/reports/route-manifest.json`. *(Dead `anthropic_client.py` → PRD-184 S2, not deleted here.)*
**Test:** per-caller behavioural test that each migrated endpoint returns equivalent results (fixture-fed, mocked); `test_no_duplicate_analytics_prefix` (the `/api/analytics` collision is gone); `test_analytics_sprawl_deleted` (source-grep guard, PRD-185 S5 shape — repoint in the same commit the route moves). Pure.
**Notes:** **Preserve the PRD-143 su-lock** — `analytics_real.py`'s `router` keeps `Depends(require_super_admin)`; cross-tenant/platform obs stays super-admin-only. **Build on** the PRD-185 S12 `ws_router` and the PRD-197 S4 `/substrate-health` tile — do not reverse them. Delete the losers in the same PR (no compat shim). Update the committed route-manifest (sorted + count bumped).

**S6 · One Health surface + domain-router adjudication — S**
**Files:** the Command Center "Health" tab wiring the *existing* `ws_router` tiles (`IsItWorkingStrip` already exists — wire, don't rebuild); the four domain routers (`llm_analytics`, `composio_analytics`, `analytics_charts`, `database_analytics`).
**Test:** `test_health_tab_reads_canonical_tiles` (SLOs + primitive-health + errors + freshness + substrate-health resolve from the one canonical `ws_router`, workspace-admin reachable); guard test that no surviving domain router re-mounts an `/api/analytics` path.
**Notes:** Reuse-first — the tiles and the strip exist; this is consolidation, not new UI. Which domain routers survive vs fold into canonical is **Gerard's call (§12)** — they are PRD-54/PRD-21 domain surfaces (LLM cost, Composio, charts, DB analytics), plausibly genuinely distinct.

---

## Sequencing

- **HARNESS is gated on real baselines existing.** S1 (baselines→DB) → S2 (proposer) → S3 (tile/lane) is a hard order. S2 depends on the PRD-185 S9/S10 eval substrate being live; the frozen `evals/baseline/*.json` are the seed baselines it measures against — if a target module has no baseline yet, HARNESS honestly reports "no baseline" (S3) rather than prescribing blind.
- **The obs consolidation (S5→S6) is independent** of HARNESS and the digest — parallel-safe, disjoint files. S6 follows S5 (canonical must exist before the Health tab points at it).
- **S4 (digest) is independent** — it consumes an event that already exists on main.
- Only shared file across themes is `config.py` (any new HARNESS/digest flag) — coordinate additions through the config module, never `os.getenv` inline.

## Verification (CI is the only gate — no local runs)

Per `feedback-no-local-servers`: write code + **pure** tests (mock the eval runners, Langfuse, DB session, and dispatch at the boundary — nothing hits a live service), commit, push, let the PR checks verify. The HARNESS propose-measure-keep lane (S3) and any eval run are **non-required** CI lanes that publish a number and **exit 0** (PRD-185 S10 posture — "the number is the deliverable", never gate CI red on it). When routers consolidate (S5/S6), update the **committed** `orchestrator/reports/route-manifest.json` (CI reads the committed manifest, never regenerates — hand-add sorted + bump the count). Migrations self-apply on boot.

## Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)

- No `os.getenv()` outside `config.py`; new flags go through the canonical config module.
- **No backward-compat shims** — delete the volume-file baseline path (S1), the losing routers + dead `anthropic_client.py` (S5), and the `review_needed` placeholder (S2) in the same PR that replaces them.
- No new tables where an existing one fits (digest reads `notifications`; baselines reuse the `harness_prescriptions` store); no parallel eval/obs stack — reuse the PRD-185 `orchestrator/evals/` runners and the `analytics_real.py` canonical surface.
- Immutable patterns; small focused functions; no silent `except` swallows.
- Canonical vocab: **Playbook** (not Recipe), **Command Center**, **Auto**, **Deliverable**, **Knowledge Graph**.
- Branch `feat/p2-w4-harness-notifications-observability`; commit, push, open a PR; CI is the gate.

## Success metrics (the definition of "the loop is real, the signal surfaces, the sprawl is one")

- **HARNESS produces ≥1 measured propose→measure→keep cycle** against a real baseline — a prescription that was kept *because a tracked number moved* (today: 0/0/0/0 ever); baselines live in the DB, not on the volume.
- **A failed playbook surfaces same-day** — a 06-16-class outage rolls up into the digest's "Needs attention" surface within the digest interval, not silently; "ok" chatter collapses to one row/workspace/interval.
- **Analytics routers N→1 consolidated surface** — 10 mounted → 1 canonical (+ any genuinely-distinct domain routers kept per §12), the `/api/analytics` collision resolved (dead `anthropic_client.py` removed by PRD-184 S2), **the PRD-143 su-lock preserved** and the PRD-185 S12 / PRD-197 S4 tiles intact and workspace-admin reachable.

## Open questions — Gerard's call (§12)

1. **Consolidation target module.** Recommendation: canonical = `analytics_real.py` (it already carries the su-locked `router` + the ws-admin `ws_router` + the substrate-health tile). Confirm — or name a different canonical.
2. **Which domain routers survive.** Fold `llm_analytics`/`composio_analytics`/`analytics_charts`/`database_analytics` into canonical, or keep them as distinct PRD-54/PRD-21 domain surfaces? Recommendation: keep llm/composio (genuinely distinct cost/integration domains), fold charts + database. Confirm.
3. **HARNESS prescription scope.** Which metrics is the proposer allowed to optimize — prompt/description shapes only (GEPA's sweet spot), or also model-tier and heartbeat-cadence config? Recommendation: prompt/description first (measurable, low blast-radius); config changes stay risk-tier-gated behind the dark `SELF_MANAGEMENT` switch.
4. **HARNESS self-management flip.** After what soak (how many weeks of real prescriptions with confirmed metric moves) do we flip `HARNESS_SELF_MANAGEMENT_ENABLED`? Recommendation: not in this PR — surface a soak criterion, decide after.
5. **Digest cadence + channel.** Reuse the existing `auto_reporting.digest_frequency` (hourly/daily) and route to the in-app bell — plus, time-box a **Novu spike** before committing to a bespoke digest engine (dossier J-P2)? Recommendation: build the in-app digest now; spike Novu in parallel as a keep-vs-adopt read, not a cutover.

---

*Traceability: `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 **P2-26**; dossiers `planning-scheduling.md` (J7/E2 — HARNESS 0 prescriptions/15mo, replace-the-brain-keep-the-skeleton), `notifications.md` (P1-digest — aggregate on `playbook_failed`), `observability-slos.md` (J5/C.4/F083 — ten-router sprawl → one, canonical `analytics_real.py`), `evals-learning.md` (T3/E2 — GEPA propose-measure-keep, baselines→DB, reuse `orchestrator/evals/`). All `file:line` re-confirmed @ `main 9dd4c848a` (10 routers mounted; `playbook_failed` live; `harness_prescriptions=0`; baselines on `WORKSPACE_VOLUME_PATH`; PRD-143 su-lock + PRD-185 S12 ws_router + PRD-197 S4 substrate tile present). Reuses PRD-185 (eval substrate, `playbook_failed`, ws health split), PRD-197 S4 (substrate tile), PRD-142 (`harness_prescriptions` store). PILOT lens; North-Star framed; no moat framing. Preserves the PRD-143 super-admin obs-lock.*
