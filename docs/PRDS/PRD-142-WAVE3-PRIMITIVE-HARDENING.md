# PRD-142 Wave 3 — Primitive Hardening (Make Each Primitive "Rock Solid")

> **Parent:** `PRD-142-CORE-DESIGN-REVIEW.md` §12 (Wave 3 + Wave 3R) and §13 (per-subsystem DoD).
> **Design companions:** `docs/architecture/GUARDRAILS.md` §H (the 7-point DoD this wave asserts), `docs/architecture/BRAIN-BLUEPRINT.md` §3 (per-primitive contracts), `docs/architecture/PLAYBOOK-ENGINE-DESIGN.md` (3R), `docs/architecture/KNOWLEDGE-GRAPH-CANONICAL.md` (moat boundary).
> **Status:** Build-PRD — drafted 2026-06-05. **Gate satisfied:** Wave 2 (Test Net) is **merged to `main`** (PR #421, the J1–J10 backbone + gap regressions + real-DB fixture + the blocking `orchestrator-tests`/`ioc-scan` CI gate). The net that guards this wave now exists and is enforced.
> **Type:** Hardening **under green** — consolidate / refactor / harden existing primitives. **Zero rewrites** (locked, §16). **No new features, primitives, endpoints-for-their-own-sake, or LLM providers.**
> **Verified against:** `origin/main` @ `86c5957ac`, code + doc reads 2026-06-05.
> **Depends on:** Wave 2 (merged) — every story in this wave is refactor-under-test; the Wave 2 net is the safety harness.
> **Reuse-first** per `CLAUDE.md` §2 / §5 and `GUARDRAILS.md` B1/B2.
> **Ralph config (authored 2026-06-06):** `scripts/ralph/prd-142-wave3.json` (14 stories, branch `ralph/prd-142-wave3-primitive-hardening`) + `scripts/ralph/PROMPT_build_prd142-wave3.md` (build prompt, agent scope) + `scripts/ralph/loop-prd142-wave3.sh` (runner). Mirrors the Wave 0 three-file pattern. Agent runs the 12 backend-safe stories; **W3-S3 (FE tile), the delete + FE-repoint half of W3-S12 (3R), and W3-S14 (prod migration) are human-gated** and excluded from the autonomous loop.

---

## 1. The founding question for Wave 3

- **Wave 0** answered *"can we measure it?"* — yes (the Command Centre vitals are live, 4 of 5 tiles).
- **Wave 1** answered *"can we stop the bleeding?"* — yes (durable execution, Mem0 async, idle-tx, honest errors).
- **Wave 2** answered *"can we **prove** it stays working and block a regression at the door?"* — yes (J1–J10 + gap regressions + a required CI gate).
- **Wave 3** answers *"is each **primitive** actually rock solid?"* — does each of the eight meet its contract, fail **visibly**, survive restart, stay tenant-isolated, keep a single source of truth, and show a **green tile** that says so right now?

Today, no. We have a **net** (Wave 2) but the primitives underneath still carry the velocity debt the Core Design Review named: a 2,300-line chat god-file, **two parallel tool loops** (G6), `os.getenv` scattered outside `config.py` (G7), a Playbook engine that **dies fire-and-forget on restart**, and Mission retries that **don't feed the verifier's critique back**. Wave 3 hardens each primitive **under the green net** — the net catches regressions while we refactor, so consolidation stops being hope.

**Goal:** all eight primitives at the `GUARDRAILS.md` §H Definition of Done, each lighting a **green per-primitive health tile**, with the structural debt the DoD forces (god-file splits, one tool loop, config discipline, durable Playbooks) cleaned up as we go — and the dead paths we replace **deleted in the same wave**.

---

## 2. What Wave 3 **is** — and is **not**

**Is** (backend primitive hardening, under the Wave 2 net):
- Each of the **eight primitives** — chat, memory, RAG, NL2SQL, graph, missions, playbooks, channels — brought to the **`GUARDRAILS.md` §H 7-point DoD** plus its **`BRAIN §3.x` contract**.
- The cross-cutting structural fixes the §H DoD *forces*: **G6** (converge the two tool loops onto one executor — A2), **G7** (sweep `os.getenv`/`os.environ` outside `config.py` — D1), **god-file splits** (chat / memory / RAG / tools — B4), **moat boundary + storage path** (§7, `KNOWLEDGE-GRAPH-CANONICAL.md`).
- The **per-primitive health instrumentation + tile** (Wave 0 **US-006**, deferred here) — the mechanism that makes DoD item #7 real for *every* primitive.
- **Mission Zero P3** (retries feed the verifier's critique back into the next attempt — E4) — folded into the **Missions** primitive.
- **Wave 3R** — the *one* consolidation: Playbook engine **consolidate + harden** (strangler-fig, restart-durable, **delete the dead `modules/workflows/` twin**).
- The **`escalation_level` migration drift** fix (deferred 2026-06-04) — small, **prod-gated**.

**Is not:**
- **Rewrites.** Zero (locked, §16). Every primitive is consolidate-and-harden, never rebuild.
- **New features / new primitives / new verticals.** The `automatos-shopify` hero workflows are unbundled to their own PRD.
- The **§11 CUT list** (dead tables/routes/`neural_field`/`chatbot_llm`) — that's **Wave 5**. The *only* deletion here is the `modules/workflows/` twin that **3R** removes as part of consolidation (plus the second tool loop G6 retires, and `os.getenv` literals).
- **HARNESS self-management** and the `knowledge_nodes/edges` learning loop — **Wave 4**, flag-gated, last. Keep the schema; do **not** wire it here.
- **Frontend Playwright** — **OPEN** (see §12, decision 1): either folds in as **WS-O** or stays a separate **Wave 2.3**. Recommendation: keep Wave 3 backend-only and route Playwright to Wave 2.3.

---

## 3. The hardening contract — what "done" means per primitive

Wave 3's acceptance bar is uniform. A primitive is **rock solid** when it satisfies its **`BRAIN §3.x` contract** *and* every box of the **`GUARDRAILS.md` §H** checklist ticks:

- [ ] **Golden-journey test** — the happy path end-to-end exists and passes (`TEST-PLAN.md`; Wave 2 built these at **API level** — Wave 3 hardens *under* them).
- [ ] **Failure path tested** — the primitive degrades or errors **visibly**, never silently.
- [ ] **Restart-safe** — no user-visible work lost on a process restart (E1).
- [ ] **Observable** — emits the telemetry the "Is it working?" dashboard needs.
- [ ] **Tenant-isolated** — proven by a cross-workspace test (A4).
- [ ] **One source of truth** — no dual write paths (F3).
- [ ] **Dashboard tile** — a number answering *"is this primitive working right now?"* (this wave builds the mechanism — **WS-M**).

The §H checklist is the *contract*; the §4 map below is the *specific work* each primitive needs to reach it.

---

## 4. Per-primitive map (current state → Wave 3 work)

Verified against `origin/main` @ `86c5957ac`, 2026-06-05. The DoD is uniform (§3); this table is the **specific gap and god-file target** per primitive.

| Primitive | `BRAIN §3.x` contract (one line) | Known gap / god-file (verified) | §H items most at risk |
|---|---|---|---|
| **Chat** | Every turn assessed → routed → streamed without dropping tool events → persisted; errors surface. | `consumers/chatbot/service.py` = **2,300 lines** (god-file); hosts one of the two tool loops (`_run_tool_loop`, **G6**). | Failure-path, One-source (tool loop), Observable |
| **Memory** | Exactly one write path per layer; budget-bounded reads; degrades when Mem0 down (circuit breaker built). | `modules/memory/unified_memory_service.py` = **2,202 lines**; `knowledge_system.py` = 1,431. Write-once-per-layer (G12) to re-prove. | One-source, Restart-safe, Observable |
| **RAG** | Ingest→chunk→embed→index atomic per doc; rerank + parent-expand; delete removes the vector. | `api/documents.py` = **1,924** (the parent review §5's named god-file — it's the API *router*; split is lower-priority), `modules/rag/ingestion/manager.py` = **1,560**, `service.py` = 1,243 (the *engine* god-files — split these first). Delete-removes-vector **proven in Wave 2** (W2-S10) — keep green. | Failure-path, Observable, Tile |
| **NL2SQL** | Generated SQL always validated/rewritten before execution (read-only); no unvalidated query; creds never logged. | Validator/router exist; end-to-end shape under-tested (Wave 2 §4.4 stretch, likely rolled here). | Failure-path, Observable, Tile |
| **Graph (moat)** | Rebuildable idempotently from sources; FBT/collection/vendor edges queryable; survives restart (persisted, not in-mem only). | **Moat boundary** — canonical store `workspace_graphs` only; **no dual-write** to `knowledge_nodes/edges` (audit referencers: `init_database.py`, `core/services/analytics_engine.py`, `api/system.py`, `api/execution_history.py`, `modules/memory/storage/knowledge_system.py`). Storage-format evolution (JSON-blob → edge-tables, parent §7) is **named-but-deferred** — not a blocker here (`KNOWLEDGE-GRAPH-CANONICAL.md`). | One-source, Restart-safe, Tile |
| **Missions** | DB-authoritative + restart-durable (**already true**, §8.2); stalled tasks re-dispatched; **retries feed critique back** (NOT true — gap). | **Mission Zero P3** (E4; parent review §5 calls it G5) — the verifier's critique is not fed into the retry prompt. | Failure-path (learning), Observable |
| **Playbooks** | An execution survives a process restart (**NOT true** — fire-and-forget); canonical noun **Playbook**, not Recipe (massively violated). | **→ WS-3R** (the consolidation). Live front door is `api/workflow_recipes.py` (`POST /api/workflow-recipes/{id}/execute`, mounted `main.py:967`, FE calls it at `api-client.ts:1475`) with `api/recipe_executor.py` the loop; `api/api_playbooks.py` is a hollow 49-line read-only stub (no execute); dead `modules/workflows/` twin. | Restart-safe, One-source |
| **Channels** | Every adapter implements the same contract; a new channel = new adapter file, zero core change; in/out counts tracked. | 11 adapters on `BaseChannelAdapter`; parametrized contract test missing (Wave 2 §4.8 stretch). | Observable, Tenant-isolation, Tile |

---

## 5. Reuse map (read before writing a line of code)

Everything below already exists. Wave 3 **adopts / extends / hardens** it; it does not rebuild.

| Concern | Reuse this | Verdict |
|---|---|---|
| The safety harness | The **Wave 2 net** — J1–J10 `golden` backbone, G1/G3/G4 gap regressions, the shared real-DB transactional fixture in `orchestrator/tests/conftest.py` | **Harden under it.** Each primitive's failure-path / restart / tenant tests **extend** the Wave 2 fixtures; do not stand up a parallel harness. |
| Playbook durability | The **Mission coordinator** durability model (DB tick, restart recovery, state in Postgres — `CoordinatorService`, grade A−) | **Port it** to the Playbook engine (3R). Missions already solved this — do **not** invent a third durability scheme. |
| Playbook front door | Live `api/workflow_recipes.py` (`/api/workflow-recipes`, mounted `main.py:967`, FE via `api-client.ts:1475/1447`, `use-unified-analytics.ts:206`) + hollow `api/api_playbooks.py` (49 lines, read-only `/api/playbooks`) | **Pick the canonical front door** (§12.6); migrate the 7 launch sites + **repoint the FE** before deleting `workflow_recipes.py`. |
| Per-primitive tile (US-006) | `api/analytics_real.py` router (exists, **no `primitive-health` route yet**) + `heartbeat_results` table + `services/heartbeat_service.py` writer | **Extend the writer** to emit primitive-mapped findings, **add one read route**. Endpoint contract is already TDD-specced (memory `prd142-wave0-us006-deferred`). |
| Config discipline (G7) | `config.py` (the only env reader) + the PRD-141 opportunistic-sweep precedent + the Wave 2 `os.getenv`-outside-config CI grep gate | **Route the ~20 stragglers through `config.py`.** Pure widening; the grep gate already guards regressions. |
| Tool executor (G6) | The two existing loops — chat `_run_tool_loop` and `AgentFactory.execute_with_prompt` | **Converge on one** (A2). Characterize both, pick the canonical, migrate, delete the loser. |
| Migration (`escalation_level`) | `alembic/versions/wave3_escalation_level.py` (`down_revision = wave1d_mission_lifecycle`; additive, nullable, idempotent) | **Apply to prod** after verifying the head chain. Already written — no new migration unless a merge head is needed. |

---

## 6. Workstreams & user stories

Story IDs are wave-local (`W3-Sn`). Workstream letters continue the sequence (Wave 2 ended at WS-I).

### WS-M — Per-primitive health instrumentation + tile *(do FIRST — every primitive's DoD #7 depends on it)*

**W3-S1 — Heartbeat writer emits primitive-mapped findings.**
- `services/heartbeat_service.py` today emits only **operational** checks (`agent_health`, `checklist`, `error`, `exec_error`, `llm_analysis`, `llm_error`) — **none map to a product primitive**. Add per-primitive `check` findings (`chat`, `memory`, `rag`, `nl2sql`, `graph`, `missions`, `playbooks`, `channels`) as each primitive is hardened in WS-J.
- **Decision (see §12.3):** (a) extend the heartbeat writer vs (b) a dedicated per-primitive probe. **Recommend (a)** — reuse the existing cadence.
- **AC:** a hardened primitive emits a primitive-mapped finding; un-hardened ones emit none (→ honest `unknown`, never a fake green).

**W3-S2 — `GET /api/analytics/primitive-health`.**
- Build the **already-TDD-specced** read endpoint on `api/analytics_real.py`: auth-gated, tenant-scoped (`ws_id = str(ctx.workspace_id)`), single query over `heartbeat_results` DESC, latest-mapped-finding-per-primitive wins. Status taxonomy: `green` / `down` / `degraded` / `unknown`. All 8 primitives always present. **Build only after W3-S1 emits findings** — never before (an all-`unknown` tile in prod is hollow).
- **AC:** returns `{primitives:[{name,status,last_checked}], generated_at}`; tenant-isolated; unmapped → `unknown,null`.

**W3-S3 — Command Centre tile.**
- Surface the 8-primitive health tile on the existing "Is it working?" dashboard (Wave 0 surface) — the 5th metric Wave 0 deferred.
- **AC:** tile renders 8 primitives; `unknown` reads as "not yet instrumented," not "broken."

### WS-K — Unify the tool loop *(G6 — A2: one tool executor)*

**W3-S4 — Converge the two tool loops.**
- Characterize both `consumers/chatbot/service.py::_run_tool_loop` and `modules/agents/factory/agent_factory.py::execute_with_prompt` (dedup / retry / truncation logic that has **drifted**), pick the canonical executor, migrate callers, **delete the loser**.
- The `modules/tools/registry/tool_registry.py` = **1,528-line** god-file (the `tools` entry in the §10 god-file metric) is **owned by this workstream** — split it (B4) as the loops converge onto one registry.
- **AC:** one tool loop remains; grep proves the other is gone; chat + agent suites green; behaviour pinned by the characterization tests first.

### WS-L — Config discipline sweep *(G7 — D1: config.py is the only env reader)*

**W3-S5 — Sweep `os.getenv`/`os.environ` outside `config.py`.**
- ~20 calls across ~8 runtime files (worst: `core/monitoring/`) + `database.py`'s import-time `load_dotenv()`. Route through `config.py`. **Opportunistic, pure widening** (PRD-141 precedent), not big-bang.
- **AC:** `grep` for `os.getenv`/`os.environ` outside `config.py` → 0 in swept files; the Wave 2 grep gate stays green; no behaviour change.

### WS-J — Per-primitive hardening loop *(the bulk — one story per primitive)*

**W3-S6…S13 — Harden each primitive to the §H DoD** (chat, memory, RAG, NL2SQL, graph, missions, playbooks→3R, channels). Each story: meet the `BRAIN §3.x` contract + the §H 7-point DoD under the Wave 2 net, **split the god-file** where named (§4), add the failure-path / restart / cross-workspace tests, and **light the primitive's tile** (WS-M). **Mission Zero P3** lands in the Missions story (W3-S11). **Playbooks** (W3-S12) **is** WS-3R.
- **Pathfinder (see §12.2):** prove the template on **one** primitive end-to-end first, then replicate. **Recommend Memory or RAG** (cleaner boundary, already partially covered by Wave 2 W2-S10) over Chat (entangled with G6).
- **AC (per primitive):** §H boxes all tick; tile green; god-file under the B4 ceiling or justified; `code-reviewer` clean on the diff.

### WS-3R — Playbook engine consolidation *(the one consolidation — `PLAYBOOK-ENGINE-DESIGN.md`)*

**W3-S12 — Consolidate + harden the Playbook engine.**
- Build behind a stable interface, **port the Mission durability model** (DB tick, restart recovery, Postgres state), migrate the **7 launch call sites** strangler-fig, consolidate the scattered `recipe_executor` loop + `workflow_recipes` launch door into one durable flow, **delete the dead `modules/workflows/` twin**, and complete the **Recipe → Playbook** rename.
- **The 7 launch sites** (verified `origin/main`, grep 2026-06-05): `api/workflow_recipes.py:905` (`/execute`), `api/workflow_recipes.py:1860` (the **second** webhook door — the one the parent review's "six" omitted), `api/composio.py:886`, `api/webhooks.py:683`, `consumers/.../handlers_playbooks.py:487`, `services/playbook_scheduler.py:208`, `services/task_reconciler.py:273`.
- **Front-door + FE-repoint delete-blocker:** the FE launches Playbooks via `api-client.ts:1475` (`POST /api/workflow-recipes/{id}/execute`) and `:1447` (`/use`), and reads stats at `use-unified-analytics.ts:206`. **Pick the canonical front door first** (§12.6 — promote `api_playbooks.py`'s `/api/playbooks` vs keep the `/api/workflow-recipes` path), **migrate the 7 launches, then repoint the FE, and delete `workflow_recipes.py` last.** `api/api_playbooks.py` (49 lines, read-only today) cannot serve launches until promoted.
- **Gated on:** the Wave 2 net guarding survivors (satisfied) **and** a passing restart-durability test **before any delete**.
- **AC:** an in-flight Playbook **recovers from DB on restart** (no longer fire-and-forget); grep shows **one** engine **and one** playbook router; the FE calls **only** the canonical path; the dead twin is gone; canonical noun is Playbook.

### WS-N — `escalation_level` migration drift *(small, prod-gated)*

**W3-S14 — Apply the unmigrated `wave3_escalation_level` migration to prod.**
- `/api/kpi/decisions-needed` swallows `column "escalation_level" does not exist` and shows an empty tile. The migration (`down_revision = wave1d_mission_lifecycle`; additive, nullable, idempotent) has **never been applied in prod** — prod may have **multiple unmigrated heads**. **Verify the head chain first** (a merge migration may be needed); **surface the exact command for Gerard's approval — run nothing against prod unprompted.**
- **AC:** `/api/kpi/decisions-needed` returns real data; the column exists; head chain is single and clean.

> **WS-O — Frontend Playwright** is **conditional** on §12, decision 1. If routed here, it is its own workstream against the full J1–J10 UI bar; if routed to Wave 2.3, it is out of scope (§9).

---

## 7. Sequencing & gates

Land in this order — each story is independently shippable; the wave can pause **between** primitives without breaking `main`:

1. **WS-M first** — the tile mechanism. Every primitive's DoD #7 needs it, and it gives a live before/after for the rest of the wave.
2. **WS-L (G7)** early — cheap, low-risk, pure widening; clears config debt out of the primitives before they're refactored.
3. **WS-K (G6)** before Chat/Agents hardening — they sit on the tool loop; converge it first so the chat god-file split lands on one executor.
4. **WS-J** — the primitive loop. **Pathfinder primitive first** (prove the §H template + tile), then replicate. Each ends with a green tile + 7 ticks.
5. **WS-3R** — the riskiest. Strangler-fig; restart-durability test **before** any delete; parity proven before the twin is removed. Run **after** the primitive loop settles (Playbooks therefore hardens last).
6. **WS-N** — anytime; **prod-gated** on Gerard's explicit per-command approval.

**Every story:** `pytest` green + **type checks pass** + `code-reviewer` on the diff (CRITICAL/HIGH addressed before merge) + the primitive's **tile goes green**. **Risky stories (3R, G6)** add a **canary soak** on one workspace before the delete lands. **No `os.getenv` outside `config.py`, no hardcoded values, no backward-compat shims, no dual write paths** — enforced by the Wave 2 gates.

---

## 8. Deletions / cleanups (delete what you replace — `CLAUDE.md` §5 / GUARDRAILS B2)

- **The dead `modules/workflows/` twin** → deleted by **3R** once parity + restart-durability are proven.
- **The losing tool loop** (G6) → whichever of `_run_tool_loop` / `execute_with_prompt` is not canonical is removed; no dual path.
- **`Recipe` naming** → renamed to **Playbook** across the consolidated engine (FE already says Playbook).
- **`api/workflow_recipes.py`** (the legacy `/api/workflow-recipes` front door) → **retired by 3R** once the 7 launch sites migrate **and the FE is repointed** to the canonical router (delete-blocker — the router is deleted **last**, never while the FE still calls it).
- **`os.getenv`/`os.environ` literals outside `config.py`** (G7) → routed through `config.py`.
- God-file fragments extracted to focused modules (B4) — the original shells shrink or split; **no `_legacy` twin survives**.

> The broader §11 CUT list (`neural_field` + `AgentExecutionManager`, `chatbot_llm`, the stream bridge, dead tables/routes) is **Wave 5**, guarded by this wave's hardened primitives — **not** this wave.

---

## 9. Out of scope

- **The §11 CUT list** beyond the 3R `modules/workflows/` twin — **Wave 5**.
- **HARNESS self-management** + wiring the `knowledge_nodes/edges` learning loop — **Wave 4** (flag-gated, after soak). Keep the schema; don't wire it.
- **New features, endpoints-for-their-own-sake, new primitives, new LLM providers, the `automatos-shopify` hero workflows** (unbundled to their own vertical PRD).
- **Frontend Playwright** *if* routed to **Wave 2.3** (§12, decision 1).
- **80% on all ~945 backend files** — coverage stays **per-tier** (touched / critical-path), long tail opportunistic.

---

## 10. Success metrics

| Metric | Current (verified 2026-06-05) | Target | How measured |
|---|---|---|---|
| Per-primitive health tiles green | 0 / 8 (US-006 deferred) | 8 / 8 | `GET /api/analytics/primitive-health` (WS-M) |
| Primitives at the §H DoD | 0 / 8 formally | 8 / 8 | §H checklist per primitive (WS-J) |
| Tool loops | 2 (`_run_tool_loop` + `execute_with_prompt`) | 1 | grep after G6 (WS-K) |
| `os.getenv`/`os.environ` outside `config.py` | ~20 across ~8 files | 0 in swept files | grep gate (WS-L) |
| Playbook engines | 3 (scattered triplet) | 1 | grep the triplet routers after 3R |
| Playbook launch sites migrated | 0 / 7 | 7 / 7 | grep the 7 launch call sites after 3R |
| Playbook front doors (routers) | 2 (`workflow_recipes` live + `api_playbooks` stub) | 1 canonical | grep routers + FE calls one path |
| Playbook restart-durability | dies fire-and-forget | recovers from DB | restart-durability test (W3-S12) |
| Mission retry feeds critique back | no | yes | E4 test (W3-S11) |
| `/api/kpi/decisions-needed` tile | errors silently (empty) | real data | WS-N + dashboard load |
| God-files (chat/memory/RAG/tools) | 2,300 / 2,202 / 1,560 / 1,528 lines | split (B4) or justified | `wc -l` + code-reviewer |

---

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hardening a primitive regresses a live path | Medium | Refactor **under the Wave 2 net**; per-story `code-reviewer`; characterization tests pin behaviour before the refactor; canary on risky stories. |
| **3R** consolidation regresses a live execution path | High | Strangler-fig behind a stable interface; restart-durability test **before** any delete; migrate the 7 call sites incrementally; delete the twin only after parity proven; canary soak. |
| **3R deletes `workflow_recipes.py` while the FE still calls it** (`api-client.ts:1475/1447`) | High | Repoint the FE to the canonical router **first**; delete the legacy router **last**; gated on a green FE Playbook-launch journey. |
| **G6** tool-loop convergence drifts behaviour (dedup/retry/truncation differ today) | Medium | Characterize **both** loops first; pin the union of behaviours; converge to one; delete the loser only when green. |
| US-006 tile reads `unknown` for un-instrumented primitives | Expected | Honest by design — instrument incrementally as each primitive hardens; **never ship a fake green**; `unknown` ≠ broken in the UI copy. |
| Scope: 8 primitives × 7-point DoD is a big wave | High | **Pathfinder-first**, then replicate; each primitive independently shippable; the wave can pause between primitives; 3R is the only tightly-coupled piece. |
| `escalation_level`: multiple unmigrated alembic heads in prod | Medium | **Verify the head chain first**; author a merge migration if needed; **prod-gated** on Gerard's explicit per-command approval. |
| Frontend Playwright scope bleeds into this wave | Medium | Decide §12.1 **before** WS-J starts; recommendation is to route it to a separate Wave 2.3 so Wave 3 stays a backend-primitive wave. |

---

## 12. Open decisions (for Gerard — settle before WS-J kicks off)

1. **Frontend Playwright — Wave 3 (WS-O) or a separate Wave 2.3?** The Core Review §12 slotted "full Playwright" into Wave 2; the Wave 2 build-PRD deferred it backend-first. **Recommend a separate Wave 2.3** — don't mix a ~670-file frontend net with backend primitive refactors. *(Locked input: frontend is at the full-Playwright enterprise bar — only the **timing** is open.)*
2. **Pathfinder primitive** — which one proves the §H template first? **Recommend Memory or RAG** (cleaner boundaries, partial Wave 2 coverage) over Chat (entangled with G6).
3. **US-006 instrumentation** — (a) extend the heartbeat writer vs (b) a dedicated per-primitive probe. **Recommend (a).**
4. **3R timing** — parallel with the WS-J loop, or after it? **Recommend after** (it's the riskiest; let the loop settle — Playbooks hardens last, since Playbooks **is** 3R).
5. **Ralph config** — **authored** as the three-file set above, mirroring the Wave 0 pattern (priority-ordered stories, `passes` as the single source of truth, human-gated frontend/prod/delete steps). Confirm the scope split (agent vs human) or adjust before the loop runs.
6. **Canonical Playbook front door** — promote `api/api_playbooks.py`'s `/api/playbooks` to the real execution router (and retire the `workflow_recipes` name), or keep the live `/api/workflow-recipes` path and just rename the file? **Recommend promoting `/api/playbooks`** — it already owns the canonical noun; migrate the 7 launches + FE onto it, then delete `workflow_recipes.py`.

---

**End of PRD-142 Wave 3 (build spec). Per the Core Review §16, the build PRD follows the approved decision — these six open items (§12) are the last calls before code.**
