# PRD-142 Wave 4 — Self-Learning / HARNESS (Auto Manages and Improves Itself, Gated)

> **Parent:** `PRD-142-CORE-DESIGN-REVIEW.md` §8 (HARNESS kept / gated / sequenced last), §7 (the five-graph taxonomy + the learning-loop roles), §12 (the Wave 4 row), and §16 (`knowledge_nodes/edges` — **fix, not cut**, 2026-05-29).
> **Design companions:** `docs/architecture/GUARDRAILS.md` §H (the 7-point DoD this wave asserts), `docs/architecture/KNOWLEDGE-GRAPH-CANONICAL.md` §4 (the learning-store boundary). Originating PRDs: **121** (HARNESS Self-Optimizing Org Loop), **141 Phase 5** (self-management US-020–026), **138/139** (the tool-routing graph).
> **Status:** Build-PRD — drafted 2026-06-07. **Gate (pending):** Wave 4 is sequenced **after** Wave 3 hardens **and soaks** the primitives this wave is allowed to self-modify (parent §8/§12). It does not start until that soak is real.
> **Type:** **Activate + extend, under green, behind flags. Zero rewrites.** HARNESS, the tool-routing learning loop, and the autonomy dial **already exist and run** — this wave **wires, widens, governs, and observes** them. **No new primitives, no new LLM providers, no rebuilds.**
> **Verified against:** worktree `automatos-ai` @ `0bcdac1c5` (`ralph/prd-142-wave3-primitive-hardening`), code reads 2026-06-07. Both flags already exist: `HARNESS_ENABLED` (default **true**) at `config.py:506`, `HARNESS_SELF_MANAGEMENT_ENABLED` (default **false**, HIGH-RISK) at `config.py:510`.
> **Depends on:** Wave 2 (merged — the test net) + Wave 3 (the hardened primitives this wave touches). Nothing in this wave may take effect in prod unless `HARNESS_SELF_MANAGEMENT_ENABLED` is true (parent §8).
> **Reuse-first** per `CLAUDE.md` §2 / §5 and `GUARDRAILS.md` B1/B2 — **this entire wave is reuse.** Nothing here is built from scratch; the net-new is two platform actions and one migration.
> **Ralph config:** authored on approval (`scripts/ralph/prd-142-wave4.json` + `PROMPT_build_prd142-wave4.md` + `loop-prd142-wave4.sh`, the Wave 0 three-file pattern). **Human-gated, excluded from the autonomous loop:** the prod migration (W4-S11), the canary flag-enablement + soak (W4-S2), the FE dashboard tile (W4-S16), and the dead-code deletion (W4-S13).

---

## 1. The founding question for Wave 4

- **Wave 0** answered *"can we measure it?"* — yes (the Command Center vitals are live).
- **Wave 1** answered *"can we stop the bleeding?"* — yes (durable execution, Mem0 async, idle-tx, honest errors).
- **Wave 2** answered *"can we **prove** it stays working?"* — yes (J1–J10 + gap regressions + a required CI gate).
- **Wave 3** answers *"is each **primitive** rock solid?"* — hardening each of the eight under the green net.
- **Wave 4** answers *"can Auto safely **manage and improve itself**?"* — can the platform **learn from its own execution** and **apply changes back onto itself**, behind a **human-governed gate**, without destabilizing the primitives Wave 3 just hardened?

The surprising finding from the design review's own code (verified 2026-06-07): **most of the machine already exists and is running.** HARNESS sweeps every workspace weekly (`main.py:521-522`, gated by `HARNESS_ENABLED`, default on). The learning loop is **already live** — the PRD-138/139 tool-routing graph records a signal after every tool execution and penalizes failed paths in real time. The governance gate already exists — the standard/full **autonomy dial**. What is missing is the **last mile**, and it is missing in five specific places:

1. The human approval path is **built but unwired** — `handle_harness_command` (`/approve`·`/reject`) has **zero non-test callers** (`api/harness_commands.py:119`), so a channel approval today goes nowhere.
2. The prescription vocabulary is **capped** — HARNESS explicitly **refuses** `power_mode_*` and `routing_rule_add` (`services/harness_service.py:1420-1425`) because the platform actions don't exist yet.
3. The live learning loop (tool-routing) and HARNESS's diagnosis are **siloed** — HARNESS never reads `fails_for_intent`, so it doesn't learn from tool failures.
4. HARNESS's memory is **flat files** — per-workspace baseline JSON under the volume, not a queryable, tenant-isolated, auditable store. The `knowledge_nodes/edges/learning_outcomes` tables that *should* hold it are **dead and unmigrated**.
5. None of it is **governed or observable as one system** — three gates (global flag, autonomy dial, per-prescription approval) exist but aren't wired into one model, and there is no "is self-learning working?" tile.

**Goal:** close that last mile — **behind the flag, on one canary workspace, with rollback proven** — so Auto can propose and (when enabled) apply approved, reversible, audited changes to itself, learning from its own tool-routing signals, all visible on one tile.

---

## 2. What Wave 4 **is** — and is **not**

**Is** (activate + extend, behind flags, on a canary):
- **Activate** the already-built HARNESS self-management (PRD-141 Phase 5): **wire** the inbound `/approve`·`/reject` command path, **canary-enable** `HARNESS_SELF_MANAGEMENT_ENABLED` on one workspace, and **prove** rollback + escalation end-to-end.
- **Widen the prescription vocabulary** — build the **two net-new platform actions** (`power_mode`, `routing_rule`) HARNESS currently refuses, so it can self-apply them behind the flag.
- **The learning loop, both roles** (parent §7): **(Role 1)** ratify the **tool-routing graph** (PRD-138/139) as the canonical tool-selection learning loop, harden it to the §H DoD, and **feed its `fails_for_intent` signals into HARNESS diagnosis**; **(Role 2)** repurpose `knowledge_nodes/edges/learning_outcomes` as HARNESS's **structured, queryable, tenant-isolated** diagnosis/prescription/outcome store, replacing the baseline JSON files.
- **Fold in the governance dial** — the untracked standard/full **autonomy** work (Auto's self-action gate) becomes Wave 4's governance layer, committed properly with its existing tests, and **coupled** to HARNESS's auto-apply threshold.
- **One observability tile** — "is self-learning working right now?"

**Is not:**
- **Building HARNESS, the learning loop, or the autonomy gate from scratch** — they exist. Zero rewrites (parent §16 lock).
- **Unbounded autonomy.** No agent self-modifies code; no budget/billing changes; the vocabulary is an **explicit, audited allow-list**, expanded by exactly two entries this wave.
- **Any change to the moat.** The `KNOWLEDGE-GRAPH-CANONICAL.md` §4 boundary holds — the learning store stays **learning-only, never business entities**; `workspace_graphs` (the moat) is untouched.
- **The §11 CUT list** (dead tables/routes/`neural_field`/`chatbot_llm`) — that's **Wave 5**. The *only* deletion here is the dead `KnowledgeGraph`/`LearningEngine` triple-store API this wave **replaces**.
- **New features, primitives, endpoints-for-their-own-sake, or LLM providers.**
- **The Auto cadence/reporting track** (`auto_cadence.py`, `actions_auto_reporting.py`, the `feat/auto-wave*` branches) — a **different wave numbering**, not PRD-142 Wave 4. Do not conflate.

---

## 3. The self-management contract — what "done" means

Wave 4's acceptance bar is the uniform `GUARDRAILS.md` §H Definition of Done, **plus two additions** the act of self-modification demands:

- [ ] **Golden-journey test** — the happy path (propose → escalate → `/approve` → apply → record) exists and passes.
- [ ] **Failure path tested** — a refused/failed/unauthorized prescription degrades **visibly**, never silently; authz **fails closed**.
- [ ] **Restart-safe** — an in-flight HARNESS run, the batched signal recorder, and the applied-tasks ledger survive a process restart (E1).
- [ ] **Observable** — emits the telemetry the self-learning tile needs.
- [ ] **Tenant-isolated** — every learning/prescription row is `workspace_id`-scoped; proven by a cross-workspace test (A4). *(Today the store has no `workspace_id` — this is a hard gap, W4-S11.)*
- [ ] **One source of truth** — HARNESS reads/writes **one** store (the DB), not JSON files **and** a DB (F3).
- [ ] **Dashboard tile** — a number answering *"is self-learning working right now?"*
- [ ] **Auditable** *(addition)* — every self-applied change records **who/what/when + the `current_value_before`** revert target.
- [ ] **Reversible** *(addition)* — auto-rollback on regression is **proven end-to-end**, not just coded.

---

## 4. Current-state map (verified 2026-06-07 → Wave 4 work)

| Component | State (verified) | Wave 4 work | §H most at risk |
|---|---|---|---|
| **HARNESS loop** | **Built + running** weekly (`main.py:521-522`); 5-phase collect/diagnose/prescribe/apply/baseline (`services/harness_service.py`). | Keep; extend. Don't rewrite. | One-source, Restart-safe |
| **Self-management apply** | Built, **gated off** — `_apply_approved_board_tasks` no-ops unless `HARNESS_SELF_MANAGEMENT_ENABLED` (`harness_service.py:1431`, flag check inside). | Canary-enable on one workspace; prove apply/rollback/escalate (WS-P). | Reversible, Auditable |
| **Approval command path** | Handler built + tested (`api/harness_commands.py:119`); **zero non-test callers** — channel approvals dead-end. | Wire at `api/webhooks.py:~437` before UniversalRouter dispatch (WS-P). | Failure-path, Auditable |
| **Prescription vocabulary** | 7 applied types; `power_mode_*` + `routing_rule_add` **refused** (`harness_service.py:1420-1425`) — actions don't exist. | Build 2 net-new platform actions; remove the refusal branch (WS-Q). | Failure-path |
| **Tool-routing learning loop** | **Wired end-to-end** — `signal_recorder.py` (batched, called `tool_router.py:614,659`) → `tool_routing_edges/affinities` → `edge_builder.py` (nightly) → `graph_router.py` (reads; `fails_for_intent` penalizes). | Ratify canonical; harden to §H; feed `fails_for_intent` into HARNESS diagnosis (WS-R). | Observable, Tenant-isolation, Tile |
| **HARNESS learning store** | **Flat JSON** per workspace under the volume; `knowledge_nodes/edges/learning_outcomes` are **dead** (zero non-test instantiations), **unmigrated** (absent from all 124 Alembic revs), **no `workspace_id`** (`modules/memory/storage/knowledge_system.py:80/96/109`). | Migration + strangler-move HARNESS storage to the DB store; delete the dead triple-store API (WS-S). | One-source, Tenant-isolation, Restart-safe |
| **Autonomy governance dial** | Built (standard/full) + executor gates + tests, but **untracked** WIP (`core/services/auto_autonomy.py`, `actions/handlers_autonomy.py`, `test_w3_*`). | Commit onto the Wave 4 branch; define + couple the three-tier gate (WS-T). | Tenant-isolation, Failure-path |
| **Self-learning tile** | None. | Build the tile; wire into the Wave 0 Command Center (WS-U). | Observable, Tile |

---

## 5. Reuse map (read before writing a line of code)

Everything below already exists. Wave 4 **wires / extends / hardens** it; it does not rebuild.

| Concern | Reuse this | Verdict |
|---|---|---|
| The HARNESS engine | `services/harness_service.py` — the 5-phase loop, `_phase_apply` (`:742`), `_auto_apply_prescription` (`:1354`), `_maybe_escalate` (`:846`), `_detect_auto_applied_regressions` (`:1294`), the applied-tasks ledger. | **Extend, never replace.** Add two `change_type` branches; swap the storage backend. The control flow stays. |
| The approval handler | `api/harness_commands.py::handle_harness_command` (`:119`) — flag check + workspace-admin authz, idempotent apply via the same `_auto_apply_prescription`. | **Wire it, don't rewrite it.** The only gap is a caller. |
| The inbound command door | `api/webhooks.py::general_workspace_webhook` (`:309`), which routes to `UniversalRouter` at `:437`. No slash-command intercept exists today. | **Intercept `/approve`·`/reject` before the UniversalRouter dispatch.** Caller identity comes from the channel reply-context already assembled there. |
| New platform actions | The **3-file registration pattern** — `actions_*.py` + `handlers_*.py` + register in `modules/tools/discovery/platform_actions.py` (HARNESS does this at `:33/:60`). `routing_rules` table already exists (`core/models/routing.py:108`). | **Add `power_mode` + `routing_rule` actions via the canonical pattern.** The routing table needs an *action*, not a new table. |
| The learning loop (Role 1) | The tool-routing graph — `signal_recorder.py`, `core/services/edge_builder.py`, `modules/tools/discovery/graph_router.py`, models `tool_routing_edges/affinities/intent_clusters` (`core/models/tool_routing.py:39/123/94`). Already migrated, already wired. | **Ratify as canonical; harden to §H.** HARNESS *reads* `fails_for_intent`; it does not get a parallel signal system. |
| The HARNESS store (Role 2) | `learning_outcomes` (`knowledge_system.py:109`) is **already HARNESS-shaped** — `success_rate_before/after`, `execution_time_before/after`, `application_count`. | **Extend it** (+`workspace_id`/`run_id`/`change_type`/`risk`/status/`current_value_before`) rather than force prescriptions into the concept-shaped `knowledge_nodes`. See §12.2. |
| The governance gate | The untracked autonomy dial — `auto_autonomy.py` (canonical reader/writer for `workspace.settings.autonomy`), the executor gates in `platform_executor.py` (`_full_autonomy` at `:445`, plus the admin + confirmation gates), tests `test_w3_auto_autonomy_service.py` + `test_w3_full_autonomy_gate.py`. | **Commit + couple, don't redesign.** It already fails safe to `standard` on corrupt input. |
| The tile mechanism | Wave 3 **WS-M** primitive-health pattern (`api/analytics_real.py` + `heartbeat_results` + the Command Center surface). | **Add a self-learning tile** alongside the primitive tiles; don't stand up a new dashboard. |

---

## 6. Workstreams & user stories

Story IDs are wave-local (`W4-Sn`). Workstream letters continue the sequence (Wave 3 ended at WS-O).

### WS-P — HARNESS activation & command-path wiring *(the last mile — do FIRST, it's the safety rail)*

**W4-S1 — Wire the inbound `/approve`·`/reject` command path.**
- In `api/webhooks.py::general_workspace_webhook`, intercept a message whose text starts with `/approve `/`/reject ` **before** the `UniversalRouter` dispatch (`:437`), parse `{command, rx_id}`, build `caller_identity` from the channel reply-context, and call `api/harness_commands.handle_harness_command(...)`; deliver its result back over the same channel.
- **AC:** a channel `/approve <rx_id>` from a workspace **admin** applies the queued prescription and records it in the ledger; from a **non-admin** it is refused **before any mutation** (fail-closed); with the flag off it returns the inert "disabled" message. No new router; the existing handler is the only executor.

**W4-S2 — Canary-enable self-management on one workspace.** *(human-gated)*
- Enable `HARNESS_SELF_MANAGEMENT_ENABLED` for **one** canary workspace; run a full sweep; confirm an approved `[HARNESS]` board task is applied via `_apply_approved_board_tasks` and recorded in `applied_tasks.json` (→ DB after WS-S).
- **AC:** approved task → applied; idempotent on a second tick; the global default stays **false**; documented enable/disable runbook.

**W4-S3 — Prove auto-rollback end-to-end.**
- Drive a low-risk auto-apply, then a regression on its target; confirm `_detect_auto_applied_regressions` emits a revert prescription that reverts to `current_value_before`.
- **AC:** a regressing auto-applied change is reverted to its pre-change snapshot within one tick; no rollback-of-rollback oscillation.

**W4-S4 — Prove escalation end-to-end.**
- A risk≥4 prescription with a connected channel notifies the workspace with `/approve`·`/reject` instructions; the round-trip (notice → `/approve` → apply) closes via W4-S1.
- **AC:** high-risk → channel notice; the embedded command applies on admin approval; no channel → silent skip (not an error).

### WS-Q — Widen the prescription vocabulary *(2 net-new platform actions — the only build-from-near-scratch in the wave)*

**W4-S5 — `power_mode` prescription + apply.**
- Build the action HARNESS needs to set power mode, **at the scope decided in §12.1** (recommend `system_settings`/`run_config` — where power mode lives today — **not** a new `Agent.power_mode` column). 3-file registration. Add the `power_mode_upgrade`/`downgrade` branch to `_auto_apply_prescription`.
- **AC:** HARNESS can prescribe and (flag-gated) apply a power-mode change; the apply is audited + reversible; fail-closed authz.

**W4-S6 — `platform_create_routing_rule` action + `routing_rule_add` prescription.**
- 3-file action writing the **existing** `routing_rules` table (`core/models/routing.py:108`); add the `routing_rule_add` branch to `_auto_apply_prescription`.
- **AC:** HARNESS can prescribe and (flag-gated) apply a routing rule; tenant-scoped; audited + reversible.

**W4-S7 — Remove the refusal branch.**
- Delete the `harness_service.py:1420-1425` "intentionally NOT handled" return once both types apply.
- **AC:** the two formerly-refused `change_type`s round-trip; grep shows the refusal gone; no `Unknown auto-apply change_type` for them.

### WS-R — Tool-routing graph → canonical learning loop *(Role 1 — mostly ratify + harden; it's already wired)*

**W4-S8 — Ratify canonical + land the in-flight WIP.**
- Document the tool-routing graph as the canonical **tool-selection** learning loop and its boundary vs the HARNESS store (Role 2). Land the in-flight **US-014** (graph-router delegation) / **US-015** (registry intent filter) WIP and the autonomy executor edits onto the Wave 4 branch with their tests.
- **AC:** the boundary is documented; US-014/015 tests green on the branch; no duplicate learning system introduced.

**W4-S9 — Harden tool-routing to the §H DoD.**
- Observability (signals/day, edges built, routing hit-rate); cross-workspace isolation test (signals/edges are workspace-keyed); restart-safety of the batched recorder drain (no signal loss on restart).
- **AC:** §H boxes tick for the loop; a cross-workspace test proves no signal/edge bleed; the recorder drains or persists its queue on shutdown.

**W4-S10 — Feed `fails_for_intent` into HARNESS diagnosis.**
- HARNESS's DIAGNOSE phase reads the tool-routing `fails_for_intent` affinities as an inefficiency signal → can prescribe a `tool_assignment_remove` (existing vocabulary). This is the cross-link that makes HARNESS **learn from tool failures**.
- **AC:** a sustained tool-failure affinity surfaces a HARNESS diagnosis; no business-entity read (boundary §4 holds).

### WS-S — Repurpose the knowledge store as HARNESS's structured store *(Role 2 — the riskiest; live-loop storage swap)*

**W4-S11 — Migration: bring the store under version control + reshape.** *(human-gated — prod migration)*
- Author the **first Alembic migration** for `learning_outcomes` (and `knowledge_nodes/edges` per §12.2), adding `workspace_id` (tenant isolation — A4), `run_id`/`baseline_id`, `diagnosis_type`, `risk_score`, `applied/rejected/rolled_back` status, `applied_at`/`rolled_back_at`, and `current_value_before`. **Or** a clean new `harness_prescriptions` table — §12.2. Verify the head chain first; **surface the exact command for Gerard's approval — run nothing against prod unprompted.**
- **AC:** the store is migration-managed, registered in `core.models`, workspace-scoped; head chain single + clean.

**W4-S12 — Strangler-move HARNESS storage JSON → DB.**
- Replace the baseline-JSON read/write paths (`_read_baseline`/`_write_workspace_file` and the ledger) with the DB store, **strangler-fig**: dual-write → read-from-DB → verify parity on the canary → drop the JSON paths.
- **AC:** HARNESS reads/writes **one** store (DB); parity proven on the canary before JSON is dropped; restart-safe; no dual write path survives (F3).

**W4-S13 — Delete the dead triple-store API.** *(human-gated — deletion)*
- Remove `KnowledgeGraph` / `LearningEngine` / `HierarchicalMemorySystem` and their dead `add_knowledge`/`learn_from_feedback` paths in `knowledge_system.py` (replaced by the HARNESS store + the tool-routing loop). Preserve the unrelated `MemoryItem` dataclass that real callers import.
- **AC:** grep for the three classes returns **zero** non-test references; `MemoryItem` importers (`api/memory_stats.py`, `api/workspaces.py`) stay green; no `_legacy` twin.

### WS-T — Governance: the autonomy dial + the three-tier gate

**W4-S14 — Commit the autonomy WIP onto the Wave 4 branch.**
- Bring `core/services/auto_autonomy.py`, `actions_autonomy.py`, `handlers_autonomy.py`, and the executor edits in `platform_executor.py` under version control with their tests (`test_w3_auto_autonomy_service.py`, `test_w3_full_autonomy_gate.py`). *(Carried from the wave3 worktree where they sit untracked — see §8/§11.)*
- **AC:** the dial + gates are committed + green; the 3-file registration is complete; fail-safe-to-`standard` behavior preserved.

**W4-S15 — Define + enforce the three-tier governance model.**
- One coherent model: **global flag** (`HARNESS_SELF_MANAGEMENT_ENABLED`) × **per-workspace autonomy** (standard/full) × **per-prescription approval** (`/approve`). **Couple** the dial to HARNESS's auto-apply threshold (the exact coupling is §12.3). Security review: both admin gates **fail closed**; cross-workspace isolation tests.
- **AC:** the three tiers are documented + enforced; `full` vs `standard` measurably changes what HARNESS auto-applies vs queues; a security-reviewer pass is clean.

### WS-U — Observability: the self-learning tile

**W4-S16 — Self-learning tile on the Command Center.** *(FE half human-gated)*
- Surface one tile: HARNESS status (last run / applied / queued / rolled-back), tool-routing signals/day + hit-rate, and the autonomy level per workspace. Wire into the Wave 0 surface alongside the Wave 3 primitive tiles.
- **AC:** the tile answers "is self-learning working right now?" with live numbers; `unknown` reads as "not yet enabled," never a fake green.

---

## 7. Sequencing & gates

Land in this order — the wave can pause **between** workstreams without breaking `main`, and **nothing touches prod behavior until the flag flips on one canary**:

1. **WS-T governance + WS-P wiring first** — the rails before anything self-applies. The dial and the approval path must exist and fail closed before the flag is ever flipped.
2. **WS-S store migration** — the riskiest; do it **early, under the Wave 2 net**, strangler-fig, parity-proven on the canary before JSON is dropped. (It blocks the canary ledger in WS-P/W4-S2 from being durable.)
3. **WS-Q vocabulary** — after the store is stable; two clean 3-file actions + two `_auto_apply_prescription` branches.
4. **WS-R tool-routing** — additive + parallel-safe; ratify, harden, then the `fails_for_intent` cross-link.
5. **WS-U tile last** — once there are real numbers to show.
6. **Canary gate (wave exit):** only after WS-P/T/S are green do we enable `HARNESS_SELF_MANAGEMENT_ENABLED` on **one** workspace, soak, and **prove rollback** — the parent §12 Wave 4 exit gate.

**Every story:** `pytest` green + type checks + `code-reviewer` on the diff (CRITICAL/HIGH addressed). **WS-T adds a `security-reviewer` pass** (it widens what Auto may do without a human). **Risky stories (WS-S storage swap, the canary enablement)** add a **canary soak before the change is irreversible**. **No `os.getenv` outside `config.py`, no hardcoded values, no backward-compat shims, no dual write paths** — enforced by the Wave 2 gates. CI is the source of truth; local full-suite is not run.

---

## 8. Deletions / cleanups (delete what you replace — `CLAUDE.md` §5 / GUARDRAILS B2)

- **The dead triple-store API** — `KnowledgeGraph` / `LearningEngine` / `HierarchicalMemorySystem` + their `add_knowledge`/`learn_from_feedback` paths (`knowledge_system.py`) → removed by **W4-S13** once the HARNESS store + tool-routing loop fully cover the learning role. Keep `MemoryItem`.
- **The baseline-JSON storage paths** — `_read_baseline` / `_write_workspace_file` (for baselines) and the file-based ledger → retired by **W4-S12** after the DB store reaches parity on the canary. No JSON+DB dual path survives.
- **The refusal branch** — `harness_service.py:1420-1425` → deleted by **W4-S7** once `power_mode`/`routing_rule` apply.
- **The untracked-WIP debt** — the autonomy files and the in-flight US-014/015 edits **stop being untracked**: they are committed properly on the Wave 4 branch (**W4-S8/S14**), not left as a permanent uncommitted shadow.

> The broader §11 CUT list (`neural_field` + `AgentExecutionManager`, `chatbot_llm`, the stream bridge, dead tables/routes) is **Wave 5** — **not** this wave.

---

## 9. Out of scope

- **The §11 CUT list** beyond the dead triple-store API this wave replaces — **Wave 5**.
- **Any moat / business-graph change.** `workspace_graphs` and the Graphify pipeline are untouched; the learning store stays learning-only (`KNOWLEDGE-GRAPH-CANONICAL.md` §4).
- **Unbounded autonomy / self-code-modification / budget or billing changes.** The vocabulary grows by exactly two audited entries.
- **New features, primitives, endpoints-for-their-own-sake, new LLM providers.**
- **The Auto cadence/reporting track** (`auto_cadence.py`, `actions_auto_reporting.py`, `feat/auto-wave*`) — different wave numbering, not PRD-142 Wave 4.
- **Prod enablement beyond one canary.** Wider rollout is a later, separate call after the canary soak.

---

## 10. Success metrics

| Metric | Current (verified 2026-06-07) | Target | How measured |
|---|---|---|---|
| `/approve`·`/reject` callers | 0 (handler unwired) | wired | grep callers of `handle_harness_command` (W4-S1) |
| Self-management enabled | 0 workspaces | ≥1 canary, **rollback proven** | canary runbook + rollback test (WS-P) |
| Refused prescription types | 2 (`power_mode_*`, `routing_rule_add`) | 0 | grep the refusal branch after W4-S7 |
| HARNESS learns from tool failures | no (`fails_for_intent` unread) | yes | DIAGNOSE reads the affinity (W4-S10) |
| HARNESS store | flat JSON + dead tables | one DB store, `workspace_id`-scoped | F3 grep (no JSON path) + a cross-workspace test (WS-S) |
| Learning rows tenant-scoped | no `workspace_id` column | every row workspace-keyed | schema + A4 test (W4-S11) |
| Dead learning API | present (zero callers) | deleted | grep `KnowledgeGraph`/`LearningEngine` → 0 (W4-S13) |
| Autonomy dial | untracked WIP | committed + governed | branch state + the three-tier model (WS-T) |
| Self-learning tile | none | live | Command Center load (W4-S16) |

---

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Swapping HARNESS's live JSON storage for the DB regresses the weekly loop | High | **Strangler-fig** (dual-write → read-DB → parity → drop); under the Wave 2 net; parity proven on the canary **before** JSON is dropped. |
| Enabling self-management destabilizes the Wave-3-hardened primitives | High | Flag **default false**; **one-workspace canary**; rollback **proven** (W4-S3) before any wider talk; the autonomy dial governs blast radius. |
| `power_mode` modeled as an agent attribute contradicts the run-scoped reality | Medium | **§12.1 decided before WS-Q builds** — recommend prescribing at `system_settings`/`run_config`, not a new `Agent.power_mode` column. |
| Autonomy `full` skips confirmation gates → privilege widening | Medium | **`security-reviewer` pass on WS-T**; both admin gates fail closed; cross-workspace isolation tests; `full` is per-workspace + still under the global flag. |
| The untracked autonomy WIP + dirty US-014/015 files don't migrate cleanly from the wave3 worktree to the Wave 4 branch | Medium | **Land them first** (W4-S8/S14); commit by explicit filename (no `git add -A`); reconcile against the wave3 branch before building on top. |
| Reshaping `knowledge_nodes` (concept-shaped) into a prescription store is an awkward fit | Medium | **§12.2** — recommend extending the already-HARNESS-shaped `learning_outcomes` + a clean `harness_prescriptions` table, deciding `knowledge_nodes/edges` fate separately. |
| Wiring `/approve` into the webhook misfires on non-command messages | Low | Strict `startswith("/approve ")`/`("/reject ")` guard **before** UniversalRouter; everything else falls through unchanged; covered by a regression test. |

---

## 12. Open decisions (for Gerard — settle before the relevant WS)

1. **`power_mode` scope (before WS-Q).** Add an `Agent.power_mode` column, or have HARNESS prescribe power mode at `system_settings`/`run_config` — **where it actually lives today** (the code comment at `harness_service.py:1421-1423` is explicit that agents have no power-mode attribute)? **Recommend the latter** — don't add an agent attribute that contradicts the current model.
2. **HARNESS store shape (before WS-S).** Reshape `knowledge_nodes/edges` per the directive, **or** extend `learning_outcomes` (already HARNESS-shaped — has `success_rate`/`execution_time` before/after) + add a clean `harness_prescriptions` table, and decide `knowledge_nodes/edges`' fate separately (keep for a future agent-concept graph, or fold into the Wave 5 cut)? **Recommend the latter** for schema fit — surfaced because the code shows `knowledge_nodes` is concept/embedding-shaped, a poor fit for prescriptions.
3. **Autonomy × HARNESS coupling (before WS-T).** Should workspace `autonomy=full` **widen** HARNESS's auto-apply risk threshold (e.g. auto-apply risk ≤3 instead of ≤2, queueing less)? Define the exact coupling so `standard` vs `full` has a precise, testable meaning.
4. **Wave 4 branch base.** Branch off `ralph/prd-142-wave3-primitive-hardening` (Wave 3 unmerged, where the autonomy WIP already sits), or off `main` after Wave 3 merges? Determines how the untracked autonomy files and dirty US-014/015 edits migrate.
5. **Canary selection + soak.** Which workspace is the canary, and how long is the soak before any conversation about wider enablement?

---

**End of PRD-142 Wave 4 (build spec). Per the Core Review §16, the build follows the approved decision — these five open items (§12) are the last calls before code, and the flag does not flip until the canary gate (§7) is met.**
