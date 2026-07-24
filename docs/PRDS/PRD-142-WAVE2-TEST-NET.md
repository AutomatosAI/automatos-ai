# PRD-142 Wave 2 — Test Net (Make "Rock Solid" Provable)

> **Parent:** `PRD-142-CORE-DESIGN-REVIEW.md` §12, Wave 2. **Design companion:** `docs/architecture/TEST-PLAN.md` (this PRD is the *build* spec for that plan).
> **Status:** Build-PRD — drafted 2026-06-02. **Gate satisfied:** Wave 1 (#404 WS-A, #394 WS-B Mem0-async, #405 WS-C durable execution, #406 W1-S8, #409 W1-S9 idle-tx) is **merged to `main`** — the surface the net is written against is now stable.
> **Type:** Test infrastructure + coverage. **Backend-first** — the frontend Playwright net is explicitly deferred (§9). The only new code is tests, fixtures, and CI; **no new product features, endpoints, or UI.**
> **Verified against:** `origin/main` @ `ca5731c3f`, code + test-tree reads 2026-06-02 (supersedes the 2026-05-29 map in `TEST-PLAN.md` §2).
> **Depends on:** Wave 1 (merged). **Reuse-first** per `CLAUDE.md` §2 / §5.
> **Ralph config:** `scripts/ralph/prd-142-wave2.json` (to add).

---

## 1. The founding question for Wave 2

- **Wave 0** answered *"can we measure it?"* — yes (the Command Centre vitals are live).
- **Wave 1** answered *"can we stop the bleeding?"* — yes (durable execution, Mem0 async, idle-tx, honest errors are merged).
- **Wave 2** answers *"can we **prove** it stays working — and stop a regression at the door?"*

Today we cannot. A hardening effort with no **enforced** net is hope. The honest, *re-verified* current state:

| Verified reading (2026-06-02) | What it really means | Wave 2 action |
|---|---|---|
| **CI runs no tests.** Only `.github/workflows/check-shopify-isolation.yml` exists. | ~1,240 backend test functions across ~89 files exist but **protect nothing at merge time** — a red suite never blocks a PR. | **WS-F** — stand up the CI test gate |
| **~13 backend tests are already failing on `main`** (`test_coordinator_parallel` ×3, `test_82c_wiring` ×7, `test_synthesis_executor` ×3) | They call `service._execute_task(...)`, a method **renamed** to `_run_agent_io`/`_prepare_task`. `TEST-PLAN.md` §2 counts these as mission coverage — they are stale, not coverage. | **WS-F** — triage/fix before flipping the gate green |
| **Wave 1's reliability fixes are mock-tested only.** `test_w1s9_idle_in_tx`, `test_w1s8_get_db_lifecycle`, `test_w1s6_boot_reaper` assert ordering/lifecycle against recording fakes. | There is **no real-DB regression** that would catch the idle-tx leak or a lost-on-restart run recurring. | **WS-G** — real-DB gap regressions (G1, G4, G3) |
| **Reasoning entry path, RAG, and mission verification have no direct test.** `AutoBrain.assess`, `IntentClassifier`, RAG ingest→retrieve, `MissionReconciler`/`VerificationService` = ❌. | The paths that decide *what the platform does* are unproven. | **WS-I** — primitive integration tests |
| **No numbered golden-journey backbone.** `tests/api/` has 12 ad-hoc journey modules; `tests/e2e/` is an empty `.auth/` stub. | We can't point at "the 10 flows that must never break" and see green. | **WS-H** — formalize J1–J10 (API level) |
| **Frontend e2e is greenfield.** Vitest installed (1 test file); **Playwright not installed**, no config. | The entire UI is untested. | **Deferred** (§9) — backend-first this wave |

**Goal:** a backend test net that proves each primitive meets its contract, the J1–J10 journeys stay green, and **a CI gate keeps it that way** — turning `TEST-PLAN.md` from design into an enforced reality.

---

## 2. What Wave 2 **is** — and is **not**

**Is** (backend-first, per the kickoff decision):
- **Make the existing 1,240 tests enforceable** — pytest config + a blocking CI gate (the single biggest cheap win).
- A **centralized real-DB transactional fixture** (TEST-PLAN §3/§7) so integration tests hit real Postgres with rollback per test.
- The **gap-regression tests** that double as reliability proof — **G1** (real-DB idle-tx/connection-leak), **G4** (mission + playbook restart-recovery), **G3** (fail-closed authz) — in TEST-PLAN §8 order.
- The **J1–J10 golden-journey backbone at API level** (TEST-PLAN §5).
- **§4 integration/unit tests for the untested primitives** — `AutoBrain.assess`, `UniversalRouter.route` tiers, `IntentClassifier`, RAG ingest→retrieve, `MissionReconciler`/`VerificationService`.

**Is not:**
- The **frontend Playwright + Vitest net** (J1–J7 UI, `api-client`, hooks) — **deferred** to a Wave 2.3 follow-on / Wave 3 frontend hardening (§9). Decided at kickoff.
- **Per-primitive DoD hardening** — that's **Wave 3**. Wave 2 writes the *characterization + golden* net; Wave 3 hardens **under green**.
- The **Playbook-engine consolidation** and `Workflow`→`Mission` table migration — **Wave 3 / 3R**.
- **80% on all ~919 backend files** — that's the long tail, reached opportunistically (TEST-PLAN §10). Coverage targets are **per-tier** (80% on touched/critical-path).
- Any new feature, endpoint-for-its-own-sake, LLM provider, or UI.

---

## 3. Relationship to `TEST-PLAN.md` (the mapping)

| TEST-PLAN section | Wave 2? | How |
|---|---|---|
| §7 Test infrastructure (CI gate, real-DB fixture, fixtures) | **In — WS-F** | The enabling layer; do first. |
| §4.9 Cross-cutting infra (G1 leak, G2 alembic, G3 authz, config/bare-except gates) | **In — WS-F + WS-G** | Gates in WS-F; G1/G3 regressions in WS-G. |
| §4.6 Missions (restart durability) / §4.7 Playbooks (G4 durability) | **In — WS-G (W2-S5)** | The restart-recovery proof for WS-C / Mission Zero P1. |
| §5 Golden journeys J1–J10 | **In — WS-H** | **API level only**; the UI half (Playwright) is deferred. |
| §4.1 Chat/reasoning entry (assess, router tiers, intent) | **In — WS-I (W2-S8)** | The biggest hole. |
| §4.3 RAG ingest→retrieve→delete | **In — WS-I (W2-S9)** | No functional test today. |
| §4.4 NL2SQL end-to-end / §4.5 Graph build idempotency / §4.8 Channels contract | **In — WS-I (W2-S10)** | Validator/router pieces exist; add the missing integration shape. |
| §4.2 Memory | **Keep + extend** | L1/L2/L3 + circuit-breaker covered; add the write-once-per-layer (G12) integration check. |
| §6 Frontend strategy | **Deferred (§9)** | Vitest exists; Playwright not stood up this wave. |

---

## 4. Reuse map (read before writing a line of code)

Everything below already exists. Wave 2 **adopts / extends / formalizes** it; it does not rebuild.

| Concern | Reuse this | Verdict |
|---|---|---|
| Golden journeys | The 12 existing `tests/api/test_*_journey.py` modules (onboarding, mission_research, daily_workflow, document, admin_config, integration_setup, …) | **Map onto J1–J10 and extend in place** — do **not** rewrite; fill only the missing journeys (J3 widget generic+shopify, J9 Shopify-moat e2e, J10 cross-workspace). |
| Real-DB fixture | The transactional pattern already in `modules/learning/tests/conftest.py` and `modules/search/tests/conftest.py` (`connection.begin()` → `SessionLocal()` → `rollback()`) | **Pull up** to a shared `orchestrator/tests/conftest.py` fixture; **de-hardcode** their `postgresql://…secure_password_123…` literals → `config.py`. |
| Schema for the test DB | `orchestrator/scripts/init_test_db.py` (`Base.metadata.create_all`) | **Reuse** for fast unit/integration schema; CI **additionally** runs `alembic upgrade head`/`downgrade base` for the G2 regression. |
| Unit-test mocks | `orchestrator/tests/conftest.py` `mock_db` + the Wave-1 recording-fake pattern (`test_w1s9_idle_in_tx`) | **Keep** for fast pure-logic tests; Wave 2 **adds** the real-DB counterparts, it does **not** replace the mock tests. |
| Router coverage | `test_graph_router*.py`, `test_us015_registry_intent_filter.py` | **Extend** toward full `route()` tier coverage; don't duplicate. |
| Env-at-import guard | `core/database/database.py` `ValueError` guard + `os.environ.setdefault` shim already used by W1-S8/S9 tests | **Reuse** the shim for unit tests; integration tests get creds from `config.py` (CI env). |
| Frontend tooling (when un-deferred) | Vitest already configured | **Reuse** at Wave 2.3 / Wave 3; nothing to install for the backend wave. |

---

## 5. Definition of Done (the whole wave)

- [ ] A **blocking** GitHub Actions workflow runs the backend `pytest` suite (against a real Postgres service) on every PR; a red suite **blocks merge**.
- [ ] A pytest config exists (`pyproject.toml [tool.pytest.ini_options]` or `pytest.ini`) pinning the async mode and declaring markers — including a **`golden`** marker that selects the J1–J10 backbone.
- [ ] The **~13 stale `_execute_task` failures are resolved** (repointed to `_run_agent_io`/`_prepare_task`, or deleted if truly superseded — verify before delete, no dual path) so the gate can be green.
- [ ] A **centralized real-DB transactional fixture** exists; integration tests hit real Postgres with rollback-per-test; **zero hardcoded creds**, **no `os.getenv` outside `config.py`**.
- [ ] **Gap regressions exist and provably fail on pre-Wave-1 code:** **G1** real-DB idle-in-tx/connection-leak, **G4** mission + playbook restart-recovery, **G3** fail-closed authz.
- [ ] **J1–J10** each have a green API-level e2e test under the `golden` marker; **J9** (Shopify-moat end-to-end) and **J10** (cross-workspace isolation) explicitly covered.
- [ ] Integration/unit tests exist for the previously-untested primitives: `AutoBrain.assess` verdict table, `UniversalRouter.route` per-tier, `IntentClassifier` vs `SmartIntentClassifier`, RAG ingest→retrieve→**delete-removes-vector**, `MissionReconciler`/`VerificationService`.
- [ ] **`alembic upgrade head` → `downgrade base`** passes on a scratch DB in CI (G2 regression) — or is explicitly deferred in §9 with a reason.
- [ ] Coverage on touched/critical-path code **≥ 80%** (per-tier, not global); per-primitive coverage is published so the dashboard can consume it.
- [ ] Every story: `pytest` green, **type checks pass**, no `os.getenv()` outside `config.py`, no hardcoded values, no backward-compat shims.

---

## 6. Workstreams & user stories

Story IDs are wave-local (`W2-Sn`). Phases map to the kickoff plan: **2.0 = WS-F**, **2.1 = WS-G**, **2.2 = WS-H + WS-I**.

### WS-F — Make the net enforceable *(infra; do first — until this lands, nothing is protected)*

**W2-S1 — Pytest config.**
- Add `[tool.pytest.ini_options]` (to `pyproject.toml`, or a new `pytest.ini`): pin `asyncio_mode` (currently the implicit pytest-asyncio default = `strict`; tests use `@pytest.mark.asyncio` — keep strict), set `testpaths`, register markers `golden`, `integration`, `slow`.
- **AC:** `pytest --markers` lists the new markers; the full suite still collects without warnings; no test behaviour changes.

**W2-S2 — Triage the ~13 stale failures.**
- `test_coordinator_parallel` (3), `test_82c_wiring` (2 in `TestWiringCoordinatorTick`; the other 5 in synthesis/template/token — verify), `test_synthesis_executor` (3) call `service._execute_task(...)`. Production has `_run_agent_io` + `_prepare_task` (verified: `def _execute_task` = 0 hits). **Repoint the mocks/asserts** to the current methods, or **delete** the test if its scenario is genuinely superseded (verify each before deleting — `CLAUDE.md` §4/§5).
- **AC:** those suites are green (or removed with justification); no `_execute_task` reference remains in `tests/`.

**W2-S3 — CI test gate.**
- `.github/workflows/test.yml`: Postgres service container → `python scripts/init_test_db.py` (and `alembic upgrade head` for G2) → `pytest orchestrator/tests`. Wire creds via env consumed by `config.py`. Make it a **required** check **after** the suite is green (W2-S2). Fold in the TEST-PLAN §7 cheap gates that aren't yet enforced: bare-`except` count (PRD-141 Phase 0 script), `os.getenv`-outside-config grep.
- **AC:** opening a PR runs the backend suite; a deliberately-broken test makes the check **red and blocks merge**; the gates fire.

**W2-S4 — Centralized real-DB transactional fixture.**
- Add a session-scoped engine (creds from `config.py`) + function-scoped `connection.begin()`/`SessionLocal()`/`rollback()` fixture to `orchestrator/tests/conftest.py`. Migrate `modules/{learning,search}/tests/conftest.py` onto it and **delete their hardcoded `postgresql://…` literals**.
- **AC:** a sample integration test uses the shared fixture and rolls back cleanly; `grep -r "postgresql://" orchestrator/**/tests` → 0 literal creds; mock-based unit tests unaffected.

### WS-G — Gap-regression tests *(reliability proof; TEST-PLAN §8 order: G1 → G4 → G3)*

**W2-S5 — G1: real-DB idle-in-transaction / connection-leak regression.**
- Using the real-DB fixture, drive a path that hydrates an `Agent` across an `await` (the W1-S9 surfaces — reconciler verify / coordinator tick, or a `get_db` request) and assert **no connection is left `idle in transaction`** (inspect `pg_stat_activity` or pool state). This is the real-DB test W1-S9 lacked.
- **AC:** green on `main`; documented to **fail** when `end_open_transaction` / the `get_db` rollback is reverted.

**W2-S6 — G4: restart-recovery durability (mission + playbook).**
- Create an in-flight mission run and an in-flight `RecipeExecution`; simulate process death (no live executor); run the **boot reaper** (WS-C, #405); assert each is **resumed or cleanly failed** (`reason="orphaned_on_restart"`), with **no** orphaned `running`/`pending` row past the threshold. This is the proof Mission Zero P1 is closed.
- **AC:** orphaned rows recovered/failed-clean; green; would fail without the reaper.

**W2-S7 — G3: fail-closed authz.**
- Unit tests proving `_check_agent_permission` / `validate_composio_action` **deny** when their check raises or errors (fail-closed, not fail-open).
- **AC:** induced error path → deny; green.

### WS-H — Golden journeys J1–J10 *(API level; UI deferred)*

**W2-S8 — Formalize the J1–J10 backbone.**
- Create the `golden`-marked backbone (`tests/api/test_golden_journeys.py` or `test_j{n}_*.py`) by **mapping the 12 existing journey modules** onto the numbered list (TEST-PLAN §5) and filling gaps:
  - J1 signup→onboarding→workspace · J2 chat→agent→tool→response · J3 widget→plugin→response (**generic *and* shopify**) · J4 mission create→plan→approve→execute→verify→deliverable (**+ restart durability, links W2-S6**) · J5 doc upload→RAG→retrieval-in-chat · J6 marketplace install→cascade→agent usable · J7 playbook schedule→run→complete(+recover) · J8 NL2SQL connect→ask→validated SQL→answer · J9 **Shopify sync→knowledge graph→FBT proactive opener (the moat)** · J10 **cross-workspace isolation across J2–J9**.
- **AC:** all 10 journeys have a green API-level test under the `golden` marker; `pytest -m golden` runs the backbone; J9 and J10 explicitly assert the moat and tenancy promises.

### WS-I — Untested-primitive integration tests *(close the §4 holes)*

**W2-S9 — Chat / reasoning entry (the biggest hole).**
- `AutoBrain.assess()` verdict table (ATOM→ORGANISM × RESPOND/DELEGATE/MISSION); `UniversalRouter.route()` per-tier selection (override / cache / rule / trigger / semantic / keyword / LLM-fallback — assert the *right tier* fires); `IntentClassifier` vs `SmartIntentClassifier` (no cross-contamination). (TEST-PLAN §4.1.)
- **AC:** crafted inputs hit the expected verdict/tier/classifier; green.

**W2-S10 — RAG functional + mission verification.**
- RAG: ingest a doc → Postgres row + S3 object + S3-Vectors entry created → retrieve returns it → **delete removes the vector** (no orphan). `MissionReconciler`/`VerificationService`: direct tests (only mocked in W1-S9 today). LLM/embeddings via **recorded fixtures** (no spend). (TEST-PLAN §4.3 / §4.6.)
- **AC:** ingest/retrieve/delete round-trips against the real-DB fixture; reconciler verdict paths covered; green.

> NL2SQL end-to-end (§4.4), graph build-idempotency (§4.5), and the parametrized channel contract (§4.8) are **stretch** within Wave 2 — fold in if WS-F–WS-I land ahead of the wave budget; otherwise they roll to Wave 3 primitive hardening.

---

## 7. Sequencing & gates

Land in this order — each is independently shippable:

1. **WS-F first** — config + triage + CI gate + real-DB fixture. Until the gate runs, no later test protects anything. Flip the gate to **required** only once the suite (post-triage) is green.
2. **WS-G** — gap regressions (G1/G4/G3). Cheap, highest reliability leverage; each doubles as a Wave-1 regression guard.
3. **WS-H** — J1–J10 backbone (the "is it working" signal).
4. **WS-I** — primitive integration tests (reasoning, RAG, reconciler).

**Every story:** `pytest` green + type checks pass. **Every workstream:** `code-reviewer` agent on the diff; CRITICAL/HIGH addressed before merge. **Integration tests** run against real Postgres in CI (mocked infra is what let the idle-tx leak survive — TEST-PLAN §3).

---

## 8. Deletions / cleanups (delete what you replace — CLAUDE.md §5)

- **Hardcoded Postgres creds** in `modules/{learning,search}/tests/conftest.py` → replaced by the shared fixture reading `config.py`.
- **Stale `_execute_task` tests** (`test_coordinator_parallel`, `test_82c_wiring`, `test_synthesis_executor`) → repointed to the current methods or removed (verify-before-delete). No `_execute_task` reference survives in `tests/`.
- The empty `tests/e2e/.auth/` stub → filled when used, or removed if the API-level golden suite covers J1–J10 without it.

---

## 9. Out of scope

- **Frontend Playwright + Vitest net** (J1–J7 UI, `lib/api-client.ts`, top `use-*-api` hooks, onboarding wizard, marketplace) — **deferred** (kickoff decision). Lands as **Wave 2.3** follow-on or folds into **Wave 3** frontend hardening. (TEST-PLAN §6.)
- **Per-primitive DoD hardening** (chat/memory/RAG/NL2SQL/graph/missions/playbooks/channels against a written definition of done) — **Wave 3**.
- **Playbook-engine consolidation** + `Workflow`→`Mission` table migration — **Wave 3 / 3R**.
- **HARNESS self-management** test expansion — **Wave 4** (flag-gated, last).
- **80% on all files / the long tail of 103 routers** — opportunistic, not this wave.
- **Nightly live-smoke** against paid LLM/Composio APIs — optional, later.
- If `alembic up/down` in CI proves heavy to stand up this wave, it may slip to Wave 3 — but the **G2 risk is then explicitly un-guarded** and must be called out, not silently dropped.

---

## 10. Success metrics

| Metric | Current (verified 2026-06-02) | Target | How measured |
|---|---|---|---|
| CI runs backend tests on PRs | none (only `check-shopify-isolation`) | blocking gate | `test.yml` (WS-F) |
| Backend tests enforced at merge | 0 of ~1,240 | all | CI required check |
| Stale failing tests (`_execute_task` drift) | ~13 | 0 | WS-F (W2-S2) |
| Real-DB reliability regressions (G1/G4/G3) | 0 | 3 | WS-G |
| Golden journeys with an enforced test | 0 numbered (12 ad-hoc) | 10 / 10 | WS-H `-m golden` |
| Previously-untested primitives (assess/intent/RAG/reconciler) | 4+ with none | 0 | WS-I |
| `alembic up/down` in CI | none | passing (or §9 deferral) | WS-F (G2) |
| Hardcoded test creds | 2 conftests | 0 | WS-F (W2-S4) |
| Frontend e2e | 0 (no Playwright) | deferred | Wave 2.3 / 3 |

---

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Standing up CI with real Postgres + the ~13 pre-existing failures makes the first gate red | High | **WS-F triages the stale tests before** the gate is flipped to *required*; introduce the workflow as non-blocking, go required only when green. |
| Real-DB integration tests are slower / flakier than mocks | Medium | Transactional rollback per test; Postgres **service container**; `slow` marker; keep the fast mock unit tests for inner-loop feedback. |
| Recorded LLM / Composio fixtures drift from the live APIs | Medium | Deterministic recorded fixtures checked in; optional nightly live-smoke subset; fixtures owned next to the test. |
| Formalizing J1–J10 churns the 12 existing journey modules | Medium | **Map first, extend in place** (reuse map §4); characterization-first — pin current behaviour before refactor. |
| Test DB creds leak into code as literals / `os.getenv` | Low | Creds via `config.py` only; CI provides them as env; WS-F (W2-S4) removes the existing literals and a grep gate guards it. |
| Coverage target read as "80% everywhere" → scope blowup | Low | Per-tier target (touched/critical-path), explicit in §2 / DoD; long tail is opportunistic. |

---

**End of PRD-142 Wave 2 (build spec).**
