# Ralph Build Prompt — PRD-164 Planning Intelligence & Integration Seams (WS-9)

You are executing **PRD-164**, one story per iteration, unattended overnight. This branch is **`ralph/prd-164-planning-intelligence`, cut from `main`** (independent track — NOT stacked on another branch). The tip must be green after every commit.

This is the **payoff PRD**: everything before it built the read-side this consumes. The platform starts to FLOW — every planner consults what the platform knows (RAG+memory+graph+roster), agent selection goes semantic, agent outputs feed back into knowledge (D6 flywheel), chat renders platform actions as real widgets, recurring agents learn. Its dependencies (PRD-157 RAG, 159 memory, 163 missions, 165 KG, 166 field-memory) are **all merged on main** — your job is to WIRE them, not reimplement them.

## Read first, every iteration

1. `scripts/ralph/prd-164.json` — the story list (`description` = BINDING contract + amendments). Pick the **first story whose ACs are not all marked DONE**.
2. `docs/PRDS/PRD-164-PLANNING-INTELLIGENCE-SEAMS.md` — full spec + binding amendments **Q21, Q22, Q58, Q60, Q61, Q62**.
3. `CLAUDE.md` — **reuse over build**; **delete what you replace**; no shims; no `os.getenv` outside `config.py`; canonical terms (**Playbook** not Recipe, **Mission** not Workflow, **Command Center** not Activity). Read the memory anti-patterns: never build custom action scoring — extend the existing matcher; never build a parallel context assembly.

## Ground truth (verified 2026-06-12 — re-grep before every edit; lines drift)

- **RAG choke point (157)**: `orchestrator/modules/rag/retrieval_filters.py` `build_retrieval_filters(...)`. The planning pack retrieves through THIS — never a parallel retrieval.
- **Planners that must converge on ONE pack (Q61)**: MissionPlanner `orchestrator/modules/coordination/planner.py`; board `plan_task` `orchestrator/api/board_tasks.py:980`; AutoBrain `orchestrator/consumers/chatbot/auto.py`.
- **AgentMatcher — EXTEND, do NOT replace**: `orchestrator/modules/coordination/agent_matcher.py` — `class AgentMatcher` with `match()` at ~line 114; it already scores `_compute_skill_match` + `_compute_model_fit` + `_build_history_map`. S2 ADDS capability-card embeddings (Qdrant) + live field signal + reasons on top of the existing scorer.
- **Field signal (166)**: `orchestrator/modules/context/adapters/vector_field.py` — the digest source for S2 (selection signal) and S4 (dispatch digest).
- **Ingestion manager (flywheel target, S3)**: `orchestrator/modules/rag/ingestion/pipeline.py` (+ `processor.py`). Route agent outputs through this — no parallel ingestion path.
- **Dead widget router (DELETE in S5, Q62)**: `frontend/components/widgets/router.ts:35` — `TOOL_WIDGET_MAP` (+ the add/delete helpers ~line 849-867). Replace with live tool-name routing.
- **Tool registration**: the canonical 3-file platform-tool pattern (see memory `prd71-tools`) for the S3 deliverable list/get tools — never register Composio actions individually.

## The execution contract

- **TDD**: failing test first, then implement, then green. Every behavioural fix needs a test that FAILS before and PASSES after.
- **Story scope**: the story's `files` list is your scope. A file outside it may be touched only when obviously required — name it in the commit body. A structural surprise (a signature change rippling across many callers, a schema surprise) → reply `RALPH_BLOCKED`, do not improvise.
- **Testing model — CI validates the backend, NOT this machine.** There is **no local database** and no containers here. Do **NOT** run `cd orchestrator && python3 -m pytest -q` — the full suite blocks on `test_82c_wiring`'s real Postgres connect and wedges the iteration. Instead:
  - **Backend** → write the failing test first (TDD), implement, then **commit + push**. The push triggers CI (`test.yml`, real Postgres) which runs the suite — CI is the authority. Locally you may run ONLY a **DB-free** isolated unit test for pure logic — e.g. `python3 -m pytest tests/<your_pure_test>.py -q` for the pack budgeter, the matcher scoring blend, the field-digest sizing — plus `python3 -m py_compile` on changed files. Anything needing a DB/Qdrant session is verified on CI after push.
  - **Frontend** (no DB) → `cd frontend && npx tsc --noEmit` AND `npm run test` (vitest) green; `npm run lint` when the story touches a lint rule. These gate the commit locally.
- **New backend test files importing `modules.*`/`consumers.*` at module level MUST start with the collection-order guard** (copy the `_sys_guard` block from `orchestrator/tests/test_prd143_selection_at_scale.py`): Linux CI collection order differs from macOS; unguarded imports die at collection even when green locally.
- **Never weaken a test to pass.** A test asserting OLD behavior is UPDATED or DELETED with the code it covers — never inverted to hide a regression.
- **Clean tree after every commit**: `git status --porcelain` must be EMPTY post-commit — an untracked new file passes locally and dies on CI checkout.
- **Protected suites (explicit gates — this PRD touches their neighborhoods): recipe (20 tests) + hint (25 tests).** They must stay green on CI.

## Browser-verify ACs (S3 deliverables tab, S5 chat widgets) — do NOT block on them

This loop is **headless with no running app** — you **cannot** satisfy "verify in browser using dev-browser skill" interactively and they **DO NOT gate completion**. Each is paired with a deterministic proxy (a reachability test, a registry-validation test, a render/behaviour vitest). Satisfy the **deterministic proxy**, then mark the browser AC `DEFERRED — morning browser check: <what to look at>` in `prd-164.json`. Implement the real fix fully; only the *visual confirmation* is deferred. **Never start a dev server, never call dev-browser.**

## Story-specific guardrails (full ACs live in `prd-164.json`)

- **S1 — Planning Context Pack**: build ONE `ContextService` "planning" mode that assembles RAG(157)+memory(159 recall)+KG(165 subgraph)+roster, token-budgeted (reuse the 157 token budgeter). This story builds the **assembler only**; the golden "seeded prior failure changes the plan" demo is proven when the planners consume it (S1 acceptance pairs the two). No parallel retrieval — call `build_retrieval_filters`.
- **S2 — Semantic agent matching (Q21)**: **EXTEND `agent_matcher.py`** — blend capability-card embeddings (Qdrant) + history (already partial) + live field signal (166) → ranked agents + **reason strings**; explicit `agent_overrides` (163 S4) **always win**. One embedding call per dispatch is acceptable. Golden matrix of 10 fixtures + override-wins test. Do NOT create a second matcher.
- **S3 — Output flywheel (Q58)**: route mission syntheses, generated documents, submitted reports through the **existing** `ingestion/pipeline.py` tagged `source_type='agent_output'`; KG incremental build learns the three source types; flywheel **ON by default, per-workspace opt-out** (test the opt-out ingests nothing); deliverable list/get tools via the 3-file pattern + mission-page tab. If `source_type='agent_output'` needs a column/enum, write a **real alembic revision** (never a shim) and note the stamp in the commit body.
- **S4 — Field-digest dispatch + replanning (Q22)**: replace the 8K-char upstream-output stuffing with the 166 field digest under a per-task budget; dispatch prompt size drops **≥60%** on the multi-task fixture while the golden task still passes. Add bounded replanning (LLMCompiler joiner) + progress ledger/stall counter (Magentic-One) with an audit trail. No new planner algorithms beyond joiner/ledger.
- **S5 — Chat renders the platform (Q62, Q60)**: **DELETE** `TOOL_WIDGET_MAP` + its helpers (delete-what-you-replace); live widget routing keyed on tool names for mission cards/board tasks/deliverables/documents/schedule/memory writes, validated against the registry in the **PRD-155 reachability test** (drift impossible). Heartbeat agents get scoped memory recall + write hooks (statelessness ends) — a heartbeat run recalls a memory written by its previous run; recall is workspace/agent-scoped (no cross-tenant reach).

## Hard NOs (human-gated — violating any is RALPH_ABORT territory)

- **NO parallel systems**: no second AgentMatcher, no parallel planning-context assembly, no parallel ingestion path, no custom action scoring. EXTEND the existing seams named above.
- **Migrations only where a story genuinely adds schema** (S3 `source_type` if absent) — a real alembic revision, noted in the commit; most stories are pure wiring and need none. NO `alembic upgrade head` against a real DB here (no DB on this machine).
- NO `os.getenv` outside `config.py`. NO hardcoded values. NO secrets in code/fixtures.
- **PUSH after each story commit to `origin ralph/prd-164-planning-intelligence` ONLY** — never force-push, never another ref, never `main`. **NO PRs mid-run** (the runner opens a draft PR at the end). **NO merges.** A NEW CI red is a bug to fix in-scope.
- Do NOT leave a replaced surface running "just in case" — the dead `TOOL_WIDGET_MAP` is deleted this PRD (CLAUDE.md §5).

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its ground truth fresh (grep — don't trust line numbers).
2. Write the failing test(s). Implement minimally. Run the story's DB-free gates + relevant pure suite.
3. Commit `feat(prd-164): <story-id> — <title>`, AC evidence in the body. Mark that story's AC lines `DONE — <evidence>` (or `DEFERRED — …` for browser ACs) in `scripts/ralph/prd-164.json` **in the same commit**. Then push.

## Completion

- All ACs DONE/DEFERRED → run `bash scripts/ralph/acceptance-prd164.sh` (DB-free local gates only). Exit 0 → reply `RALPH_COMPLETE`.
- **The backend suite is NOT in the local gate — it runs on CI.** Make sure your final commit is **pushed** so CI has a run to evaluate; the runner records the CI result in the night report and the morning human merges only on green CI.
- Local DB-free gate red, or a backend bug you can prove with a DB-free unit test → fix it in the owning story. Out-of-scope cause, ambiguity, or something only a DB/Qdrant can confirm and you can't → reply `RALPH_BLOCKED` with one line of why (the tip stays green by construction, so the chain continues).
