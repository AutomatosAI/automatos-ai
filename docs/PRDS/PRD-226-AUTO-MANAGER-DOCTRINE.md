# PRD-226 — The Manager's Doctrine: Auto's Soul, Lanes, and Dispatch Contracts

> **Status:** Draft for rollout planning — written 2026-08-27, not yet scheduled.
> **Origin:** Munder Difflin deep review (2026-08-27). Their entire management style is one tuned paragraph (`hive.ts:1426` in their tree); ours is missing. Review artifact:
> https://claude.ai/code/artifact/f31677a8-f2cb-47fe-b7dd-f705d764418b
> **Type (per CLAUDE.md §3):** Extension — prompt/seed content and planner prompt structure. Near-zero mechanism.

## 1. Overview

Munder Difflin's GOD agent feels like a manager because its prompt encodes a management doctrine: maintain awareness, delegate rather than implement, check the roster before creating anything, dispatch work as a self-contained contract, own only the high-leverage calls, narrate decisions. The mechanism under it is thin; the doctrine carries the demo. Automatos has the opposite balance — a strong mechanism (matcher, dispatcher, leases, budgets) and no doctrine: AutoBrain classifies complexity and picks RESPOND/DELEGATE/MISSION, and nothing tells Auto *how to behave as a manager*. This PRD ports the doctrine, adapted to our lanes and surfaces.

## 2. Current reality (grounded)

- Auto's persona is seeded per workspace from `core/seeds/auto-cto-custom-soul.txt` plus a `platform-management` skill (`core/seeds/platform-management-skill.md`), by `core/seeds/seed_auto_agent.py:29-84`. Personas live in the DB (house rule: no file hacks for DB data — the seed files are the source, rows are the runtime).
- AutoBrain: `consumers/chatbot/auto.py:587 AutoBrain`, `:613 assess()` → complexity (ATOM…ORGANISM) + action (`RESPOND|DELEGATE|MISSION`) + `tool_hints` + `target_agent_id`. No management guidance in the assessment prompt.
- Mission task specs: `MissionPlanner.decompose` (`modules/coordination/planner.py:341,488`, prompt at `:756`) emits goal-decomposed tasks with no required contract structure; verification (`modules/coordination/verification.py`) scores without an explicit per-task definition of done.
- The matcher already *scores* reuse (skills, tool coverage, history, busyness — `agent_matcher.py:578`), but nothing tells Auto to prefer routing to an existing agent over creating one (`platform_create_agent` is one tool call away).
- Board-as-ledger: agent tools exist (`actions_board_tasks.py`), but Auto is never instructed that multi-step work belongs on the board.

## 3. Goals

- G1: Auto's seeded persona carries the manager doctrine (below), versioned in the seed files and backfilled to existing workspace Auto rows.
- G2: Every dispatch — PRD-224 ticket descriptions and mission task specs — is a **4-part contract**: OBJECTIVE · OUTPUT · TOOLS (use/avoid + references to read instead of re-deriving) · BOUNDARIES (scope limits + definition of done). References, not pasted content.
- G3: Verification and PRD-224 watches score against the contract's definition of done instead of inferring one.
- G4: Doctrine changes are gated by the local gold-set evals before shipping (prompt-regression risk; gold sets never committed — repo is public).

## 4. The doctrine (the content itself)

Adapted from their orchestrator prompt; ours, adjusted to Automatos lanes:

1. **Awareness.** Know the floor before acting: roster, board, in-flight missions, watches. (Grounded by PRD-228's fleet tool once it lands; until then `platform_board_summary` + `platform_list_missions` + `platform_list_agents`.)
2. **Three lanes, chosen deliberately.** DELEGATE = the specialist answers this conversation. ASSIGN (PRD-224) = a named/single agent does work off-thread, on the board, supervised. MISSION = multi-agent project, planner-staffed. Say which lane and why in one line.
3. **Delegate, don't implement.** Auto owns decomposition, dispatch, sign-off, conflict resolution, and QA — not the grunt work its agents exist for.
4. **Reuse before creating.** Check the live roster first; honor named routing ("have Jim…" goes to Jim); create a new agent only when nothing fits, and say that you checked. One capable owner beats a duplicate.
5. **Dispatch as a contract.** The 4-part structure (G2) on every handoff, short, referencing artifacts rather than inlining them.
6. **Board as ledger.** Any multi-step ask gets a board card first — the Command Center must always show what Auto believes the floor is doing.
7. **Asks are decisions, not reports.** When raising a human question (PRD-225): one bold sentence stating the need, options as bullets, ≤ ~700 chars, markdown; rewrite asks that originate in agent reports — never make the human read the investigation to find the decision. Never idle-wait for an answer.
8. **Recurring work becomes a playbook.** On a repeated ask, propose `platform_create_playbook` + schedule; supervision via watches already exists.
9. **Narrate.** Every assignment, escalation, and sign-off gets a one-line explanation in the thread — visibility is the product.

## 5. Design

- **Component A — seed update + backfill.** Doctrine block into `auto-cto-custom-soul.txt` (identity-level) and lane/contract mechanics into `platform-management-skill.md` (procedure-level). Auto rows are per-workspace (`auto-{workspace_id}`): ship an idempotent backfill (migration or seed-sync path — follow whichever mechanism `seed_auto_agent.py` already uses for updates) so existing workspaces get the new soul. Old text is replaced, not appended (no compat shims).
- **Component B — assessment guidance.** The AutoBrain assessment prompt gains the three-lane rubric (with PRD-224's ASSIGN) and reuse-before-create signals; `tool_hints` steer accordingly.
- **Component C — planner contract.** `MissionPlanner.decompose` prompt (`planner.py:756`) requires the 4-part structure per task; parser stores `definition_of_done` alongside the spec; `verification.py` and PRD-224 watch scoring consume it when present. Missing DoD ⇒ verification falls back to today's inference (no hard dependency between waves).
- **Component D — eval gate.** Extend the local gold sets with lane-selection and contract-shape cases; doctrine PRs run the eval before merge. (CI is the only gate; evals run in CI with local-only fixtures per the public-repo rule.)

## 6. Waves & acceptance criteria

**Wave 0 — draft + eval offline.**
- [ ] Doctrine text drafted; gold-set cases written (lane selection incl. "have my accountant…"→ASSIGN, "research and build…"→MISSION, "what does X mean"→RESPOND/DELEGATE; contract-shape checks).
- [ ] Baseline eval run recorded for comparison.

**Wave 1 — soul + skill + backfill.**
- [ ] Seeds updated; backfill applied; a fresh workspace and an existing workspace both serve the new persona (verified via seeded-agent fetch in CI tests).
- [ ] Lane-selection eval ≥ baseline on non-lane cases (no regression), passes new lane cases.

**Wave 2 — planner contracts.**
- [ ] Decomposed mission tasks carry the 4-part structure; `definition_of_done` persisted; verification consumes it when present (unit tests on parse + score paths).
- [ ] PRD-224 ticket descriptions produced by the ASSIGN lane follow the same shape (shared prompt fragment, not duplicated text).

## 7. Technical considerations

- Prompt fragments shared between the ASSIGN lane (PRD-224) and the planner live in one place (seed/skill), not copy-pasted into code — single source, per house rules.
- Token cost: the doctrine block is O(300 tokens) in CHATBOT context; acceptable against the existing section budget (`modules/context/modes.py:40-50`), but measure in Wave 1.
- No routes, no migrations beyond the persona backfill mechanism.

## 8. Open questions (Gerard)

1. Doctrine tone: Auto's soul is "CTO" today — keep CTO voice or shift toward "chief of staff / floor manager" for the management sections?
2. Should narration (doctrine §9) be a per-workspace dial (some users may find one-line explanations noisy)?
3. Backfill policy: force-replace all workspace Auto souls, or only workspaces that haven't customized theirs?
