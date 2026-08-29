# Ralph Build Prompt — PRD-230 Packages & Vertical Onboarding (+ W0 PRD-222 fixes)

You are executing **PRD-230**, one story per iteration, unattended. Branch **`ralph/prd-230-packages` ← `origin/main`** (post-#633). Tip green after every commit.

**THE INVARIANT — read twice (PRD §2 D1/D2/D3).** Anything installed from the marketplace registers to the workspace with its FULL dependency closure, workspace-owned and editable. Agent A with 3 tools + 2 skills + 1 LLM ⇒ exactly 7 registrations. A 6-agent package ⇒ all six closures. Nothing half-installed, nothing platform-dangling. Every install path honors this or the wave fails.

**SCOPE.** 10 stories (`scripts/ralph/prd-230.json` = the BINDING contract; the seeded PRD `docs/PRDS/PRD-230-PACKAGES-VERTICAL-ONBOARDING.md` = intent). US-001/002 are PRD-222 W0 fixes (`fix(prd-222)` prefixes — chat trial metering, doctrine v2); US-003..010 are packages (`feat(prd-230)` prefixes).

## Read first, every iteration

1. `scripts/ralph/prd-230.json` — first story with un-DONE ACs; ACs are binding.
2. The seeded PRD — §2 decisions, §4 design, §8 traps.
3. `CLAUDE.md` — reuse over build; no new tables when one fits; canonical terms.

## The execution contract

- **RE-VERIFY anchors by grep** (kit-time, they drift): factory seam `modules/agents/factory/agent_factory.py:506/:546` + gate `core/llm/manager.py:~718`; chatbot factory import `consumers/chatbot/service.py`; registration tables `core/models/marketplace_plugins.py:186/:216/:243`, `composio_cache.py:171`; marketplace migrations `20260131_add_marketplace_to_agents` + `20260214` LLM tables (re-verify their SHAPES before designing agent/LLM registration); `marketplace-grid.tsx` type/category props; the v2 section's proposal block.
- **ONE migration** (US-003, additive DDL only — no assumptions about existing constraints; the `workspaces_plan_check` prod-drift incident is why).
- **Reuse registration patterns**; new surfaces only where a type has none — cite the reused tables in commit bodies.
- **No auto-connect**: closure app requirements return as `required_connects` for the guided step.
- **PRD-222 surfaces are load-bearing** (opener, power-up, pill, banner, intake card, connect card, checklist, reset + dev page, v2 section, PLAN_TIERS/exposure, trust guards). Breaking any is a hard failure.
- JSONB rebuild-don't-mutate; no `os.getenv` outside `config.py`; walker/schema-truth green after every tool story; public-repo seeds (no customer data; every member ref must resolve).
- PURE tests locally; real-Postgres in CI per-story push. Frontend: vitest + build green after every frontend story.
- **STAGING DISCIPLINE:** explicit paths only; NEVER `git add -A`/`.`/`-u`; never `git stash -u`.

## Hard NOs

- NO second migration; NO parallel registration mechanism where a pattern exists; NO auto-connecting apps.
- NO package pricing/billing; NO verticals beyond the two Shopify packages; NO general web-search capability.
- NO weakening trust guards / the stage validator; NO touching Wave-3 or plans/tiers code beyond reading exposure.
- PUSH each story commit to `origin ralph/prd-230-packages` ONLY. No PRs mid-run, no merges.

## Per-iteration protocol

1. First un-DONE story; re-verify its anchors fresh.
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ frontend vitest/build for frontend stories).
3. Commit (`fix(prd-222)` for US-001/002, `feat(prd-230)` otherwise) with evidence in the body; mark ACs `DONE — <evidence>` in `scripts/ralph/prd-230.json` same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd230.sh`. Exit 0 → reply `RALPH_COMPLETE` (final line, alone).
- Hard-NO conflict → `RALPH_BLOCKED` (final line) + one-line why + grep evidence in the last commit.
