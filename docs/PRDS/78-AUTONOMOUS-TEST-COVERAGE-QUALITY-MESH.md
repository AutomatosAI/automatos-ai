# PRD-78: Autonomous Test Coverage & Quality Mesh

**Version:** 1.1
**Status:** Active
**Priority:** P0
**Author:** Gar Kavanagh + Auto CTO
**Created:** 2026-03-10
**Updated:** 2026-03-12
**Dependencies:** PRD-05 (Memory & Knowledge), PRD-55 (Autonomous Assistant Platform), PRD-68 (Progressive Complexity Routing), PRD-69 (Agent Intelligence Layer), PRD-72 (Activity Command Centre), PRD-73 (Observability & Monitoring Stack), PRD-77 (Agent Self-Scheduling & Memory Dashboard)

---

## Executive Summary

Automatos already has the foundations for autonomous quality engineering: API tests, workflow execution, memory, scheduling, skills, platform observability, and agents that can raise Jira tickets and fix bugs. What it does not yet have is a **deliberately designed quality mesh**: a structured testing architecture that grows with the platform, runs at different cadences, produces machine-readable artifacts, and allows specialist agents to cooperate reliably.

**Current baseline (2026-03-12):** 124 API integration tests across 23 domain files, 3 runner scripts, 1 audit/gap-finder tool. No unit tests, no browser tests, no regression-pin tests.

Today, testing is partially present:

1. `tests/run_nightly.py` runs the broad suite and produces summaries.
2. `tests/run_health_regression.py` runs a curated high-signal subset.
3. `tests/run_gap_finder.py` inventories the suite and detects gaps.
4. Skills exist for QA analysis, Jira administration, and automated bug fixing.

The missing piece is the **system design** that turns these into a coordinated quality program:

- A test taxonomy that grows organically as the platform stabilizes
- Scheduled execution lanes by speed, risk, and purpose
- Consistent artifacts that downstream agents can consume
- A clear role for browser automation, internal tool validation, API contracts, and worker-level regressions
- Optional specialist agents and model routing for quality tasks

This PRD defines that system and its growth milestones.

### Growth Philosophy

This is a **living architecture**, not a "write 500 tests" project. The test suite grows in lockstep with platform maturity:

1. **Pilot Readiness (now)** — Flush bugs, pin known regressions, deepen P0 coverage so 10-15 pilot users hit a stable platform.
2. **Post-Pilot Hardening** — As real users surface issues, each bug becomes a regression test. Coverage grows from usage, not quotas.
3. **Mature Platform** — When the platform stops changing rapidly, fill out the full taxonomy. The 500+ target is a north star, not a Phase 1 deliverable.

### What We're Building

1. **A scalable test coverage architecture** spanning API integration tests, regression pins, internal tool validation, stateful journeys, and eventually browser tests — growing as the platform matures
2. **Three production testing recipes** with clean runner scripts, schedules, artifact outputs, and downstream handoffs:
   - Nightly Self-Test Suite
   - API Health Check & Regression Detector
   - Weekly Test Coverage Gap Finder
3. **A specialist-agent quality workflow** where QA, Jira, and Bug Fixer agents cooperate with rigid artifact contracts
4. **A scheduling model** that separates PR checks, hourly health, nightly full confidence, weekly gap analysis, and release validation
5. **A quality artifact standard** that lets agents create Jira issues, classify severity, trace source files, and drive bug-fixing without ad hoc reasoning

### What We're NOT Building

- A separate test repository right now
- Full load/performance engineering in this PRD
- Chaos testing or production incident automation
- A new CI platform
- A full synthetic monitoring platform for deployed environments
- 500 tests before the platform is stable enough to warrant them

Those can become future PRDs once the in-repo testing mesh is stable.

---

## 1. Problem Statement

Automatos needs more than "more tests." It needs **structured, scalable, agent-operable testing**.

Current pain points:

1. **Coverage is present but uneven.** Many current tests are smoke checks per endpoint. There are fewer stateful user journeys and fewer end-to-end contract validations than the platform now needs.
2. **No formal test taxonomy exists.** Without grouping by risk, speed, and schedule, a 500+ suite will become slow, noisy, and hard to reason about.
3. **The QA -> Jira -> Bug Fixer handoff was previously implicit.** Artifact names, scratchpad payloads, and runner outputs were not rigid enough for reliable automation.
4. **Browser simulation is underdeveloped.** The product has rich UI and multi-step agent workflows, but browser-level user journeys are not yet a first-class layer.
5. **Internal tools need deeper validation.** Platform actions, scratchpad contracts, scheduler behavior, memory logic, recipe execution, and agent orchestration require test coverage beyond HTTP route checks.
6. **Testing must stay version-aligned with the product.** Since routes, payloads, workflows, runner outputs, and agent behaviors change rapidly, test logic should evolve in the same repo and release cycle as the code.

---

## 2. Strategic Decision: Keep Testing In-Repo

The test system remains inside `automatos-ai`.

### Why

1. **Version alignment matters more than separation right now.**
   - API routes, response schemas, internal tools, UI behavior, and workflow contracts change together.
   - Bug fixes and test fixes often belong in the same branch and PR.

2. **The downstream agents depend on repo-relative evidence.**
   - `qa-engineer` emits repo-relative `source_files`
   - `jira-admin` puts those references into tickets
   - `bug-fixer` reproduces directly from those paths

3. **Agent workflows are easier when code and tests share one lifecycle.**
   - A separate repo would add drift, duplicated review overhead, and slower iteration.

### Future Split Line

Only split into a separate quality repo later for:

- load testing
- deployed-environment black-box validation
- synthetic monitoring
- chaos testing
- cross-repo certification

Core regression, journey, API, worker, and internal tool tests stay in `automatos-ai`.

---

## 3. Coverage Architecture

We will build a **test pyramid plus schedule matrix**.

### 3.1 Layer A: Fast deterministic tests

**Target:** 180-220 tests

Purpose:
- Catch refactor regressions quickly
- Validate pure logic and service behavior
- Keep PR feedback fast

Coverage examples:
- memory classification and formatting
- model config validation
- severity and category mapping
- runner output builders
- route payload validators
- tool registration and formatting
- scratchpad contract builders
- workflow stage/event formatting
- scheduler utilities

### 3.2 Layer B: API integration tests

**Target:** 140-180 tests

Purpose:
- Validate backend route contracts
- Verify real state changes and resource lifecycles

Coverage examples:
- agents CRUD + model config + execute
- chat create/history/rename/delete
- memory stats/search/recent
- workflows execute/status/cancel
- heartbeat endpoints
- routing, channels, tools, skills, personas
- documents, knowledge, webhooks, keys, analytics

### 3.3 Layer C: Browser / Playwright journeys

**Target:** 80-120 tests

Purpose:
- Simulate real user behavior
- Catch UI regressions, stale state, broken navigation, and role-based visibility issues

Coverage examples:
- login / workspace landing
- create and configure agent
- edit model config
- enable heartbeat
- create and run recipe
- open chat, follow up, rename, revisit history
- document upload + processing + search
- routing rule management
- activity feed and command centre flows
- settings pages

### 3.4 Layer D: Internal tools, worker, and orchestration tests

**Target:** 60-90 tests

Purpose:
- Validate Automatos-specific machinery not visible through simple route smoke tests

Coverage examples:
- `platform_*` executor behavior
- tool routing
- scratchpad read/write contracts
- scheduler / recipe scheduler / heartbeat service
- memory daily logs and access logging
- workflow execution streaming and SSE/AI SDK event shape
- task runner queued vs local behavior
- Jira/QA/Bug Fixer artifact compatibility

### 3.5 Layer E: Regression contracts

**Target:** 30-50 tests

Purpose:
- Pin the expensive bugs
- Prevent repeated break/fix cycles

Coverage examples:
- memory scoping regressions
- null-handling bugs
- response-shape regressions
- bad SQL column/path regressions
- runner artifact contract regressions
- Jira evidence handoff regressions
- workflow execution handle regressions

### 3.6 Coverage Growth by Milestone

Tests grow with the platform. Targets are approximate and driven by need, not quotas.

| Layer | Pilot Readiness | Post-Pilot | Mature Platform |
|---|---:|---:|---:|
| API integration (Layer B) | 160+ | 200+ | 250+ |
| Regression contracts (Layer E) | 15+ | 40+ | 60+ |
| Internal tools/worker (Layer D) | 10+ | 30+ | 70+ |
| Stateful journeys (Layer B+) | 10+ | 30+ | 50+ |
| Playwright/browser (Layer C) | 0 | 10+ | 80+ |
| Fast deterministic (Layer A) | 0 | 50+ | 150+ |
| **Total** | **~195** | **~360** | **~660** |

**Pilot Readiness** = what we need before 10-15 users hit the platform.
**Post-Pilot** = bugs surfaced by real usage become regression tests; stable areas get deeper journeys.
**Mature Platform** = full taxonomy, browser coverage, unit test layer, release validation lane.

---

## 4. Test Grouping by Risk

### P0 Critical Areas

These get the deepest coverage first:

- authentication and workspace scoping
- chat and orchestration
- memory and Mem0 integration
- workflows and recipes
- heartbeat and scheduler
- internal tool execution
- runner artifact contracts
- document and knowledge retrieval

### P1 High-Value Product Areas

- agent configuration and model config
- routing and channels
- skills/plugins assignment
- analytics and activity
- settings pages
- workspace file/exec

### P2 Nice-to-Have Areas

- UI polish states
- rare edge cases
- non-blocking admin screens
- long-tail validation and degraded fallback flows

---

## 5. Schedule Matrix

Not all 500+ tests run all the time.

### 5.1 PR / Pre-Merge Lane

**Runtime target:** 5-10 minutes

Run:
- fast deterministic tests
- critical API smoke and key regressions
- selected internal contract tests
- optional tiny browser smoke pack

Purpose:
- block obvious breakage
- fast developer feedback

### 5.2 API Health Check & Regression Detector

**Script:** `python3 tests/run_health_regression.py`

**Runtime target:** 3-8 minutes

Run:
- curated high-signal API subset
- chat, agent, memory, workflow, heartbeat checks
- critical user journeys
- required orchestrator regression tests

Outputs:
- `health-regression-report.json`
- `health-regression-summary.json`
- `qa-report.json`

Purpose:
- detect fresh regressions
- drive QA analysis and Jira filing

### 5.3 Nightly Self-Test Suite

**Script:** `python3 tests/run_nightly.py`

**Runtime target:** 20-45 minutes

Run:
- full API suite
- required orchestrator regressions
- growing internal platform validation
- selected slower integration tests

Outputs:
- `test-report.json`
- `test-summary.json`

Purpose:
- broad nightly confidence
- historical trend and run-level status

### 5.4 Weekly Test Coverage Gap Finder

**Script:** `python3 tests/run_gap_finder.py`

**Runtime target:** 1-5 minutes for audit mode, longer if later expanded with execution checks

Run:
- test inventory scan
- domain coverage analysis
- journey vs smoke classification
- missing-domain and weak-coverage detection

Outputs:
- `coverage-gap-summary.json`

Purpose:
- identify test debt
- create weekly planning work

### 5.5 Release Validation Lane

**Runtime target:** 45-120 minutes

Run:
- nightly suite
- full browser pack
- role/permission matrix
- worker/scheduler deep checks
- document and channel flows
- optional deployment-specific validation

Purpose:
- release confidence

---

## 6. Runner Scripts and Artifact Contracts

### 6.1 Existing / Planned Runner Scripts

| Recipe | Script | Purpose |
|---|---|---|
| Nightly Self-Test Suite | `tests/run_nightly.py` | Broad full-suite validation |
| API Health Check & Regression Detector | `tests/run_health_regression.py` | Fast high-signal regression lane |
| Weekly Test Coverage Gap Finder | `tests/run_gap_finder.py` | Coverage inventory and planning |

### 6.2 Artifact Contracts

#### `test-summary.json`

Audience:
- nightly recipes
- high-level status reporting

Shape:
- total tests
- pass/fail counts
- duration
- failures with `nodeid`, `assertion_message`, `source_files`

#### `health-regression-summary.json`

Audience:
- run-level health reporting
- Jira overview

Shape:
- targeted suite summary
- curated test target list
- high-signal failure list

#### `qa-report.json`

Audience:
- `qa-engineer`
- `jira-admin`
- `bug-fixer`

Shape:

```json
{
  "run_date": "...",
  "total": 0,
  "passed": 0,
  "failed": 0,
  "skipped": 0,
  "pass_rate": "0%",
  "status": "PASS|FAIL",
  "platform_logs": null,
  "log_fetch_required": true,
  "bugs": [
    {
      "test": "tests/api/test_memory.py::test_memory_stats_real",
      "severity": "P1",
      "title": "[memory] Example failure",
      "error": "AssertionError: ...",
      "traceback": "...",
      "server_log": null,
      "source_files": ["orchestrator/api/memory_stats.py:35"],
      "category": "memory"
    }
  ]
}
```

#### `coverage-gap-summary.json`

Audience:
- weekly planning recipes
- `jira-admin`
- future test planner agents

Shape:
- covered domains
- missing expected domains
- journey files
- smoke files
- module inventory
- action items

---

## 7. Specialist Agents and Skills

### 7.1 Core Specialist Agents

#### QA Engineer

Primary responsibilities:
- execute runner scripts
- classify failures
- enrich failures with logs
- generate `qa_report`

Primary skill:
- `automatos-skills/qa-engineer`

Preferred model profile:
- balanced reasoning, high reliability, moderate cost

Why:
- needs structured analysis, classification, and evidence correlation

#### Jira Admin

Primary responsibilities:
- create/update issues from test artifacts
- map severity to Jira priorities
- export rich `issue_details`

Primary skill:
- `automatos-skills/jira-admin`

Preferred model profile:
- concise, schema-following, low hallucination

Why:
- operational accuracy matters more than creativity

#### Bug Fixer

Primary responsibilities:
- reproduce
- write failing test
- apply minimal fix
- verify and prepare PR

Primary skill:
- `automatos-skills/bug-fixer`

Preferred model profile:
- stronger code reasoning and repo navigation

Why:
- this is the most expensive but most value-dense step

### 7.2 Future Specialist Agents (Post-Pilot)

These agents are **not needed until the suite exceeds ~300 tests and real users are generating regression data**. Do not build them prematurely.

#### Playwright Runner (Post-Pilot)

Responsibilities:
- browser automation
- UI regressions
- journey validation
- screenshot and trace capture

#### Contract Auditor (Post-Pilot)

Responsibilities:
- verify response schemas
- compare API output against expected structures
- detect artifact drift

#### Flake Hunter (Mature Platform)

Responsibilities:
- detect unstable tests across repeated runs
- classify infra flake vs product regression
- quarantine candidates

#### Coverage Planner (Mature Platform)

Responsibilities:
- read `coverage-gap-summary.json`
- group missing areas into actionable themes
- propose next test additions and schedules

### 7.3 Model Routing by Task Type

| Agent / Task | Model Tier | Why |
|---|---|---|
| QA Engineer | Standard | classification + evidence synthesis |
| Jira Admin | Economy/Standard | structured summarization and ticketing |
| Bug Fixer | Premium | code reasoning, test-first fixes |
| Playwright Runner | Standard | interaction reliability |
| Coverage Planner | Standard | structured planning |
| Flake Hunter | Standard/Premium | pattern detection across failures |

---

## 8. Skill and Recipe Coordination

### 8.1 Dynamic Skill Use

The skills are **context-aware**, not hardcoded to only the three internal test recipes.

Rules:

1. If dedicated runner outputs exist, prefer them.
2. If they do not exist, operate from raw pytest, route output, browser output, or platform logs.
3. Skills must remain useful outside the recipe chain.

### 8.2 Preferred Recipe Flow

#### Flow A: Regression Bug Flow

1. `qa-engineer` runs `run_health_regression.py`
2. `qa-engineer` enriches `qa-report.json` with platform logs when failures exist
3. `jira-admin` reads `qa_report`
4. `jira-admin` creates Jira issues and exports `issue_details`
5. `bug-fixer` reads `issue_details`
6. `bug-fixer` reproduces, tests, fixes, verifies, PRs

#### Flow B: Nightly Oversight Flow

1. `qa-engineer` runs `run_nightly.py`
2. `qa-engineer` reviews `test-summary.json`
3. `jira-admin` creates aggregated nightly bug/tasks as needed

#### Flow C: Weekly Planning Flow

1. `qa-engineer` or `coverage planner` runs `run_gap_finder.py`
2. `jira-admin` creates Tasks/Stories from `coverage-gap-summary.json`
3. planning agent groups future coverage work by domain and priority

---

## 9. Browser and Playwright Strategy

### 9.1 Browser Test Groups

Organize by journey, not by page:

- `tests/playwright/smoke/`
- `tests/playwright/chat/`
- `tests/playwright/agents/`
- `tests/playwright/recipes/`
- `tests/playwright/knowledge/`
- `tests/playwright/admin/`
- `tests/playwright/error_states/`

### 9.2 Browser Priorities

Phase 1:
- login / landing
- create agent
- chat basic flow
- create/run recipe
- document upload

Phase 2:
- routing/channels/settings
- activity feed
- memory dashboard
- browser error handling and retries

Phase 3:
- role matrix
- long multi-step workflows
- mobile and cross-browser coverage

---

## 10. File and Folder Strategy

### 10.1 Recommended In-Repo Structure

```text
tests/
├── api/                         # HTTP integration tests
├── playwright/                  # Browser journeys
├── contracts/                   # Artifact / schema / output contracts
├── regressions/                 # Bug-pin tests
├── journeys/                    # Multi-step stateful backend journeys
├── fixtures/                    # Shared data builders and fakes
├── reports/                     # Local fallback outputs
├── run_nightly.py
├── run_health_regression.py
├── run_gap_finder.py
├── audit_suite.py
├── RECIPE_RUNNERS.md
└── README.md
```

### 10.2 Naming Rules

- `test_<domain>.py` for domain suites
- `test_<journey>.py` for stateful multi-step flows
- `test_<bug_or_contract>.py` for regressions
- browser suites grouped by user intent, not component name

---

## 11. Implementation Plan

### Milestone 1: Pilot Readiness (current priority)

**Goal:** Flush bugs and pin regressions so 10-15 pilot users hit a stable platform.

**Exit criteria:** All P0 areas have at least one stateful journey test. Known regressions from memory system, multi-tenancy, and cloud doc sync are pinned. Runners produce clean artifacts. ~160-200 total tests.

1. Stabilize the three runner scripts and lock artifact schemas
2. Create `tests/regressions/` directory and pin known bugs:
   - Memory scoping regressions (Mem0 search, user_id format mismatch)
   - Multi-tenancy isolation (workspace fallback bug)
   - Cloud document sync (silent processing failure)
   - AgentFactory tool source divergence
3. Deepen P0 API tests with stateful journeys:
   - Chat: create → send → history → rename → delete
   - Memory: store → search → stats consistency
   - Workflows: create → execute → status → cancel
   - Heartbeat: enable → trigger → verify results
   - Agents: create → configure → assign tools → execute → delete
4. Add test data cleanup strategy (teardown fixtures or dedicated test workspace)
5. Add internal tool contract tests for `platform_*` executor paths
6. Ensure `run_health_regression.py` targets include new regression tests
7. Verify QA → Jira → Bug Fixer artifact handoff works end-to-end

### Milestone 2: Post-Pilot Hardening

**Gate:** 10-15 users actively using the platform and generating feedback.

**Goal:** Every user-reported bug becomes a regression test. Deepen coverage in areas users actually exercise.

8. Convert each pilot bug into a regression test in `tests/regressions/`
9. Expand API journeys to cover user-reported flows
10. Add internal tool and worker tests for scheduler, recipe execution, scratchpad
11. Begin Playwright smoke pack (login, create agent, basic chat) — only if UI bugs are a real problem
12. Grow to ~300-360 tests

### Milestone 3: Mature Platform

**Gate:** Platform is stable, release cadence is predictable, API surface is not changing weekly.

13. Stand up full Playwright browser journey suite
14. Add fast deterministic unit tests for pure logic
15. Add flake detection and release validation lane
16. Reach 500+ tests across all layers
17. Evaluate specialist agents (Contract Auditor, Flake Hunter, Coverage Planner)
18. Review split-line for external quality repo only if lifecycle divergence demands it

---

## 12. Success Metrics

### Pilot Readiness (Milestone 1)

| Metric | Target |
|---|---:|
| P0 domains with at least one stateful journey test | 100% |
| Known regressions pinned as tests | 100% |
| `qa-report.json` directly usable by Jira Admin | 100% of runs |
| Health regression runtime | < 8 min |
| Nightly full runtime | < 30 min |
| Total tests | ~160-200 |

### Post-Pilot (Milestone 2)

| Metric | Target |
|---|---:|
| User-reported bugs encoded as regression tests | 90%+ |
| Bug Fixer able to reproduce from `issue_details` | 90%+ |
| All P1 domains covered | 100% |
| Total tests | ~300-360 |

### Mature Platform (Milestone 3)

| Metric | Target |
|---|---:|
| Total automated tests | 500+ |
| Browser journey coverage for critical flows | 80%+ |
| PR lane runtime | < 10 min |
| Nightly full runtime | < 45 min |
| Weekly gap finder output contains actionable tasks | 100% |

---

## 13. Open Questions and Decisions

### Resolved

1. **Should platform logs be fetched inside runner scripts or remain agent-enriched?**
   **Decision:** Runner-enriched (deterministic). Runner scripts fetch logs when failures occur and embed them in `qa-report.json`. Agents can augment further but the baseline artifact must be self-contained.

2. **Do we need a dedicated Coverage Planner agent now or later?**
   **Decision:** Later (Milestone 3). The weekly gap finder output is consumed manually or by `jira-admin` until the suite is large enough to warrant a dedicated planner.

3. **When do we add the 4 new specialist agents?**
   **Decision:** Not until Milestone 2 at earliest. The 3 existing agents (QA Engineer, Jira Admin, Bug Fixer) handle Milestone 1. New agents are gated on real need, not PRD ambition.

### Still Open

4. **When should browser smoke move into the health regression lane?** Only after Playwright tests are stable for 2+ weeks with <5% flake rate.
5. **How do we quarantine flaky tests without hiding real instability?** Needs a dedicated flake policy — likely a `@pytest.mark.quarantine` marker with a weekly review cadence.
6. **When do we add load/performance lanes?** After Milestone 2, only if user feedback indicates performance issues.
7. **Test data cleanup strategy:** Dedicated test workspace? Teardown fixtures? Or both? Needs decision before Milestone 1 regression tests create persistent test data.

---

## 14. Bottom Line

This PRD does not propose "write 500 tests." It proposes a **quality operating system** that grows with Automatos:

- tests grouped by purpose and risk
- scripts grouped by schedule
- artifacts grouped by downstream agent
- specialist skills aligned with those artifacts
- coverage that deepens as the platform stabilizes and users surface real issues

**Right now**, the priority is Milestone 1: flush bugs, pin regressions, deepen P0 coverage, and make the platform solid for 10-15 pilot users. Everything else follows from that foundation.

The result is not a test count. It is a testing platform that your own agents can run, interpret, escalate, and repair — growing naturally from pilot to production.
