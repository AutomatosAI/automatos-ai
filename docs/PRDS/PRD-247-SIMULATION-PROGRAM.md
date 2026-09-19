# PRD-247 — The Simulation Program: would a customer be happy?

**Status:** DRAFT for Gerard's decisions (D1–D12 below) · **Date:** 2026-09-18 · **Author:** Gerard Kavanagh + Claude
**Grounded @** `origin/main af214bc7b` (Studio merged) with the PRD-245 stack (#758…#775) on top.
**Extends:** PRD-78 (autonomous test mesh — the root `tests/` suite is its output), PRD-14 (repeatable benchmarks), PRD-103 (verification & quality), PRD-142 W2 (test net), PRD-182 (CI bar), PRD-185 S9/S10 (eval substrate), PRD-204 (Auto Watcher), PRD-212 (HARNESS rebuild, draft), PRD-223 (model governance), PRD-231 (context diet), PRD-232 (Intent Graph), PRD-245 (session tools bridge).

## Framing (CLAUDE.md §3)

**Extension**, with one small net-new piece. The platform already records almost everything a tester needs (cost per call, tool trails, refusals by kind, report grades, verification verdicts, selection telemetry) and already has a proven journey harness, a nightly API suite and an unattended-loop kit. What does not exist is the thing that runs thousands of scenarios against a workspace and turns those records into one answer per scenario: *did the customer get what they came for, and what did it cost?* That runner, its scenario packs and its scorer are the net-new. Everything else is reuse, and two things are retired.

## The question this PRD answers

Not "is it built" — "does it work as it should, smoothly, at a cost and quality a paying customer would accept, and can we make it better every night?" Every pack below scores four things per scenario:

| Row | Meaning | Where the number comes from |
|---|---|---|
| **Usability** | turns, steps and questions it took to get the outcome; whether Auto's replies were clear | chat turn count, tool-call trail, `approval_grants` kind=question, judge |
| **Cost** | dollars, tokens and seconds per successful outcome; wasted calls (refused, retried, duplicate) | `llm_usage` by lane/agent, `board_tasks.runtime_ref.permission_denials` / `recent_tools`, `tool_execution_logs` |
| **Quality** | complete, correct, actionable, clear, on template | the six-dimension rubric in `modules/coordination/run_verdict.py`, freed from watches; deterministic checks; the human 1–5 grade on `agent_reports` for calibration |
| **Usefulness** | did the outcome match the stated goal | judge against the scenario's goal + expected-effects manifest |

Rolled up per pack and per persona, that is the scorecard. Changes are judged by the delta it moves.

## What the platform already gives us (verified 2026-09-18)

| Asset | Where | State | Reused for |
|---|---|---|---|
| Nightly live-HTTP API suite: `X-Api-Key` auth, a dedicated "TEST - Nightly Suite" workspace, cleanup registries that delete what they created, `run_nightly.py` → `test-summary.json` | `tests/`, `tests/api/conftest.py`, `tests/run_nightly.py` | runnable locally, not in CI, nobody runs it | the API pack's skeleton (PRD-78's lanes) |
| Throwaway-workspace provisioner: idempotent by slug, mints a service SDK key | `orchestrator/scripts/create_test_workspace.py` | local | isolation |
| Purge with dynamic FK discovery, sparing identity/credentials/system agents | `orchestrator/services/workspace_purge.py`, `POST /api/admin/workspaces/{id}/purge` | live | teardown |
| The journey harness: stage-driven persona runs, effect-based scoring, zero model tokens in the runner, already has the local-edition switch | `~/.automatos-eval/persona_run.py`, `suite.py`, `personas.py`, `clerk_session.py` | local, prod-shaped | the runner pattern; the onboarding pack |
| The unattended loop: caffeinate, pinned model, iteration caps, usage-limit backoff to the reset, state files, night report | `scripts/ralph/overnight-prd*.sh` (`run_claude`, `handle_usage_limit`, `RALPH_MODEL`) | local, PRD-shaped | the overnight scheduler |
| 197 platform actions, callable in-process (`PlatformActionExecutor.execute`) or over HTTP; local edition needs no credential (anonymous → operator, super admin) | `modules/tools/discovery/action_registry.py`, `platform_executor.py`, `core/auth/hybrid.py` | live | the driver |
| Per-call cost/tokens/latency/tier per lane; sessions booked at $0 `tier=subscription` | `llm_usage`, `core/llm/usage_tracker.py`, `ReportService.compute_execution_metrics` | live | the cost row |
| Per-ticket run telemetry: denials by kind, tools called, files touched, platform calls | `board_tasks.runtime_ref`, `services/session_denials.py`, `services/session_report.py` | live (PRD-245) | wasted-call metrics |
| Quality scorers: six-dimension business rubric (watch-bound), cross-model verifier (mission-bound), deterministic checks (8 types), playbook five-dimension score, `grade_report` 1–5 | `modules/coordination/run_verdict.py`, `verification.py`, `deterministic_checks.py`, `core/services/playbook_quality_service.py`, `PATCH /api/reports/{id}/grade` | live, bound | the quality row |
| Selection telemetry and its metric: hit/fallback rate, `__tool_gap__` / `__tool_shown__` rows, nightly edge builder | `tool_execution_logs.router_decision`, `GET /api/analytics/selection-health` (su-only), `core/services/edge_builder.py` | live | Auto's-brain pack |
| Offline selection eval with cost/latency/accuracy per model, 1,224-row baseline in tree; the graph-flip gate (≥5 points) | `orchestrator/scripts/eval/tool_routing/` (`run_eval.py`, `score.py`), `orchestrator/evals/operating_graph_uplift.py` | local / CI non-required | Auto's-brain pack, regression gate |
| Retrieval, memory, NL2SQL gold sets and scorers | `orchestrator/evals/*.py`, `orchestrator/scripts/eval/*/gold_set.jsonl`, `tests/nl2sql_eval/` | CI non-required | Knowledge pack |
| Per-turn prompt assembly record: sections, tokens, trims, cacheable prefix | `messages.context_trace` (PRD-201) | live, **write-only** | Auto's-brain pack (SQL read) |
| Governors: turn cost ceiling, tool-step caps, budget rails, rate limiter, concurrency guard, breaker | `config.py`, `services/budget_ceiling.py`, `modules/policy/budget.py`, `core/security/rate_limiter.py` | live, some off by default | cost ceilings; Governance pack |
| Board/calendar/questions/governance routes and tables, with SLOs already computed | `api/board_tasks.py`, `api/activity.py`, `api/approval_grants.py`, `api/governance.py`, `services/slo_metrics.py` | live | Command Center pack |
| HARNESS: weekly five-phase sweep, status vocabulary, board-task prescriptions | `services/harness_service.py`, `GET /api/harness/self-learning`, `platform_harness_*` | live, **0 prescriptions in 15 months** (baselines on a non-persistent volume, PRD-212) | Harness pack, after PRD-212's fix |
| Auto Watcher: ticks every 300 s, claims due watches, verdicts + actions, cost attributed to `watch-<id>` | `services/watch_*.py`, `GET /api/v1/watches` | live | Harness pack |

**Retired by this PRD:** `automatos-testing/` (a year stale, 200 PRDs behind; only its journey-class shape survives) and the `e2e/` scaffold's scenario endpoints, which name routes that do not exist (`/api/board/summary`, `/api/questions`); its YAML scenarios are folded into the packs with the real paths.

**Missing today (the net-new):** a scripted driver of the Auto surface as sequences; an iteration engine; a campaign-level result store and scorer; a standalone "score this deliverable against this brief" function; a campaign id on `llm_usage` (encode in `execution_id`, the `watch-<id>` precedent); a throwaway-workspace lifecycle a script can drive end to end; an unattended answer to human gates; briefs with expected outcomes; a local latency instrument; a load driver and a mock model.

## Decisions (defaults proposed; each is Gerard's to change)

- **D1 · Where it lives.** Code in `automatos-ai/tests/sim/`, beside the nightly API suite whose auth pattern it borrows (Gerard, 2026-09-18: "we have built test stuff in automatos-ai/tests"). Corpora, personas, gold manifests and results under `~/.automatos-sim/`, matching the existing `orchestrator/scripts/eval/**/live/` + `*.gold.jsonl` gitignore rule. Scenario *shapes* (the TOML packs) may be committed; anything derived from real tenants may not.
- **D2 · Two driving modes, one scorer.** *Direct*: platform actions and routes, no model tokens — tests the platform. *Chat*: `POST /api/chat` with a persona, the way a human uses Auto — tests Auto's understanding, planning and tool selection, and spends Auto's model every turn. A pack declares which mode each scenario uses; most packs carry both.
- **D3 · Eval layer: adopt promptfoo.** Our runner is its custom provider; it supplies the scenario × model matrix, assertions from exact checks to LLM-graded rubrics, cost and latency columns, a viewer and run-to-run comparison. DeepEval/Ragas for the Knowledge pack's faithfulness and hallucination metrics. Not Langfuse or Phoenix yet: they mean instrumenting the product, and PRD-185 S9's Langfuse adoption never shipped; revisit when JSON results outgrow the viewer. *P0 as built writes `run.json`, `scorecard.md`, `trace.jsonl` and a SQLite campaign store; emitting promptfoo's results format is an open item under P0 below, not yet done.*
- **D4 · Load is separate and mocked.** A tiny local OpenAI-compatible server returning canned completions and tool calls (`sim/mockllm/`), so load runs are free and repeatable; Locust as the driver (Python, like the repo). Numbers on the Mac are relative, for regressions, never capacity claims about Railway.
- **D5 · Isolation.** One throwaway workspace per pack per night via `create_test_workspace.py` (now takes `--slug/--name/--key-name/--purpose`); purged at the end through the admin route's own two steps — soft-delete, then `services.workspace_purge.purge_workspace_sync` — run inside the backend container by `orchestrator/scripts/purge_test_workspace.py`, because the route itself needs a signed-in admin and an API key is not one (`_is_admin` is False without `ctx.user`). The purge script refuses any workspace not stamped `settings.managed_by=create_test_workspace.py` with a `sim:` purpose. Never `DEFAULT_WORKSPACE_ID`; `wipe_built` never against the operator's workspace (the persona harness's own trap note). Runtime-agent packs are the exception: they run in the host's paired workspace, serialised, with their own tickets.
- **D6 · Human gates.** A sim-mode answerer (`sim/answerer.py`) polls `GET /api/v1/approval-grants?status=pending`, answers questions and holds by pack rule (allow read-only, deny side effects, answer asks from the scenario's script), records every decision in the run, and never touches a workspace it did not create. Approvals on tickets use run-now consent. No product change; nothing auto-approves outside a sim workspace.
- **D7 · Cost.** A per-night dollar ceiling on each sim workspace, enforced by the runner: before every scenario it sums the workspace's `llm_usage` and stops at the ceiling (default $5 via `SIM_BUDGET_USD`; a pack's `budget_usd` may only lower it), on top of the existing turn ceiling and step caps. `plan_limits.budget` was not verified to exist and is not relied on. The cheap model is pinned two ways: `PUT /api/agents/{id}/model-config` on every agent the pack seeds, and the global `chatbot`/`*_llm` rows of `system_settings` (that table has no workspace column, so the runner snapshots them to the run directory first and restores them in `finally`). Volume runs on the cheapest Auto in the registry; a fixed sample re-runs on the premium model for the quality row. Runtime packs are bounded by subscription windows: tens of tickets a night, not thousands; the Ralph backoff makes that unattended.
- **D8 · Composio.** Sim packs may call read-only actions only (`*_FIND_*`, `*_GET_*`, `*_LIST_*`) until the session hold-versus-allow decision from PRD-245 is made. A simulated OPS must never send anyone an email.
- **D9 · Report-only.** The overnight loop proposes; it does not apply. Its output is a scorecard, a ranked list of wasted-call and stall findings, and proposed changes as diffs or draft PRs, each naming the metric it should move. Skills changes go to `automatos-skills`; code goes through CI. Report-only stays until a week passes with no false accusation against the product — the eval harness's own hard lesson.
- **D10 · Trust.** Assert on effects, never on tool names; capture full payloads; label every finding trace-backed or inferred. The same three rules that were learned the expensive way in August.
- **D11 · Pinned models.** Any unattended loop pins its model explicitly; the transcript is checked afterwards (the July fallback incident).
- **D12 · HARNESS prerequisite.** The Harness pack cannot prove anything until PRD-212's baseline persistence lands; until then the pack asserts only that the sweep fires and reports its status honestly. Two weeks of simulated runs are exactly the history the prescribe phase needs afterwards.

## Stories

Each is small and independently shippable. Files are the intended homes; tests are what proves the story.

### P0 — The instrument (build first, run nothing)

> **Status 2026-09-18 — built** (PR stacked on this one) as `tests/sim/`: `config` · `api` (stdlib HTTP, every exchange traced) · `sse` (the AI SDK v4 frames, tool calls included) · `packs` · `workspace` · `answerer` · `driver` + `driver_task` + `driver_chat` · `cost` · `judge` · `score` · `store` · `night` · `schedule`, two packs (`smoke`, `agents`), unit tests in `tests/sim/tests/` (CI job `sim-units`), and the two container-side scripts. Where the build differs from the stories below, the story text has been corrected in place. **Credential (found in review):** `get_request_context_hybrid` accepts `X-Api-Key` only as the single static `ORCHESTRATOR_API_KEY` (empty on the local edition) and honours the per-workspace `ak_srv_` keys only as a Bearer token on the board-task read routes — so the harness runs as the anonymous local operator (super admin) and `X-Workspace-ID` is the whole boundary: sent on every call, every seeded agent must echo the sim workspace id, the default workspace is refused by id, and a non-local `DOCKER_HOST` is refused. The same fact means `create_test_workspace.py`'s docstring claim that its minted key can be "dropped into `tests/.env`" does not hold for `X-Api-Key` routes — flagged, not fixed here. **Open after P0 (owner decisions, not deferrals):** (1) emit promptfoo's results format so its viewer and comparison work (D3) — one more writer in `store.py`, done once the format is verified against a real promptfoo run; (2) the mock model and Locust (D4) belong to P11, so S0.6's dry run uses the cheap real model under a $1 ceiling instead; (3) S0.2's in-process *direct* mode — the runner lives outside the container, so direct mode is the HTTP routes with the workspace key; `PlatformActionExecutor.execute` in-process would need the runner piped into the container like the two scripts are.

**S0.1 · Workspace lifecycle (S).** `sim/workspace.py`: create (wraps `create_test_workspace.py`, returns id + key), seed (agents/skills from a pack manifest through the install actions so closure is exercised), purge (admin route). Refuses `DEFAULT_WORKSPACE_ID` by name.
**Test:** create → purge leaves zero rows for the id across `agents`, `board_tasks`, `agent_reports`, `deliverables`, `llm_usage`; a second create by slug is idempotent.

**S0.2 · Driver (M).** `sim/driver.py`: *direct* mode is the routes over HTTP with the workspace key (in-process is open, see the status note); *chat* mode posts to `/api/chat`, parses the SSE stream (text, `prompt_tokens`, subtool names, effects — the persona harness's parser), and applies the escalation ladder keyed on observed effects, never repeating a sentence. Every call records the full request and full response.
**Test:** a scenario of five actions leaves five `tool_execution_logs` rows tagged `telemetry_source='eval'`; a chat scenario yields a transcript with per-turn tokens.

**S0.3 · Scenario packs as data (S).** `tests/sim/packs/<pack>.toml` (TOML: `tomllib` is standard library, so the scheduled job needs no virtualenv): scenarios with `mode`, `persona`, `steps` or `goal`, `expected_effects` (rows that must exist, statuses, files, sections), `budget`, `composio: readonly`. Seeds: the six PRD-245 tickets, the fifteen onboarding personas, PRD-14's four benchmark workflows, the 59-query tool-routing set, the munder-wave scenarios with real paths.
**Test:** every pack validates against a schema; every referenced agent/skill exists in the seed manifest.

**S0.4 · Scorer (M).** `sim/score.py`: per run, the four rows. Usability from turn/step/question counts; cost from `llm_usage` rows — the platform stamps `execution_id` itself (`board_task:<id>` for tickets), so attribution is by the ids the run created plus the workspace's own window, which also catches spend the scenarios did not ask for (heartbeats, retries); plus wasted-call classification from the tool trail (refused / retried / duplicate / gap); quality from a standalone judge (`sim/judge.py`, the `run_verdict` rubric lifted into a function taking brief + output + criteria, cross-model, output-hash cached, writing the 1–5 grade to the report so your grade sits beside it); usefulness from expected effects. Writes JSONL per run and a SQLite campaign store; emits promptfoo results so the viewer and comparison work.
**Test:** a golden run scores identically twice (deterministic), the judge is skipped when no model is configured, and a run with a refused call shows it as wasted, not as failure.

**S0.5 · Human-gate answerer (S).** D6 as code. **Test:** a held command in a sim workspace is answered within one poll; a hold in a non-sim workspace is never touched.

**S0.6 · Overnight runner (S).** `tests/sim/night.py` (Python, not the Ralph shell skeleton — the runner needs the drivers in-process): pinned model, per-pack budget, partial `run.json` after every scenario, night report; one pack per night by rotation (`packs/rotation.toml`); report-only. `tests/sim/schedule.py` installs it as a launchd job under `caffeinate`, the CLI host's own pattern.
**Test:** a dry run with the mock model completes a pack and writes the scorecard.

**S0.7 · Corpora (M, local only).** Under `~/.automatos-sim/`: briefs with expected-effects manifests for every agent persona; personas for the collective pack; the report-structure schema per `report_type`. Built from the six tickets and their deliverables, yesterday's baseline copies, and the onboarding personas. This is night one's whole job.

### P1 — Marketplace pack

Scenarios in both modes: goal-shaped searches ("I need someone to write weekly posts"), browse, install agent / skill / plugin / package / model. **Assert:** the installed closure equals the artifact's declared manifest every time (`package_installer` D2/D3 as an invariant checked from outside); the right result is first for each goal; descriptions and tags are non-empty and distinct; Auto's pick from a listing matches the gold pick. **Cost:** none in direct mode.

### P2 — Onboarding pack

The persona harness moved into `sim/` with the local switch, one persona per throwaway workspace. **Assert:** stages advance monotonically, turns and cost to done, package correctness, recording-not-reciting, the funnel events in the onboarding document. **Cost:** Auto's turns.

### P3 — Agents pack

Create, edit, delete with every field (persona, tags, skills, capabilities, tools, model, heartbeat); assign a brief to every agent type; API agents fan out across workspaces, runtime agents serialise on the host. **Assert:** closure on create; heartbeat fires within its window (`heartbeat_results` advances); end status; wasted calls; refusals by kind (a `hold` is the only kind that may put a ticket in review); deliverable registered; report has its type's required sections; quality and usefulness rows. **Cost:** subscription for runtime, cheap model for API volume.

### P4 — Command Center pack

Direct mode, invariants from the tables, all cheap:

- **Board:** an `assigned` task gets a lease and `in_progress` within `POLL_SECONDS + ε`; no `in_progress` row outlives `lease_until` by more than one sweep without renewal or requeue; `attempts <= MAX_ATTEMPTS` and the cap lands `failed`, never `done`; every mutation reaches an open SSE within `board_event_freshness_seconds`; `sla_breach_notified` flips once; a rejected task never reaches `done` without an approve; run-now on an `always_ask` workspace produces a grant, not an execution.
- **Calendar:** every item with `next_run_at` in the window produces a real run record inside its misfire grace; the schedule payload is byte-identical across workers; `scheduler-health` never says healthy with a stale last fire; the per-agent and per-workspace caps are refused, not silently accepted.
- **Questions:** every held or parked subject yields exactly one pending `kind=question` grant, listed by the route, with a `question_pending` notification; answering flips status and unparks the subject in one transaction; dismiss leaves it blocked and executes nothing; a Telegram reply from the wrong chat never answers.
- **Governance:** a denied or revoked grant is never followed by an execution; every pending grant has a notification; `status.grants.by_status` reconciles with a direct count; `oversight.requires_approval` recomputes from `oversight_for_risk` for every row; every grant/deny/revoke writes one `audit_logs` row with a real actor.

### P5 — Harness and Watcher pack

**Assert now:** the sweep fires (`GET /api/harness/self-learning` → `completed`, `last_run_at` within 7 days, every artifact `ok`), `platform_harness_status` agrees with the route; the Watcher claims due watches every 300 s, every terminal target produces exactly one terminal `watch_event` and one verdict notification, judge cost is attributed to `watch-<id>`. **Assert after PRD-212:** a workspace with two weeks of simulated runs produces at least one prescription, each prescription names a metric and a baseline, risky ones queue as `harness`-tagged board tasks and an approval path exists. The known dead-loop signature (completed, iterations > 1, zero prescriptions ever) is a named failure.

### P6 — Governance, blueprints and standards pack

The sim workspace runs the policy plane in `shadow` so verdict rows exist without blocking; assertions then read `audit_logs` where `action LIKE 'policy:%'`. **Assert:** every fail-closed risk class produces a verdict row; budget admission refuses past the ceiling and the refusal is a row; rate limits throttle `platform_write` at 60/min and the 429 is counted by the runner (there is no table); the concurrency guard refuses the eleventh pending playbook; model policy refuses a quarantined orchestrator model at all three writer paths; a `strict` blueprint blocks an agent missing `min_tools`, and `advisory` only warns.
**Declared but not enforced, made explicit** (each becomes a decision, not a silent gap): `max_budget_per_run` is in the blueprint schema and never read; `required_tags` and `allowed_models` warn even in `strict`; `LLMModel.requires_plan` is a dead column; the tool-schema `required[]` rule lives only in a pytest walker; `scripts/check_hierarchy_gate.py` is invoked by no workflow; report standards are prose. The pack asserts the current truth and flags each for Gerard to promote to enforced or to delete.

### P7 — Auto's brain pack

Chat mode against the tool-routing query set and the collective persona. **Assert and trend:** first-try selection rate (`selection-health` hit and fallback rates, recomputed per workspace from `router_decision->'selection'` since the route is super-admin-only and platform-wide); prompt tokens per turn from `messages.context_trace.token_estimate` **and** `llm_usage.input_tokens`, with their divergence as its own assertion; cache hit rate from `cache_read_tokens` by provider, expected zero where `prompt_cache_control` is false and non-zero on Anthropic-routed and OpenRouter-passthrough calls; wrong-tool rate from `tool_execution_logs.status`; gap rows (`__tool_gap__`) as "hunted for a tool it wasn't given"; retries by `router_decision.turn_id` grouping; skill activation from the one log line until it has a surface. **Offline gates each night:** the tool-routing eval against the 1,224-row baseline (accuracy, in-set, $/correct, p95), and `operating_graph_uplift --from-telemetry` for the ≥5-point flip gate. The known numbers become the baseline: sonnet-4.6 89.4% full vs 93.6% filtered-schema at a third of the tokens; in-set 93.6%, meaning the ranker drops a correct action on one query in sixteen.

### P8 — Playbooks pack

Multi-agent playbooks with handoffs, same briefs. **Assert:** one solid result on the rubric; handoff losses (facts present in step N absent in N+1); duplicate work; team cost versus the best single agent; the playbook quality score moves with the rubric; breaker trips are visible. Gated on P3 clearing a bar.

### P9 — Knowledge pack

Ingest each source type, then briefs that require it, including questions whose answer is not there. **Assert:** the committed retrieval, memory and NL2SQL gold sets at their baselines; citation correctness; made-up answers (DeepEval faithfulness); cost per query.

### P10 — Missions pack

Plan, approve, run, verify. **Assert:** verifier pass rate, replans, `RUN_AWAITING_HUMAN` stalls answered by the sim answerer, budget events, the final rubric. Gated on P8.

### P11 — Load pack (mock model)

Locust against the mock: concurrent running tasks before p95 dispatch latency doubles, claim fairness, Redis connection errors at turn start (a known local symptom), SSE fan-out, and how many concurrent Claude sessions the Mac tolerates (six ran cleanly on 2026-09-18). Relative numbers only.

### P12 — Collective pack (the scorecard night)

One persona's week compressed into a night, on the premium model: onboard, find and install a team, give it work, review the board and calendar, answer its questions, run a playbook, ask about the knowledge it ingested, run one mission. Scored as one customer. This is the number that answers the PRD's question.

### P13 — Regression mode and the rotation

`sim/night.sh --pack <name> --against <last baseline>` prints the delta table. Rotation: P1+P2 · P3 · P9 · P4+P6 · P8 · P11 · P12, then repeat; P5 and P7 run as short read-only sweeps every night since they cost nothing.

## Surfaces a tester cannot read today (small product stories, each optional)

Reads that exist only in SQL or logs; each is one small route or column and is listed so Gerard can pick, not so the program depends on them:

- `messages.context_trace` and `retrieval_context` have no read route; prompt tokens per turn and cache hit rate have no endpoint or tile.
- `selection-health` is super-admin-only and platform-wide; a workspace cannot read its own selection quality.
- HARNESS state is JSON on the volume with no table and no HTTP status route; `next_scheduled_run` is a hardcoded string; `platform_harness_trigger` needs no confirmation while self-management is off.
- No dispatcher liveness surface; `requeue_expired_leases` counts are dropped; no SSE metrics; `heartbeat_results` has no API.
- Rate-limit 429s leave a log line only; fail-open policy events are greppable but uncounted; spend-to-date against the budget ceiling has no endpoint; consent events cannot be listed by reason.
- Skill activation is one log line; `__tool_gap__` / `__tool_shown__` rows have no read route; the AutoBrain phrase-map `matched_tools` is dead.

## Not in this PRD (owner decisions)

PRD-212's HARNESS rebuild (a prerequisite for P5, not part of this); promoting any declared-but-unenforced rule to enforced (P6 surfaces them); the PRD-245 Composio hold decision (D8 works around it); Playwright UI smoke beyond five post-deploy checks; SaaS-edition runs against Clerk (the persona harness already has that path; the packs are edition-agnostic).

## Test plan (owner, local edition)

1. **After P0:** a dry run of one pack with the mock model creates a workspace, runs ten scenarios, writes the scorecard and purges; nothing touches the operator's workspace.
2. **After P1–P3:** the first real night on the cheap model produces a scorecard you can read in ten minutes and a ranked findings list; the six PRD-245 tickets score against yesterday's baseline copies.
3. **After P7:** the tool-routing and prompt-token baselines are dated in `benchmarks/`; a deliberate regression (re-add a promoted schema) is caught by the delta.
4. **After P12:** one collective night on the premium model; you grade five reports by hand and the judge's grades correlate.

## Verify at build (no spend)

- promptfoo's custom-provider contract accepts a Python script returning `{output, cost, latency}` per test, and its comparison view diffs two result files.
- `create_test_workspace.py` + the admin purge route round-trip cleanly, including `composio_connections` (dynamic FK discovery).
- The anonymous local principal resolves to the operator with super admin, so a sim workspace must always be addressed by its own key, never by default.
- The `run_verdict` rubric can be called with a synthetic `Watch` shell or is lifted into a function without changing its cached behaviour.
- Locust can drive the SSE chat route; the mock model's tool-call shape satisfies `openai_compatible_client`.

## Merge notes

Docs first (this PRD), then P0 as one PR with the mock model, then packs as separate PRs each carrying its YAML and its assertions. Corpora never land in the repo. DCO sign-off on every commit. The night report and the scorecard are files under `~/.automatos-sim/reports/`, linked from the handoff.
