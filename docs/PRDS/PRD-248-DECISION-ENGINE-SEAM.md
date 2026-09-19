# PRD-248 — The decision seam: typed, calibrated decisions beside Auto's classifier and the tool router (PoC)

**Status:** PoC, local edition first. Default OFF everywhere. Nothing goes live in production from this PRD.
**Owner decision (Gerard, 2026-09-18):** "Let's do the work in a separate worktree, and we try it out."
**Branch:** `feat/prd-248-decision-seam` · worktree `.worktrees/automatos-ai/feat+prd-248-decision-seam`

## Framing (CLAUDE.md §3)

**Net-new seam, justified.** Nothing in the platform returns a typed decision with a calibrated probability. The three places that make a pre-completion decision today are a chat completion parsed with a regex (`AutoBrain._llm_classify`, `UniversalRouter._classify_with_llm`) or a heuristic (regex intent, blind top-K on cosine). Reuse: the usage tracker (one row per decision, no schema change), `system_settings` for the dials, the rerank manager's httpx-per-loop pattern, the tool-routing eval harness and the uplift harness for benchmarks.

## Overview

TypeSafe AI's Jev (released 2026-09-15) answers typed questions — Choice over described options, Score on ordered levels, Noul yes/no — in one non-autoregressive pass with a probability per option. Measured independently: p50 about 0.33 s, p95 up to 1.3 s; accuracy at mid-tier-LLM level; calibration honest on easy questions and drifting on hard ones; $0.042 per million input tokens, output free. It cannot index, generate, or reason.

This PRD builds the **seam**, not the vendor: one `decide()` contract with three backends (Jev through OpenRouter on the platform's existing key, Jev direct, and the platform's own system LLM through a JSON adapter as the no-cloud-key baseline), and wires it to the first hook — Auto's complexity classifier — in shadow mode, so the comparison is measured before anything decides a turn.

## Current reality (grounded 2026-09-18, local stack, last 14 days)

| Lane | Calls | Avg input tokens | p50 | p95 | Cost |
|---|---|---|---|---|---|
| chat (Opus 4.6) | 75 | 32,397 | 4.2 s | 17.7 s | $8.64 |
| complexity_assessor (Gemini Flash) | 3 | 1,982 | 1.57 s | 1.74 s | $0.003 |
| embedding (Qwen3-8B via OpenRouter) | 143 | 3,382 | 1.27 s | 10.4 s | $0 |

- `AutoBrain.assess` reaches the Tier-3 completion on the tail past a 24 h Redis cache and regex fast-paths; its `confidence` is a constant 0.85 (`consumers/chatbot/auto.py`).
- The tool surface is a blind top-15 on in-process cosine (relevance floor 0 = off); the uncached query embedding is the live weakness (2.5 s budget; observed 37–67 s when upstream degrades) and the fallback is open-full.
- No per-request model choice exists; models resolve once at `activate_agent`. Out of scope here (prompt-cache and PRD-223 governance interactions).

## Design

**Package `core/llm/decisions/`**
- `questions.py` — `Choice` (2..255 options), `Score` (2..10 levels), `Noul`; `to_wire()` produces the System One request shape; `parse_answers()` the response shape; `DecisionAnswer.certainty` is one 0..1 number to threshold on whatever the type.
- `typesafe_client.py` — one route per instance (`typesafe` → `TYPESAFE_API_URL` with `TYPESAFE_API_KEY`; `openrouter` → `OPENROUTER_DECISIONS_URL` with the platform's OpenRouter key resolved the way the embedding client resolves it). One attempt, hard timeout, `None` on any failure, one `llm_usage` row per call (lane `decision`, input tokens at `DECISION_ENGINE_USD_PER_MTOK_IN`, output free).
- `llm_adapter.py` — the same contract over `system_llm` (service `decision_adapter`) via a JSON prompt; probabilities are the model's stated numbers. The baseline and the local-edition path.
- `engine.py` — reads the dials (category `decision_engine`, defaults, 30 s cache, fail-soft), builds the backend, `decide()` bounded and never raising, `record_shadow()` appending JSON lines to `DECISION_SHADOW_LOG_PATH`.

**Dials (`system_settings.decision_engine`, seeded at boot, editable in Settings → System):** `classifier_mode` off|shadow|live · `tool_rerank_mode` off|shadow|live · `provider` openrouter|typesafe|llm · `model` (empty = pinned default: `typesafe/jev-1.13` via OpenRouter, `jev-1.13.0` direct) · `timeout_seconds` 2.5 · `min_confidence` 0.7 (classifier floor) · `rerank_candidates` 30 · `rerank_min_probability` 0.5 · `rerank_min_keep` 5.

**Hook 2 — the tool surface (S4).** After the embedding index ranks the dispatcher allow-list, the engine judges a wider top-N (one Noul per action: "calling `name` (description) would help with the request"). Shadow: a fire-and-forget row per narrowed turn with the embedding top-K, the wider list, every probability, the cut, `nothing_fits`, and the overlap/dropped/added comparison. Live: the cut (probability ≥ floor, ordered, topped up to the minimum, capped at top-K) replaces the embedding top-K; a miss keeps it. The pins, page and onboarding priors apply after, unchanged. Offline: `python -m scripts.eval.tool_routing.run_eval --mode jev_rerank` scores the same rule on the 77-query set beside `filtered_schema`; `python -m evals.operating_graph_uplift --ranker jev [--from-telemetry]` reports a Choice-over-candidates challenger beside the gate's three rankers.

**Hook 1 — AutoBrain (`consumers/chatbot/auto.py`, questions in `auto_decisions.py`).** State = the message (hard-truncated), the turn count and the active roster. Questions = `complexity` Choice (the rubric's five levels), `action` Choice (the four lanes), `tool_domain` Choice (eight domains incl. none), `needs_memory` and `needs_multi_agent` Noul, and `target_agent` Choice over roster names plus `none` when a roster exists.
- *shadow*: the engine call starts beside the tiers and never blocks the turn; when it lands, the comparison against whichever tier answered (1, 2 or 3) is appended to the shadow log with the engine's latency, tokens, answers and what it would have decided at the floor.
- *live*: after the regex tier and before Tier 3; a decision above `min_confidence` (the lesser of the complexity and action certainties) replaces the Tier-3 completion and is cached exactly as a Tier-3 verdict would be; below the floor, or on any failure, Tier 3 runs unchanged. A named `assign` target is resolved against the roster with the existing grounding rule (the name must be in the user's words).

## Stories

| Story | What ships | Acceptance |
|---|---|---|
| **S1 seam** | the package, the three backends, config knobs, the settings category + seed, the usage lane, the service-tier entry | every backend returns `None` rather than raising; a missing key leaves the engine idle; one `llm_usage` row per HTTP decision with the right lane, provider and cost; dials default off and survive a broken settings read |
| **S2 classifier shadow/live** | Tier 2.5 and the shadow beside every tier | off = byte-identical behaviour; shadow never changes a verdict and writes one row per turn; live replaces Tier 3 only above the floor and caches; assign resolves the roster name |
| **S3 shadow scorer** | `scripts/eval/decision_shadow/score.py` | agreement per field, per tier, per confidence band; latency p50/p95; would-decide count |
| **S4 tool rerank + eval modes** | `core/llm/decisions/rerank.py` (pure: one Noul per candidate, the cut above the floor with a minimum kept, "nothing fits"), `modules/tools/discovery/decision_rerank.py` (the runner: shadow = fire-and-forget comparison row, live = the cut replaces the embedding top-K, fail-open), the hook in `tool_router._narrow_dispatcher_actions_async`; `--mode jev_rerank` in `scripts/eval/tool_routing`; `--ranker jev` challenger in `evals/operating_graph_uplift.py` | off is byte-identical; shadow never changes the surface; live never turns a list into None; curated replay: in-set and top-1 vs `filtered_schema` at ≤15 surfaced; telemetry replay: the challenger is reported beside the gate, never part of it — the PRD-232 rule (≥5 points per tenant over the best baseline) still decides any flip |

## How to try it (local edition)

1. Check the branch out where the backend mounts (`automatos-ai/orchestrator` is bind-mounted with `--reload`); the startup seed creates the `decision_engine` rows.
2. OpenRouter route needs nothing new (the platform's OpenRouter key). Direct route: `TYPESAFE_API_KEY` in the compose override environment, never the shared `.env`.
3. Settings → System → `decision_engine.classifier_mode` = `shadow`. Chat normally for a week.
4. `docker exec automatos_backend python -m scripts.eval.decision_shadow.score`; `llm_usage` rows with `request_type='decision'` carry latency and cost.
5. Flip `provider` to `llm` for the same week to get the baseline in the same log.

## S5 — four more decision points, shadow only (Gerard, 2026-09-19: "add the extra… let them run for a few days")

Each is judged beside the platform's own decision and logged with what the platform did; none has a live mode (`live` on these dials reads as `shadow` with a warning). The pure half is `core/llm/decisions/judgements.py`; each call site checks its dial, gathers plain values and hands the coroutine to `engine.shadow()`, which runs it as a task on the running loop or on a daemon thread from a sync call site. Every one is tied to a night-1 finding.

| dial | where | the questions | what the row compares |
|---|---|---|---|
| `ticket_assign_mode` | `AgentMatcher.rank` (mission tasks; also the watch re-match) | one Choice over the roster plus `none` | the engine's pick against the platform's top, and the platform rank of the engine's pick |
| `session_end_mode` | `cli_host_service.apply_result` (a Claude Code session posts its result) | three Noul: work complete · nothing done · owner needed | the engine's verdict against the status the board applies (joined by task id and attempt); night 1's 188 re-dispatches would read `nothing_done` |
| `hold_risk_mode` | `core/services/approval_grants.create_grant` (every question and approval, held commands included) | Score on five blast-radius levels · Choice of intent · Noul "a non-technical owner could judge this" | logged with the grant id; the human's allow/deny/answer is joined later from `approval_grants` |
| `report_triage_mode` | `ReportService.create_report` and `heartbeat_service._dispatch_heartbeat_notification` | Noul "the owner should act on this today" · Choice of severity | against where the platform sent it (`report_submitted`, `requires_approval`, `report_to=…`) |

The scorer summarises each purpose (`purposes` in `--json`): coverage, latency, agreement where the row carries one, the engine's probabilities and picks per question, and the platform side as a distribution. Question wording follows the vendor's failure-mode list: direct, no negations, no arithmetic, state trimmed to the fields the question needs. The verdict rule for the programme: a hook earns a live-mode design only when its shadow rows show it would have caught real misses at a useful rate; three of four earning it is a win, two tells us which decisions are not this shape.

## Benchmarking through PRD-247 (the simulation program)

PRD-247's **P7 "Auto's brain pack"** is this PoC's benchmark lane. Its chat-mode driver produces the classified turns the shadow needs in one night instead of a week of the operator's own traffic; its four-row scorecard (usability, cost, quality, usefulness) answers the question the shadow's agreement rate cannot — whether a *different* decision is a *better* outcome; and its nightly offline gates are the two harnesses this PRD extended (`--mode jev_rerank`, `--ranker jev`). How the two connect:

- **Campaign attribution.** The sim encodes `sim:<campaign>:<scenario>:<run>` in the usage scope's `execution_id`. Every decision receipt in `llm_usage` and every shadow row carries that id, so `scripts.eval.decision_shadow.score --only sim` and `--only real` never mix a simulated night with the operator's turns, and a decision's cost lands on the same row family as the turn it served. One dependency on PRD-247's own plumbing: the chat lane opens its scope inside the streaming service (`chat:<id>`), after `AutoBrain.assess` has run, so the classifier's receipts see the campaign id only when the driver's id is set at the route — which is where a campaign id arriving over HTTP has to be set anyway.
- **Shadow is inert, provably.** Run one pack with both dials off, then again in shadow: the four rows must be identical within noise. That invariant is the first thing the sim should assert about this PRD.
- **Live is a delta.** Off versus live on the same pack: usability (turns, steps, questions), cost per successful outcome (including the decision rows), quality (the rubric), usefulness (expected effects). Prompt tokens per turn and the `context_trace` versus `llm_usage` divergence that P7 already trends will show the rerank's effect on the tool block directly.
- **What stays separate.** The uplift gate reads production rows only (`telemetry_source` production or null); the driver tags its rows `eval`, so a simulated night fills the shadow log and the scorecard but never the flip gate. That is by design — eval rows never grade themselves.
- **Trend lines P7 can add from this PRD:** classifier agreement per tier and per engine-confidence band; rerank overlap, dropped and added actions, `nothing_fits` rate; decision latency p50/p95 and cost per turn. All from the shadow log; no product change.
- **A possible give-back, the owner's call:** the sim's standalone judge (`sim/judge.py`, brief + output + rubric → 1–5) is a Score question per rubric dimension in this seam's terms — cheap enough to grade every deliverable every night, with the operator's hand grades (P12's test plan step 4) as the calibration check.

## Traps (pre-verified)

- **Never register Jev as an LLM provider.** It generates no text; PRD-236's registry has no `decision` kind, and the seam stays beside it.
- **A schema-valid answer can still be wrong.** The floor gates *acting*, not *being right*; only the labelled comparison says which classifier is right. Tier 3 has no gold set — label ~100 real turns before believing either.
- **Question shape decides accuracy.** Keep instructions terse; a wide single Choice underperforms decomposition on confusable classes; evaluate per hook, never assume.
- **Data egress.** The message text and roster names leave the platform on every shadow/live call. Local edition = the operator's own data; SaaS = a sub-processor decision, which this PRD does not make (default off in both editions).
- **OpenRouter's decisions endpoint is on an alpha path** (`/api/alpha/decisions`) and may move; `OPENROUTER_DECISIONS_URL` is the one knob.
- Per-turn model switching is out of scope: no runtime seam, PRD-223 governance, and prompt caching (#745) would be defeated.

## Open questions for the owner

| # | Question | Proposal |
|---|---|---|
| Q1 | Route for the PoC week: OpenRouter beta on the existing key, or a TypeSafe key from the waitlist? | OpenRouter first (zero setup); direct if the alpha path misbehaves |
| Q2 | Label the ~140 local user turns for classifier accuracy (about an hour)? | Yes, after the first shadow week, from the `message_preview` column of the log |
| Q3 | Should `llm` (the adapter) be the default provider so the seam works with no cloud key in every install? | Keep `openrouter` for the PoC; decide from the two shadow weeks |

## Traceability

- Research and measurements: session 2026-09-18 (memory `jev-system-one-assessment-2026-09-18`).
- Precedents reused: `core/llm/rerank_manager.py` (httpx per loop, usage receipt), PRD-223 `model_policy` dials, PRD-232 shadow logger and uplift gate, PRD-240 per-edition posture (default-off cloud seam).
