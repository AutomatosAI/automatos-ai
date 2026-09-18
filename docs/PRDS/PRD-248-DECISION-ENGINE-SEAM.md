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

**Dials (`system_settings.decision_engine`, seeded at boot, editable in Settings → System):** `classifier_mode` off|shadow|live · `tool_rerank_mode` off|shadow|live (S4) · `provider` openrouter|typesafe|llm · `model` (empty = pinned default: `typesafe/jev-1.13` via OpenRouter, `jev-1.13.0` direct) · `timeout_seconds` 2.5 · `min_confidence` 0.7.

**Hook 1 — AutoBrain (`consumers/chatbot/auto.py`, questions in `auto_decisions.py`).** State = the message (hard-truncated), the turn count and the active roster. Questions = `complexity` Choice (the rubric's five levels), `action` Choice (the four lanes), `tool_domain` Choice (eight domains incl. none), `needs_memory` and `needs_multi_agent` Noul, and `target_agent` Choice over roster names plus `none` when a roster exists.
- *shadow*: the engine call starts beside the tiers and never blocks the turn; when it lands, the comparison against whichever tier answered (1, 2 or 3) is appended to the shadow log with the engine's latency, tokens, answers and what it would have decided at the floor.
- *live*: after the regex tier and before Tier 3; a decision above `min_confidence` (the lesser of the complexity and action certainties) replaces the Tier-3 completion and is cached exactly as a Tier-3 verdict would be; below the floor, or on any failure, Tier 3 runs unchanged. A named `assign` target is resolved against the roster with the existing grounding rule (the name must be in the user's words).

## Stories

| Story | What ships | Acceptance |
|---|---|---|
| **S1 seam** | the package, the three backends, config knobs, the settings category + seed, the usage lane, the service-tier entry | every backend returns `None` rather than raising; a missing key leaves the engine idle; one `llm_usage` row per HTTP decision with the right lane, provider and cost; dials default off and survive a broken settings read |
| **S2 classifier shadow/live** | Tier 2.5 and the shadow beside every tier | off = byte-identical behaviour; shadow never changes a verdict and writes one row per turn; live replaces Tier 3 only above the floor and caches; assign resolves the roster name |
| **S3 shadow scorer** | `scripts/eval/decision_shadow/score.py` | agreement per field, per tier, per confidence band; latency p50/p95; would-decide count |
| **S4 tool rerank shadow + eval modes** (next PR) | one Noul per top-30 candidate beside today's cut, logged in the tool-shadow line; a `jev_rerank` mode in `scripts/eval/tool_routing`; a Jev ranker for `evals/operating_graph_uplift.py` | curated replay: in-set and top-1 vs `filtered_schema` at ≤15 surfaced; telemetry replay: the PRD-232 rule, ≥5 points per tenant over the best baseline, or it does not flip |

## How to try it (local edition)

1. Check the branch out where the backend mounts (`automatos-ai/orchestrator` is bind-mounted with `--reload`); the startup seed creates the `decision_engine` rows.
2. OpenRouter route needs nothing new (the platform's OpenRouter key). Direct route: `TYPESAFE_API_KEY` in the compose override environment, never the shared `.env`.
3. Settings → System → `decision_engine.classifier_mode` = `shadow`. Chat normally for a week.
4. `docker exec automatos_backend python -m scripts.eval.decision_shadow.score`; `llm_usage` rows with `request_type='decision'` carry latency and cost.
5. Flip `provider` to `llm` for the same week to get the baseline in the same log.

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
