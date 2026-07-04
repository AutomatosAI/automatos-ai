# Thesis T3 — One eval/measurement harness so quality is a tracked number, not a vibe

**Reviewer:** Phase-2 deep review (Opus) · **Written:** 2026-07-04 · **Tree:** main @ pinned `77bc9c6d5` (tool-selection cites verified against `e040d9b53`)
**Inputs:** dossiers `{evals-learning, nl2sql, memory, rag-retrieval, tool-selection}.md` + `evidence/real-data-inventory.md` + live source (`file:line`) + web (competitor/tool currency, cited).
**Lens:** pilot-phase (§2) — empty tables / synthetic seed / cold-start are **not** failures; the value of T3 is a tracked number **going forward**, not today's pass rate. Genuine brokenness (dead paths, write-only tables, placebo metrics) is still flagged.
**Security/adversarial lens deliberately excluded** — runs as the separate Opus hardening pass.

---

## VERDICT (one line)

**ADOPT Langfuse (self-hosted, MIT core) as the trace/score/dataset/experiment substrate; EXTEND the three real in-house harnesses onto it as thin domain gold-sets (do not build an eval platform); use RAGAS + DeepEval as metric libraries; and make the first concrete metric a memory one — a LongMemEval-category-style gold set over the platform's *own* memory API plus a with/vs/without task-lift experiment — because the binding constraint is signal capture, not tooling.**

---

## 1. What the review actually found (the state T3 must fix)

The `evals-learning` dossier already did the enumeration; this thesis resolves it. Restated as the decision inputs:

- **Quality is nowhere a tracked, current number.** ~3.9k LOC of eval code exists across five disconnected assets; **every recorded number is synthetic, stale, placeholder, or on one laptop** (`evals-learning.md` C.1). There is no trace store, no score store, no datasets-from-production, no online evaluators, no judges, no dashboards, no trends.
- **The learning plane is armed and unfed, end-to-end.** From the live Railway Postgres (2026-07-04, read-only, `real-data-inventory.md`): `tool_execution_logs` 2,341 rows **100% `telemetry_source='synthetic'`** frozen 2026-05-05; `rag_feedback` **0 rows ever**; `harness_prescriptions` **0**; `database_query_audit` **0**; `nl2sql_benchmark_runs/_results` **0/0**; `memory_items` (L3) **0**; `nl2sql_training_examples` **2**. Two feedback tables exist — **`rag_feedback` has a reader and no writer; `votes` has a writer and no reader** (`evals-learning.md` C.2; `rag-retrieval.md` C-1; verified: chat thumbs post to `/api/chat/vote` at `frontend/lib/chat/api.ts:47`, not `/api/rag/feedback`; zero frontend callers of the latter).
- **Eval coverage is inversely correlated with what clients touch.** The two CI evals cover NL2SQL (0 audited prod queries ever) and graph tool-routing (synthetic-only). The daily-driver surfaces — chat (1,278 msgs), playbooks, deliverables (2,242), **memory** (the seed frustration of this whole review), RAG retrieval quality, context assembly — have **no eval of any kind** (`evals-learning.md` C.4).
- **One eval has ever gated a decision, and it worked.** The W7 operating-graph uplift eval measured **−32.9 points** vs its +5 gate and `TOOL_ROUTING_GRAPH` stayed dark *on evidence* (`operating_graph_uplift.py`, verified: exits 0 always, docstring pre-commits to "the number is the deliverable… the caller must not flip the flag on it"). This is the platform's single genuine instance of eval-driven development, and it is the pattern to make the norm. *(Note: that file's docstring still uses legacy "moat" language internally at `:4-6`; per the review's framing this is described as a tool-selection quality gate, not a moat.)*

**Implication for T3 that reshapes the whole thesis:** all four commercial eval platforms assume *you have production traffic to score*. Here the binding constraint is **signal capture, not tooling** (`evals-learning.md` §D closing caveat). Adopting a platform before fixing the starved loops produces beautiful empty dashboards. **Feeding the loops is Phase-0 of T3, not a follow-on.**

---

## 2. Adopt vs build — the substrate decision (cited)

The commodity layer (trace store, score store, datasets, judge orchestration, annotation queues, experiment gates, dashboards) is **not earned in-house**: the reuse rule (§2) bites hardest here, and the in-house track record is five one-shot artifacts plus an empty `modules/evaluation` scaffold that has waited "to be built fresh" indefinitely (`modules/evaluation/__init__.py:20-29`). Candidate substrates and metric libraries:

| Tool | Kind | License / cost | What it gives T3 | Verdict for Automatos |
|---|---|---|---|---|
| **Langfuse** | OSS platform | **MIT core** (except `ee/`); self-host $0 license + ~$20–60/mo infra; Cloud free tier to start ([github.com/langfuse/langfuse](https://github.com/langfuse/langfuse), [self-hosting](https://langfuse.com/self-hosting)) | Tracing + **LLM-as-judge** + **code evaluators** + datasets/experiments + **`langfuse/experiment-action` CI gate** (May 2026 — fails a PR when experiment scores drop below threshold, posts result as a comment, [May update](https://langfuse.com/blog/2026-05-31-langfuse-may-update)) + human annotation queues + scores API for arbitrary online signals + OTel ingestion ([overview](https://langfuse.com/docs/evaluation/overview), [LLM-as-a-judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge)) | **ADOPT — the substrate.** Best shape-fit: MIT-licensed (compatible with the open-core direction), self-hostable next to the existing Railway Postgres, covers the online/offline/CI triad out of the box. Caveat: SCIM / audit-log / data-retention modules need a commercial licence when self-hosting ([discussion #13737](https://github.com/orgs/langfuse/discussions/13737)) — not needed for T3's core. |
| Braintrust | Closed platform | Free tier (1 GB / 10k scores); Pro **$249/mo** flat ([pricing](https://www.cekura.ai/blogs/braintrust-pricing)) | Online scoring over prod traces, experiments-over-time, GH-Actions per-case regression comments ([how-to-eval](https://www.braintrust.dev/articles/how-to-eval)) | Better *product* if a managed closed platform is acceptable; rejected here on open-core fit + flat fee. Reasonable fallback if self-hosting Langfuse proves operationally heavy. |
| LangSmith | Closed platform | Plus **$39/seat/mo** + trace overage ([pricing](https://www.langchain.com/pricing)) | Strongest human-in-the-loop: annotation queues, online evaluators, pairwise ([evaluation](https://www.langchain.com/langsmith/evaluation)) | Rejected: per-seat cost scales, LangChain-ecosystem gravity, closed. |
| Arize Phoenix | OSS platform | **Elastic License 2.0** (not MIT) ([github.com/arize-ai/phoenix](https://github.com/arize-ai/phoenix)) | OTel-native tracing + evaluator catalog (Hallucination/QA/Relevance/Toxicity) | Rejected vs Langfuse: non-MIT (open-core friction), thinner experiments/datasets story. |
| **RAGAS** | metric library | OSS (`pip install`) | Reference-free RAG quartet: faithfulness, answer relevancy, context precision, context recall ([metrics docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)) | **ADOPT as a library** for the missing RAG/memory metrics. Don't invent metric math. |
| **DeepEval** | metric library | OSS, pytest-native | 14+ metrics incl. **G-Eval** (criteria-based LLM-judge w/ CoT), agentic metrics (task completion, tool correctness); **pytest CI metric gates** ([github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)) | **ADOPT as a library** for the judge/agent-quality metrics + CI thresholds. |
| promptfoo | OSS harness | MIT (OpenAI-acquired Mar 2026, still MIT) | Declarative YAML evals, model-matrix, large assertion library, **red-team** 50+ categories, CI gate ([github.com/promptfoo/promptfoo](https://github.com/promptfoo/promptfoo)) | **OPTIONAL.** Everything `run_eval.py`+`score.py` hand-roll (1,074 LOC) is promptfoo config — a ~1k-LOC *simplification*, not required. Also the natural home for adversarial/red-team eval later. |

The 2026 industry pattern the searches confirm: **run two of these in parallel** — a red-team CI gate (promptfoo) + a metric CI gate (DeepEval/RAGAS) — logged to one observability platform ([genai.qa comparison](https://genai.qa/blog/promptfoo-vs-deepeval-vs-ragas/), [helpmetest](https://helpmetest.com/blog/llm-evaluation-frameworks/)). For Automatos that maps to **Langfuse (substrate) + RAGAS/DeepEval (metrics) + thin in-house gold-sets (domain) + promptfoo-when-red-teaming.**

**What is explicitly NOT adopted:** an in-house "EvaluationEngine" (the `modules/evaluation` docstring's "engine/, benchmarks/, assessment/ … Sellable as: automatos-evaluation" plan) — that is the build-it-here path the reuse rule exists to stop.

---

## 3. Consolidate the scattered assets — the mapping

T3's explicit ask: fold PRD-108 `experiment.py`, the NL2SQL eval, and the W7 uplift eval into **one harness**. The consolidation is **not "rewrite them into one file"** — it is **"point them all at one store, behind one CI gate, on one dashboard."** Each asset keeps its irreducibly domain-specific gold-set; all of them stop emitting gitignored CSVs and start logging run-records + scores to Langfuse.

| Asset | Today | Consolidated target | Action |
|---|---|---|---|
| **Tool-routing model-matrix** (`scripts/eval/tool_routing/`, 1,074 LOC: `run_eval.py`+`score.py`) | Runs by hand; `results/`+`benchmarks/` gitignored, on one laptop; metrics top-1 / in-set / $/correct / p95 are **well-designed** (`README.md:109-125`, verified) | A Langfuse **dataset** (the 47-query `eval_set.jsonl`) + **experiment**; runner logs a run-record + scores per model×mode | **EXTEND.** Keep the runner (works; problem is persistence). Optional: reimplement on promptfoo (−~1k LOC). Grow set 47→few-hundred with abstain/multi-turn/paraphrase/non-English + AST-score on action+params like BFCL ([gorilla leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)). |
| **Operating-graph uplift** (`orchestrator/evals/operating_graph_uplift.py`, 456 LOC) | Pure-stdlib decision harness; CI runs self-tests as a gate + the number `continue-on-error` (`test.yml:218-259`, verified); measured −32.9 on the **fixture** because per-tenant prod telemetry doesn't exist | Keep as-is as the **decision harness**; log its per-tenant uplift as Langfuse scores; **re-run against organic telemetry** once loops are fed — the "real gate" its own docstring demands | **EXTEND (highest-integrity asset — do not touch its logic).** Its exit-0-always + do-not-flip design is the template for every decision gate. |
| **NL2SQL regression** (`tests/nl2sql_eval/`, ~350 LOC) | Self-contained SQLite execution-accuracy harness (`harness.py`, verified generation-agnostic `evaluate(generate_fn)`); always-on goldens integrity; LLM eval gated by `RUN_NL2SQL_EVAL=1` which **CI never sets**; `baseline.json` still `{"accuracy": 0.0}` (verified, 2026-06-12 placeholder) | A Langfuse dataset (20→30+ questions) + experiment; **one real `RUN_NL2SQL_EVAL=1` run** to bump `baseline.json` off 0.0 and make the regression gate non-vacuous | **EXTEND.** Best eval artifact on the platform and the T3 template. This should be the **first CI-gated lane** (smallest, cheapest, already built). |
| **NL2SQL product benchmark** (`modules/nl2sql/benchmarks/runner.py`, 154 LOC) | The only *product-surfaced* eval; but scores **normalized exact-match** (`execution_match` is a TODO mirroring `exact_match`, `runner.py:84-87`) and test set is a random subset of the same verified examples the retriever trains on — **train/test contamination by construction** | Retire exact-match; re-point the endpoint at the execution-accuracy harness pattern | **FIX.** Wrong metric + contaminated split — a placebo product number today. |
| **Field-memory A/B** (`tools/benchmark_field_memory.py` 962 LOC + `compare_benchmarks.py` + `modules/context/experiment.py`) | Only *committed* results (9 JSONs, 2026-03-30); but the `vector_field` run records **`field_queries:0, field_injects:0` in all 5 trials** — the artifact **can't attribute its 0.88 coverage to the field** (`evals-learning.md` C.5); `experiment.py` **verified zero callers repo-wide** | Superseded by PRD-166's field rework; its numbers describe a dead system | **ARCHIVE/DELETE** the stale results + orphaned `experiment.py`; **replace** with the memory gold-set of §4 (Gerard's call per §12 — flagged, evidence here). |

**Net:** one Langfuse project holding N datasets + N experiments + one CI experiment-gate job (non-required first, matching the platform's honest-CI posture), replacing five orphaned artifacts. Zero net-new eval-*platform* code.

---

## 4. The first concrete metric — memory first (T3's explicit starting point)

The memory dossier makes memory the correct first target on evidence, not sentiment: the founder's seed symptom ("memory saves low-quality memories") is **confirmed, quantified, and under-stated** — durable tier **0% available** (mem0 edge-404), **~87% of L2 is duplicated operational chatter**, **0 L3 promotions in the table's lifetime**, recall last logged **2026-03-11**, L2 recall is an **ILIKE substring match gated behind a temporal regex** so "what did we learn about the Shopify sync?" matches **zero rows by construction** (`memory.md` C.1–C.3, G).

### 4a. Which datasets — and an honest negative on the public ones

T3's brief names LongMemEval / LOCOMO / DMR. The currency check changes the recommendation:

- **LOCOMO is contested — do not chase its leaderboard.** Only **81 QA pairs**; **Zep's own 84% LOCOMO claim was corrected down to 58.44%** ([getzep/zep-papers issue #5](https://github.com/getzep/zep-papers/issues/5)); the "100%" class of claims used `top_k=50` = retrieving the whole conversation; honest performance at reasonable k is ~60% Recall@10 ([mem0 benchmarks 2026](https://mem0.ai/blog/ai-memory-benchmarks-in-2026), [preuve.ai stats](https://preuve.ai/blog/ai-memory-systems-statistics-2026)). Using LOCOMO as *the* memory number would import a broken yardstick.
- **DMR** (the MemGPT-lineage set Zep reports 94.8% on) has **template-drift** criticism and is small/saturating.
- **LongMemEval (v1)** is the more robust conversational-memory choice: **500 manually-curated questions across 6 categories** — information-extraction (single-session), multi-session reasoning, temporal reasoning, **knowledge-update**, **abstention**, single-session-preference — with histories up to ~1.5M tokens; commercial assistants drop ~30% accuracy on it, which is exactly the failure mode a memory loop must catch ([arXiv:2410.10813](https://arxiv.org/abs/2410.10813), [project page](https://xiaowu0162.github.io/long-mem-eval/)).
- **LongMemEval-V2** (May 2026): **451 questions over web-agent trajectories** (up to 115M tokens), 5 agentic abilities (static-state recall, dynamic-state tracking, workflow knowledge, environment gotchas, premise awareness); best system 72.5% vs 48.5% RAG baseline ([arXiv:2605.12493](https://arxiv.org/html/2605.12493v1)). Closer to Automatos's actual agent-memory use, but **harder and heavier — a later target, not the first metric.**

**Recommendation:** the first memory metric is **not** a public-leaderboard number. It is **two internal numbers** that measure *this platform's* loop on *this platform's* data:

1. **Retrieval quality — recall@5 / MRR on a ~50-question workspace gold set** built from real workspace history, authored in the **LongMemEval category shape** (a few of each: "what is the user's brand?" [user-fact], "why did the cron playbook fail?" [knowledge-update over the OpenRouter-402 spam], "did we already sync Shopify orders?" [multi-session], plus **abstention** cases where the answer isn't in memory). Runnable **offline against a store snapshot** — no live traffic needed, so it works during the pilot. This is the number that would have exposed the ILIKE-can't-match defect.
2. **End-to-end task-lift — with vs without the memory section** on a fixed ~30-task set (the number Zep/Mem0 publish and Automatos has never produced). Same experiment shape as the W7 uplift eval — an A/B with an honest gate — reused, not reinvented. This directly answers "does yesterday's work make today's work better?", the North-Star test the memory dossier says today fails.

Then, once the loop is live and producing recall numbers, **run LongMemEval-v1 once to baseline against the field**, and consider LongMemEval-V2 as the agentic-memory target. Sequence matters: **a benchmark over dead plumbing is still dead** — fix the durable tier + semantic recall (`memory.md` J.1–J.4) *before* the public benchmark, or you measure a corpse.

### 4b. First metric per module (the T3 deliverable — each becomes a Langfuse dataset/experiment)

| Module | First offline metric | First online signal | Today's number | Grounding |
|---|---|---|---|---|
| **Memory** (first) | recall@5 / MRR on 50-Q workspace gold set (LongMemEval category shape) + with/without task-lift A/B | promotions/week > 0; % of chat turns whose prompt contained ≥1 memory; Explorer delete success-rate | recall invocations lifetime **6** (last 2026-03-11); L3 promotions **0**; delete success **0%** | `memory.md` G, C.3 |
| **NL2SQL** | **execution accuracy** vs a bumped `baseline.json` (harness exists) + BIRD-mini slice for external comparability (top pipelines ~80%, human ~93%) | success-rate / correction-rate / human-verify-rate from the audit choke-point | offline **0.0 placeholder**; online **n/a (0 audited queries)** | `nl2sql.md` G; [BIRD leaderboard](https://llm-stats.com/benchmarks/bird-sql-(dev)) |
| **RAG / retrieval** | RAGAS **faithfulness + context-recall** on 50–100-Q gold set from the real 19k-chunk corpus; recall@k with enhancement on/off (finally answers "does HyDE/RRF pay for its 4 LLM calls") | **zero-result rate + retrieval-error rate** (today error masquerades as empty — would have caught the ~06-16 embedding-402 silent outage in one day) | none exists; `rag_feedback` **0 rows** | `rag-retrieval.md` G, C-2 |
| **Tool-selection** | top-1 / in-set on the existing 47-Q set, grown, on **organic-traffic replay**; uplift eval re-run on real telemetry | **organic telemetry rows/day** (the single most diagnostic canary — 0 today, would have flagged the blindness in May) | `filtered` **93.6%** top-1 (synthetic, one laptop); uplift **−32.9** | `tool-selection.md` G |
| **Missions / deliverables** | **G-Eval-style LLM-judge** quality score calibrated on ~50 human labels + human-accept rate | deliverable accept/revise (2,242 deliverables have **no feedback affordance** — highest-value human label the platform can collect); board `done-with-error`(194)/`done-no-result`(484) as a **free eval dataset going unread** | none | `evals-learning.md` I.3, C.2 |
| **Chat** | — | vote-rate + positive share (requires votes→scores wiring) | votes write-only (nothing reads `Vote`) | `evals-learning.md` C.2 |

### 4c. Eval-coverage as the meta-metric

This module is the meta-module; its own quality metric is *whether the platform has metrics*. Three trackable numbers, all baselined today:

1. **Eval coverage** — fraction of the 28 capability-map modules with (a) a gold-set/offline eval and (b) an online signal, each having produced a number in the last 30 days. **Today: 0/28** on the 30-day test.
2. **Signal liveness** — organic rows/week into the five learning stores (non-synthetic `tool_execution_logs`, `rag_feedback`, votes, verified NL2SQL examples, memory promotions). **Today: 0/week across all five.**
3. **Decision linkage** — shipped changes gated by an eval number. **Today: 1, ever** (the W7 flag-hold).

---

## 5. The harness design (concrete)

**Instrument two chokepoints, not everywhere:**
- `core/llm/manager.py:612` (`generate_response` / `generate_response_sync`) — every model call → a Langfuse trace.
- `orchestrator/modules/tools/execution/unified_executor.py` finally-block (verified: already fires `fire_telemetry(...)` with `caller_context`, right beside `capture_tool_outcome`) — every tool call → a trace + the existing `tool_execution_logs` write, unchanged.

**Push existing online signals as Langfuse *scores*** (they already exist — they just have no reader): chat votes, `rag_feedback` (once wired — §6), board task outcomes (`done-with-error` / `done-no-result`), playbook step failures, NL2SQL verify events, memory promotions.

**LLM-judge + code evaluators** sampled over production traces for the fuzzy qualities (deliverable quality, mission-synthesis faithfulness, RAG faithfulness) — calibrated against a small human-labelled set via Langfuse annotation queues; use RAGAS/DeepEval metric implementations rather than hand-rolled judge math.

**CI experiment gate** via `langfuse/experiment-action` (or promptfoo) as a **non-required job first** — matching the platform's existing honest-CI posture (the uplift eval and nl2sql-eval already run `continue-on-error`/non-required, `test.yml:218-259`) — promoted to required per-lane once green for two weeks.

**One surface:** a **Command Center → Quality tab**, one row per module (current value, 30-day sparkline, last-run date, red/green vs gate), sourced from Langfuse's API — **do not rebuild eval UI in the frontend**; deep-link into self-hosted Langfuse for drill-down (trace → scores → judge reasoning). This is also *Auto's* read surface: expose the numbers as a platform tool so HARNESS/Auto can reason over quality trends, not just ops metrics. (Today HARNESS is collection-ON / actuation-OFF with 0 prescriptions — `config.py:623,627`, verified — and measures ops metrics, not output quality.)

---

## 6. Sequencing — signal capture is Phase-0, not a follow-on

Ordering is load-bearing: adopting the substrate before feeding the loops yields empty dashboards.

1. **Feed the loops (S–M, strictly first).** Verify on live traffic that the W7 telemetry write (`unified_executor.py`) produces organic `tool_execution_logs` rows across **all** lanes — it is **production-unproven**: it merged ~2026-07-01+, *after* the last recorded live traffic, and the write path swallows every failure at `logger.debug` so "no rows" and "failing writes" are indistinguishable (`tool-selection.md` C.1). Raise that logging to WARNING + a boot-probe + an "organic rows/day = 0" alert (XS). Wire **chat thumbs → `rag_feedback`** for retrieval turns (the chat service already knows the doc ids — one wiring closes both half-tables). Emit board outcomes + playbook step failures as scores. **Success test:** signal-liveness > 0 for four stores within two weeks of real traffic.
2. **Adopt the substrate (M).** Self-host Langfuse; instrument the two chokepoints; push the §5 online signals as scores; stand up the CI experiment gate non-required.
3. **Re-point the three real harnesses at the store; make their numbers real (S).** NL2SQL: one `RUN_NL2SQL_EVAL=1` run → bump `baseline.json` off 0.0 (first CI-gated lane). Uplift: re-run on organic telemetry (its docstring's "real gate"; the actual decider for `TOOL_ROUTING_GRAPH`). Routing: commit a `benchmarks/` snapshot + log runs.
4. **Author the missing evals, memory first (M–L, phased).** The §4 memory gold-set + task-lift A/B (the North-Star payload — the seed complaint becomes a number); then RAG (RAGAS faithfulness/context-recall on the real corpus); then deliverables/missions (G-Eval calibrated on ~50 human labels).
5. **Kill the decoys (S, Gerard's call per §12).** `modules/evaluation`, `modules/learning`, `api/api_playbooks.py` (500s on every call today), the `modules/__init__` re-exports; archive the stale 2026-03-30 field-benchmark results + orphaned `modules/context/experiment.py`. Evidence in this thesis + `evals-learning.md` E; the July kill-list PRD-184 was never authored.
6. **Decide the HARNESS posture (S to decide, Gerard's call).** Once (1) gives it a real diet, either enable `HARNESS_SELF_MANAGEMENT_ENABLED` for the pilot workspace under the existing risk≤2 ceiling, or explicitly park it — not a unilateral deferral.

---

## 7. Cost note (informational; not a gate)

- **Substrate:** Langfuse self-host **$0 licence + ~$20–60/mo infra** (or Cloud free tier). Alternatives: Braintrust Pro $249/mo flat; LangSmith $39/seat/mo + overage.
- **Online judge scoring:** ~10% trace sampling × ~1k-token judge on a mini-class model ≈ **$0.0002–0.001/scored trace** → tens of $/month at current pilot volumes. Industry reference: 10k RAG traces/day with DeepEval+RAGAS ≈ $200–600/mo on GPT-4o ([genai.qa](https://genai.qa/blog/promptfoo-vs-deepeval-vs-ragas/)) — Automatos is far below that.
- **Domain evals:** routing sweep ~$2–5/full run (filtered ~10× cheaper); uplift **$0** (pure stdlib); NL2SQL LLM tier ~$0.10–0.50/run; memory gold-set recall@k is embedding-only (sub-cent); task-lift A/B ~$1–3/run.

---

## 8. Honest negatives / risks (required)

- **The substrate is the easy half; the diet is the hard half.** Nothing in T3 produces a real number until §6.1 lands and *real pilot traffic* flows. The review is explicit that low/zero usage is expected in pilot — so the deliverable is the *instrumented, gated harness that will produce numbers going forward*, not a dashboard of current pass rates. Do not oversell "quality is now tracked" before the loops are fed.
- **LOCOMO/DMR are weak public yardsticks** (§4a) — a plausible failure mode is chasing a contested leaderboard instead of measuring the platform's own loop. The internal gold-set + task-lift A/B is the primary; public benchmarks are a *secondary, later, honest-baseline* exercise.
- **Langfuse self-hosting is real ops** (it runs Postgres + ClickHouse + Redis + object storage). If that operational surface is unwelcome, Langfuse Cloud free tier or Braintrust are the honest fallbacks — the *decision* (adopt a substrate, don't build one) holds regardless of hosting choice.
- **A judge is only as good as its calibration set.** LLM-judge numbers for deliverables/missions are untrustworthy until anchored to ~50 human labels; budget that labelling as part of item §6.4, not after.
- **This thesis does not, by itself, fix any module** — it makes each module's quality *measurable* so the per-module upgrade paths (memory J.1–J.9, RAG J.1–J.11, nl2sql J.1–J.7, tool-selection J.1–J.9) can be judged before/after by a number instead of a vibe. That is the point.

---

## Sources

**Internal (`file:line`):** `orchestrator/evals/operating_graph_uplift.py` (:4-6 legacy language, :49 gate, exit-0) · `orchestrator/tests/nl2sql_eval/{harness.py, baseline.json}` · `orchestrator/scripts/eval/tool_routing/README.md:109-125` · `orchestrator/modules/nl2sql/benchmarks/runner.py:84-87` · `orchestrator/modules/context/experiment.py` (0 callers, verified) · `tools/benchmark_field_memory.py` + `tools/benchmark_results/*.json` · `.github/workflows/test.yml:149-158,210-259` · `orchestrator/core/llm/manager.py:612` · `orchestrator/modules/tools/execution/unified_executor.py` (telemetry finally-block, verified) · `orchestrator/modules/evaluation/__init__.py:20-29` · `orchestrator/config.py:623,627` · `frontend/lib/chat/api.ts:47` (vote path, verified) · dossiers `{evals-learning, nl2sql, memory, rag-retrieval, tool-selection}.md` + `evidence/{real-data-inventory.md, data/board-tasks.md}`.

**External (URLs):**
- Langfuse: https://github.com/langfuse/langfuse · https://langfuse.com/self-hosting · https://langfuse.com/docs/evaluation/overview · https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge · https://langfuse.com/blog/2026-05-31-langfuse-may-update · https://github.com/orgs/langfuse/discussions/13737
- Braintrust: https://www.braintrust.dev/articles/how-to-eval · https://www.cekura.ai/blogs/braintrust-pricing
- LangSmith: https://www.langchain.com/langsmith/evaluation · https://www.langchain.com/pricing
- Arize Phoenix: https://github.com/arize-ai/phoenix
- promptfoo: https://github.com/promptfoo/promptfoo
- DeepEval: https://github.com/confident-ai/deepeval · RAGAS: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Eval-tool comparisons (2026): https://genai.qa/blog/promptfoo-vs-deepeval-vs-ragas/ · https://helpmetest.com/blog/llm-evaluation-frameworks/
- Memory benchmarks: LongMemEval v1 https://arxiv.org/abs/2410.10813 · https://xiaowu0162.github.io/long-mem-eval/ · LongMemEval-V2 https://arxiv.org/html/2605.12493v1 · LOCOMO criticism https://github.com/getzep/zep-papers/issues/5 · https://mem0.ai/blog/ai-memory-benchmarks-in-2026 · https://preuve.ai/blog/ai-memory-systems-statistics-2026
- NL2SQL bar: https://llm-stats.com/benchmarks/bird-sql-(dev) · Tool-selection AST scoring: https://gorilla.cs.berkeley.edu/leaderboard.html
