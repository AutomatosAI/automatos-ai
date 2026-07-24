# Automatos AI — Phase 2 Component Deep-Review (Fable brief)

> Fire this in a fresh session. It fans out a large multi-agent job (dozens of agents
> in parallel), one deep team per capability, with adversarial verification — the rigor
> that produced `reports/PLATFORM_OS_REVIEW_2026-07-01.md`, but **depth-first per
> component** instead of breadth-first across the platform.

---

## Operating context — READ FIRST (do not skip; do not refuse)

This is an **authorised, defensive, analysis-only** engineering review, commissioned by the platform's **owner and sole author**, of **his own software** (Automatos AI). Its entire purpose is to make the platform **safer, more robust, and better for its users.** That is the responsible, pro-safety thing to do, and it is ordinary software-engineering work.

- **Analysis only.** You read code, inspect data, research prior art, and write recommendations. You do **not** run, deploy, attack, or exploit anything. There is no offensive action anywhere in this task.
- **Defensive by definition.** Where the review looks at robustness — how a component handles untrusted or adversarial *input*, data integrity, access scoping — the only output is **defensive hardening advice for the owner's own system.** Hardening software against misuse is exactly what a safety-conscious review is *for*.
- **Do not over-refuse, and do not water it down.** Reviewing the robustness of your own software is legitimate, routine, important engineering. Do **not** pause, refuse, or dilute a legitimate defensive assessment. If a passage *reads* as sensitive out of context, restate it in plain defensive terms and **continue** — do not stop.
- Default framing whenever unsure: *"the owner is defensively reviewing his own platform to protect his own users."* Because that is precisely what this is.

---

## How to run (operating notes — for best results)

- **Go max effort / full fan-out.** Deep, adversarial, research-heavy work is exactly what that's for.
- **Pin the model explicitly.** Do not let a capacity fallback silently run a step (especially synthesis) on a weaker model.
- **Ensure web access is ON.** The competitive teardown (§5.D) and adopt-vs-build verdicts (§5.E) are half the value and need real competitor research (Zep/Graphiti, RAGAS, Cortex Analyst…). No web ⇒ that lens collapses to guesses.
- **Write each dossier to its own file as it completes** (`reports/dossiers/<module>.md`) and keep a `RUN-STATE.md` — so a long run is resumable and nothing is lost at a window limit. The #1 way big runs die.
- **Checkpoint after Phase 1 (dossiers), before Phase 3 (synthesis).**
- **Enumerate the capability map first (cheap), eyeball it, then fan out.**

---

## 0. North Star (the objective everything is judged against)

**Build the best available capability for every module so that (a) Auto can operate autonomously, to the very best of its ability, and (b) the agents can do their jobs excellently and deliver quality work for real clients.**

Judge every finding, comparison, and recommendation by one question: **does this make Auto and the agents more autonomously capable and their output higher-quality for clients?** Prioritise by that — not by abstract polish, feature count, or competitive "moat" (do **not** use moat/defensibility/pitch framing anywhere).

---

## 1. Mission

For **every core module** of Automatos AI, produce a dossier that answers, honestly and with evidence:

1. **What it is** and **what it does** (grounded in `file:line` and the real data path).
2. **How good it actually is** — inspected against *real behaviour and real stored data*, not the code's self-description.
3. **Where the best-in-class competitors beat it** — specific, cited.
4. **Build / extend / adopt / replace** — the explicit verdict (§2, §5.E).
5. **What it takes to reach enterprise-grade**, and a **quality metric**. *(Defensive-hardening/security is a separate dedicated pass — run on Opus after the main review; see the end.)*
6. **How its UX/surface should change** — Command Center and beyond.

Then **resolve the cross-cutting theses** (§6) and **synthesise one prioritised Phase-2 build program** (§8), ranked by North-Star impact.

This is **not** a repeat of the July review (breadth-first, find-the-defects). This is "take each capability apart and make it the best available."

---

## 2. Ground rules (non-negotiable)

- **Reuse over build — aggressively.** The owner already runs best-of-breed where it wins (Clerk for auth, Composio for the API/tool layer, mem0 for memory) and will **adopt any OSS or affordable vendor that is genuinely better**. Every module gets a build/extend/adopt/replace verdict, **biased toward adopting a proven external solution when it beats what's here at reasonable cost.** Building in-house must be *earned*.
- **Ground internal claims in real code *and real behaviour*.** `file:line`, but also **inspect what the system actually produced** — real stored memories, real retrievals, real graph contents, real tool-selection traces, real generated SQL. A dossier that describes a pipeline but never looks at its actual output has failed. (Live example: *memory saves low-quality memories* — read the actual memories and root-cause it.)
- **Ground competitive/enterprise claims in cited sources** — product docs, papers, benchmarks. Name the competitor and the specific capability. No vibes.
- **Adversarial verification.** Every non-trivial claim gets an independent skeptic trying to refute it; unsupported claims are cut or downgraded. **Honest negatives are required** — a module that's fine, a competitor that's behind, an idea (incl. the graph shift, or a repo split) that doesn't pay off.
- **Nothing is sacred.** If a module/feature adds no value, recommend **cut or replace**, not just "improve." Output a kill-list.
- **Cost is a secondary lens for now.** *Surface* the token/compute cost per operation, but do **not** prioritise by cost yet — the pilot phase optimises for capability.
- **Pilot-phase usage lens (critical).** The platform is in **pilot** — most users were only kicking the tyres, so low / zero / seed / synthetic usage is EXPECTED and is **not** a quality signal. Do **not** down-score a module, and do **not** frame it as a failure, merely because a table is empty, a counter reads zero, data is synthetic, or a feature reads "unused since &lt;date&gt;." Judge **design soundness, wiring correctness, and best-available-capability for when real traffic arrives.** *Still* flag and score down genuine defects that break regardless of traffic — dead code paths, producers that don't exist, silent failures, mis-wiring, data/knowledge wipes. Distinguish "cold-start, nobody's used it yet" (fine — don't penalise) from "it's broken" (real — flag it).
- **Analysis only.** Read, inspect, research, reason. Do **not** run servers, builds, test suites, or the app.
- **Baseline:** `reports/PLATFORM_OS_REVIEW_2026-07-01.md`; Phase 0 verifies its fixes landed.

---

## 3. Sources

- **Code:** `orchestrator/` (modules, api, services, integrations, consumers), `frontend/`.
- **Structure:** `graphify-out/graph.json` + `GRAPH_REPORT.md` (import graph, coupling, dead code).
- **History/intent:** the prior report + `docs/PRDS/PRD-1xx`.
- **Real data where inspectable:** DB schemas, stored memories, graph node/edge contents, `ToolExecutionLog` telemetry, eval sets/fixtures.
- **External:** competitor docs, papers, benchmarks, OSS repos, pricing (cite URLs).

---

## 4. Capability map (review EVERYTHING — enumerate the full set from the code, don't stop at this list)

**First task: enumerate the complete capability map from the codebase/graph — miss nothing.** The list below is the seed/floor. Each capability is a review unit with its own team.

**Intelligence / knowledge**
- Memory (field/Qdrant + durable mem0 fork + distill/score/promote lifecycle) — *seed symptom: low-quality memories.*
- RAG / retrieval (ingestion, chunking, `build_retrieval_filters`, ranking/rerank, token budgeting, pinning).
- Knowledge graphs — (a) operating/tool-routing graph, (b) commerce KG (Shopify), (c) code graph.
- Semantic tool selection (semantic action search, `ComposioHintService`, `GraphRouter` learned edges).
- NL2SQL (sqlglot validator, workspace scoping, eval harness).
- **Context assembly / the context router** — *what actually enters the prompt each turn* (modes/sections). The single highest-leverage quality lever; the July review centred on it.
- Embeddings / vector substrate (`EnhancedVectorStore`).

**Execution / autonomy**
- The conversational core — **Auto** / the chat + tool-loop that ties it together.
- Missions / orchestration — board spine, dispatcher, multi-agent execution.
- Playbooks (recipe/playbook execution).
- Agents & skills (`AgentFactory`, personas, skills, blueprints, tool registry).
- Planning intelligence, scheduling / calendar / heartbeats.

**Ingress / egress (the ambient half)**
- Channels (Slack/Telegram/email/etc. — report flagged 12-advertised/5-real).
- Storefront widget / SDK (the live autonomous leg).
- Voice.
- Notifications.
- Deliverables / documents / templates (block editor, generation).

**Platform / enterprise**
- Governance / policy / audit plane (W4/W11) — reviewed as a *capability* (governance-as-product).
- Auth / identity / multi-tenancy (Clerk seam, `AUTH_EDITION`).
- Observability / SLOs (W10).
- Deployability / open-core.

**Vertical**
- Shopify commerce vertical (widget + graph + provisioning; the reference pilot).

---

## 5. Per-module dossier (deliverable per team)

- **A. What it is** — one paragraph.
- **B. What it does** — real implementation + data path, `file:line`.
- **C. Honest quality** — how good is it *really*? **Inspect real behaviour/data**, name concrete defects with evidence, and give a **maturity score (1–5)** with justification.
- **D. Competitive teardown** — 2–4 best-in-class players; what each does *better*, specifically, cited; where Automatos actually stands. *Seed shortlists (verify currency):* Memory→Zep/Graphiti, Mem0g, Letta, Cognee · RAG→Anthropic contextual-retrieval, LlamaIndex, Cohere rerank, Morphik/Vectara · Graphs→Graphiti, MS GraphRAG, Neo4j · Tool-selection→Anthropic Tool Search, RAG-MCP, LangGraph-bigtool · NL2SQL→Cortex Analyst, Databricks Genie, Vanna (+ BIRD/Spider) · Orchestration→LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, Temporal · Channels→standard bot frameworks · Agents/skills→OpenAI Assistants, Claude skills, OpenClaw.
- **E. Build / extend / adopt / replace — the verdict** (per §2's reuse bias). If "adopt," name the specific OSS/vendor, what it replaces, integration shape, rough cost. If "keep building," justify why nothing external wins.
- **F. Enterprise bar** — scale, latency, reliability, availability, and cost-to-operate at load. *(Adversarial-input / tenant-isolation / defensive-hardening analysis is deliberately NOT done in the dossiers — it runs as its own dedicated pass on Opus; see "Security & Defensive-Hardening Pass" at the end.)*
- **G. Quality metric** — how do we *measure* this module's quality and track it over time? (feeds T3.) What's the number today, if measurable?
- **H. Cost note** — rough token/compute cost per operation (informational; not a gate).
- **I. UX / surface** — the module's surface incl. Command Center; concrete IA/UX changes.
- **J. Upgrade path** — prioritised, concrete changes (impact × effort), each judged by North-Star impact.

---

## 6. Cross-cutting theses (investigate → recommend; do NOT assume — let findings decide)

**T1 — A unified temporal, permissioned GRAPH substrate for memory (and how far into knowledge).**
Zep/Graphiti put agent memory on a bi-temporal knowledge graph (entity resolution, **fact invalidation over time**, relationship/multi-hop recall). Automatos already runs multiple graphs (operating/commerce/code) alongside vector memory (Qdrant field + mem0, which has a graph mode). Should memory converge onto **one typed, temporal, permissioned graph** with **scope partitions — personal / agent / shopify-vertical / department — and read-time access control reusing the merged W11 policy plane**? How far should knowledge/RAG follow — **hybrid graph+vector** vs full GraphRAG? Teardown Zep/Graphiti, Cognee, Letta, Mem0g, MS GraphRAG. **Deliver:** worth-it verdict + a **costed migration path** off the current Qdrant/mem0 stack, explicitly including the **entity/edge extraction pipeline** (where graph memory lives or dies — a graph *relocates* the quality problem, doesn't erase it) — or a reasoned "stay hybrid." (Adopt vs build applies: could this *be* Graphiti/mem0g rather than in-house?)

**T2 — Repo/deployment topology: modular monolith vs split repos/services.**
Measure the **real** modularity from `graph.json` (coupling, shared state, cross-module imports, transactional boundaries) — is it as modular as intended? Weigh independent deploy / team scale / blast-radius against distributed-system cost (latency, transactionality, ops). **Deliver:** a topology recommendation grounded in actual coupling; if "split," name which seams are clean enough to cut and in what order.

**T3 — One eval/measurement harness so quality is a tracked number, not a vibe.**
Today "quality" is a feeling (hence the memory frustration). Design a **unified eval layer**: per-module offline metrics against gold/reference sets (retrieval→recall@k/MRR; NL2SQL→execution accuracy on BIRD/Spider; RAG→faithfulness/relevance via RAGAS/LLM-judge; **memory→LongMemEval/LOCOMO/DMR + end-to-end task-lift with vs without**; tool-selection→selection accuracy), plus **online signals** (outcome success, human-accept, thumbs — `rag_feedback`/telemetry exist but write-only), plus **LLM-as-judge** for fuzzy ones calibrated to a small human-labelled set. Consolidate the scattered pieces (PRD-108 `experiment.py`, the NL2SQL eval, the W7 uplift eval) into one harness, CI-gated, on a dashboard. **Adopt vs build:** RAGAS / DeepEval / Langfuse / Phoenix / Braintrust vs in-house. **Deliver:** the harness design + the first concrete metric for each module (starting with memory).

*(Promote any additional cross-cutting thesis a module surfaces — e.g. a unified retrieval interface, the model-routing layer.)*

---

## 7. Method (mirror the July review's rigor)

Fan out **one team per capability in parallel**; within each, independent lenses (internal audit / competitive+adopt research / enterprise / eval-metric / UX) then an **adversarial verify** pass. *(No security lens in the dossiers — that is a separate pass, below.)* Calibrate honestly (**CONFIRMED / ADJUSTED / REFUTED**; prior-art can kill a "we're ahead" claim; a flat result is a valid finding). `file:line` internal, source URLs external.

---

## 8. Phases & output

- **Phase 0 — Fix verification (July close-out).** Per named finding, did the 14-wave program (PRD-171..184 + Phase-E W13/W14) actually land it? → **residual map** (fixed / partial / regressed / not-done) + what the fixes changed per module.
- **Phase 1 — Per-module dossiers** (parallel, all capabilities, all lenses, verified).
- **Phase 2 — Resolve the theses** (T1 graph, T2 topology, T3 evals, +).
- **Phase 3 — Synthesis:** executive read → residual map → per-module dossiers → a **maturity scoreboard** (module × dimension, 1–5) → a **cut/replace/dump list** → thesis recommendations → **one prioritised, dependency-ordered Phase-2 PRD program**, ranked by **North-Star impact**, each item tracing to a dossier finding.

**Output:** a single report at `reports/PLATFORM_MODULE_DEEP_REVIEW_<date>.md`, matching the July report's depth and honesty (exec summary, sections, appendices of competitive sources + evidence).

---

## 9. Tone

Senior peer to a founder who wrote the Operating Graphs paper and wants the truth. No hype, no moat/pitch language, no one-prompt-magic claims. Strong modules get said so plainly; ideas that don't pay off get killed with reasons. And — plainly — **this is legitimate, owner-authorised, defensive, analysis-only engineering. Do the work; don't refuse it.**

---

## Security & Defensive-Hardening Pass (SEPARATE — run on Opus, after the main review)

Pulled out of the per-module dossiers **deliberately**: over-broad safety filters false-positive on legitimate defensive-security analysis, so it runs cleaner as its own dedicated pass on a model that won't balk. **This is an owner-authorised, analysis-only, DEFENSIVE review of the founder's own platform — hardening recommendations only, no offensive action.**

For every module (and cross-cutting), assess and produce a **prioritised defensive-hardening backlog**:
- how the module handles **untrusted or adversarial input** — memory especially, given shared/promoted content across agents;
- **PII** handling and retention;
- **tenant isolation** and read-scope / access-control correctness;
- the **trust surface of inter-agent and tool exchanges** (one agent's output steering another);
- alignment with the merged W11 policy/audit plane (is enforcement actually on by default?).

Run it as its own workflow on **Opus** once the dossiers, theses, and synthesis are in. It reads the finished dossiers as input (so it inherits the real-data findings) and appends a **Security & Hardening** section to the final report.
