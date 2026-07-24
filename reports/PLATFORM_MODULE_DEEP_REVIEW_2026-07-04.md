# Automatos AI — Platform Module Deep-Review (Phase 2)

**Date:** 2026-07-04 · **Model:** Opus (maximum rigor, analysis-only) · **Reviewed tree:** `origin/main @ 77bc9c6d5` (all 14 hardening waves merged; a handful of dossiers cite `e040d9b53`, verified byte-identical for their paths) · **Real data:** Railway production Postgres, read-only, 2026-07-04

**North Star (the objective everything is judged against):** *build the best available capability for every module so that (a) Auto can operate autonomously to the very best of its ability, and (b) the agents deliver quality work for real clients.* Everything below is ranked by that, not by feature count, polish, or "moat."

**How to read this.** It synthesises 28 per-module dossiers, 3 cross-cutting thesis verdicts, and a dedicated Opus security pass — all in `reports/dossiers/`. The **PILOT lens** is applied throughout: the platform is in pilot, so empty tables, synthetic seed data, and cold-start counters are **not** failures and are **not** scored down. What *is* scored is design soundness, wiring correctness, the capability ceiling, and genuine brokenness — dead code paths, producers that don't exist, silent failures, mis-wiring, data wipes. The distinction that runs through the whole review is **"cold-start, nobody's used it yet" (fine) vs "it's broken" (flagged).**

---

## 1. Executive summary

**The platform's engineering taste is consistently good and its execution is consistently unfinished at the last mile.** Across 28 modules the pattern repeats almost without exception: the hard, easy-to-get-wrong architecture is sound — one converged tool loop, one context assembler, a Postgres-native board spine with exactly-once claims, fail-closed tenant resolution, a curate-then-store memory design, a taint-gated promotion job, a genuinely honest eval harness for the operating graph — and then the loop that would make it *work* is dead, unmeasured, or switched off. Nothing here needs a rebuild. Almost everything needs **arming, wiring, and a number.**

Read against the North Star, the two halves resolve like this:

**Autonomy (can Auto run on its own?) — the machinery exists; the nervous system is severed in three specific places.** Since W1 fixed F001, all ten headless execution surfaces funnel through one tested `AgentFactory.execute_with_prompt` path, and the daily heartbeat/playbook clocks genuinely fire in production. But three defects break the feedback that autonomy depends on: (1) **the universal tool-telemetry write is type-poisoned** — `ToolExecutionLog.user_id = Column(Integer)` but the chat lane binds a Clerk **string** id, so every logged-in chat tool call fails its INSERT and is swallowed at DEBUG (`telemetry.py:107-112`); this is *the* root cause that the operating graph has 0 organic edges across 21 real workspaces after two months, and it starves the learning plane, the SLOs, and the uplift eval all at once. (2) **The platform's one live autonomous production line — playbooks — failed silently every day for ~2.5 weeks** (OpenRouter 402) while marking its board tasks `done` and telling no one, because there is no `playbook_failed` notification type and `board_task_bridge.py:114` hard-codes `status='done'`. (3) **The governance plane that would let Auto safely take destructive/external actions is off by default, fails open when on, and is bypassed by four lanes** — so on a real deployment there is no enforced boundary, and the only route through the confirmation gate is an all-or-nothing "full autonomy" dial. Auto today writes memories nobody reads, takes actions nothing records, and cannot complete a supervised destructive action.

**Client-quality (do the agents deliver good work?) — the substrate is real but the quality is unproven and, in the two most-client-facing places, actively suspect.** The RAG corpus is real (19,130 chunks), the workspace Knowledge Graph is the freshest learning-adjacent surface in the DB (155 blobs rebuilt 2026-07-01), and the deliverables engine has a clean editor-independent block schema. But: the durable-memory tier has been **dead in production** (mem0 host 404s on every path) and Auto has run without durable memory with nothing on any dashboard saying so; the document-vector plane **may have been silently empty since W2** (the committed config can't construct the F005-guarded backend) with a random-vector embedding fallback that would make retrieval *quietly meaningless* rather than empty; the live Shopify pilot's cross-sell edges are **wiped by every catalog re-sync** so the graph the shopper-facing opener cites has zero `frequently_bought_with` edges while its status block boasts 16; and the only informative memories in production are the founder telling Auto it is lying. The two-stage mission verifier is advisory-only — it pays judge tokens on every task and gates nothing. Quality is nowhere a tracked, current number: every eval figure the platform has ever produced is synthetic, stale, placeholder, or on one laptop.

**What is genuinely strong and should be said plainly** (PILOT lens — these are real assets, not cold-start noise): the converged tool loop with dedup / stuck-loop-that-learns / length-recovery is ahead of what most frameworks ship by default; field memory (decaying multi-agent shared context, unit-tested, taint-gated promotion) is a distinctive capability no competitor offers in that form; the RAG choke-point + token budgeter + citation assembly + agentic `read_document`/`grep_documents` tools are the right OS-shaped integration a bolt-on vendor can't give; the operating-graph design (per-tenant, success-weighted learned routing) is genuinely ahead of what Anthropic/LangChain/Composio sell — it is simply unfed and its only eval reads negative; the deterministic Shopify mappers with privacy-by-design FBT are exemplary; the tenant-resolution spine (advisory-lock race fix, cross-tenant guards) and the credential store (Fernet + BOLA + audit) are careful, real security engineering. The security *culture* is sound — the gap is consistently **arming and enforcing**, not designing.

**The honest bottom line.** This is a mature, well-architected platform in a **pilot** where almost nothing has met real traffic, and where the small number of things that *have* run reveal that the last mile — the write that records, the flag that enforces, the number that measures — was never closed. The single highest-leverage fortnight in the whole review is not a feature: it is **make the loops observable and fed** (fix the telemetry write, surface playbook failure, verify/relight the retrieval plane, stand up the memory eval), because until "better" is a measured number, every larger investment — a graph substrate, a learned router, a new vertical — is spent blind, and this platform already has too many blind dead planes.

---

## 2. Maturity scoreboard (all 28 modules)

Maturity 1–5 judged by the North Star (capability ceiling + real brokenness, **not** cold-start usage). Verdict is the module's own build/extend/adopt/replace call. "Top defect" is the single most North-Star-damaging finding from each dossier's C/E sections.

### Intelligence / knowledge

| Module | Mat. | Verdict | Headline | Top defect (file:line) |
|---|:---:|---|---|---|
| **memory** | **2** | Replace deploy-shape · extend field/distill · kill relics | Right taste (curate-then-store, shared field, taint gate); the parts that work feed the part that's dead | Durable tier dead in prod — `MEM0_API_URL` 404s on every path; Auto ran w/o durable memory, silently (`mem0_client.py:158-164`; 0 L3 promotions ever) |
| **rag-retrieval** | **3** | Extend spine · adopt Cohere rerank + contextual chunks + RAGAS | Best-integrated slice architecturally; quality half unproven and partly fictional | Silent quality cliff — embedding failure → empty result indistinguishable from success; RAG likely returning nothing since ~06-16 (`service.py:981-983`) |
| **knowledge-graphs** | **2.5** | Keep commerce/code · adopt-trial Graphiti for doc leg (gated on T1) | Operationally ahead of stock OSS; content-quality machinery a generation behind | Catalog sync `import_graph(merge=False)` wipes all accumulated knowledge on every product edit (`api/shopify.py:568`) |
| **tool-selection** | **2.5** | Extend embedding spine · adopt search-tool + Composio Router · hold learned edges dark | Proven baseline layer; learning superstructure unfed, unread, measured-negative | Learning loop never ate a real datum (100% synthetic telemetry); uplift eval **−32.9** vs +5 gate (`operating_graph_uplift.py`) |
| **nl2sql** | **2** | Extend (narrowed) · adopt semantic-model + benchmark-in-product patterns | Best validator + tenancy in class; accuracy machinery entirely unplugged | Semantic layer is a 4-layer placebo — can't load, save crashes on 2 nonexistent methods, wrong format for the reader (`service.py:654-678`) |
| **context-assembly** | **3** | Extend · adopt context-editing/memory-tool + tiktoken + prompt caching · delete rivals | Genuinely singular assembler; on real data most intelligence sections inject noise or nothing | Memory section injects 402-spam / recorded lies with no relevance floor or type filter (`sections/memory.py:261-267`) |
| **vector-substrate** | **2** | Extend seams · consolidate onto Qdrant+pgvector · replace S3 Vectors · kill zombies | Model choice + Qdrant design textbook; operated as a system it can't prove its main plane is on | Committed prod config can't construct the F005-guarded S3 backend → document plane plausibly dark since W2 (`config.py`/`s3_vectors_backend.py:51-55`) |

### Execution / autonomy

| Module | Mat. | Verdict | Headline | Top defect (file:line) |
|---|:---:|---|---|---|
| **auto-core** | **3** | Extend · adopt streaming/parallel-tools at provider seam | Real, robust, tested live agentic loop; a great engine with a painted-on speedometer | No real streaming — full completion then fake typewriter (`streaming.py:309-376`); + every chat belongs to user id=1 (`api/chat.py:82-89`) |
| **missions-orchestration** | **3** | Extend · Temporal only on a named trigger | Well-engineered mechanical layer; autonomy loop dead-ends at a one-shot approval notification | Verification is advisory-only — FAIL verdicts pass anyway; retry machinery is dead code (`reconciler.py:457-479`) |
| **playbooks** | **2.5** | Extend · replace miner vertical | Platform's only live autonomous line; also the most damning failure story | Failed-marked-done + no `playbook_failed` event — line down ~2.5wks, told no one (`board_task_bridge.py:114`; `notification_dispatcher.py:54-55`) |
| **agents-skills** | **2.5** | Extend plane · adopt Agent Skills open standard · kill dead third | Hardened converged spine; the layer that makes agents *capable* is corrupted in prod | Flagship agent's prompt corrupted — 5× `platform-management` skill, ~5k dup tokens/turn (`sections/skills.py:48-100`; no `UNIQUE(agent_id,skill_id)`) |
| **planning-scheduling** | **2.5** | Extend firing (DB-first) · replace HARNESS heuristics with evals · keep heartbeats | Most-alive plane in the platform; makes Auto autonomously *noisy*, not useful | ~90% of the 148k-row heartbeat table is probe spam; HARNESS produced 0 prescriptions in 15 months; F041 locks config to super-admin (`heartbeat_service.py:244-254`; `api/heartbeat.py:28-32`) |
| **tool-runtime** | **3** | Extend · adopt approval-grant pattern | Best-engineered seam in the platform; nervous system severed in 3 places | **Telemetry `user_id` type poison** — Integer col, Clerk-string bound → every chat tool call writes 0 rows, swallowed at DEBUG (`composio_cache.py:215`; `telemetry.py:91,107-112`) |
| **composio-integration** | **3** | Keep vendor · extend+consolidate wrapper | Deep, pilot-hardened core; several load-bearing guardrails broken/bypassed | Destructive-gate metadata sync is a no-op (hardcoded 8-app placeholder + `await` on a sync method) → gate runs blind forever (`composio_action_sync.py:330-340,207`) |
| **llm-core** | **2.5** | Extend manager · adopt OpenRouter failover + LiteLLM price map + prompt caching | Clean provider abstraction + BYOK + per-call telemetry; the operational half failed a live test | Let the whole autonomous output lane die silently for ~2.5wks on OpenRouter 402 — failure data in its own table, fuel gauge unread; no retry/failover/streaming (`manager.py:573,610,624-625`) |

### Ingress / egress

| Module | Mat. | Verdict | Headline | Top defect (file:line) |
|---|:---:|---|---|---|
| **channels** | **2** | Extend core · adopt email via Composio/Chatwoot · kill 7 dead adapters | Correct architecture, broken last mile, unused in practice | Reply gate keyed on legacy `integrations` bag → the one active channel can't reply, endpoint reports success anyway (`webhooks.py:417…`; F026 NOT DONE) |
| **storefront-widget** | **3** | Extend · adopt measurement patterns (Fin/Rep) | Best-built ingress module; well-architected, unproven, uninstrumented | Zero effectiveness metric + headline opener rides untested extraction pipeline (F091) that can confidently fabricate to shoppers |
| **voice** | **2** | Adopt transport (Vapi/Retell) · keep agent bridge | Deep brain, shallow (cold) voice wrapper; correct reuse architecture | Non-streaming end-to-end (full LLM before 1 TTS byte) + 500-char truncation silently mutilates spoken answers (`chat_voice.py:126-150,284-293`) |
| **notifications** | **2** | Extend inbox · evaluate Novu for delivery engine | Architecture right; plane is honest about successes and silent about failures | Went blind to the outage it exists to surface — `playbook_complete` stopped 06-16, no failure event exists (`recipe_executor.py:1588`; `_fail_execution` dispatches nothing) |
| **deliverables-documents** | **3** | Extend fusion · adopt Carbone loop model + Plate editor | The fusion (brand+vars+tool+registry+flywheel) is the real value; unproven, incomplete | Durable client link dies after 1h — persists raw presigned URL as `download_url`, `ExpiresIn=3600` (`generation_service.py:642-645,697-705`; F030) |
| **onboarding-intake** | **2** | Extend · lean on Firecrawl harder | Well-engineered async pipeline; dark-launched, single-vertical, untested, never run | Entry point commented out (`welcome-modal.tsx:116-140`) + 1 archetype + 0 tests → never ran for a real client |

### Platform / enterprise

| Module | Mat. | Verdict | Headline | Top defect (file:line) |
|---|:---:|---|---|---|
| **governance-policy** | **2** | Extend (arm) · selectively adopt Oso | Best-architected governance plane; off by default, cost-blind, headless | Whole W4 plane behind `AUTOMATOS_POLICY_PLANE` default OFF, set nowhere; even ON fails open (`config.py:645`; `unified_executor.py:271-280`) |
| **auth-identity** | **3** | Keep Clerk · consolidate authZ · flip flag · kill fossil | Auth solved + tenant-resolution strong; authorization is the unfinished half | Editor/viewer decorative on 85/92 routers (a viewer can create agents/launch missions/delete deliverables) — `@require_permission` on ~6 (`permissions.py:12-36`) |
| **observability-slos** | **2** | Adopt Langfuse/OTel · extend W10 primitives · delete sprawl | Measurement primitives good and honest; almost none reach the operator | Command Center "is-it-working" strip 403s to blank for non-super-admins (`analytics_real.py:38-42`); flagship SLI structurally dead on prod data |
| **deployability-open-core** | **2** | Extend (arm last mile) · adopt Supabase-style schema diff | Right-shaped scaffolding; fresh clone can't boot, CI green-while-broken | `docker-entrypoint.sh` is non-executable in git (100644) → container dies at start; smoke lane masks it green (F009) |

### Vertical

| Module | Mat. | Verdict | Headline | Top defect (file:line) |
|---|:---:|---|---|---|
| **shopify-vertical** | **2** | Extend (fix) · adopt post-purchase support | Most complete vertical by construction; client-facing intelligence not yet trustworthy | Catalog re-sync wipes FBT edges — pilot graph has **0** `frequently_bought_with` while status says 16; F032 auto-runs it on every inventory change (`shopify.py:568`; `graph_service.py:428-461`) |

### Cross-cutting (discovered unit)

| Module | Mat. | Verdict | Headline | Top defect |
|---|:---:|---|---|---|
| **evals-learning** | **2** | Adopt Langfuse substrate · extend 3 real harnesses · kill decoys | The meta-module; quality is nowhere a tracked number, and the packages named "learning"/"evaluation" are empty theatre | 0/28 modules have a live quality number; every eval figure is synthetic/stale/placeholder; 2 feedback tables (one reader-no-writer, one writer-no-reader) |
| **code-canvas** | **2** | Extend (close loop) · adopt sandbox runtime + MCP · ask Gerard on scope | Well-built scaffolding; the assembled system can't do its one job | No prompt-ingress path — nothing ever calls `client.query(...)`; the started SDK session connects and idles (`canvas_session_service.py:289,452`) |

**Scoreboard shape:** 0 modules at 1; **13 at 2, 6 at 2.5, 9 at 3**; 0 above 3. The absence of any 4+ is the headline — no module clears "proven, measured, client-quality" — but it is a *pilot* absence: it reflects "unproven and unmeasured," not "badly built." The clustering at 2–3 with strong sub-component engineering is the review's central finding in one number: **good bones, open loops.** (Every module's component-engineering would score materially higher in isolation — memory's field/distill are 3–4, tool-runtime's spine ~4, the assembler ~4; the deployed-system scores are dragged down by dead loops, not sloppy code.)

---

## 3. The three thesis verdicts

### T1 — Unified temporal, permissioned GRAPH substrate for memory & knowledge → **HOLD; FIX-THEN-DECIDE**

Do **not** adopt a graph memory substrate now. Decompose the thesis into four separable decisions:

- **T1-a (conversational memory on a temporal graph): HOLD / no.** The memory failure is **operational, not representational** — a dead store, an 87%-spam write side, and a recall path (ILIKE gated behind a temporal regex) that fires 6 times in the table's life. *A temporal graph over dead plumbing is still dead*, and would ingest the same spam at higher LLM cost on an account already hitting daily 402 credit limits.
- **T1-b (document/agent-output KG on Graphiti): ADOPT-TRIAL, gated.** This leg genuinely lacks entity resolution + temporal invalidation + hybrid retrieval — Graphiti's exact core, Apache-2.0, embeddable, `group_id`→workspace, FalkorDB-Lite to bound infra. Run a **time-boxed trial on this leg only**, with the **exit criterion = the memory/KG eval (T3) showing measured uplift over the repaired baseline.** If it doesn't beat the fixed hybrid baseline, build resolution+invalidation in-house.
- **T1-c (scope partitions + ACL): keep the W11 policy plane as the ACL, not `group_id`.** Graphiti's `group_id` is a coarse *partition*, weaker than the existing PRD-124 team ACL. Add a typed scope axis as metadata enforced by W11 at read time; the partition field is an implementation detail beneath it.
- **T1-d (how far into RAG): HYBRID, vector-first — no full GraphRAG.** The cheapest measured retrieval wins are BM25 + Cohere rerank (−67% failure) + contextual chunks, not a graph. Never pay full-GraphRAG up-front indexing for a global-query capability the product doesn't yet consume.

Honest currency check that reshapes the "adopt the leader" instinct: the 2026 LongMemEval leaderboard has **moved past Zep/Graphiti** (OMEGA ~95.4%, Mastra ~94.87% vs Zep ~63.8–71.2%), so vendor benchmarks are upper bounds, not guarantees — which is exactly why T1 must be decided on Automatos's own eval number.

**First step:** stand up the memory/KG eval (T3's memory metric) and make the loop live enough to feed it — memory dossier J.1–J.4 + J.7, in that order. T1 is undecidable today because there is no number.

### T2 — Repo/deployment topology: modular monolith vs split → **STAY A MODULAR MONOLITH**

Do **not** split RAG (or memory, or nl2sql) into its own repo/service. An AST parse of all 754 backend `.py` files (3,059 internal imports — the authoritative signal, not the graph's INFERRED edges) shows the feature modules are **already cleanly decoupled from each other**: only **3.0% of imports are one feature module reaching directly into another.** What binds the system is a **shared kernel** (ORM + DB session + config + LLM client, 80.5% of cross-module coupling) and one Postgres (152 tables, FK graph, the board dispatcher's `FOR UPDATE SKIP LOCKED` exactly-once claim) — the textbook signature of a *healthy* modular monolith, not a ball of mud straining against its process boundary. A split's benefits (independent deploy cadence, blast-radius ownership) are **organizational** and accrue to teams that don't exist (Gerard = 2,417 of ~2,900 commits). Its costs land squarely on the platform's two worst recurring bugs: **silent failure across a network hop** and **loss of the single-DB transactional model.** And the platform already ran the experiment: extracting mem0 as an HTTP service produced a silent weeks-long outage and version drift, and its own dossier verdict is to **un-split it back in-process.**

**First step:** harden the boundaries *in-repo* — an `import-linter`/`grimp` CI contract locking the 3.0% lateral-coupling in ("no feature module imports another except through `modules/tools` or `api`"), plus un-splitting mem0. Re-run T2 only on a named trigger: a second team, a measured throughput bottleneck (`codegraph` is the one pre-cleared clean seam), a hard isolation need, or a data-residency requirement.

### T3 — One eval/measurement harness so quality is a tracked number → **ADOPT Langfuse; EXTEND 3 harnesses; feed the loops FIRST**

**ADOPT Langfuse** (self-hosted, MIT core) as the trace/score/dataset/experiment substrate; **EXTEND** the three real in-house harnesses (tool-routing, NL2SQL, uplift) onto it as thin domain gold-sets — do **not** build an eval platform; use **RAGAS + DeepEval** as metric libraries. But the thesis-reshaping finding is that all four commercial platforms assume *you have production traffic to score*, and here the binding constraint is **signal capture, not tooling** — adopting a substrate before feeding the starved loops produces beautiful empty dashboards. **Feeding the loops is Phase-0 of T3, not a follow-on.**

The first concrete metric is a **memory** one (the seed frustration of this whole review): recall@5 / MRR on a ~50-question workspace gold set in the LongMemEval category shape (runnable offline against a store snapshot — works during the pilot), plus a with-vs-without task-lift A/B reusing the W7 uplift eval's honest-gate shape. Then, once the loop is live, one LongMemEval-v1 run to baseline against the field. Do **not** chase LOCOMO (contested; Zep's own 84% was corrected to 58.44%).

**First step:** verify the W7 telemetry write actually produces organic rows across all lanes (it is production-unproven and swallows failures — see the tool-runtime type-poison bug in §4), wire chat thumbs → `rag_feedback`, emit board outcomes as scores. Success test: signal-liveness > 0 for four stores within two weeks of real traffic.

**One eval has ever gated a decision, and it worked** (the W7 flag-hold on a −32.9 number). That honest-gate pattern — exit-0-always, "the number is the deliverable," do-not-flip — is the template to make the norm.

---

## 4. Quick wins (1-day / high-leverage fixes, ranked)

Every item is a small change with disproportionate North-Star payoff. Ranked by (impact × how-cheap). All `file:line` verified against the live checkout during this synthesis.

| # | Fix | file:line | Payoff | Effort |
|:--:|---|---|---|:--:|
| **1** | **Fix the telemetry `user_id` type poison.** Resolve Clerk-id → integer `User.id` before insert (or retype the column); raise the swallow from `logger.debug` to WARNING + a boot-probe + an "organic rows/day = 0" alert. | `core/models/composio_cache.py:215` (Integer col); `modules/tools/execution/telemetry.py:91,107-112` | **Restores all tool logs.** Un-starves the *entire* learning plane — operating-graph edges, affinities, intent clusters, selection-health, the W7 uplift eval, and SLI-1 — with real data for the first time. Single biggest unblock in the review; one line + a CI canary. | XS |
| **2** | **End the 17-day silent playbook outage.** Add `playbook_failed` to `VALID_EVENT_TYPES` + defaults; dispatch it from `_fail_execution`; change the board bridge to set `failed` (not `done`) on failure. | `core/services/notification_dispatcher.py:54-55` (no `playbook_failed`); `services/board_task_bridge.py:114` (`status='done'`); `api/recipe_executor.py:1588` | Converts the platform's one live autonomous line from "fails daily, tells no one, marks itself green" into same-hour signals. The board stops lying over a production outage. | S |
| **3** | **Repair the severed playbook learning loop + import guard.** Fix the 3 imports of the March-renamed `recipe_memory_service`/`recipe_learning_service` (now `playbook_*`); add a regression test importing every `core.services.*` referenced anywhere. | `api/recipe_executor.py:1118,1742`; `api/workflow_recipes.py:1231` (modules confirmed gone from disk) | Playbooks have written memories but read none since March, `auto_learning` no-ops, `/learn` 500s — a rename + swallowed `ImportError` that survived 3.5 months. | S |
| **4** | **Stop retrieval from silently returning noise.** Remove `DeterministicEmbeddingProvider` from all production selection paths — fail loud on a missing/failed embedding key instead of returning hash-seeded random vectors; make failure typed (`empty` vs `error`) with a zero-result/error-rate metric. | `core/llm/clients/base.py:225` (`rng.standard_normal`); `embedding_manager.py:90-93`; `modules/rag/service.py:981-983` | On the same OpenRouter key that 402-failed daily since ~06-16, retrieval has plausibly been empty-or-meaningless with **nothing on any dashboard**. This is the single biggest silent client-quality risk on the platform. | S |
| **5** | **Fix the durable-deliverable link rot.** Persist the app-relative `/api/documents` path as `download_url`; let the existing re-mint endpoint own presigning. | `modules/documents/generation_service.py:642-645,697-705` (`ExpiresIn=3600`) | The artifact a client is handed **404s after one hour** — the most damaging bug on the output plane. Fix is cheap and half-built. | S |
| **6** | **Stop catalog re-sync wiping the cross-sell graph.** Make `_product_sync_impl` preserve `frequently_bought_with` edges (mirror the orders path's strip-then-remerge, or `merge=True` with catalog-node replacement); add an FBT-persistence integrity check to CI. | `api/shopify.py:568` (`merge=False`); `graph_service.py:428-461` | The live pilot's marquee feature is broken *by construction*: 0 FBT edges present while the status block claims 16, and F032 auto-wipes on every inventory change. Restores the one thing the shopper actually sees. | S |
| **7** | **Give the RAG feedback loop a mouth.** Wire chat votes (which write the read-only `Vote` table) to also write `rag_feedback` with the turn's retrieved document ids. | `frontend/lib/chat/api.ts:47` (posts to `/api/chat/vote`); `api/rag/feedback` has 0 callers | The W9 negative-feedback penalty is correct code reading a table that has **0 rows ever**; one wiring closes both half-tables and turns the "learning from retrieval feedback" claim from fiction into fact. | S |
| **8** | **Thread real identity into chat.** Use `ctx.user` for `chats.user_id`, message saves, vote checks, and the PRD-163 approval `_driving_clerk`. | `api/chat.py:82-89` (`get_user_id` returns id=1); `service.py:1268-1275` | Every chat + every mission-approval attribution currently resolves to user 1; the ownership check compares a constant to itself. Cheapest trust fix; unblocks per-user memory signal. | S |
| **9** | **Kill the noise at the source (memory pollution).** Move the mem0 30s probe writes out of `heartbeat_results`; stop the daily-summary double-write + the fabricated `User:/Assistant:` heartbeat "conversation"; add a memory-injection relevance floor + content-type exclusion (`playbook_summary`/`heartbeat_log`). | `heartbeat_service.py:244-254,1049-1089`; `modules/context/sections/memory.py:261-267` | Flips the 87%-junk memory ratio and stops 402-spam / recorded lies from reaching every prompt. The assembly-side floor protects clients *now*, independent of the write-side fix. | S |
| **10** | **Make the operator's cockpit visible.** Split the super-admin-locked analytics router so own-workspace health tiles (primitive-health, errors, SLOs, activation) are reachable by workspace admins; wire `/api/analytics/slos` into the "is-it-working" strip. | `api/analytics_real.py:38-42` (`require_super_admin`); no frontend caller of `/slos` | The Command Center "is it working?" strip 403s to blank for every non-super-admin, and three tracked SLOs render nowhere. The operator sees nothing today. | S |
| **11** | **Reject-on-mismatch webhook verification.** Delete the two "allow through for debugging" fall-throughs; 401 on mismatch/exception; require a signature when a secret is configured. | `api/composio.py:630,633`; `api/webhooks.py:59,67-69` | A forged Jira/agentic event currently dispatches a real agent or playbook on the platform's autonomous-inbound leg. The sharpest edge on the platform; a one-file deletion. | S |
| **12** | **Truth-in-tools for NL2SQL + the `generate_document` id gap.** Rewrite the `query_database`/`smart_query_database` cards to describe only what executes; add `template_id` to the `generate_document` registry ToolSpec. | `modules/tools/registry/tool_registry.py:658-757,1185-1218` | Tool cards actively mislead the agent LLM (advertising a dead PandasAI lane, a deleted main-DB path, dead params) — for an agent platform, the tool card *is* the UI; and non-chat agents can't pass the template id they were just handed. | S |

**Top-3 (the ones to do first):** **#1 telemetry type-fix** (unblocks the whole learning plane), **#2 playbook failure visibility** (ends the silent outage on the one live autonomous line), **#4 embeddings fail-loud** (stops silent-noise retrieval on the client-facing grounding path). Items #1–#12 together are roughly one to two weeks and move the platform from "blind and unmeasured" to "observable and honest" — the precondition for every larger investment.

---

## 5. Kill-list (cut / delete, from the dossiers' "nothing is sacred" verdicts)

Grep-proven dead or actively-dishonest surface. These are the never-authored PRD-184 deletions plus per-module additions; cutting them removes decoy surface that misleads reviewers, agents, and the founder. Grouped by confidence.

**Delete now (dead-on-arrival / fabricating / zero real callers):**

- **`modules/learning/` + `modules/evaluation/` + `api/api_playbooks.py`** — the packages literally named "learning"/"evaluation" contain hardcoded demo data behind a crashing mounted API (`db.execute(raw_string)` no `text()`), and an empty TODO scaffold. They signpost *away* from the real loops. (evals-learning E; F069/F082/F080)
- **`api/permissions.py` + `AgentToolPermission`/`Tool`/`PermissionAuditLog` tables** — workspace-blind by schema (no `workspace_id`), disconnected from the real Composio/tool-runtime gating, untouched since pre-waves. A fake "RBAC pillar." (auth-identity C.5)
- **Placebo/fabricating endpoints:** `POST /api/agents/{id}/execute` (invented `execution_id`, hardcoded 2025 timestamp, executes nothing), `/api/agents/active` + `/health` (global-factory fiction, reports ~0 active while dozens run), `/{id}/performance` (hardcoded 85/99.8). (agents-skills C.4/C.5)
- **`nl2sql/intelligence/`** (1,687 LOC, 0 callers, still exported) + the legacy PRD-21 limb in `nl2sql/service.py:680-883` (calls a nonexistent `_build_connection_string`) + `SchemaLinker` (0 callers, false "embedding" docstring). (nl2sql C.8)
- **`EnhancedVectorStore` + `SearchService` + `ContextRetrievalEngine`** (the F079 trio; the namesake store's table was dropped in PRD-135 and its "cosine" uses the L2 operator) + the in-process FAISS leg in `/api/v1/memory`. (vector-substrate C.5; context-assembly F079)
- **Memory relics:** `AdvancedMemoryManager` vertical + `api/memory.py` router (fake-success delete) + `MemoryKnowledgeSystem` models whose tables are 0-rows-forever. (memory E.4)
- **`llm-core` dead scaffolding:** `function_executor.py`/`function_registry.py`/`response_parser.py`/`semantic_skill_matcher.py` (~1,400 LOC, 0 external callers) + the misfiled broken `api/anthropic_client.py`. (llm-core C.10)
- **`exec_planning.py` stub vertical** — 8 "planning" tools that are hardcoded template writers with 0 LLM (routed but exposed to no agent). (planning-scheduling B5/§12)
- **`modules/tools/execution/concurrency.py`** (0 callers; even if wired, its unprefixed action names never match real `platform_*` actions) + `modules/tools/service.py` `ToolService` (two latent crashes) + `composio_tool_router.py`'s crash-on-`db_session` delegate. (tool-runtime C.6)
- **7 driverless legacy channel adapters** (teams/google_chat/signal/imessage/irc/matrix/line, 1,589 byte-identical lines) + `_ping_platform_legacy` (0 callers). (channels F081)
- **Frontend relics:** `/api-control` (placebo panel for the PRD-168-deleted mock system), `/styleguide` (442 lines routed in prod), two of three tracked lockfiles (nondeterministic builds), the `workspaceMeta='pilot · 11 op'` fabricated pill, the `/chat/[id]` zombie route + its 3 live `router.push` callers. (deployability F084; auto-core/observability F036/F038)
- **Discarded KG output:** `graph.html` export (0 consumers), `surprising_connections`/`score_all` dead computation, hyperedge prompting (parsed then dropped), `knowledge_nodes`/`knowledge_edges` tables + their forever-zero analytics tiles. (knowledge-graphs E.5)

**Retire (migrate then delete — replaces a real surface):**

- **The legacy workflow engine** (`api/workflows.py` 1,424 lines + `workflow_templates`) — a fifth execution engine, Composio-webhook-reachable via `jira_bug_triage`, with none of the missions/board hardening. Migrate the jira recipe onto the Mission/Playbook path, then unmount. (missions C.8/J.10; F078)
- **The `PlaybookMiner` vertical** (`api_playbooks` + `modules/learning/playbooks/miner.py` + the stubbed `PlaybooksPanel` miner UI) — mining recurring step-sequences is a fine future PRD against real `recipe_executions`, not this demo scaffold. (playbooks E)

**Decide-then-cut (Gerard's call per §12 — surface, don't defer unilaterally):** the RAG dark features (pinning UI or delete S5; multimodal search tools over an unfed store); the `TOOL_SIGNAL_RECORDER_ENABLED` dark flag (default-true or delete — "dark forever" is the one wrong state); the field-benchmark stale 2026-03-30 results + orphaned `modules/context/experiment.py`; the two caller-less `ContextMode`s (COORDINATOR, ORCHESTRATOR_STAGE) + `tone.py`.

Author these as **PRD-184** (the never-written July kill-list). Net: several thousand lines of decoy/fabricating surface removed, and the codebase stops lying to the humans and agents that read it.

---

## 6. Prioritised Phase-2 PRD program

Ordered by North-Star impact (autonomy + client quality), dependency-aware, grouped into waves. Each item traces to its module dossier and folds in the security appendix's top risks where they coincide. **Wave 0 is load-bearing for everything after it** — it makes the platform observable and fed; without it, every later investment is spent blind.

### Wave 0 — Make the loops observable, honest, and fed (the precondition)
*Nothing larger is worth building until quality is a number and the nervous system is connected. This wave is mostly Quick-Wins §4 promoted to a program, ~2 weeks.*

| PRD | What | Dossiers | Why first |
|---|---|---|---|
| **P2-01** | **Telemetry write repair + per-lane CI canary + fail-loud embeddings** | tool-runtime J1, evals-learning J1, rag-retrieval J1, vector-substrate J2 | The type-poison fix un-starves the *entire* learning plane; the embedding fail-loud stops silent-noise retrieval. Together: the platform can finally see what its agents did and whether grounding worked. (Security §5.1) |
| **P2-02** | **Playbook failure visibility + severed-learning repair + failure circuit-breaker** | playbooks J1–J3, missions J6, notifications P0 | Ends the ~2.5-week silent-outage class on the one live autonomous line; a paused-not-refiring playbook saves the daily 402 spam. |
| **P2-03** | **Verify/relight the document-vector plane** | vector-substrate J1, rag-retrieval C2 | One AWS probe (is prod dark? what index dimension?) then fix env + re-embed, or fold into the Qdrant move. If dark, every agent answer over workspace documents is currently ungrounded — this gates all RAG work. |
| **P2-04** | **Stand up the eval substrate + memory eval (T3 Phase-0)** | T3, memory J7, evals-learning J2–J4 | Self-host Langfuse; instrument the two chokepoints; author the ~50-Q memory gold-set + task-lift A/B. The seed complaint ("memory saves low-quality memories") becomes a tracked number. Feed the loops *before* the dashboard. |
| **P2-05** | **Operator cockpit reach + honest tiles** | observability-slos J1–J3, governance I | De-super-admin the own-workspace health tiles; wire SLOs into the "is-it-working" strip; deliverable-freshness tile (would have caught the 06-16 outage day one). |

### Wave 1 — Resurrect the dead client-facing loops
*The parts that work feed the parts that are dead. Bring the dead ones back, in-process, measured against Wave-0's numbers. ~4–5 weeks.*

| PRD | What | Dossiers | Why |
|---|---|---|---|
| **P2-06** | **Un-split memory: in-process durable store on Qdrant + distill the write side + semantic always-on recall + fix promotion** | memory J1–J4, T2 §4 | Retires the dead Railway fork, the HTTP hop, the breaker apparatus; flips the 87%-junk ratio; un-deadlocks promotion. Converts memory from "subtracts from autonomy" to functional. |
| **P2-07** | **RAG quality stack: Cohere rerank on + contextual chunk annotations + Postgres BM25 leg + retrieval eval** | rag-retrieval J3–J6, vector-substrate J8 | The cited −49% to −67% retrieval-failure stack, gated by the eval. Turns "hybrid is decorative" into real hybrid, cheaply. |
| **P2-08** | **Shopify integrity: stop the FBT wipe + debounce webhooks + mapper behavioral tests + un-skip golden journeys** | shopify J1–J3, knowledge-graphs J1–J2, storefront-widget J2 | Restores the pilot's marquee cross-sell feature and stops the widget confidently fabricating to shoppers. (Security §5.3) |
| **P2-09** | **Deliverables: fix link-rot + `template_id` on the autonomy lane + unresolved-gate + clean-render metric** | deliverables J1–J3, J5, J7 | The client-facing artifact stops rotting after an hour; non-chat agents can use templates; clients stop receiving `[[unresolved]]` documents. |
| **P2-10** | **Agents-skills data repair: dedupe skills + `UNIQUE(agent_id,skill_id)` + concurrency-safe seeders + real skill priority** | agents-skills J1–J3, J5 | Removes ~5k dup tokens/turn from the flagship agent's prompt — Auto's per-turn quality is the product. (Security §3.2.a: skill-attach visibility check) |

### Wave 2 — Arm autonomy safely (governance, approvals, identity)
*Auto can't be trusted with client-facing autonomous action until there's an enforced boundary and a scoped principal. Fold in the security appendix's top-2 cluster here. ~3–4 weeks.*

| PRD | What | Dossiers | Why |
|---|---|---|---|
| **P2-11** | **Staged policy-plane rollout + fail-closed-for-destructive + price the budget gate + close the 4 bypass lanes** | governance J1/J3–J5, tool-runtime J3, auth-identity J1 | The single highest-leverage security action — closes the F040/F042/F043 authorization cluster and turns scaffolding into guardrails. (Security §2.1, §2.2 — CRITICAL) |
| **P2-12** | **Tool-level approve→grant→resume + in-chat approval card** | tool-runtime J3/E1, governance I1 | Converts the 12 dead-end confirmation actions from "impossible unless full autonomy" into supervised autonomy that actually completes. The biggest single unlock for client-safe autonomous operation. |
| **P2-13** | **Close the untrusted edges: webhook reject-on-mismatch + widget origin fail-closed/CORS boot-guard/Redis limiter + Composio destructive-gate feeder fix** | channels, storefront-widget, composio | The three surfaces that go hot first with real customers. (Security §1.1/§1.2/§1.4 — CRITICAL/HIGH) |
| **P2-14** | **AuthZ consolidation: flip flag + `@require_permission` sweep (85 routers) + collapse 5 role vocabularies → 1 + credential NULL-workspace deny** | auth-identity J1–J4 | Turns editor/viewer from decorative to real; a viewer stops being a functional editor. (Security §3.1/§3.2.b) |
| **P2-15** | **Governance operator surface + audit retention + GDPR subject-tags** | governance I1–I5, J9–J10 | Art.14 "effective human oversight" is a UI claim; the durable-grant machinery is unreachable without a front door. |

### Wave 3 — Close the capability gaps (the substrate + editor + engine questions)
*Now that quality is measured, spend on the gaps the numbers justify. ~4–8 weeks, several gated on Wave-0 numbers.*

| PRD | What | Dossiers | Why |
|---|---|---|---|
| **P2-16** | **Consolidate vector stores onto Qdrant + pgvector (retire S3 Vectors) + Qdrant snapshots + config-integrity CI gate** | vector-substrate J3–J7, T2 | One engine less; hybrid + snapshots unlocked; makes open-core RAG actually work; fixes the "four dimension knobs" and settings-placebo. |
| **P2-17** | **Graphiti trial on the document/agent-output KG leg (gated on T1 exit criterion)** | T1-b, knowledge-graphs J8 | Closes entity-resolution + temporal-invalidation + hybrid-retrieval with proven OSS — *only if* it beats the repaired hybrid baseline on the T3 eval. |
| **P2-18** | **NL2SQL productization: fix the semantic layer end-to-end + run the eval for real + kill-list** | nl2sql J1/J3/J4 | The proven ~20% accuracy lever (semantic model) currently can't be written; the eval is stuck at a 0.0 placeholder. Convert advertised → measured capability. |
| **P2-19** | **Mission verification gates once + real resume (wire or delete checkpoints) + approvals inbox** | missions J1–J3 | The flagship quality feature (cross-model judge) gates nothing today; the flagship recovery feature (checkpoints) doesn't exist at runtime; 47% of missions ever created are parked at a one-shot approval. |
| **P2-20** | **Context assembly: persist the trace + one tokenizer + model-aware budgets + prompt caching + adopt context-editing/memory-tool** | context-assembly J1–J3, J6–J7, auto-core J12 | Makes "what did Auto know?" answerable, makes every budget honest, and captures the ~85% prefix-cache cost saving on the highest-volume plane. |
| **P2-21** | **Adopt the Agent Skills open standard (trigger-based L2 activation + L3 script execution via workspace-worker)** | agents-skills J4 | Ecosystem-scale skills + an order-of-magnitude prompt-cost cut per skill; keeps the genuinely differentiating tenanted/governed/scanned distribution. |
| **P2-22** | **Ship onboarding intake (entry point + tests + generalise beyond Shopify) + streaming voice (adopt Vapi/Retell) + code-canvas prompt loop** | onboarding, voice, code-canvas | Three dark-launched surfaces that are architecturally sound but unproven; each is a small unblock (an uncommented CTA, a vendor swap, a `client.query` wire) that turns dead code into a live capability. Voice/code-canvas scope decisions are Gerard's per §12. |

### Wave 4 — Deployability, topology hardening, and the honest-CI floor
*Second-order but it protects every other module's quality. Mostly one-line/one-command arming actions + human repo-admin. ~1 week + human steps.*

| PRD | What | Dossiers | Why |
|---|---|---|---|
| **P2-23** | **Fix fresh-clone boot (exec-bit + alembic stamp/squash, as a coupled pair) + de-mask the smoke lanes + schema-drift CI check** | deployability J2/J4/J5, T2 §6 | "Open-core" is a claim, not a fact, until a stranger can clone and boot. The exec-bit fix without the schema stamp turns silent drift into a hard boot abort — they must land together. |
| **P2-24** | **[human] Arm the gates: branch protection `strict:true` + require CI/security lanes + coverage ratchet + purge the tracked Clerk artifact** | deployability J1/J3, auth-identity J7 | The 30-second repo-admin action that gives every other module's CI teeth; the July review attributes real red-main incidents to exactly this gap. (Security §6.4/§6.1) |
| **P2-25** | **In-repo topology discipline: import-linter contract + un-split mem0 + `/api/tasks` under the policy gate** | T2 §6, memory E1, code-canvas C7 | Locks in the 3.0% lateral-coupling that makes the monolith healthy; closes the ungoverned unattended-execution ingress. (Security §4.1) |
| **P2-26** | **HARNESS rebuild on evals (baselines→DB, GEPA-style propose-measure-keep) + notifications digest + observability sprawl consolidation** | planning-scheduling J7, notifications P1, observability-slos J5 | The self-optimization loop becomes real (0 prescriptions in 15 months today); the notification plane stops burying the signal; the 10-router analytics sprawl collapses to one. |

**Program shape:** ~26 PRDs, 5 waves. The dependency spine is strict — **Wave 0 gates Wave 1 gates the T1/HARNESS bets in Wave 3.** The security appendix's top-3 (webhook trust, policy-plane-on, widget edges) are folded into Wave 2 (P2-11/P2-13/P2-14) because they must precede the widget and channels going hot with real customers, not because current traffic is high. If only one wave ships, it must be **Wave 0** — it is the difference between improving the platform on evidence and improving it on faith.

---

## 7. Security & hardening

A dedicated Opus defensive-hardening pass read all 28 dossiers and re-framed their defects as a prioritised, owner-authorised, **defensive** backlog (analysis-only, no offensive action). Full detail, per-item fixes, and the honest-credit section: **`reports/dossiers/security-hardening-appendix.md`**.

**The one-line posture:** *"careful engineering, wired but not armed."* The hard, easy-to-get-wrong primitives are done well and are credited below; what is missing is **enforcement-by-default at the untrusted edges.** Because the pilot's real traffic is low, the exposure window is small *today* — but the storefront widget, channels, and Composio lane are exactly the surfaces that go hot first when real customers arrive, so these land before scale, not after.

**Top-3 risks (fix first):**

1. **Forged webhook → autonomous agent/playbook execution `[CRITICAL]`.** The Composio trigger webhook *logs and proceeds* on a V3 signature mismatch (`api/composio.py:630,633`) and skips verification entirely when signature headers are absent; the channels/workspace/playbook lanes skip HMAC when no signature is present. A forged Jira/agentic event dispatches a real agent or recipe. **Fix:** delete the "allow through" fall-throughs, 401 on mismatch/exception, require a signature when a secret is set, add a replay guard. One-file deletions. *(= Quick-win #11.)*
2. **The runtime authorization boundary is off by default, fails open, and is bypassed `[CRITICAL, enabler]`.** `AUTOMATOS_POLICY_PLANE` defaults false (`config.py:645`) and is set nowhere; even ON it fails open on any internal error (`unified_executor.py:271-280`); and four lanes execute tools around the chokepoint. Net: no enforced boundary on external side-effects, and the widget empty-permission key is a live god-key. **Fix:** staged `=on` rollout with a fail-closed branch for destructive/external risk classes, then close the four bypasses. *(= Wave 2 P2-11 — the highest-leverage security action in the review.)*
3. **The live storefront widget has an origin bypass + default-open CORS + inert rate limiting `[HIGH]`.** A public `ak_pub_` key skips the domain check when `Origin`/`Referer` is absent (`api/widgets/auth.py:182-183`); CORS allows all origins when the allowlist is unset (the default); the per-process limiter is largely inert. A scraped public key replays from anywhere, burning the merchant's budget. **Fix:** deny public keys when origin is absent (fail-closed), require the allowlist in production, move the limiter to Redis. *(= Wave 2 P2-13.)*

**The rest of the backlog** (23 ranked items in the appendix) clusters into: the four bypass lanes, the destructive-gate blind spot (metadata sync no-ops), the five-way authZ fork (editor/viewer decorative on 85/92 routers), cross-tenant read/attach holes (skill-attach injects another tenant's prompt content; credential NULL-workspace; file-hash dedup; `_agent_id` bleed), the `/api/tasks` ungoverned shell/git ingress, content-injection paths (random-vector embedding fallback; unfiltered memory injection; commerce-KG conversation pollution), and config hygiene (tracked Clerk artifact, credential prod→dev fallback, CI gates that don't bite). Several coincide exactly with the Wave-0/Wave-1 quick wins (embeddings fail-loud, chat identity, memory floor) and are earned twice.

**What is already handled well (honest credit — real assets, not cold-start):** fail-closed tenant-resolution with an advisory-lock provisioning race fix and audience-pinned JWT verification; **server-minted actor identity** that closes the agent-impersonation door at the dispatch layer; a Fernet-encrypted credential store with per-row BOLA checks; SSRF-blocked document rendering + a `SandboxedEnvironment` anti-SSTI; a sanitised git-clone path with a 42-pattern content scan; a two-stage plugin security scanner that ran on all 73 marketplace plugins; the workspace-worker exec sandbox (allowlist + non-root + safe-path); the callback GDPR posture (phone never stored, salted per-Site hash); the **taint-gated memory promotion** (untrusted-provenance patterns never promote — preserve this ordering when memory is consolidated); and a policy-plane design that consciously rejects shell-string hooks as RCE-by-configuration. The gap is consistently arming and enforcing, not designing — which is why the backlog is config flips, deletions of "allow through" fall-throughs, and wiring, not new systems.

---

*Prepared as the Phase-2 capstone. Sources: 28 module dossiers + 3 thesis verdicts + the security-hardening appendix + evidence packs, all in `reports/dossiers/`. Internal claims cite `file:line` in the pinned tree `77bc9c6d5` (top quick-wins and kill-list items spot-verified against the live checkout during synthesis); external competitor/benchmark claims cite source URLs in the dossiers. Baseline: `reports/PLATFORM_OS_REVIEW_2026-07-01.md`; Phase-0 residual map: `reports/dossiers/evidence/phase0-residual-map.md` (38 fixed / 22 partial / 27 not-done / 0 regressed / 3 unverifiable). PILOT lens applied throughout — cold-start/empty/synthetic is not penalised; genuine brokenness is.*
