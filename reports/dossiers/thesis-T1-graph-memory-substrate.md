# Thesis T1 — A unified temporal, permissioned GRAPH substrate for memory & knowledge

| | |
|---|---|
| Thesis | Should Automatos converge memory + knowledge onto **one typed, temporal, permissioned graph** (Zep/Graphiti-style) — scope partitions (personal / agent / shopify-vertical / department) with **read-time ACL via the W11 policy plane**, **hybrid** (graph+vector, not full-GraphRAG) for documents, off the current Qdrant + mem0 split — with the **entity/edge extraction pipeline as make-or-break**? |
| Verdict | **HOLD the substrate migration; FIX-THEN-DECIDE.** Do **not** adopt a graph memory substrate now. Repair the current loop, stand up the memory eval (T3), then run a *time-boxed Graphiti trial on the document/agent-output KG leg only* — gated on measured recall. A full graph convergence for conversational memory is **not yet earned** and, on current benchmarks, may not pay off. |
| Reviewed at | dossiers pinned `origin/main @ 77bc9c6d5 / e040d9b53` (2026-07-04); real-data evidence read-only prod Postgres 2026-07-04 |
| Inputs | dossiers: `memory`, `knowledge-graphs`, `rag-retrieval`, `vector-substrate`, `context-assembly`, `tool-selection`; evidence: `real-data-inventory.md`, `data/{memory-short-term,mem0,qdrant,workspace-graphs,operating-graph-edges}.md` |
| Lens | **Pilot-phase** (§2): 0 organic edges, 87% memory chatter, 0 promotions are **cold-start** signals — judged as design soundness + capability ceiling, not usage failure. No moat/pitch framing (North Star §0). |

---

## 0. One-line verdict

**HOLD** — fix the dead memory loop and stand up the eval first; then trial Graphiti on the document/agent-output KG leg **only**, gated on recall numbers. Converging conversational memory onto a temporal graph is not yet earned, and on current benchmarks Zep/Graphiti is no longer the clear leader to adopt.

---

## 1. What T1 is actually asking, decomposed

The thesis bundles **four separable decisions**. Treating them as one is the trap; the honest answer differs per part:

| # | Sub-decision | Verdict | Why |
|---|---|---|---|
| **T1-a** | Put **conversational/agent memory** (the mem0 + `memory_short_term` half) on a temporal graph | **HOLD / no** | The failure is operational, not representational; a graph relocates the quality problem onto an unmeasured extraction pipeline. |
| **T1-b** | Put the **document/agent-output Knowledge Graph** (the extraction+temporal core) on Graphiti | **ADOPT-TRIAL, gated** | This leg genuinely lacks entity resolution + temporal invalidation + hybrid retrieval, which are exactly Graphiti's core and Apache-2.0. |
| **T1-c** | **Scope partitions** (personal/agent/vertical/department) + **read-time ACL via W11** | **KEEP the ACL, DON'T adopt group_id as the ACL** | Graphiti's `group_id` is a *partition*, not row-level team ACL; Automatos already has PRD-124 team ACL that group_id cannot replicate. |
| **T1-d** | How far into RAG — **hybrid graph+vector vs full GraphRAG** | **HYBRID, and vector-first** | Full GraphRAG pays LLM indexing up-front for a global-query capability the product doesn't yet use; the cheapest retrieval wins are BM25 + rerank + contextual chunks, not a graph. |

The brief's own warning is the spine of this whole thesis: **"a graph relocates the quality problem, doesn't erase it."** Every part of the verdict below turns on that sentence, because the real data proves the quality problem is currently *unmeasured and upstream* of any storage choice.

---

## 2. Where Automatos actually stands (grounded, not self-described)

Automatos is not starting from "flat memory, no graph." It **already runs three graph substrates plus two vector memory stacks** — the convergence question is landing on a platform that is, if anything, *over*-fragmented, and whose graph machinery is a generation behind on content quality while its memory loop is operationally dead.

**The memory half (T1-a target):**
- **Durable tier (mem0 fork) is dead in production.** `MEM0_API_URL` answers Railway edge `404 "Application not found"` on every path — the deployment no longer exists (`evidence/data/mem0.md`). Every L3 read/write silently no-ops (memory dossier C.1). Auto has effectively run **without durable memory**, and no dashboard says so.
- **`memory_short_term` is 2,211 rows, ~87% operational chatter** (playbook_summary 1,255 + heartbeat_log 346 + transcript 336 = 1,937), the *same* OpenRouter-402 failure memorised twice per run daily (`evidence/data/memory-short-term.md`).
- **Zero L2→L3 promotions ever** across the table's life; `memory_items` = 0 rows; `memory_access_log` = 6 reads ever, last **2026-03-11**. Memories are written, never read, never promoted (memory dossier C.3).
- **mem0 runs with `infer=False` everywhere** (`unified_memory_service.py:345,836`) — its headline value-adds (LLM extraction, ADD/UPDATE/DELETE reconciliation, graph memory) are **switched off by design**; the fork is a thin HTTP vector store (memory dossier C.4).

**The knowledge half (T1-b target):**
- The workspace KG **is real and maintained**: 155 `workspace_graphs` blobs rebuilt 2026-07-01 (`evidence/data/workspace-graphs.md`) — the freshest learning-adjacent surface in the DB.
- But it has **no entity resolution** (merge = exact-snake_case-id last-write-wins, `graph_service.py:1060-1061`), **no temporal model** (add-only, no valid-from/valid-to, `:1431-1440`), and **weak NL query** (term-overlap + BFS, no hybrid, no global/theme query) — knowledge-graphs dossier C.2/C.3/C.9.
- **Extraction quality on real workspace data is unmeasured** (knowledge-graphs dossier C). Given the memory module's *confirmed*-low quality from the identical LLM-write-side pattern, assuming graph extraction is fine is unjustified.
- A latent **catalog-sync clobber** (`import_graph(merge=False)`, `api/shopify.py:568`) wipes accumulated knowledge on every product edit (knowledge-graphs dossier C.1 defect).

**The scope/ACL half (T1-c target):**
- Scope partitions today exist only as **team_access lists** (PRD-124 filters every KG read, `graph_service.py:148-181`) and workspace_id columns — there is no personal/agent/vertical/department *typed* scope axis.
- The **W11 policy plane exists** (merged, flag `AUTOMATOS_POLICY_PLANE` default OFF per MEMORY) — read-time ACL is a real seam to reuse, but it is not yet the memory/graph enforcement path.

**The substrate half (T1-d target):**
- **Five-and-a-half vector substrates** run in parallel (vector-substrate dossier B): S3 Vectors (documents), Qdrant (field memory), pgvector (codegraph live + 2 dead legs), JSONB sidecars, in-process FAISS relic, plus mem0's own store. The document-vector plane **may be dark** (committed env can't construct the F005-guarded backend, vector-substrate C.1).
- The operating graph (tool routing) has **0 organic edges across 21 real workspaces**; all 29 edges are synthetic seed frozen 2026-05-05 (`evidence/data/operating-graph-edges.md`).

**The one-sentence reality:** the pieces with the *right taste* (field memory, distill taxonomy, deterministic commerce mappers, taint-gated promotion, per-tenant scoping) are sound; the pieces that **feed and read** them are dead, unmeasured, or unfed. A graph substrate changes the storage; it does not, by itself, resurrect the loop.

---

## 3. The case FOR a temporal permissioned graph substrate

This is a real and well-motivated thesis; it is not a strawman. The strongest points, cited:

**3.1 Temporal fact invalidation is a genuine capability Automatos lacks entirely.** Zep/Graphiti stores every fact as a graph edge with a validity interval (`valid_at`/`invalid_at`); when new knowledge contradicts an existing fact, the old fact is **invalidated, not deleted** — preserving auditable history ([Zep arXiv 2501.13956](https://arxiv.org/abs/2501.13956); [Neo4j — Graphiti](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)). Automatos has **no temporal axis at all**: its consolidation is `SequenceMatcher`/Jaccard string-matching (`contradiction.py:44-82`) that cannot represent "was true, now isn't," and its KG is add-only with contradicted facts coexisting indefinitely (knowledge-graphs C.3). "Pricing is $49" → "$99" produces two live nodes today. For an agent doing client work over evolving facts, this is a correctness gap, not a nicety.

**3.2 Entity resolution is where graph quality lives — and Automatos has none.** Graphiti dedupes/resolves entities against the existing graph per episode, constrained by developer-defined Pydantic entity/edge types ([github.com/getzep/graphiti](https://github.com/getzep/graphiti), verified 2026-07-04: Apache-2.0, 28.4k stars, embeddable `graphiti-core` library, custom entity/edge types, `group_id` partitioning). Automatos resolves by exact string id — "Acme Corp"/"Acme"/"acme_corporation" are three nodes (knowledge-graphs C.2). This is the single biggest content-quality lever for everything agents read from the graph, and it is absent.

**3.3 "Co-located facts" beat top-K vector crowding on interconnected data.** An independent 2026 head-to-head (Graphiti vs Mem0) found Graphiti scored **4.75/5 vs 3.25/5 on knowledge coverage and 4.75/5 vs 3.0/5 on contradiction handling**, because vector-only retrieval suffers "context blindness" — the LLM only sees what similarity returns, and emotionally/lexically weighted embeddings crowd out structurally relevant facts ([dev.to — Graphiti vs Mem0](https://dev.to/juandastic/i-benchmarked-graphiti-vs-mem0-the-hidden-cost-of-context-blindness-in-ai-memory-4le3)). Automatos's memory recall is *worse* than plain vector: L2 recall is an **ILIKE full-substring match** gated behind a temporal regex (memory dossier C.3), so "what did we learn about the Shopify sync?" matches zero rows by construction.

**3.4 The scope-partition idea maps cleanly onto `group_id`.** Graphiti's `group_id` on every node/edge, filterable in all queries, maps 1:1 onto workspace_id and *could* express the personal/agent/vertical/department axis T1 wants ([DeepWiki — Graphiti multi-tenancy](https://deepwiki.com/getzep/graphiti/7.3-multi-database-and-multi-tenancy)). This is a real architectural fit for the partitioning half of T1-c.

**3.5 Convergence would reduce a genuine fragmentation cost.** Cognee's pitch is precisely this — one ECL (Extract-Cognify-Load) engine over relational+vector+graph, the consolidation Automatos lacks across three vector substrates and three memory stacks ([github.com/topoteretes/cognee](https://github.com/topoteretes/cognee); Apache-2.0, ~12k stars, $7.5M seed, 70+ prod deployments). The vector-substrate dossier independently recommends *collapsing* the substrate count. A graph convergence is one way to do that.

**3.6 Adopt-not-build is strongly available for the graph core.** If a graph is chosen, Automatos should **not** build fact-invalidation/entity-resolution in-house: Graphiti is Apache-2.0, embeddable, works with Anthropic/OpenAI/Gemini structured output, and now offers **FalkorDB Lite** (embedded, file-based, zero-server) to cut the infra burden ([Graphiti issue #1240](https://github.com/getzep/graphiti/issues/1240), Feb 2026). The knowledge-graphs dossier's E.3 verdict (ADOPT-TRIAL Graphiti for the doc leg) and the memory dossier's E.2 (Graphiti conditional on T1) both point here.

---

## 4. The honest case AGAINST (a graph shift that doesn't pay off IS a valid verdict)

This is the heavier side of the ledger, and it wins for now.

**4.1 The failure is operational, not representational — a graph over dead plumbing is still dead.** This is decisive. The memory module does not have a *storage-model* problem; it has a **dead store, a flooded write side, and a recall path that can't match** (memory dossier C.1-C.3). Every one of those defects survives a migration to a graph:
- A temporal graph still needs a **live deployment** — the mem0 fork died silently; a Neo4j/FalkorDB could die the same way with the same lack of alerting.
- A temporal graph still ingests **whatever the write side sends it** — feed it playbook-402 spam twice a day and you get a beautifully temporal graph of duplicated failure edges.
- A temporal graph still needs a **recall path that fires** — Automatos's recall is gated behind a temporal regex and never runs (6 lifetime invocations).

Fixing these is ~2 weeks of work (memory dossier J.1-J.4) and is a **prerequisite to even evaluating** T1. Spending the graph-migration budget first would relocate a dead loop into a more expensive dead loop.

**4.2 Extraction is make-or-break, and Automatos's extraction is unmeasured with strong priors it is poor.** The brief names this correctly. Automatos already extracts typed facts (distill taxonomy for memory; LLM entity/relation extraction for the KG) — so it *has* the hard half a graph needs. But the memory dossier **confirmed** that write side produces low-quality output (87% chatter, recorded lies as "memories"), and the KG's extraction quality on real data is **never measured** (knowledge-graphs C). Graphiti would inherit the *same* LLM extraction problem — worse, it fires **more** LLM calls per episode (node extraction, dedup, edge extraction, per-edge resolution, timestamping, attributes) so garbage-in is *more* expensive ([Graphiti issue #1193](https://github.com/getzep/graphiti/issues/1193)). You cannot buy your way out of an extraction-quality problem by changing where the extractions are stored.

**4.3 On current benchmarks, Zep/Graphiti is no longer the clear leader to adopt.** The dossiers cited Zep's 2025 numbers (94.8% DMR, 63.8% LongMemEval vs Mem0 49%). Verified 2026-07-04, the **LongMemEval leaderboard has moved past Zep**: leading systems now include OMEGA ~95.4%, Mastra ~94.87%, EverMind/EverOS ~83% LongMemEval-S, MemPalace ~96.6% R@5, with **Zep/Graphiti at ~63.8-71.2%** ([OMEGA benchmarks](https://omegamax.co/benchmarks); [innobu — Agent Memory 2026](https://www.innobu.com/en/articles/agent-memory-2026-mem0-letta-zep-hermes-openclaude-comparison.html); [mem0 — AI memory benchmarks 2026](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)). Two consequences: (a) "adopt the benchmark leader" no longer uniquely points at Graphiti; (b) self-reported memory benchmarks are treated as upper bounds, not guarantees — which is *exactly* why T1 must be decided on **Automatos's own eval numbers**, not vendor leaderboards. The temporal-graph *approach* is validated; the specific *product leadership* is contested and fast-moving.

**4.4 Ingestion cost and latency are real and land on every write.** The independent head-to-head measured Graphiti at **1.36-2.25× Mem0's token cost** because its dedup pipeline checks each new edge against the entire existing graph ([dev.to benchmark](https://dev.to/juandastic/i-benchmarked-graphiti-vs-mem0-the-hidden-cost-of-context-blindness-in-ai-memory-4le3)); community reports flag LLM cost and observability as "first-class concerns… very expensive at scale" ([Graphiti issue #1193](https://github.com/getzep/graphiti/issues/1193)). Automatos already has a **daily OpenRouter-402 credit outage** in the data (`real-data-inventory` §3) — layering a more LLM-hungry ingestion path onto an account already running out of credits is the wrong sequencing.

**4.5 `group_id` is a partition, not the ACL T1 wants — and Automatos already has better.** T1-c wants **read-time ACL via the W11 policy plane** with personal/agent/vertical/department scopes. Graphiti's `group_id` gives *partitioning* (one id per node/edge, filterable) — it does **not** give row-level team-membership ACL. Automatos's PRD-124 team_access lists already do finer-grained read scoping than group_id offers (knowledge-graphs D notes group_id is "partition, not row-level team lists"). Adopting group_id-as-ACL would be a **downgrade** dressed as an upgrade. The correct design — keep the W11 policy plane as the enforcement point and use group_id (or equivalent) only as a coarse partition beneath it — is more work than "the graph gives you ACL for free," which it does not.

**4.6 Full GraphRAG is the wrong economics for the document leg.** Microsoft's own work shows classic GraphRAG pays LLM extraction up-front on every document; **LazyGraphRAG** achieves comparable quality at ~0.1% of indexing cost by deferring LLM work to query time ([MS Research — LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)). Automatos currently pays LLM extraction up-front **and** gets no global-query capability for it — the worst quadrant (knowledge-graphs D). But the fix for document retrieval quality is not "more graph"; the rag-retrieval dossier's measured-best levers are **BM25 + Cohere rerank (−67% failure) + contextual chunk annotations** ([Anthropic contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval)) — none of which require a graph. Hybrid, vector-first, is the answer; full GraphRAG is not.

**4.7 There is a cheaper "do less" alternative that partly obsoletes the build.** Anthropic now ships a server-side **memory tool** + **context editing** with published evals (+39% over baseline, 84% token reduction on a 100-turn task) as request-parameter integrations ([Anthropic context management](https://claude.com/blog/context-management); context-assembly dossier D). For the *conversational-memory* half (T1-a), a large fraction of the value (persistence across turns, note-taking) is adoptable at the API layer without any graph, any new infra, or any extraction pipeline — on the Claude-routed lanes the platform already uses. This weakens the case for building conversational memory on a self-hosted graph at all.

**4.8 Migration blast radius is large and the platform is mid-pilot with dead planes.** Memory is read by the context assembler on *every* turn (context-assembly B); the KG feeds five agent tools + the live Shopify widget (knowledge-graphs B). Swapping the substrate touches the highest-traffic read path in the system while three planes are already dark (mem0, document-vector-plane-maybe, tool telemetry). The right move under the pilot lens is to **make the existing loop live and measured**, then change substrates from a position of knowing what "better" means — not to re-plumb blind.

---

## 5. Verdict per sub-decision (the actual T1 answer)

**T1-a (conversational/agent memory on a temporal graph): HOLD — do not adopt.**
The problem is a dead store + spam write-side + broken recall, none of which a graph fixes. Fix the loop (in-process durable store on existing Qdrant, distill the write side, semantic always-on recall, promote-on-write), stand up the eval, adopt Anthropic's memory-tool/context-editing on Claude lanes for the cheap win. **Re-open T1-a only if**, after the loop is live and measured, recall numbers plateau in a way that co-located-facts/temporal-invalidation would specifically address.

**T1-b (document/agent-output KG core on Graphiti): ADOPT-TRIAL, gated on T3.**
This leg genuinely lacks entity resolution + temporal invalidation + hybrid retrieval — Graphiti's exact core, Apache-2.0, embeddable, `group_id`→workspace, Pydantic ontology, FalkorDB-Lite to bound infra. Run a **time-boxed trial on this leg only** (episodes per workspace), with the **exit criterion = the memory/KG eval (T3) showing measured recall/answer-quality uplift over the repaired baseline**. If it doesn't beat the fixed hybrid baseline, build entity-resolution + invalidation in-house instead — justified *only then* because a proven OSS implementation exists.

**T1-c (scope partitions + read-time ACL): KEEP the W11 policy plane as the ACL; use partitions only as a coarse index.**
Do not adopt `group_id` as the access-control mechanism — it is weaker than the existing PRD-124 team ACL. Add a typed scope axis (personal/agent/vertical/department) as *metadata* enforced by W11 at read time; a graph's partition field is an implementation detail beneath that, not a replacement for it.

**T1-d (how far into RAG): HYBRID, vector-first — no full GraphRAG.**
The cheapest measured retrieval wins are BM25 + rerank + contextual chunks, not a graph (rag-retrieval E). Keep documents on vector + BM25; let the KG answer the *relationship/multi-hop/global-theme* questions vectors can't, once T1-b proves out. Never pay full-GraphRAG up-front indexing for a global-query capability the product doesn't yet consume.

---

## 6. Costed migration sketch (what T1-b would actually take, if the gate passes)

Presented as the brief requires — a costed path, explicitly including the extraction pipeline — but **sequenced behind the fix-the-loop work**, because a migration now would relocate a dead system.

**Phase 0 — Prerequisites (do regardless of T1; ~2-3 weeks; memory + vector-substrate dossiers).** *Non-negotiable before any graph spend.*
- Resurrect durable memory **in-process on existing Qdrant** (memory J.1; ~1-2 wk; retires mem0 fork, HTTP hop, breaker apparatus).
- Stop the write-side spam (memory J.2; distill/noise-gate the playbook+heartbeat writers; kill fabricated heartbeat "conversations"; ~2-4 days).
- Semantic always-on L2 recall via existing `field_scoring` stack (memory J.3; ~1 wk).
- Fix promotion semantics (drop the access-count AND-gate deadlock; memory J.4; ~1-2 days).
- Verify/relight the document-vector plane (vector-substrate J.1; is prod dark? one AWS probe).
- **Stand up the memory/KG eval (T3): gold-set recall@k + one LongMemEval/LOCOMO baseline** (memory J.7). *This is the gate for everything below.*

**Phase 1 — Repair the KG's live damage (this week; knowledge-graphs J.1-J.3).**
- Scoped catalog re-sync (kill `merge=False` clobber). Mapper behavioral tests. F064 two-line fix. *~S each; independent of any substrate choice.*

**Phase 2 — Graphiti trial on the doc/agent-output leg (gated; ~2-4 weeks eng).**
- Stand up **FalkorDB (or FalkorDB-Lite for the local edition)** as the graph store — one container, or embedded file-based; no Neo4j licensing.
- Route `ingest_agent_output` + document-ingest pendings to **Graphiti episodes per workspace** (`group_id = workspace_id`); encode the node taxonomy as Pydantic entity/edge types.
- **Extraction pipeline (the make-or-break):** reuse the existing typed-extraction prompts as Graphiti's structured-output schema; budget the **higher LLM spend** (1.36-2.25× a flat store; many calls/episode) — and *fix the OpenRouter-402 credit outage first* or the trial can't run.
- Keep `DbWorkspaceClient` exports (or a projection job) feeding the existing drill-in UI + `platform_graph_*` tools during transition — **do not run two permanent substrates**.
- **Exit criterion:** eval shows measured entity-resolution/temporal/answer-quality uplift over the repaired hybrid baseline. **If not → build resolution + invalidation in-house (knowledge-graphs J.4), do not keep Graphiti.**

**Phase 3 — Scope + ACL (only if Phase 2 passes; ~1-2 weeks).**
- Add typed scope axis (personal/agent/vertical/department) as metadata; wire **W11 policy plane** as the read-time enforcement point (not group_id). Reuse merged W11 (flag it on for memory/graph reads).

**Rough cost envelope:** Phase 0-1 ~4-5 weeks (mandatory, pays for itself in a live memory loop). Phase 2 trial ~2-4 weeks + a graph container + ~1.5-2× the current KG-extraction LLM spend on the doc leg + one full re-ingest. Re-embedding the corpus is trivial (~$0.08, vector-substrate H). The dominant true cost is **engineering time on the highest-traffic read path** and **extraction LLM spend on an account already hitting credit limits** — which is why the gate and the credit fix come first.

---

## 7. The single highest-leverage first step

**Stand up the memory/KG eval harness (T3's memory metric) and make the memory loop live enough to feed it — i.e. memory dossier J.1-J.4 + J.7, in that order.**

Rationale: T1 is undecidable today because **there is no number.** "Quality is LOW" is confirmed by inspection but there is no recall@k, no LongMemEval baseline, no promotion-rate, no extraction precision — so any substrate choice (graph or not) would be made on faith. The eval is the instrument that:
1. Converts "should we adopt a graph?" from a vibe into a measured before/after (the whole point of T3, and the brief's stated fix for "the memory frustration");
2. Would immediately expose that the durable tier is dead (0% availability) and recall never fires — the operational failures that must be fixed *first*;
3. Becomes the **exit criterion** for the gated Graphiti trial (T1-b) — no graph ships without beating the repaired baseline on this number;
4. Is cheap and reused across modules (RAG, KG, tool-selection all need the same shape).

Concretely: a ~50-question gold set from real workspace history ("what is the user's brand?", "why did the cron playbook fail?", "pricing was X, is it still?"), runnable offline against a store snapshot, plus one LongMemEval/LOCOMO run to baseline the end-to-end loop — as a non-required CI job first (the NL2SQL harness pattern), promoted to required once green. **Everything else in T1 is sequenced behind having this number.**

---

## 8. Competitive sources (verified 2026-07-04)

| Player | What it does better (relevant to T1) | Source |
|---|---|---|
| **Graphiti / Zep** | Bi-temporal fact invalidation; entity resolution per episode; hybrid retrieval (semantic+BM25+graph) P95 ~300ms; `group_id` partition; Pydantic ontology; Apache-2.0, 28.4k★, embeddable `graphiti-core`; FalkorDB/FalkorDB-Lite/Neptune/Neo4j backends | [arXiv 2501.13956](https://arxiv.org/abs/2501.13956); [github.com/getzep/graphiti](https://github.com/getzep/graphiti); [DeepWiki multi-tenancy](https://deepwiki.com/getzep/graphiti/7.3-multi-database-and-multi-tenancy); [issue #1240 FalkorDB-Lite](https://github.com/getzep/graphiti/issues/1240) |
| **LongMemEval leaderboard (moved past Zep)** | OMEGA ~95.4%, Mastra ~94.87%, MemPalace ~96.6% R@5, EverOS ~83%; **Zep/Graphiti ~63.8-71.2%** — "adopt the leader" no longer uniquely = Graphiti; treat vendor numbers as upper bounds | [OMEGA benchmarks](https://omegamax.co/benchmarks); [innobu 2026](https://www.innobu.com/en/articles/agent-memory-2026-mem0-letta-zep-hermes-openclaude-comparison.html); [mem0 benchmarks 2026](https://mem0.ai/blog/ai-memory-benchmarks-in-2026) |
| **Graphiti vs Mem0 (independent)** | Graphiti wins coverage (4.75 vs 3.25) + contradiction (4.75 vs 3.0) via co-located facts; but **1.36-2.25× Mem0 token cost**; Mem0 wins efficiency/simplicity for ~90% of cases | [dev.to head-to-head](https://dev.to/juandastic/i-benchmarked-graphiti-vs-mem0-the-hidden-cost-of-context-blindness-in-ai-memory-4le3) |
| **Mem0 (upstream/graph)** | 4 graph backends (Neo4j/Memgraph/Kuzu/AGE); `infer=true` extraction+reconciliation loop — the half Automatos switched off; graph mode now "entity-aware ranking," relations not directly traversable | [mem0 changelog](https://docs.mem0.ai/changelog/highlights); [FalkorDB mem0](https://www.falkordb.com/blog/graph-memory-llm-agents-mem0-falkordb/); [mem0 research](https://mem0.ai/research) |
| **Cognee** | One ECL engine over relational+vector+graph (the consolidation Automatos lacks); Apache-2.0, ~12k★, $7.5M seed, 70+ deployments; self-host + on-prem €1,970/mo | [github.com/topoteretes/cognee](https://github.com/topoteretes/cognee); [cognee.ai](https://www.cognee.ai/) |
| **Anthropic (the "do less" alt)** | Server-side memory tool + context editing; +39% over baseline, 84% token reduction on 100-turn task; API-parameter integration, no graph/infra | [context management](https://claude.com/blog/context-management); [contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval) |
| **MS GraphRAG / LazyGraphRAG** | Global/theme query via community reports; **LazyGraphRAG ~0.1% of indexing cost** by deferring LLM to query time — the argument *against* full-GraphRAG up-front indexing | [MS Research LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) |

---

## 9. Bottom line (senior-peer, no hype)

The temporal-permissioned-graph thesis is **intellectually right and prematurely actionable.** Its central capabilities — temporal invalidation, entity resolution, co-located facts — are genuine gaps Automatos has, and if a graph is chosen the adopt-vs-build math favors **Graphiti on the document leg** decisively (Apache-2.0, embeddable, group_id, FalkorDB-Lite). But the platform's memory failure is **operational, not representational**: a dead durable tier, an 87%-spam write side, and a recall path that never fires — and *a temporal graph over dead plumbing is still dead*. The extraction pipeline the brief flags as make-or-break is **unmeasured**, with strong priors (from the confirmed-low memory quality) that it is poor — and Graphiti would inherit that problem at higher LLM cost, on an account already hitting credit limits. On top of that, the 2026 benchmark picture no longer crowns Zep/Graphiti, so "adopt the leader" is no longer a clean call.

So: **fix the loop, build the number, then decide.** Adopt Anthropic's memory tool for the cheap conversational-memory win; repair the KG's live damage this week; run a gated Graphiti trial on the document leg with the eval as its exit criterion; keep the W11 policy plane (not group_id) as the ACL; stay hybrid and vector-first for documents. The highest-leverage first step is the **memory eval harness** — because until "better" is a measured number, every graph-migration dollar is spent blind, and this platform has too many blind dead planes already.
