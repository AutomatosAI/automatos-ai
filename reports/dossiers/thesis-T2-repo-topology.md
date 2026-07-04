# Thesis T2 — Repo / Deployment Topology: modular monolith vs split repos/services

| | |
|---|---|
| Question | Should Automatos stay a modular monolith, or split into separate repos/services (e.g. RAG as its own repo)? |
| Method | Measured the **real** import coupling by AST-parsing all 754 backend `.py` files (not the graph's inferred edges — see "Measurement note"). Cross-checked against `graphify-out/graph.json`, the runtime topology (`docker-compose.yml`, `main.py`), the prod DB census (152 tables, one Postgres), and the deployability / auth / memory / composio / rag dossiers. |
| Verified against | live tree `origin/main` @ 2026-07-04; `orchestrator/` = 244,176 backend LOC; graph.json dated 2026-06-09 (used for shape only). |
| **VERDICT** | **STAY A MODULAR MONOLITH.** Do **not** split RAG (or memory, or nl2sql) into its own repo/service. The coupling data says the modules are *already* cleanly separated from each other (3.0% lateral peer coupling) but *all* welded to one shared kernel (ORM + DB session + config + LLM client) and one Postgres — the exact shape where a service split pays negative dividends. The only defensible extractions are **isolation-driven** (the code-execution sandbox, already split) or **vendor-defined** (Clerk, Composio, Gotenberg — already external). One in-house seam (`codegraph`) is clean enough to extract *if ever forced*, but nothing today forces it. And the platform's own recent experiment — extracting mem0 as an HTTP service — is the cautionary tale: it silently killed durable memory and its dossier verdict is to **un-split it back in-process**. |

---

## 0. The verdict in one paragraph (North-Star framing)

The North Star is "Auto operates autonomously and the agents deliver client-quality work." A repo/service split does **nothing** for either — it is pure operational topology, invisible to Auto and to clients. Its only real benefits are organizational (independent team deploy cadence, per-team blast-radius ownership, polyglot freedom) and those benefits accrue to **teams that do not exist**: this is a solo codebase (Gerard = 2,417 of ~2,900 commits; the rest are his own agents/bots). Meanwhile the costs of a split are immediate and land squarely on the two failure modes the dossiers already flag as this platform's worst recurring bug: **silent failure across a network hop** (the dead mem0 Railway service that ran Auto without durable memory for weeks; the OpenRouter 402 that silently emptied RAG) and **loss of the single-DB transactional model** (the board dispatcher's `FOR UPDATE SKIP LOCKED` exactly-once claim, the fail-closed tenancy filters, the FK graph across 152 tables). Splitting converts cheap, loud, in-process function calls into expensive, silent, cross-service calls — buying a benefit with no beneficiary at the price of the platform's known Achilles' heel. **Stay monolith; keep the module boundaries clean in-repo; extract only for isolation or vendor reasons.**

---

## 1. Measurement note — why the graph's headline number is a red herring (and what the real number is)

The prompt says "measure the real coupling; don't assume." Doing that honestly requires flagging that the **most obvious number is untrustworthy**, then computing a trustworthy one.

**The trap.** `graphify-out/graph.json` has 63,575 edges. If you count `calls`+`uses`+`imports_from`+`inherits` and bucket by module, **74.8% cross module boundaries** — which *looks* like a tightly-coupled ball of mud that wants splitting. But 84% of those edges are `confidence: INFERRED` (`uses` = 21,092 and `calls` = 17,911, all heuristic 0.8 — name-matching across the whole graph, which manufactures spurious cross-module edges wherever two modules happen to share a symbol name). The **EXTRACTED** (AST-real, confidence 1.0) edges tell the opposite story: backend-to-backend, cross-module EXTRACTED dependency edges number **11**. And the graph's own `imports_from` extraction is broken for this purpose — only 356 edges total, mostly test-helpers and frontend, with import *targets* almost never resolving to a source module. **The graph is reliable for shape and community structure; it is not reliable for a cross-module import census.** (This matches the memory-file caveat that graph.json is "shape/coupling-indicative, not exact.")

**The real measurement.** I AST-parsed every backend `.py` file (754 files, 0 parse errors), extracted every `import` / `from … import`, and bucketed importer-file → imported-module at `modules/<x>`, `core/<x>`, `api`, `services`, `consumers`, `channels`, `integrations`, `config` granularity. **3,059 internal import statements.** This is the authoritative coupling signal for a split decision, because a repo split's cost is *exactly* "how many import edges become network calls."

| Metric | Value | What it means for T2 |
|---|---:|---|
| Within-module imports | 746 (**24.4%**) | modules have real internal cohesion |
| Cross-module imports | 2,313 (**75.6%**) | high cross-module traffic *in aggregate*… |
| …of which **into shared cores** (`core/*`, `config`, `services`) | 1,861 (**80.5% of cross**) | …but 4 out of 5 cross edges are "everyone depends on the shared kernel" — the ORM, DB session, config, LLM client. **This is shared-kernel coupling, not feature entanglement.** |
| …**feature ↔ feature** (excl. cores) | 452 (19.5% of cross) | genuine inter-feature traffic is small |
| …of feature↔feature **routed through aggregators** (`api`/`tools`/`consumers`) | 361 (**79.9%**) | and most of *that* is the API/tool layer wiring features together, not features reaching into each other |
| **True lateral peer coupling** (feature→feature, neither an aggregator nor a core) | **91 (3.0% of ALL internal imports)** | **the modules barely touch each other directly.** |

**The finding that decides the thesis:** the feature modules are **already cleanly decoupled from one another** — only 3.0% of imports are one feature module reaching directly into another. What binds the system is a **shared kernel** (`core/models` imported by 31 modules / 603 stmts; `config` by 36; `core/database` by 27 / 276 stmts; `core/llm` by 22; `core/auth` = 216 stmts) plus a **central aggregation layer** (`api` imports from 42 modules; `modules/tools` from 28). That is the textbook signature of a **healthy modular monolith**: low coupling between features, high cohesion within them, everything sharing one data/runtime substrate. It is *not* the signature of a system straining against its process boundary.

---

## 2. Runtime & data topology — what a "split" would actually have to cut

A repo split is cheap; a *service* split has to sever runtime and transactional boundaries. Here is what those look like today.

**One process, one database.** The backend is a single FastAPI ASGI app (`orchestrator/main.py` is the only `FastAPI(...)` instance outside tests). All 28 backend modules run in that one process and talk to **one Postgres** (`DATABASE_URL`) holding **152 tables** ([census](evidence/data/census.md)). There is no per-module database, no bounded-context data ownership — `core/models` defines 99 ORM classes in a shared namespace that every module imports and joins across (RAG's team filter joins `documents`; the board dispatcher locks `board_tasks`; tenancy filters span the whole schema). A service split would either (a) keep the shared DB — giving you distributed *compute* over a shared database, the worst of both worlds (network latency + no data isolation + cross-service migrations) — or (b) partition the DB, which means unwinding an FK/join graph across 152 tables. Neither is remotely justified at 21 workspaces / 22 users.

**The async spine is Postgres-native, not broker-native.** The mission/board engine (`services/board_dispatcher.py`) decouples producers from workers using **`FOR UPDATE SKIP LOCKED` + `LISTEN`/`NOTIFY`** on the shared DB (`board_task_available` channel, `:37-60,100,142`). Its exactly-once guarantee *is* the row lock. There is no Celery/RQ/Kafka/RabbitMQ anywhere in the module tree (grep-confirmed). So the system's concurrency model is intrinsically single-database; you cannot lift a module out of the process without either giving it its own queue infrastructure or leaving it reaching back into the origin DB.

**Zero internal RPC exists today.** Every `httpx`/`requests`/`aiohttp` client in `modules/` calls an **external** service — mem0, firecrawl, GitHub, voice/ElevenLabs, cloud-file providers. **No module calls another module over HTTP.** There is no partial service seam already in flight to widen; a split would be greenfield distributed-systems work, introducing the first-ever internal network boundary into a codebase whose dossiers repeatedly document that its worst outages are silent failures across exactly such boundaries.

**What is *already* out-of-process (and why each is legitimate):**

| Split that exists | Reason | Verdict |
|---|---|---|
| `frontend` (Next.js container) | UI/API is the one universally-correct split | **Keep — correct.** |
| `gotenberg` (PDF render sidecar) | third-party binary, HTML→PDF | **Keep — vendor sidecar.** |
| `workspace-worker` (`profiles:["workers"]`, off by default) | runs **untrusted agent-authored code** — needs process/container **isolation** (`canvas_confinement.py`, `evaluate_tool_confinement`) | **Keep — isolation-mandated.** Note it *still* shares `DATABASE_URL` (`main.py:385`) — even the one legitimate compute-split doesn't decouple data; it isolates execution. |
| `mem0` fork (`MEM0_API_URL` HTTP → Railway) | was adopted as an external memory server | **UN-SPLIT (dossier verdict).** See §4. |
| Clerk (auth), Composio (tools) | vendor SaaS, correctly adopted | external by nature — not "our repos". |

The pattern is unambiguous: **on this platform, out-of-process boundaries are drawn for isolation or vendoring, never to decompose an in-house capability.** That is the correct instinct, and the coupling data explains why.

---

## 3. The specific candidate — "RAG as its own repo" — costed

The prompt names RAG explicitly, so here is the concrete extraction cost, from the measured imports.

**What RAG drags with it (outbound deps).** `modules/rag` imports from: `core/database` (12), `config` (10), `core/models` (8), `core/llm` (7), `core/cache` (4), `core/composio` (4), `core/team_access` (2), `core/context_guard` (1), `services` (1), `modules/search` (5), `modules/knowledge` (1), `core/math` (1). To become its own repo it must carry or re-consume the **shared ORM models** (it joins `documents`/`document_chunks`), the **DB session factory**, the **central config**, the **embedding/LLM client**, and — critically — `core/team_access` + `core/context_guard`, which are the **fail-closed tenancy fabric** the RAG dossier names as its actual differentiator ("no external platform drops into the workspace/team/policy fabric this is welded to"). You cannot extract RAG without either duplicating that fabric (drift risk on a security boundary) or calling back into the monolith for every retrieval (network hop on the hot path).

**What breaks on the caller side.** RAG is called in-process by `api` (21 imports), `modules/tools` (8), `consumers` (3), `modules/context` (2), `modules/agents` (2), `modules/search` (1), `modules/nl2sql` (1). Every one of those becomes an **HTTP client + serialization boundary + new failure mode**. Today a `search_knowledge` tool call is a Python function call that fails loudly if RAG throws. Post-split it is a network call — and the RAG dossier already documents (defect C-2) that RAG's failures get *swallowed into empty results* and that a network-dependency outage (OpenRouter 402) silently emptied retrieval for ~2.5 weeks with nothing on any dashboard. **Adding an internal network hop to the retrieval path multiplies the exact silent-empty-retrieval failure mode the module is already worst at.**

**Cost / risk summary for RAG extraction:**

- **Migration cost:** high — stand up a new service + its own deploy/CI; give it a copy of (or a client to) `core/models`, `core/database`, `config`, `core/llm`, `core/team_access`; retrofit ~38 inbound call-sites across 7 modules with HTTP clients; either share the Postgres (no isolation win) or migrate `documents`/`document_chunks`/`rag_feedback`/`document_usage` out (FK/join surgery). Realistically **weeks**, all of it plumbing, none of it improving retrieval quality.
- **Latency:** worse — retrieval is already 3–10 s (4 sequential enhancement LLM calls + up to 5 vector searches, dossier §H); a network hop adds RTT and a serialization tax to every agent turn.
- **Reliability:** worse — new partial-failure surface on the module with the worst silent-failure record.
- **Benefit:** ~none for the North Star. Independent deploy of RAG helps a RAG *team*; there is no RAG team. Independent scaling helps if RAG is a QPS bottleneck; at 19,130 chunks / 21 workspaces it is not (dossier §D: S3 Vectors is "fine and cheap" at this scale).

**RAG verdict: do not extract.** Keep it as the well-bounded in-repo module it already is (self-containment 0.37 — dragged down *entirely* by its shared-core deps, not by feature entanglement; its only feature deps are `modules/search` ×5 and `modules/knowledge` ×1).

The same math holds, more strongly, for **memory** (drags `core/database` 12, `config` 10, `core/models` 3; called by `services` 12, `api` 11, `tools` 9 — and see §4, it *was* split and it failed) and **nl2sql** (drags `core/models` 11, `core/database` 10, `core/credentials` 8; and it executes SQL *against the shared DB by design* — a service split would have it reach across the network to query the database it's meant to be co-located with).

---

## 4. The natural experiment — mem0 was split, and it is the cautionary tale

Automatos already ran the exact experiment T2 asks about: it extracted durable memory (L3) into a **separate service** — the `automatos-mem0` fork, an OpenMemory/FastAPI server on Railway, reached over HTTP via `MEM0_API_URL` with its own `MEM0_DATABASE_URL`. This is the strongest available evidence, and it points hard **against** splitting:

1. **The split enabled a silent, weeks-long outage.** The memory dossier (C, `data/mem0.md`) found `MEM0_API_URL` answers **edge-404 "Application not found" on every path — the deployment no longer exists**. Because the client "skips silently" when unreachable and every caller catches-and-returns-empty, **"Auto has effectively been running without durable memory, and nothing on any dashboard said so."** That is the split-induced failure mode in its purest form: a boundary that turns "a bug" into "invisible silent degradation."
2. **The split created deploy/version debt that a monolith cannot have.** The auth + metadata fixes (PRD-156/159) exist only on an un-merged fork branch (`16b27eb2`), never an ancestor of the fork's `origin/main`; the human runbook step to merge+redeploy+re-pin was never executed. A separate repo/service **multiplied the release surface** and the pieces drifted out of sync — the canonical distributed-systems tax, paid by a solo maintainer for whom it is pure overhead.
3. **The split bought nothing, because the hop carries no value.** Every platform write passes `infer=False`, so the fork is "a thin HTTP vector store" — "running a *forked server* to get worse-than-library behavior is the single most questionable architectural position in this module." The dossier's **E.1 verdict is REPLACE-by-un-splitting**: move durable memory **in-process** onto the already-running Qdrant, "killing the dead Railway service, the fork-merge debt, the HTTP hop and its 3–5 s timeouts, and the circuit-breaker apparatus in one move… reuse *your own* already-running store rather than resurrect a dead fork of someone else's server."

**This is T2 answered empirically.** The one in-house-ish capability that got pulled out of the monolith produced silent outages, version drift, and latency, delivered no offsetting benefit at this scale, and the recommendation is to pull it back in. Every prospective split (RAG, nl2sql) shares the same structural properties that made mem0-as-a-service a net loss.

---

## 5. Honest steelman — where the split case has any merit, and why it still loses

An honest thesis has to argue the other side. The real arguments for splitting, and the rebuttals grounded in this codebase:

- **"244k LOC is a lot for one repo / one deploy."** True, it's large. But size alone argues for **in-repo modularity discipline** (which the 3.0% lateral-coupling number shows already exists), not for distribution. A 244k-line monolith with clean module seams is a maintainability question; a 244k-line system fragmented into 6 services with a shared DB is a *distributed-systems* question — strictly harder for a solo founder to debug, deploy, and reason about. Bigger codebase → *more* reason to avoid distributed failure modes, not less.
- **"Independent deploy cadence / blast-radius isolation."** The genuine benefits of splitting — and they are real for the right org. But they are **organizational** benefits (Conway's Law: services mirror teams). With one human author, there is no team topology to mirror; "independent deploy" means the same person deploying more moving parts, and "blast-radius isolation" is outweighed by the *new* blast radius of silent cross-service failures (§4). Reassess **if and when there are ≥2 teams owning disjoint capabilities** — that is the real trigger, and it is a hiring event, not an architecture event.
- **"Independent scaling of a hot capability."** Legitimate *in principle* — if one capability were a throughput bottleneck, extracting it to scale horizontally would pay off. But no capability is CPU/QPS-bound at pilot scale (RAG dossier: S3 Vectors cheap at 19k chunks; auth dossier: 22 users; the DB census shows tiny row counts). The one workload that genuinely needs a separate runtime — **untrusted code execution** — is *already* isolated in `workspace-worker`. Scaling-driven extraction is a "when a specific module is measurably the bottleneck under real load" decision; T3's eval/telemetry harness is what should trigger it, and today it triggers nothing.
- **"Open-core / self-host wants smaller deployables."** The deployability dossier actually shows the opposite is the current pain: the fresh-clone **monolith** `docker compose up` doesn't even boot yet (exec-bit + unstamped-schema blockers). Adding *more* services to stand up would make the self-host story worse, not better. The comparators it cites (Dify, n8n, Cal.com) all ship a **single compose bundle** and win on *one-command* boot — i.e. the monolith-in-a-compose-file is the self-host-friendly shape, and "one writer of schema truth" (n8n) is the lesson, not "more services."

None of these clear the bar at pilot scale for a solo team. All of them are **future triggers to re-run this thesis** (see §7), not present-tense reasons to split.

---

## 6. What to do instead — harden the boundaries *in-repo* (the monolith's real to-do list)

Rejecting a split is not "do nothing." The coupling data points to concrete, cheap improvements that capture the *good* part of "modularity" without paying the distributed-systems tax:

1. **Protect the shared kernel as the actual contract.** 80.5% of cross-module coupling flows into `core/models` / `core/database` / `config` / `core/llm` / `core/auth`. These are the platform's real internal API. Keep them small, stable, and typed; changes here ripple to 20–36 modules. This is where "modular" is won or lost — not at a process boundary.
2. **Enforce the module boundaries you already have with an import-linter, not a network.** The modules are 97%-decoupled laterally; lock that in with an `import-linter`/`grimp` contract in CI (e.g. "no feature module may import another feature module except through `modules/tools` or `api`"). This preserves every architectural benefit a split would claim — enforced boundaries, controlled blast radius — at zero runtime cost and with loud, compile-time failure instead of silent network failure.
3. **Un-split mem0** (memory dossier E.1): retire the HTTP fork, move durable memory in-process onto the existing Qdrant. This is a *reduction* in topology that directly fixes a live outage.
4. **Make the shared boundaries fail loud, since they're in-process (an advantage — use it).** The recurring bug class (mem0 silent-skip, RAG silent-empty) is *worse* across a network but exists in-process too. Because it's all one process, these can be surfaced with plain exceptions + a health tile rather than distributed tracing. Monolith is the *easier* place to fix observability — lean into that (feeds the observability-slos dossier's "is-it-working" strip).
5. **`codegraph` is the one clean cut — bank the option, don't spend it.** It's the single genuinely self-contained backend module: 5 outbound cross-imports (`config` 2, `core/llm` 2, `core/security` 1), **zero feature deps**, inbound only via `api`/`tools`/`agents`/`workflows`. *If* a future forces one extraction (e.g. it needs a heavyweight language-server runtime, or a separate scaling profile for large-repo indexing), it is extractable at low cost. Nothing forces that today. Keep it as a well-bounded in-repo module and note it as the pre-cleared seam.

---

## 7. Re-evaluation triggers (when to re-run T2)

Stay-monolith is the right answer *for now*; name the conditions that would change it, so this isn't a forever-decision made blind:

- **A second team.** ≥2 people/teams owning disjoint capabilities and blocked on each other's deploy cadence. (Organizational trigger — the real one.)
- **A measured throughput bottleneck.** T3's telemetry shows one capability is CPU/QPS-bound under real load and needs an independent scaling profile (most plausibly `codegraph` on large-repo indexing, or the widget/storefront leg if it goes high-QPS). Extract *that one*, at the pre-cleared seam, and only that one.
- **A hard runtime/isolation need.** A capability needs a runtime the monolith can't host (a GPU model server, a language-specific toolchain, another untrusted-execution surface) — extract for *isolation*, like `workspace-worker`, sharing the DB if needed.
- **A compliance boundary.** A tenant/data-residency requirement forces physical separation of a data domain — the only reason to partition the shared Postgres.

Absent one of these, the monolith wins on every axis that matters to a solo founder shipping to pilot users: one mental model, one deploy, one DB, loud in-process failures, and module boundaries enforced by a linter instead of by a network.

---

## 8. Evidence index

- **Coupling census** (this analysis): AST parse of 754 backend `.py` files → 3,059 internal imports; 75.6% cross-module but 80.5% of that into shared cores, only **3.0% true lateral peer coupling**. Per-module fan-in/out and extract-cost tables in §1–§3.
- **Graph** `graphify-out/graph.json`: used for shape; its 74.8% "cross-module" figure shown to be an INFERRED-edge artifact (§1 Measurement note). EXTRACTED cross-module backend edges = 11.
- **Runtime:** `orchestrator/main.py` (single ASGI app), `docker-compose.yml` (9 services; the only compute-splits are frontend/gotenberg/workspace-worker), `services/board_dispatcher.py:37-142` (Postgres `SKIP LOCKED`+`LISTEN/NOTIFY` spine).
- **Data:** [census](evidence/data/census.md) — one Postgres, 152 tables, 99 shared ORM classes; pilot-scale row counts.
- **The mem0 natural experiment:** `reports/dossiers/memory.md` §C/§E.1 + `evidence/data/mem0.md` — extracted-as-service → dead deployment → silent memory outage → verdict "un-split, go in-process."
- **RAG boundary:** `reports/dossiers/rag-retrieval.md` §A/§C-2/§E — welded to workspace/team/policy fabric; silent-empty-on-network-failure is its worst failure mode.
- **Deployability:** `reports/dossiers/deployability-open-core.md` — single-compose self-host is the target shape; more services would worsen the boot story.
- **Solo authorship:** `git log --format=%an` — Gerard 2,417 commits; remainder are his own agents/bots. No team topology to mirror.
