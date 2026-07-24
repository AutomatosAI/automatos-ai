# Observability & SLOs — module dossier

**Module key:** `observability-slos` · **Tier:** standard · **Status:** live
**Pinned tree:** `origin/main @ 77bc9c6d5` (2026-07-03). All `file:line` refer to that tree.
**Baseline:** `reports/PLATFORM_OS_REVIEW_2026-07-01.md` (§5; F007/F036/F037/F038/F083/F090). W10 = PRD-180.
**North Star:** does this make Auto and the agents more autonomously capable, and their client output higher quality? Observability's answer is indirect but real: it is the layer that lets Auto (and its operator) *know* whether the autonomous loop is actually working — the July review's central lesson was that the platform silently ran broken for weeks and nothing said so.

> Scope note: this dossier deliberately contains **no security / adversarial-hardening lens** (section F is omitted by instruction — it runs as a separate Opus pass). Auth findings are named only where they change whether observability *functions*, not as a threat model.

---

## A. What it is

The observability plane is Automatos's answer to "is the platform actually working?" — assembled almost entirely from telemetry the platform already writes, not a parallel metrics stack. It has three layers that were built at different times and do not yet form one coherent product:

1. **W10 (PRD-180) SLO + honest-health layer** — three concrete SLIs computed on demand from existing tables (`services/slo_metrics.py`), an 8-primitive health tile fed by best-effort heartbeat findings, a per-subsystem `error_events` sink, and the running/error tool chips in chat. This is the layer the July review commissioned to kill "fabricated green."
2. **The Command Center surface** — `/command-center` (frontend), whose `StatsStrip` + `IsItWorkingStrip` tiles are the operator's at-a-glance "is it working?" view, plus the activity feed (`activity_service.py`) that lists what ran.
3. **A large analytics/statistics router sprawl** — seven-plus routers (`analytics.py`, `analytics_api.py`, `analytics_real.py`, `analytics_charts.py`, `statistics.py`, `kpi_api.py`, `execution_history.py`, `database_analytics.py`, `llm_analytics.py`, `composio_analytics.py`) serving overlapping dashboard endpoints, which this unit must adjudicate (F083). Behind it, the `core/monitoring/` alerts+logs+metrics layer (Loki/Prometheus-adjacent) and the separate `automatos-monitoring` repo (Docker-Compose observability stack, out of this tree).

The honest one-line summary: **the measurement primitives that W10 added are genuinely good and honest; almost none of them reach the operator's screen, and the analytics layer around them is a decade of accumulated duplication.**

---

## B. What it does — real implementation + data path

### B.1 The three SLIs (`services/slo_metrics.py`)

`compute_slos(db, workspace_id, window_seconds)` (`slo_metrics.py:178-200`) returns a stable envelope `{generated_at, window_seconds, slos:[…]}` of three SLIs, each an immutable dict `{sli, description, value, unit, target, target_comparator, window_seconds, sample_size, meets_target}`:

- **`tool_call_success_rate`** (`:61-95`) — fraction of `ToolExecutionLog` rows with `status=='success'` over the window, **filtered to `telemetry_source=='production'`** (`:77`). Target ≥ 99.0%. `value=None` when nothing ran (honest empty, never fabricated 0).
- **`board_dispatch_latency_p95_seconds`** (`:98-140`) — p95 of `BoardTask.started_at − created_at` over tasks that actually started in the window, via `StatisticalAnalysis.calculate_percentile` (`:128`). Target ≤ 5.0s.
- **`board_event_freshness_seconds`** (`:143-175`) — age of `max(BoardTask.updated_at)` for the workspace; a proxy for LISTEN/NOTIFY SSE freshness. Target ≤ 30.0s.

Targets are module constants (`:45-47`); the default window is config-driven — `SLO_DEFAULT_WINDOW_SECONDS = 86400` (`config.py:583`). Time handling is careful: `BoardTask` is tz-aware and uses `_aware_utc_now()` (`:110,205`); `ToolExecutionLog.executed_at` is a naive `Column(DateTime)` (`core/models/composio_cache.py:246`) and the SLI-1 filter correctly uses naive `datetime.utcnow()` (`:73`) — no tz-mismatch crash. `_as_aware` (`:209-211`) coerces the naive `updated_at` for SLI-3 arithmetic.

**Exposure:** `GET /api/analytics/slos?window=24h` (`api/analytics_real.py:122-143`) → `compute_slos(...)`. `_parse_window` (`:146-160`) tolerates malformed windows (falls back to 24h, never 400s).

### B.2 Primitive health tile (`emit_primitive_finding` → `/primitive-health`)

Eight canonical primitives — `chat, memory, rag, nl2sql, graph, missions, playbooks, channels` (`heartbeat_service.py:35-44`, single source of truth `PRIMITIVE_NAMES`). Each has its own **writer module** that calls the best-effort `emit_primitive_finding(ws, primitive, status, detail)` (`heartbeat_service.py:53-125`), which inserts one row into `heartbeat_results` with a `primitive_check`-shaped JSONB `findings` payload (no schema change). Writers found and confirmed wired:

| primitive | writer | on the real path? |
|---|---|---|
| chat | `consumers/chatbot/primitive_heartbeat.py:_emit_chat_primitive` | **yes** — called 4× in `consumers/chatbot/service.py:2173-2287` (per-turn success/failure) |
| playbooks | `services/playbook_engine_heartbeat.py:_emit_playbooks_primitive` | yes — terminal playbook transition |
| memory | `services/heartbeat_service.py:254` (mem0 probe) | yes — heartbeat cycle |
| rag | `modules/rag/ingestion/manager.py:1398-1400` | yes — ingest path |
| nl2sql | `modules/nl2sql/primitive_heartbeat.py` | yes — validation path |
| graph | `modules/knowledge/primitive_heartbeat.py` | yes |
| missions | `modules/coordination/primitive_heartbeat.py` | yes |
| channels | `channels/primitive_heartbeat.py` | yes |

`GET /api/analytics/primitive-health` (`analytics_real.py:285-358`) reads the **latest** `primitive_check` finding per primitive (`ORDER BY created_at DESC`, latest-wins dedup at `:335-340`), always returns all 8, and renders `{status:"unknown", last_checked:null}` for a primitive with no finding (`:345`) — honest gap over fake green. This is a **direct, well-built answer to July F038** ("fabricated metrics by default").

### B.3 Error sink (`record_error` → `error_events` → `/errors/by-subsystem`)

`record_error(subsystem, operation, error, …)` (`core/utils/exception_telemetry.py:47-109`) logs a structured record on the `automatos.errors` logger **and** best-effort-persists an `ErrorEvent` row (`:112-160`, `_persist_error_event`). ~15 real call sites: boot reaper (`core/boot/reaper.py` ×5), startup tasks, `smart_tool_router.py:249`, `verification.py:581,749`, `planner.py:478`, `board_tasks.py:1090`, `wizard.py:787`, `signal_recorder.py:465`, background tasks. `GET /api/analytics/errors/by-subsystem?window=24h` (`analytics_real.py:168-222`) aggregates by subsystem, workspace-scoped, with proper index paths.

### B.4 Command Center surface

`/command-center` is real and mounted (sidebar entry `frontend/components/layout/sidebar.tsx:46`; layout maps the route `main-layout.tsx:67`). `command-center-shell.tsx:148` renders `<IsItWorkingStrip />` under `<StatsStrip />`. `IsItWorkingStrip` (`frontend/components/command-center/is-it-working-strip.tsx`) shows five cells — ACTIVATION, MISSIONS, ERRORS, WIDGET, PRIMITIVES — via `use-analytics-api` hooks (`useActivationMetrics`, `useMissionSuccessRate`, `useErrorsBySubsystem`, `useWidgetEngagement`, `usePrimitiveHealth`). It is honest by construction: a cell with no data renders `—` and a muted tone, and PRIMITIVES shows "awaiting checks" rather than a fake green (`:88-98`).

### B.5 The activity feed (`services/activity_service.py`, 1046 lines)

`ActivityService.get_feed` (`:72`) and `get_stats` (`:124`) union five sources — chats, routines (heartbeat_results), recipes (recipe_executions), board_tasks, agents — into one workspace-scoped feed with working-now / completed / needs-attention counts. This is the "what ran" ledger behind the Command Center board/activity tabs.

---

## C. Honest quality — inspected against real data

**Maturity score: 2 / 5** (Emerging). The measurement *primitives* are a 4; the *product* around them — reach, consolidation, and whether the numbers can even fire on real data — is a 1–2. Averaged and weighted toward "does the operator actually see the truth," it lands at 2.

### C.1 The flagship SLI is structurally dead on real data — CONFIRMED

`tool_call_success_rate` only counts `telemetry_source=='production'` rows (`slo_metrics.py:77`). The real DB has **2,341 `tool_execution_logs`, 100% `telemetry_source='synthetic'`, frozen 2026-05-05** — not one organically-recorded execution (`evidence/data/tool-telemetry.md`). Two months of daily playbooks, heartbeats, and chat through 06-27 wrote **zero** production tool rows. Therefore SLI-1 returns `value=None, sample_size=0` in production **forever**, until some code path actually writes `telemetry_source='production'` to `ToolExecutionLog`. The SLO's own honesty rule ("None, never a fabricated number") is working exactly as designed — and the result is that the platform's headline "do Auto's actions land?" objective is **permanently unmeasurable with the current writers**. The problem isn't `slo_metrics.py`; it's that the tool runtime doesn't feed it (a `tool-runtime` / `evals-learning` cross-dependency, but it lands here because this is where the number is supposed to appear).

### C.2 The SLOs reach nothing — CONFIRMED (the seed symptom)

Grep for any SLO reference across the entire `frontend/` tree returns **zero hits** — no `getSlos`, no `useSlo`, no `/slos`, no rendering of `success_rate`/`meets_target`. `GET /api/analytics/slos` is wired end-to-end on the backend and **called by no client**. The three tracked objectives the July review commissioned exist only as an endpoint a human could curl. `IsItWorkingStrip` — the literal "is it working?" tile — does not consume them; it uses a *different* five metrics. So the platform measures three SLOs and shows none of them.

### C.3 The whole Command Center strip is super-admin-gated — CONFIRMED, and it's the sharper version of F036

`analytics_real.py` is **router-wide super-admin-locked**: `dependencies=[Depends(require_super_admin)]` (`:38-42`). `require_super_admin` (`core/auth/super_admin.py:17-29`) is fail-closed — it 403s unless `system_role=='super_admin'` (API-key, service, and ordinary workspace users all refuse). That router serves `/primitive-health`, `/errors/by-subsystem`, `/activation`, `/dashboard/success-rate`, `/widget-engagement` — i.e. **four of the five `IsItWorkingStrip` cells and the entire PRIMITIVES tile**. Consequence: for any non-super-admin user, the Command Center's "is it working?" strip **403s on every tile and renders all `—`**. The tile is honest about missing data, so it won't lie — but it will show an operator nothing. This is F007's fix (obs-tier lock, correctly landed) colliding with a workspace-facing surface that was never re-scoped: the obs tier is locked to super_admin, yet the Command Center is a per-workspace screen. The July review closed the *unauthenticated-read* hole (good) but left the *tile can't read its own data* hole. This is the single biggest reason to score the module low against the North Star: **the operator's cockpit is, for most principals, blank.**

### C.4 Analytics router sprawl is real and duplicative — CONFIRMED (F083 PARTIAL)

Ten-plus analytics/statistics routers are mounted (`main.py:962-1014`), several serving the same conceptual dashboard. Two routers mount the **identical prefix `/api/analytics`**: `analytics_api.py:22` and `analytics_real.py:39`, included consecutively (`main.py:983-984`). FastAPI serves both, first-match-wins by include order — so overlapping paths (`/dashboard/*`, `/system/health`) resolve to whichever was included first and the other's version is shadowed and untestable-in-place. `analytics.py:27` still mounts the anomalous bare `/analytics` prefix (no `/api`) exactly as F083 filed (residual map: F083 PARTIAL — only the `workspace_exec` leg was fixed by W14; the bare-mount + `anthropic_client.py` legs remain). `analytics_api.py` also exposes write endpoints (`/track/agent-execution`, `/track/context-optimization`, `/track/learning-progress` at `:133-193`) whose target tables are among the empty learning-plane tables. This is exactly the "200 tables / 103 routers happened because we kept both" debt the project CLAUDE.md §5 warns against.

### C.5 What is genuinely good (honest positives)

- **Primitive-health is well-engineered and honest.** 8 primitives, one writer each, latest-wins, `unknown` over fake-green, best-effort emit that never breaks the caller, defence-in-depth name filter. This is a real answer to F038 and it's on the real chat/playbook/ingest paths. Score this piece a 4.
- **The `record_error` → `error_events` sink is real and current** (12 rows in prod, W10 table live) with ~15 honest call sites and proper indices. Errors are no longer invisible.
- **F037 (Composio chips) and F038 (Studio sidebar lies) are genuinely fixed** with guard tests (residual map confirms both FIXED). The running/error indicators now show real external action names.
- **F090 (board "Streaming live") is real LISTEN/NOTIFY** now, with the Command Center subscribed and the fake poll deleted — SLI-3 (`board_event_freshness`) is a coherent proxy for that stream's health.
- **Honest-empty is the house style.** Every surface here returns `None`/`—`/`unknown` rather than a fabricated number. Given the July finding was *fabrication*, this cultural shift is the most important thing W10 delivered, and it's real.

### C.6 Residual lies of the same species — CONFIRMED

The fabricated workspace pill `workspaceMeta='pilot · 11 op'` still renders (`studio-sidebar.tsx:44,96`; `main-layout.tsx:161` passes no props) — F038's leftover (residual map §4.C.17). And the `/chat/[id]` zombie route 404s every SaaS request yet three activity surfaces `router.push` into it (F036 NOT DONE) — an observability surface (the activity feed) links to a dead page. Small, but exactly the "surface tells the user something false" debt this module is supposed to eliminate.

### C.7 Reliability smell in the emit path — PLAUSIBLE

Both `emit_primitive_finding` (`heartbeat_service.py:89`) and `_persist_error_event` (`exception_telemetry.py:135`) open a **fresh synchronous `SessionLocal()`** and `commit()` inline on hot paths (per chat turn, per tool-router error, per planner failure). They are best-effort and swallow exceptions, so they can't break the caller — but each is a blocking DB round-trip added to the request/execution path, and under a DB slowdown they add latency to exactly the paths you least want slowed. Not a correctness bug; a load-behavior note (see F).

---

## D. Competitive teardown

The reference bar for "LLM/agent observability" is now a well-defined category. Automatos's in-house plane is far below it on breadth, and — more importantly — below it on *reach* (nobody sees the numbers).

### D.1 Langfuse (OSS, MIT) — the closest adopt candidate

- **What it does better:** full request-lifecycle **tracing** (every LLM call, retrieval, tool exec, nested spans with timing/inputs/outputs/metadata), cost & latency & quality **dashboards with automated alerts**, LLM-as-a-judge + heuristic **evals on production traces**, prompt management, datasets. Ingests over **OpenTelemetry** and 100+ integrations. ([overview](https://langfuse.com/docs/observability/overview), [repo](https://github.com/langfuse/langfuse))
- **Licensing/cost:** *all* core features (tracing, dashboards, evals, alerts, prompt mgmt) are **MIT, self-hostable with no usage metering or phone-home**; in June 2025 they open-sourced the last cloud-only features (managed LLM-judge, annotation queues, prompt experiments) under MIT ([open-source strategy](https://langfuse.com/docs/open-source), [self-host pricing](https://langfuse.com/pricing-self-host)). The real cost is infra: v3 requires a **ClickHouse** cluster for trace storage (~$200-800/mo cloud depending on volume) + Postgres ([Langfuse pricing analysis](https://coverge.ai/blog/langfuse-pricing)).
- **Where Automatos stands:** Automatos has **no request tracing at all** — no span tree, no per-turn LLM/tool/retrieval waterfall. It has three point SLIs and a primitive tile. Langfuse would give Auto's operators (and Auto itself, if fed back) a real "why was this turn slow/wrong" view Automatos entirely lacks.

### D.2 Arize Phoenix (OSS, self-hostable) — the eval-heavy alternative

- **What it does better:** native **OpenTelemetry (OTLP)** ingestion + auto-instrumentation for LangChain/LlamaIndex/OpenAI/Anthropic/Bedrock; **built-in evaluators** (faithfulness, relevance, hallucination, toxicity); RAG-specific metrics; embeddings/UMAP analysis; dataset + replay to prove a change helped before rollout. Runs locally, in a notebook, or containerized. ([Phoenix](https://arize.com/phoenix/), [repo](https://github.com/Arize-ai/phoenix/), [docs](https://arize.com/docs/phoenix))
- **Where Automatos stands:** Phoenix's RAG/embedding observability and hallucination evals are directly relevant to Automatos's RAG + memory modules, which today have **no faithfulness/hallucination measurement whatsoever** (the `evals-learning` unit confirms `modules/evaluation` is a TODO stub). Phoenix pairs naturally with the T3 eval-harness thesis.

### D.3 OpenTelemetry GenAI semantic conventions — the standard Automatos ignores

- **What it is:** a converging standard for `gen_ai.*` spans/metrics/events (model, tokens, latency, tool calls, MCP) — the lingua franca every vendor above now ingests. Status: **Development/beta, not yet stable**, but the transition baseline (v1.36) is set and Datadog/Langfuse/Phoenix all consume it. ([OTel GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/), [GenAI observability blog](https://opentelemetry.io/blog/2026/genai-observability/))
- **Where Automatos stands:** Automatos emits **nothing** in this shape. Every metric it has is a bespoke Postgres aggregate. Adopting OTel GenAI spans on the LLM/tool path is the single highest-leverage integration move: it makes the platform legible to *any* of these tools for free, instead of hand-rolling each dashboard.

### D.4 Datadog LLM Observability — the enterprise ceiling (not an adopt target)

- **What it does better:** hallucination/prompt-injection/PII evals out of the box, **hierarchical topic clustering of production traffic** (coverage-gap discovery), APM correlation, custom LLM-judge. Prices per **LLM span only** (tool/retrieval/agent spans free): free 40K spans/mo, Pro $160/mo for 100K ([Datadog LLM Obs](https://www.datadoghq.com/product/ai/llm-observability/2/), [metrics docs](https://docs.datadoghq.com/llm_observability/monitoring/metrics/)).
- **Where Automatos stands:** this is the "what enterprise-grade looks like" reference, not something to adopt (SaaS, per-span billing, and Automatos is itself a product that would want to *own* this layer for its clients). Cited to calibrate the gap: Automatos has ~5% of this surface.

**Net:** on breadth Automatos is ~1/5 of Langfuse/Phoenix. On *honesty* it is arguably ahead of a naive dashboard (it refuses to fabricate), but honesty of a number nobody can see is worth little.

---

## E. Build / extend / adopt — the verdict

**Verdict: ADOPT (Langfuse, self-hosted, MIT) for tracing + dashboards + alerts; EXTEND the W10 primitive/SLO layer for the platform-specific tiles; DELETE the analytics-router sprawl.** Do **not** keep building a bespoke metrics stack — the category has a free, MIT, self-hostable winner and Automatos is reinventing a fraction of it in ten overlapping routers.

Concretely, three moves in priority order:

1. **ADOPT Langfuse (self-hosted).** It replaces the entire "trace + cost + latency + quality dashboard + alerting" ambition that the analytics sprawl gestures at but never delivers. Integration shape: instrument the LLM manager (`core/llm/manager.py`) and the unified tool executor (`tool-runtime`) to emit **OpenTelemetry GenAI spans**; point the OTel exporter at a self-hosted Langfuse. This is a *few* instrumentation points, not a rewrite, because every LLM/tool call already funnels through one manager and one executor. Rough cost: engineering ~1-2 weeks to instrument + a Langfuse deployment (Postgres you already run + a ClickHouse instance, ~$200-400/mo at pilot volume). This is the correct "reuse over build" call per ground-rule §2 and CLAUDE.md §2. Langfuse's evals also seed the T3 harness.
   - *If ClickHouse ops is unwelcome at pilot scale,* **Phoenix** is the lighter self-host (no ClickHouse cluster) and stronger on RAG/hallucination evals — adopt Phoenix instead and revisit Langfuse when trace volume justifies it. Either way: adopt an OSS OTel-native tool; don't hand-roll traces.

2. **EXTEND, don't replace, the W10 primitive-health + SLO layer.** These are genuinely good and platform-specific (the 8 Automatos primitives, board-dispatch latency, board-event freshness are *your* domain concepts — no external tool knows them). Keep `slo_metrics.py` and `emit_primitive_finding`; they are the right in-house 20%. But (a) **fix the reach problem** — wire `/slos` into the Command Center and move the obs tiles off the super-admin-only router, and (b) **make SLI-1 fire** by writing real `telemetry_source='production'` rows from the tool runtime.

3. **DELETE the sprawl** (a Refactor/Consolidation PRD per CLAUDE.md §3/§5). Pick `analytics_real.py` as canonical, migrate the handful of live callers off `analytics_api.py`/`analytics.py`/`kpi_api.py`/`statistics.py`/`execution_history.py`, and delete the losers plus the dead `api/anthropic_client.py` and the bare `/analytics` mount. Resolve the two-routers-on-`/api/analytics` collision. This is pure debt reduction with a clear canonical path.

**Why not "keep building"?** Nothing platform-specific is lost by adopting: the in-house layer that *is* differentiated (primitives, board SLIs) is kept and extended. What's replaced is the generic tracing/dashboard/alert plane where a free MIT tool already beats ten hand-rolled routers on every axis (tracing, evals, alerts, OTel-native, battle-tested at billions of observations/mo).

---

## G. Quality metric — how we measure this module over time

The module's own quality is measured on **reach and truthfulness**, not endpoint count:

1. **SLO-coverage %** = (SLIs surfaced in a UI a workspace user can actually see) / (SLIs computed). **Today: 0/3 = 0%** — three computed, none rendered, and the router that would render adjacent tiles is super-admin-locked. Target: 3/3 visible to the workspace operator.
2. **SLI-fireability** = fraction of defined SLIs that return a non-`None` value on production data. **Today: at most 2/3** — `tool_call_success_rate` is structurally `None` (no production telemetry rows ever); board latency/freshness fire only for the one workspace that runs board tasks. Target: 3/3 non-null.
3. **Primitive-health liveness** = of 8 primitives, how many show a non-`unknown` status in prod right now. Measurable directly from `/primitive-health`; today unknown-from-this-tree (needs a live super-admin call), but all 8 writers exist and 3+ are on daily-active paths (chat, playbooks, heartbeat). Target: 8/8 green-or-honest.
4. **Fabrication count** = number of surfaces rendering a hardcoded/fake metric. **Today: 1 confirmed** (`workspaceMeta='pilot · 11 op'`) plus the `/chat/[id]` dead-link. Target: 0.
5. **Analytics-router count** = mounted routers under an analytics/statistics prefix. **Today: 10+.** Target: 1 canonical (+ the domain-specific llm/composio ones if genuinely distinct).

Metric 1 (SLO-coverage) is the headline and feeds T3.

---

## H. Cost note (informational)

Cheap. Every SLI/tile is an on-demand Postgres aggregate over indexed columns, run only when a dashboard is opened — **zero LLM tokens, no background compute**. `slo_metrics` is three indexed queries; `/primitive-health` is one workspace-narrowed JSONB read; `errors/by-subsystem` is one grouped count. The only ongoing write cost is one extra `heartbeat_results` row per primitive emit and one `error_events` row per caught error — negligible against 148K existing heartbeat rows. The synchronous-`SessionLocal`-on-hot-path pattern (C.7) is a *latency* cost, not a token cost. Adopting Langfuse adds infra cost (ClickHouse ~$200-400/mo pilot) but no per-request token cost — traces are structured data, not model calls.

---

## I. UX / surface — concrete changes

The measurement exists; the surface is where this module fails the North Star. Concrete IA/UX changes:

1. **Add an SLO tile/row to the Command Center.** `IsItWorkingStrip` is the natural home — add a sixth concern that renders the three SLIs with their target + pass/fail (green/amber on `meets_target`). Right now the "is it working?" tile is the one place SLOs belong and they're absent.
2. **Move the operator obs tiles off the super-admin-only router.** Split: keep *cross-tenant* / staff analytics super-admin-locked, but the *own-workspace* "is it working?" tiles (`primitive-health`, `errors/by-subsystem`, `activation`, `slos`) must be reachable by a normal workspace admin — otherwise the Command Center is blank for the people who run the workspace. This is the highest-impact single change in the whole dossier.
3. **One SLO/health page, not ten dashboards.** Consolidate the analytics sprawl behind a single "Health" Command Center tab: SLOs + primitives + recent errors + activity, in that priority order. Kill the duplicate dashboards.
4. **Fix the two residual lies:** delete the `workspaceMeta='pilot · 11 op'` default and either revive or unlink `/chat/[id]` from the activity feed (a health surface must not link to a 404).
5. **When Langfuse/Phoenix lands,** deep-link from a trace-heavy surface (e.g. a slow/failed activity item) into the Langfuse trace for that request — turn "something failed" into "here's the span waterfall."

---

## J. Upgrade path — prioritised (impact × effort), judged by North-Star impact

| # | Change | Impact | Effort | North-Star rationale |
|---|---|---|---|---|
| 1 | **Re-scope the obs tiles: own-workspace health endpoints reachable by workspace admins** (split the super-admin router; keep cross-tenant staff-only) | **Very high** | Low | Today the operator's cockpit 403s to blank for non-super-admins. Fixing this is what turns "we measure it" into "the operator sees whether Auto is working." |
| 2 | **Write real `telemetry_source='production'` tool rows from the unified executor**, so SLI-1 (and learned routing, uplift evals) finally have a real diet | **Very high** | Med | The flagship SLI and the entire learning plane are starved (0 production rows in 2 months). This one write path unblocks observability *and* `evals-learning` *and* `tool-selection`. |
| 3 | **Wire `/slos` into the Command Center** (SLO tile in `IsItWorkingStrip`) | High | Low | Three tracked objectives that no one can see are worth nothing. Trivial once #1 lands. |
| 4 | **ADOPT Langfuse/Phoenix via OTel GenAI spans** on the LLM manager + tool executor | High | Med | Gives Auto's operators the request-trace/why-was-this-wrong view the platform completely lacks, and seeds the T3 eval harness — reuse over building a bespoke tracer. |
| 5 | **Consolidate the analytics-router sprawl** to one canonical router; delete losers + bare `/analytics` mount + dead `anthropic_client.py`; resolve the `/api/analytics` prefix collision | Med | Med | Pure debt reduction; removes a whole class of "which endpoint is real?" confusion and shadowed routes. |
| 6 | **Kill the residual fabrications** (`workspaceMeta='pilot · 11 op'`, `/chat/[id]` dead link from activity) | Med | Low | Finishes the anti-fabrication job W10 started; a health surface must never mislead. |
| 7 | **Move the two hot-path emits off synchronous `SessionLocal`** (batch/async the primitive + error sink writes) | Low | Low | Removes a blocking DB round-trip from the chat/tool paths under DB slowdown; hygiene, not correctness. |

**Cross-thesis note (T3):** this module is where T3's "quality is a tracked number" becomes visible. The W10 SLO layer is the seed of the right in-house metric surface; the missing pieces are (a) real telemetry to compute from (#2), (b) a place it's shown (#1, #3), and (c) an adopted trace/eval backend (#4). All three trace to findings above.

---

### Evidence index

- Real data: `evidence/data/tool-telemetry.md` (100% synthetic, frozen 05-05), `evidence/data/census.md` (error_events=12, 0-row learning tables), `evidence/data/notifications.md` (playbook_complete stopped 06-16 — a real outage observability should have screamed about), `evidence/real-data-inventory.md`, `evidence/phase0-residual-map.md` (F007 FIXED, F036 NOT DONE, F037/F038 FIXED, F083 PARTIAL, F090 FIXED).
- Code: `services/slo_metrics.py`, `api/analytics_real.py:122/168/285`, `services/heartbeat_service.py:35/53`, `consumers/chatbot/primitive_heartbeat.py`, `core/utils/exception_telemetry.py:47/112`, `services/activity_service.py`, `frontend/components/command-center/is-it-working-strip.tsx`, `frontend/components/command-center/command-center-shell.tsx:148`, `core/auth/super_admin.py:17`, `main.py:962-1014`, `config.py:583`.
- External (cited inline): Langfuse (docs/repo/open-source/pricing), Arize Phoenix (site/repo/docs), OpenTelemetry GenAI semantic conventions, Datadog LLM Observability.
