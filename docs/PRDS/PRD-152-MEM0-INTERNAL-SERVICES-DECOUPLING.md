# PRD-152: Memory & Internal-Services Decoupling — Self-Hosted mem0 Default + the `railway.internal` Sweep

**Status:** Draft
**Author:** Gerard Kavanagh (with Auto)
**Date:** 2026-06-09
**Type:** Extension + Consolidation (compose-host an existing OSS service; make cloud-internal defaults local-safe)
**Related:** PRD-150 (Auth — open-core slice 1), PRD-151 (Storage — slice 2; this is slice 3), PRD-153 (One-command local run — hosts the services this PRD wires), PRD-141 US-004/US-006 (mem0 circuit breaker + health probe — the resilience this PRD leans on)

---

## 1. Introduction / Overview

Long-term memory (L3) is served by **mem0**, running as a separate Railway service (`automatos-mem0-server`, a fork of mem0 OSS). The platform's client (`modules/memory/integrations/mem0_client.py`) speaks the **standard OpenMemory REST API** (`/api/v1/memories/` add/search/get/delete) — *not* fork-custom endpoints. That means any mem0/OpenMemory server satisfies it, including one running in the local compose network.

Unlike auth (PRD-150) and storage (PRD-151), **no decoupling refactor is needed here**. The work split is:

1. **Host mem0 locally by default** — fold the existing `infrastructure/docker-compose.memory.yml` services (mem0-server + its dedicated pgvector) into the canonical compose (PRD-153), so OSS users get *working memory*, not a degraded platform. The user's directive: don't cut mem0 off at the neck — ship a running default.
2. **Sweep the `*.railway.internal` defaults out of `config.py`** — seven service URLs currently default to Railway's private network, which silently fails (or noisily retries) on any non-Railway deployment.

The platform is already proven safe without mem0 (per-workspace circuit breaker, 3s-bounded reads, fire-and-forget writes, `test_memory_degrades_when_mem0_down.py` asserts L2 survives L3 loss). That degradation contract stays as the safety net — but it is the *fallback*, not the OSS experience.

### Why now

PRD-150/151 make login and storage work locally; without this PRD the "working platform" still ships with amnesia (L3 empty) and background log-shipping errors (log-relay defaults **on**, pointed at a Railway-internal host). This is the cheapest of the four open-core slices — mostly compose + config, near-zero backend code.

---

## 2. Current-State Map (code-verified 2026-06-09)

### mem0 integration (healthy — keep as is)

| Aspect | Reality |
|---|---|
| Client | `mem0_client.py`, async httpx, standard OpenMemory API, `Token {api_key}` auth, workspace/agent scoping via `metadata` |
| Reads | `search_long_term()` — Redis cache (5min TTL) → mem0 search, 3s timeout; on failure returns `[]`, chat proceeds |
| Writes | fire-and-forget (`store_long_term*`, daily logs); loss is non-fatal |
| Resilience | per-workspace circuit breaker (3 failures → open, 60s cooldown) + 30s health probe that trips/resets all breakers (PRD-141 US-004/US-006) |
| Degradation proof | `test_memory_degrades_when_mem0_down.py` — L2 (Postgres) transcript persists when L3 raises |
| Existing compose | `infrastructure/docker-compose.memory.yml` already defines `mem0-server` + dedicated `mem0-pgvector` — built, never folded into the root compose |

### The `*.railway.internal` inventory in `config.py`

| Config (line) | Default | Local impact |
|---|---|---|
| `MEM0_API_URL` (659) | `http://automatos-mem0-server.railway.internal` | L3 dead locally; breaker churn until probe trips |
| `LOG_RELAY_URL` (451) | `http://log-relay.railway.internal:8080/push` — **`LOG_RELAY_ENABLED=true` by default** | every local boot ships logs into a black hole |
| `LOKI_URL` (438/469) | `http://loki.railway.internal:3100` | optional, fails open |
| `PROMETHEUS_URL` (439) | `http://prometheus.railway.internal:9090` | optional |
| `AGENT_OPT_WORKER_URL` (636) | `http://agent-opt-worker.railway.internal:8080` | FutureAGI optimization no-ops |
| `VOICE_SERVICE_URL` (778) | `http://voice-service.railway.internal:8300` | voice features dangle |
| internal API/frontend (367–368) | `automatos-ai*.railway.internal` | informational; not consumed by orchestrator |

The pattern to fix: **cloud-topology values living in code defaults.** Defaults must describe the *local* topology (compose service DNS) or be empty-and-feature-off; Railway values belong in Railway env vars.

---

## 3. Default replacement — recommendation

**Run the mem0 fork as a compose service (default-on), backed by its own pgvector container.** Considered:

| Option | Verdict |
|---|---|
| **mem0 server in compose (fork image)** | ✅ Zero backend changes (client API-compatible); memory genuinely works offline; `docker-compose.memory.yml` already wrote most of it |
| Upstream `mem0ai` image instead of the fork | Viable fallback if fork-image publishing is a blocker (client speaks the standard API) — decide in Q1 |
| Degradation-only ("OSS has no L3") | ❌ Ships a worse product and tells contributors memory is SaaS-only — exactly the perception we're fixing |
| Replace mem0 with a Postgres-native L3 | ❌ Large refactor of `unified_memory_service` contracts for zero functional gain; rejected |

Backing store: keep the **dedicated `mem0-pgvector`** container from `docker-compose.memory.yml` rather than sharing the platform Postgres — mem0 manages its own schema/extensions, and isolation keeps `alembic` ownership clean.

---

## 4. Goals

- **G1** — OSS compose includes `mem0` (+ its pgvector) by default; `MEM0_API_URL` resolves over compose DNS; L3 memory works on a clean clone.
- **G2** — Zero `*.railway.internal` defaults remain in `config.py`; every such service is (a) a compose service, (b) profile-gated, or (c) default-off with a clear flag.
- **G3** — `LOG_RELAY_ENABLED` defaults **off**; observability (loki/prometheus/log-relay/promtail) becomes an opt-in compose profile.
- **G4** — The no-mem0 degradation contract stays tested and green (breaker, `[]` reads, L2 survival) — removing the container must never break the platform.
- **G5** — Zero behavioral change for SaaS/Railway (env vars there already override every default this PRD touches).

---

## 5. User Stories

### Phase 0 — Config truthfulness

#### US-001: Local-safe defaults sweep
- [ ] `MEM0_API_URL` default → `http://mem0-server:8765` (compose DNS); Railway keeps its env override.
- [ ] `LOG_RELAY_ENABLED` default → `false`; `LOG_RELAY_URL` default → `""` (boot validates: enabled ⇒ URL required).
- [ ] `LOKI_URL` / `PROMETHEUS_URL` / `AGENT_OPT_WORKER_URL` / `VOICE_SERVICE_URL` defaults → `""`; each consumer already treats absence as feature-off — verify per consumer, add the guard where missing.
- [ ] No `os.getenv()` outside `config.py`; grep `railway.internal` under `orchestrator/` → **zero** (env files excepted).

#### US-002: Feature-off behavior is explicit, not accidental
- [ ] For each swept service: one log line at boot stating the feature is disabled and which var enables it (no retry spam, no stack traces).
- [ ] Unit tests: empty URL ⇒ consumer no-ops cleanly (voice, opt-worker, log-relay, loki/prom).

### Phase 1 — mem0 in the box

#### US-003: Fold `docker-compose.memory.yml` into the canonical compose (with PRD-153)
- [ ] `mem0-server` + `mem0-pgvector` services land in the PRD-153 compose, default profile, healthchecked; `MEM0_API_KEY` wired via `.env`.
- [ ] `infrastructure/docker-compose.memory.yml` is **deleted** (replace cleanly).
- [ ] Fresh `docker compose up`: chat stores and recalls an L3 fact across two sessions (manual + automated check).

#### US-004: mem0 image provenance
- [ ] Decide Q1 (fork image on GHCR vs upstream image) and implement: a pullable image reference in compose — contributors must not need the private fork repo to boot.
- [ ] Document the fork-vs-upstream delta (if any beyond config) in `docs/architecture/`.

#### US-005: mem0's own LLM key
- [ ] mem0-server's fact-extraction LLM key wired from the same single `.env` LLM credential the platform uses (no second signup); documented in QUICKSTART.

### Phase 2 — Observability as a profile

#### US-006: `observability` compose profile
- [ ] loki + prometheus + promtail (+ grafana if present in `infrastructure/docker-compose.monitoring.yml`) move behind `--profile observability`; their config.py URLs are set by that profile's env, not code defaults.
- [ ] `infrastructure/docker-compose.monitoring.yml` is deleted after the fold.

### Phase 3 — Proof

#### US-007: Both-modes CI
- [ ] CI lane A (default): compose with mem0 up — L3 store/recall test green.
- [ ] CI lane B (degraded): mem0 container stopped — `test_memory_degrades_when_mem0_down.py` + chat golden path still green, breaker trips and recovers.

---

## 6. Functional Requirements

- **FR-1** — `config.py` contains no `railway.internal` defaults.
- **FR-2** — mem0 + mem0-pgvector run in the default compose profile; removing them degrades gracefully (existing contract).
- **FR-3** — `LOG_RELAY_ENABLED=false` by default; all observability shipping is opt-in.
- **FR-4** — Absent optional services (voice, opt-worker, loki/prom) ⇒ feature-off log line, zero error noise.
- **FR-5** — Compose mem0 image is publicly pullable.
- **FR-6** — SaaS/Railway behavior unchanged (env overrides already present there).

---

## 7. Non-Goals (Out of Scope)

- **Replacing mem0 or refactoring `unified_memory_service` / L-stack contracts** — the integration is healthy.
- **A `MemoryProvider` interface** — explicitly rejected; mem0's API is the seam, the breaker is the fallback. (Pluggable = CI tax; optional = cheap. This is the "optional" pattern done right.)
- **L3 input-curation changes** (PR #407 territory) — untouched.
- **Voice service decoupling** — gets a profile + empty default here; making voice run locally is its own later PRD if wanted.
- **agent-opt-worker / FutureAGI** — profile + default-off only; the stale-SDK fix is separate (see memory: futureagi-prompt-opt-stale-sdk).

---

## 8. Technical Considerations

- **Reuse over build:** `docker-compose.memory.yml` already encodes the mem0 topology — this PRD promotes it to canonical and deletes the orphan, rather than authoring anything new.
- **Compose DNS vs Railway DNS** is pure env: same image, same client, different `MEM0_API_URL`. No edition flag needed for memory.
- **Health probe noise:** with mem0 in-network the 30s probe is cheap; in degraded lane B it must not log-spam (breaker + probe already rate-limit — assert in US-007).
- **Resource footprint:** mem0 + pgvector adds ~2 containers to the default stack. Acceptable; anything heavier (observability, voice) is profiled off by default to keep `docker compose up` lean.
- **Sequencing:** US-001/002 are independent and can land any time (small, test-guarded). Phase 1–2 land with/after PRD-153's compose consolidation to avoid editing compose files twice.

---

## 9. Success Metrics

- **M1:** Clean clone, default compose: tell the assistant a durable fact, new session recalls it (L3 round-trip) — zero Railway, zero mem0 cloud.
- **M2:** `grep -rn "railway.internal" orchestrator/config.py` → 0.
- **M3:** Local boot log contains zero connection errors to optional services (today: log-relay retries on every boot).
- **M4:** Degraded lane (mem0 down) stays green in CI.
- **M5:** SaaS memory behavior unchanged (existing memory test suite passes untouched).

---

## 10. Open Questions

- **Q1 — Image provenance.** Publish the fork to GHCR (preferred — pinned, ours) or point compose at upstream `mem0ai/openmemory`? Depends on the fork delta — enumerate it first (the client uses only standard endpoints, so the delta may be config-only).
- **Q2 — mem0 LLM provider.** Fork/upstream supports OpenAI-compatible endpoints — confirm it accepts the platform's OpenRouter key so the "one LLM key" QUICKSTART promise holds for memory too.
- **Q3 — `MEM0_API_KEY` in OSS.** Generate per-install in `.env` (like `API_KEY`) and pass to both server and client? (Proposed: yes — never ship a fixed default secret.)
- **Q4 — Internal API/frontend URL configs (lines 367–368)** — consumed by anything in orchestrator? If dead, delete with this sweep.

---

## 11. Phase Summary

| Phase | Stories | Character | Gate |
|---|---|---|---|
| 0 — Config sweep | US-001–002 | Small code + tests, independent | — |
| 1 — mem0 in compose | US-003–005 | Compose + image publishing | lands with PRD-153 consolidation |
| 2 — Observability profile | US-006 | Compose | after Phase 1 |
| 3 — Proof | US-007 | CI | both lanes required |

**Estimated blast radius:** `config.py` + ~6 consumer guards + compose files. Near-zero risk to SaaS (env overrides already authoritative there).
