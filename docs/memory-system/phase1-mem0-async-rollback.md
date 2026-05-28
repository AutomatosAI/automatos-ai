# Phase 1 (Mem0 async) — Rollback Runbook

**PRD:** 141 Platform Reliability · **Gate:** US-008 · **Scope:** Phase 1 (US-003…US-007 + US-008 review fixes)

Phase 1 rewrote `Mem0Client` from a synchronous `requests` client (one thread held per
call, retries via `time.sleep`) to an async `httpx.AsyncClient`, added a per-workspace
circuit breaker, a proactive health probe, and tightened timeouts. It touches the hottest
path on the platform (every widget/chat turn enriches from Mem0), so this runbook gives an
on-call two rollback tiers: a **fast env-flag mitigation** that needs no deploy, and a
**full git revert** that has been verified to apply cleanly.

---

## When to roll back

During the INBUILD canary soak (or after a wider rollout), trigger a rollback if you see any
of:

- Widget/chat **error-rate regression** vs the pre-deploy baseline — watch
  `record_error(subsystem="memory")` counts in Loki/Grafana.
- Chat turns **hanging** or timing out on memory enrichment (event-loop starvation).
- A workspace's Mem0 breaker **stuck open** — memory silently empty for that tenant despite
  Mem0 being up.

Pick the tier by symptom (see below). **Tier 1 first** if the symptom maps to a tunable —
it is a single env change and a restart, no code deploy.

---

## Tier 1 — env-flag mitigation (no deploy, no code revert)

All Phase 1 behaviour that is *tunable* is gated behind config in `orchestrator/config.py`
(read via env). Set these on the affected Railway service and restart — no rebuild needed.

| Symptom | Env var | Set to | Default (Phase 1) | Effect |
|---|---|---|---|---|
| Health probe is tripping/resetting breakers wrongly | `MEM0_HEALTH_PROBE_ENABLED` | `false` | `true` | Disables the 30s heartbeat probe; breakers go back to per-call open/close only |
| Slow (5–7s) writes timing out | `MEM0_WRITE_TIMEOUT_SECONDS` | `15.0` | `5.0` | Restores the pre-US-007 write budget |
| Breaker re-opening too aggressively after a blip | `MEM0_CIRCUIT_COOLDOWN_SECONDS` | `300` | `60` | Restores the pre-US-007 5-min cooldown |
| Probe too chatty | `MEM0_HEALTH_PROBE_INTERVAL_SECONDS` | higher (e.g. `120`) | `30` | Slows the probe cadence |

These revert cleanly because every value is read through `config.py` with an env override —
no code change re-reads them elsewhere (the `os.getenv` ban outside `config.py` guarantees a
single source of truth).

**Tier 1 does NOT undo the structural change** (async client, per-workspace breaker,
`workspace_id` threading). If the problem is the async path itself — event-loop starvation,
a coroutine-scheduling bug — go to Tier 2.

---

## Tier 2 — full git revert (structural rollback)

This reverts the entire Phase 1 code change back to the synchronous `requests` client.
**Verified 2026-05-28:** the revert below applies with **zero conflicts** and the reverted
tree compiles; `import requests` is restored in `mem0_client.py`.

Phase 1 is the contiguous commit range **`7145efaea` … `ce3ee3a69`** on
`ralph/prd-141-platform-reliability` (Phase 0 telemetry `cfa16c56c` is *not* in scope —
Phase 2+ depends on it; do not revert it).

```bash
# from a clean working tree on the deploy branch
git revert --no-edit 7145efaea^..ce3ee3a69
# -> 6 revert commits, newest-first; restores the sync Mem0Client

# sanity-check before pushing
python -m py_compile orchestrator/modules/memory/integrations/mem0_client.py \
                     orchestrator/modules/memory/unified_memory_service.py
grep -c "import requests" orchestrator/modules/memory/integrations/mem0_client.py   # expect 1

git push   # Railway auto-rebuilds the service from the deploy branch
```

### Commit manifest (newest → oldest)

| SHA | Story | What it did |
|---|---|---|
| `ce3ee3a69` | US-008 fixes | 404→breaker success, `workspace_id` threading, load test |
| `8b942e903` | US-007 | write timeout 15s→5s, cooldown 300s→60s |
| `e09403cb0` | US-006 | proactive health probe in heartbeat |
| `92e62a3c9` | US-005 | thread `workspace_id` into UnifiedMemoryService |
| `0412b8817` | US-004 | per-workspace circuit breaker |
| `7145efaea` | US-003 | async httpx client + drop executor wrappers |

To revert a **single** story instead of the whole phase (e.g. keep async but drop the probe),
`git revert --no-edit <that SHA>` — but check for conflicts, since later commits build on
earlier ones (US-006 references the breaker registry from US-004).

---

## Verify after rollback

1. **Build is green** on Railway and the service is healthy.
2. **Error rate** returns to the pre-deploy baseline — `record_error(subsystem="memory")` in
   Grafana stops regressing.
3. **A chat/widget turn** completes and enriches from memory (smoke the canary workspace).
4. If Tier 2: confirm `mem0_client.py` is the sync client again (`import requests` present,
   no `httpx.AsyncClient`).

---

## After a rollback — do NOT batch the fix into Phase 2

Per the US-008 gate notes: if the canary regresses, **file a fix user-story and re-soak**.
Phase 2 must not start on top of an unproven Phase 1. Capture the regression signal (Grafana
screenshot / Loki query) on the fix US so the re-soak has a concrete pass/fail bar.
