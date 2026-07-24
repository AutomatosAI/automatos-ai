# mem0 (durable memory server) — NOT REACHABLE

Attempted 2026-07-04 with GET-only probes. This is a finding, not a failure.

## What was tried

`MEM0_API_URL` is present in `automatos-ai/orchestrator/.env` (value = a `*.up.railway.app` hostname, no scheme; credentials/value elided per policy — host class: `automatos-mem0-server-production.up.railway.app`).

```bash
curl -m 8 https://<MEM0_API_URL>/                       # 404
curl -m 8 https://<MEM0_API_URL>/health                 # 404
curl -m 8 https://<MEM0_API_URL>/docs                   # 404
curl -m 8 https://<MEM0_API_URL>/api/v1/stats/          # 404
curl -m 8 https://<MEM0_API_URL>/api/v1/memories/?page=1&size=5   # 404
```

Response body for the API paths:

```json
{"status":"error","code":404,"message":"Application not found","request_id":"..."}
```

`"Application not found"` is Railway's **edge** error — the request never reached an app. The Railway service that used to answer on this hostname has been deleted, renamed, or unexposed. (Endpoints probed are real: they exist in the checked-in spec `orchestrator/mem0_openapi.json` — e.g. `GET /api/v1/stats/`, `GET /api/v1/memories/`.)

## First look / implication

The env file the orchestrator reads locally points at a dead mem0 deployment, so mem0-held durable memories could not be sampled at all. Production may use a different (Railway-internal) URL, but that could not be verified from this machine. Two consequences for dossier teams: (1) the founder's "memory quality is LOW" impression can only be judged from the platform-side `memory_short_term` table (see memory-short-term.md — it confirms the impression); (2) if the orchestrator anywhere falls back silently when mem0 is unreachable, recall may have been quietly degraded for some time — `memory_access_log` last recorded a recall on 2026-03-11.
