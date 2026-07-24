# Qdrant (field memory) — NOT INSPECTABLE from this machine

Attempted 2026-07-04. This is a finding, not a failure.

## What was tried

```bash
curl -s -m 8 http://localhost:6333/collections   # -> exit 000, connection refused (no local Qdrant)
```

## Why no remote attempt was possible

- Neither `automatos-ai/orchestrator/.env` nor `automatos-ai/.env` contains any `QDRANT_*` variable (verified by listing all variable names in both files).
- The orchestrator's config default is local: `QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")` — pinned tree `orchestrator/config.py:859`; consumed at `orchestrator/modules/context/adapters/vector_field.py:82`.
- The only committed value found anywhere is the compose-internal hostname in `envs/api.defaults` (`QDRANT_URL=http://qdrant:6333`, pinned tree `envs/api.defaults`) — resolvable only inside the docker/Railway private network.
- No public Qdrant endpoint or API key exists on this machine, so field-memory collections/point counts/payloads (PRD-166/W8 promotion surface) could not be sampled.

## First look / implication

Field memory contents (workspace-persistent field, provenance payloads, promotion candidates) are invisible to local inspection; verifying W8 (PRD-178 taint-guarded promotion) against real points requires either a Railway-internal shell or a temporary read-only public endpoint. Given `memory_items` = 0 rows and zero L3 promotions in Postgres (see memory-short-term.md), the Qdrant side of the promotion pipeline is the missing half of that story and should be a priority for whoever has network access.
