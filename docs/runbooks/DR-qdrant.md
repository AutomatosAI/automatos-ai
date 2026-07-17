# DR Runbook — Qdrant Memory Snapshots & Restore (PRD-197 S3)

**Scope:** Disaster recovery for the two Qdrant **memory planes** Automatos
cannot regenerate: `durable_memory` (PRD-187 L3 durable agent memory) and
`field_memory` (the shared multi-agent field). The **document plane is out of
scope** — documents live on S3 Vectors and their DR is PRD-186's.
`DR-postgres.md` covers the primary database; this runbook is the Qdrant arm
of the same effort.

**Tooling:** the `memory_qdrant_snapshot` job on the shared memory-jobs
scheduler (`services/memory_jobs.py` → `services/qdrant_snapshots.py`).
Daily at `MEMORY_SNAPSHOT_CRON_HOUR_UTC` (default 04:00 UTC — after the
03:00 L2→L3 promotion, so each snapshot includes the night's promotions) it:

1. creates a node-side snapshot per collection (Qdrant snapshot API),
2. uploads it to `s3://$MEMORY_SNAPSHOT_S3_BUCKET/$MEMORY_SNAPSHOT_S3_PREFIX/<collection>/<snapshot>`
   (bucket defaults to `S3_DOCUMENTS_BUCKET`),
3. prunes node-side snapshots **and** object-store copies past
   `MEMORY_SNAPSHOT_RETENTION_DAYS` (default 7).

Disable with `MEMORY_SNAPSHOT_ENABLED=false`. The job is fail-soft per
collection: one collection's failure never blocks the other's snapshot, and
a failed cycle logs a WARNING and retries next cron.

## 1. Stated RPO / RTO

| Metric | Target | Basis |
|---|---|---|
| **RPO** | **≤ 24 h** | One snapshot per day, mirroring the DR-postgres nightly baseline. Memories lost since the last 04:00 snapshot are re-earnable (distill/promotion re-runs on live traffic); tighten the cron if that stops being acceptable. |
| **RTO** | **≤ 15 min per collection** | Snapshot recover is a single Qdrant API call; the memory collections are small (thousands of points). |

## 2. Restore (the one-liner, expanded)

Qdrant restores a collection from a snapshot **file** via its recover API.
From a machine that can reach the Qdrant node and the object store:

```bash
# 1. Pull the newest snapshot for the collection out of the object store
aws s3 cp "s3://$BUCKET/qdrant-snapshots/durable_memory/<snapshot-name>" ./durable_memory.snapshot

# 2. Recover the collection from the uploaded file (destructive: replaces the collection)
curl -X POST "$QDRANT_URL/collections/durable_memory/snapshots/upload?priority=snapshot" \
  -H "api-key: $QDRANT_API_KEY" \
  -F "snapshot=@durable_memory.snapshot"
```

Repeat for `field_memory`. Verify: `GET $QDRANT_URL/collections/<name>`
reports the expected `points_count`, and the Command Center memory primitive
tile returns green on the next heartbeat probe (≤ 30 s).

## 3. Open decision for Gerard (§8-Q3 — built to proposal, not settled)

Cadence (daily), retention (7 days), and destination (the platform object
store, `S3_DOCUMENTS_BUCKET`) are the PRD-197 §8-Q3 **proposal**, shipped as
defaults so the planes are recoverable now. Flip the `MEMORY_SNAPSHOT_*`
knobs to adjust — no code change needed. If a tighter RPO than 24 h is ever
required, raise the cron frequency before reaching for anything richer.
