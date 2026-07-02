# DR Runbook — PostgreSQL Backup & Restore (PRD-176 F050)

**Scope:** Disaster recovery for the two durable datastores Automatos cannot
regenerate: the **primary pgvector database** (`orchestrator_db`) and the
**separate mem0 instance** (durable agent memory). Redis is a cache and is out
of scope (it is rebuildable and holds no source-of-truth state).

**Tooling:** `scripts/dr/backup.sh` (`pg_dump -Fc`) and `scripts/dr/restore.sh`
(`pg_restore`). Both read connection strings from the environment — no
credentials are stored in the repo.

**Baseline chosen:** nightly `pg_dump` (custom format). See
[DR depth — open decision](#dr-depth--open-decision-for-gerard) for the
richer WAL/PITR option, which is **surfaced for decision, not silently
deferred.**

---

## 1. Stated RPO / RTO

| Metric | Target (nightly-`pg_dump` baseline) | Basis |
|---|---|---|
| **RPO** (max data loss) | **≤ 24 h** — worst case, everything since the last nightly dump | One dump per day. A workspace losing up to a day of documents/missions/memory is the accepted baseline for open-core + single-instance. Tighten by running `backup.sh` more often (e.g. every 6 h → RPO ≤ 6 h) or by adopting PITR (§5). |
| **RTO** (time to restore) | **≤ 30 min** for the primary DB | Provision a fresh Postgres → `CREATE EXTENSION vector` → `pg_restore` the latest dump → run the wait-migrate-seed entrypoint to head → `/health` green. Dominated by `pg_restore` time, which scales with DB size; 30 min is comfortable for a DB in the low-GB range. |
| **RTO (mem0)** | **≤ 30 min**, restored in parallel with the primary | Same `pg_restore` path against `MEM0_DATABASE_URL`. mem0 is independent of the primary, so the two restores run concurrently. |

These are **baseline** targets for the open-core / single-instance deployment.
A production SaaS SLA that requires a tighter RPO than 24 h is the trigger for
the PITR decision in §5.

---

## 2. What is backed up (and the two gotchas)

`pg_dump -Fc` captures the **full schema and data**, including:

- All SQLAlchemy-modelled tables.
- **Raw-DDL tables** that `create_all()` cannot build (e.g. `document_chunks`,
  the pgvector chunk store created in prod via `init_complete_schema.sql`).
  These live in the database, so the dump contains them — the restore rebuilds
  them from the dump, not from `create_all()`.

Two things the restore must handle that a naive `pg_restore` misses:

1. **pgvector extension.** The schema declares `vector` columns. The extension
   must exist in the **target** database *before* the restore. `restore.sh`
   runs `CREATE EXTENSION IF NOT EXISTS vector` first.
2. **Role/ownership drift.** Dumps are taken `--no-owner --no-privileges` and
   restored the same way, so a recovery instance with a different Postgres role
   restores cleanly.

---

## 3. Backup procedure

### 3.1 Manual / on-demand

```bash
# Primary DB only:
DATABASE_URL='postgresql://USER:PW@HOST:5432/orchestrator_db' \
  scripts/dr/backup.sh /backups

# Primary + mem0 (both durable stores):
DATABASE_URL='postgresql://USER:PW@HOST:5432/orchestrator_db' \
MEM0_DATABASE_URL='postgresql://USER:PW@MEM0_HOST:5432/mem0' \
  scripts/dr/backup.sh /backups
```

Output: `/backups/primary-<UTC timestamp>.dump` (and `mem0-<...>.dump`).

### 3.2 Schedule (nightly)

Run `backup.sh` once per day via the platform's scheduler (cron / Railway
scheduled job / CI cron). Recommended: **02:00 UTC**, off peak.

### 3.3 Where dumps are stored

- **Local/dev:** the `./backups` directory (or `DR_BACKUP_DIR`).
- **Production:** ship each dump to durable off-host object storage
  (S3/Backblaze) immediately after it is written, and set a retention policy
  (e.g. 7 daily + 4 weekly). **A dump that lives only on the database host is
  not a backup** — a host loss takes both the DB and its dumps.

---

## 4. Restore procedure

1. **Provision a fresh, empty target** Postgres (pgvector image, e.g.
   `pgvector/pgvector:pg16`), reachable at `RESTORE_DATABASE_URL`.
2. **Restore the dump:**

   ```bash
   RESTORE_DATABASE_URL='postgresql://USER:PW@NEWHOST:5432/orchestrator_db' \
     scripts/dr/restore.sh /backups/primary-<timestamp>.dump
   ```

   `restore.sh` creates the pgvector extension, then `pg_restore`s the dump
   (`--exit-on-error`, so a real failure is loud).
3. **Bring schema to head.** Point the backend at the restored DB and start it;
   the wait-migrate-seed entrypoint (F051) runs `alembic upgrade heads`. A dump
   is stamped with its `alembic_version`, so any migrations authored after the
   dump apply on top.
4. **Restore mem0** in parallel against its own `RESTORE_DATABASE_URL` using the
   `mem0-<timestamp>.dump`.
5. **Verify:** backend `/health` returns 200; spot-check row counts on a few
   tables (`documents`, `workspaces`, `agents`) against expectations.

---

## 5. DR depth — open decision (for Gerard)

**This runbook ships the nightly-`pg_dump` baseline (RPO ≤ 24 h).** The richer
alternative is **continuous WAL archiving + Point-In-Time Recovery (PITR)**,
which lowers RPO to near-zero (recover to any second) at the cost of:

- a WAL archive (continuous shipping of write-ahead logs to durable storage),
- a base-backup + WAL-replay restore path (more moving parts than `pg_restore`),
- provider support (e.g. managed PITR, or `pgBackRest`/`wal-g` operationally).

**Decision needed:** does the target RPO/RTO require WAL/PITR, or is the
nightly-dump RPO acceptable for this baseline?

- If **nightly-dump RPO (≤ 24 h, or ≤ 6 h at 4×/day) is acceptable** → this
  wave's `pg_dump`/`pg_restore` path is the complete answer.
- If a **tighter RPO is required** → PITR is a larger, separable build (WAL
  archiving + base backups + replay tooling). Per PRD-176 §6 it is **built in
  this wave if that is the call** — it is not being deferred here, it is being
  surfaced for the owner's decision (CLAUDE.md §12).

---

## 6. Test coverage

The restore path is CI-tested (a backup that has never been restored is not a
backup): `orchestrator/tests/test_dr_restore.py` dumps a populated DB
with `pg_dump -Fc`, restores it into a fresh database, and asserts **row-parity**
on a sampled table set. See PRD-176 §5.7.
