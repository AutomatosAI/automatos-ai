# PRD-176 — Deployability & Reliability Baseline (Wave 6)

**Status:** Draft v1 — pending approval
**Type:** Infrastructure / Reliability (open-core deployability + DR)
**Priority:** P0 — "`git clone && docker compose up`" must yield a working local instance
**Owner:** Gerard Kavanagh
**Author:** Gerard Kavanagh + Claude (Opus 4.8)
**Date:** 2026-07-02
**Phase:** B — Policy plane & deployability · **Size:** M–L · **Risk:** medium
**Depends on:** **Wave 5 (auth decoupling)** — boot-before-auth must land first (compose still hard-requires Clerk keys today; a fresh clone cannot reach a working instance until W5 makes login optional)
**Parent:** [PLATFORM-OS-ROADMAP.md](./PLATFORM-OS-ROADMAP.md)
**Source:** review §13 Wave 6 (the absent PRDs 151–153); §2 Reliability + Deployability pillars; §7 "do not do" (the alembic two-step)
**Findings register pinned to:** `37fdecc4e` — re-confirm each `file:line` on current `main` before editing
**Findings in scope:** F009, F010, F051, F089, F068, F050

---

## Operating Principle

> **A fresh clone boots itself, with no external credentials, into a working local instance — and the
> database it boots can be replayed from zero and restored from a backup.** Deployability and reliability
> are one wave because they share one substrate: a single, honest schema lifecycle. Today schema truth is
> split across four mechanisms that mutate overlapping tables, none of which a new contributor can run
> end-to-end. This PRD collapses the boot path to **one wait-migrate-seed entrypoint**, makes Alembic
> replay from zero to **exactly one head**, gives outputs a real local object store, makes the SaaS-topology
> defaults local-safe, and — for the first time in the repo — makes the primary database **backup-able and
> restorable** with a stated RPO/RTO. It adds no product capability; it makes the product the open-core
> thesis promises **actually installable and operable**.

---

## 1. Purpose

The open-core thesis — "`git clone && docker compose up` → a working local instance, no login, zero
external SaaS" (roadmap §2, Deployability pillar) — is **undeployable on `main` today**, and the database
underneath it has **no disaster-recovery story at all**.

Concretely: a fresh clone can't boot because `docker-compose.yml:35` mounts the initdb schema from a path
that does not exist (`./orchestrator/database/init_complete_schema.sql`; the real file is at
`./orchestrator/core/database/init_complete_schema.sql`) — so the Postgres init volume runs nothing, the
backend comes up against an empty DB, and because the frontend is gated `depends_on: backend:
service_healthy` (`docker-compose.yml:153-155`), **the frontend never starts** (F009). Even with that path
fixed, Alembic cannot replay the schema from an empty database: 133 revision files form a **four-headed
forest** (`alembic heads` → `20260612_nl2sql_example_embedding`, `prd158_cloud_default_team`,
`prd161_sla_breach`, `prd164_doc_source_type`), core tables are `ALTER`ed by migrations but `CREATE`d by
none, and **no CI ever runs from-zero** — production only ever boots via `alembic upgrade heads` against
already-stamped incremental databases, so this is a **thesis / OSS blocker, not a production-runtime crash**
(F010, ADJUSTED). The boot script that does exist (`docker-entrypoint.sh`) waits for Postgres and seeds, but
**never runs a migration** — schema arrives only from the (broken) initdb mount, which is precisely the
"schema truth split across four mechanisms mutating overlapping tables" the review names (F051).

On top of the boot failure: there is **no local object storage**, so the knowledge flywheel fail-softs to
`None` on every generated output (`services/knowledge_flywheel.py:159,166,223`) and documents live only on
ephemeral container disk (F089); **nine `railway.internal` hostnames** are hardcoded as config defaults and
`LOG_RELAY_ENABLED` defaults `"true"` — SaaS topology leaking into the local default (F068); and there is
**no backup or disaster recovery anywhere in the repo** — no `pg_dump`/`pg_restore` tooling, no
volume-snapshot script, no restore runbook for the primary pgvector database or the separate mem0 instance;
the only DDL tool that exists is a dev helper (`scripts/snapshot_table_ddl.py`) (F050).

**W6 makes the two deployability bars and the reliability bar pass:** a fresh clone boots to a green
`/health` with no external credentials, Alembic replays from zero to exactly one head under CI, and the
primary DB has a tested restore with a stated RPO/RTO. It is the review's "absent PRDs 151–153," now scoped
as one wave on top of the Wave-5 auth seam.

---

## 2. Background

### 2.1 What's working today (must not break)

- **The compose topology is sound.** `docker-compose.yml` already wires pgvector Postgres, Redis, the
  FastAPI backend, and the Next.js frontend with health checks and `depends_on` conditions; the backend
  health gate targets `/health` (`docker-compose.yml:131-136`) and Postgres has a proper `pg_isready`
  healthcheck (`:36-41`). W6 fixes what the topology *mounts and runs*, not the topology.
- **An entrypoint already exists** (`docker-entrypoint.sh`, mounted at `docker-compose.yml:127`). It waits
  for Postgres (`wait_for_postgres`), verifies the connection (`check_database`), and idempotently loads seed
  data (`load_seed_data` → `database/load_seed_data.py`). W6 **inserts the missing migrate step into this
  existing script** — it does not introduce a competing entrypoint.
- **Production boot is fine.** `alembic upgrade heads` against a stamped, incrementally-migrated database
  works in prod today. F010 is about *from-zero replay*, which only the OSS/fresh-clone and CI paths exercise
  — do not "fix" a production path that isn't broken.
- **Config is centralized.** All env reads already route through `orchestrator/config.py` (CLAUDE.md §4);
  the `railway.internal` defaults and `LOG_RELAY_ENABLED` all live there in one file, so F068 is a
  bounded edit to defaults, not a scattered hunt.
- **The S3/object-store abstraction exists.** `config.py` already models S3 (`S3_VECTORS_BUCKET`,
  `S3_VECTORS_ENABLED`, with a workspace-embed guard at `config.py:905-930`); the flywheel already calls an
  object-storage upload path. W6 adds a **local MinIO endpoint** behind the existing seam (`S3_ENDPOINT_URL`),
  it does not introduce a new storage subsystem.

### 2.2 What's broken / missing

- **F009 — a fresh clone can't boot.** `docker-compose.yml:35` mounts
  `./orchestrator/database/init_complete_schema.sql` — a path that does not exist. The real schema file is
  `./orchestrator/core/database/init_complete_schema.sql` (72 KB, present on `main`). The init volume runs an
  empty mount, the backend starts against an empty DB, and the frontend (gated on backend health,
  `docker-compose.yml:153-155`) never starts.
- **F010 (ADJUSTED) — Alembic can't replay from zero.** 133 revisions, **four heads** (confirmed via
  `alembic heads` on `main` — note: the review counted 132/four heads at `37fdecc4e`; re-confirm the exact
  count). Core tables are `ALTER`ed but never `CREATE`d, and **no CI exercises from-zero**. This blocks the
  OSS/thesis path and any contributor's clean install; it is **not** a prod-runtime crash (prod boots via
  `upgrade heads` on stamped DBs).
- **F051 — schema truth split across four mechanisms.** The initdb SQL mount, `docker-entrypoint.sh`'s seed
  loader, the Alembic revisions, and ad-hoc `scripts/` DDL all touch overlapping tables, and no single
  lifecycle owns "bring an empty database to a correct, seeded, migrated state." Needs **one wait-migrate-seed
  entrypoint / single lifecycle**.
- **F089 — no local object storage.** With no `S3_ENDPOINT_URL` / MinIO (PRD-151's local object store), the
  knowledge flywheel's upload fail-soft returns `None` on **every** generated output
  (`services/knowledge_flywheel.py:159,166,223` — "output flow unaffected" logged, document lost), and
  generated documents live only on ephemeral disk. The flywheel — the moat's input — silently ingests nothing
  locally.
- **F068 — SaaS topology hardcoded into local defaults.** Nine `railway.internal` hostnames are config
  defaults (`config.py:407,408,506,507,519,537,730,753,884`), and `LOG_RELAY_ENABLED` defaults `"true"`
  (`config.py:521`) — so a local instance tries to reach Railway-internal Loki/Prometheus/mem0/log-relay/
  workers/voice by default. Local default should be **local-safe**; `LOG_RELAY_ENABLED` should default
  **false** locally.
- **F050 — no backup or disaster recovery anywhere.** `grep` across the repo finds no `pg_dump`,
  `pg_restore`, volume-snapshot script, or restore runbook for either the primary pgvector DB or the separate
  mem0 instance. The only DDL-adjacent tool is `scripts/snapshot_table_ddl.py` (a schema-diff dev helper, not
  a backup). There is **no stated RPO/RTO** and no tested recovery path.

### 2.3 Why now

This is **Phase B, gated on Wave 5**. W5 mounts Clerk only in SaaS behind an `AppAuth` facade so a local
instance runs no-login; until that lands, compose still hard-requires `CLERK_SECRET_KEY` / `CLERK_JWKS_URL`
(`docker-compose.yml:118-121`) and a fresh clone cannot reach a working instance regardless of the boot fix.
With W5 in place, W6 is the wave that turns "open-core" from a claim into a `docker compose up`. It also
underwrites the **Reliability** pillar (roadmap §2): Alembic-replays-from-zero-with-one-head and
tested-restore-with-stated-RPO/RTO are two of that pillar's three bars (the third, exactly-once under lease
expiry, is Wave 1). Everything downstream that a contributor or an eval must run — W7's moat eval, W12's CI
bar, W13's Shopify pilot — assumes a database that can be built from zero and recovered.

---

## 3. Findings in scope

| ID | Sev | Location (pinned `37fdecc4e`) | Defect | Fix |
|---|---|---|---|---|
| **F009** | High | `docker-compose.yml:35` | initdb mount points at `./orchestrator/database/init_complete_schema.sql` — path does not exist; fresh volume gets an empty DB and the frontend (gated on backend health) never starts | Repoint the mount to `./orchestrator/core/database/init_complete_schema.sql`; add a fresh-clone smoke test (see §5) |
| **F010** | High (ADJUSTED — thesis/OSS blocker, not prod crash) | `orchestrator/alembic/versions/` (133 revs, **4 heads**) | Cannot replay from an empty DB: four-headed forest, core tables `ALTER`ed but never `CREATE`d, no from-zero CI | **Two steps (review §7):** author **one merge revision now** to collapse the four heads; **squash to a single baseline later** with a from-zero replay CI job asserting **exactly one head** |
| **F051** | High | `docker-entrypoint.sh` (waits + seeds, no migrate) + initdb SQL mount + Alembic + `scripts/` DDL | Schema truth split across four mechanisms mutating overlapping tables; no single lifecycle owns empty→correct | One **wait-migrate-seed** entrypoint: extend the existing `docker-entrypoint.sh` to run `alembic upgrade heads` between wait and seed; make the initdb SQL mount and migrations non-overlapping (one source of `CREATE`) |
| **F089** | High | `orchestrator/services/knowledge_flywheel.py:159,166,223`; `config.py` (S3 seam) | No local object store → flywheel upload fail-softs to `None` on every output; generated docs on ephemeral disk | Add a local **MinIO** service to compose wired via **`S3_ENDPOINT_URL`** behind the existing S3 config seam; flywheel writes to it locally |
| **F068** | Medium | `config.py:407,408,506,507,519,537,730,753,884` (9× `railway.internal`); `config.py:521` (`LOG_RELAY_ENABLED` default `"true"`) | SaaS topology hardcoded as local default; log relay on by default locally | Make the `railway.internal` defaults **local-safe** (localhost/opt-in, unset in local profile); default **`LOG_RELAY_ENABLED` false** |
| **F050** | High (missing) | repo-wide (only `scripts/snapshot_table_ddl.py` exists) | No `pg_dump`/`pg_restore` tooling, no volume-snapshot script, no restore runbook for primary pgvector DB **or** the separate mem0 instance; no RPO/RTO | Add a **`pg_dump` backup + restore runbook and DR** with a **tested restore** and stated **RPO/RTO**, covering both the primary DB and the mem0 instance |

---

## 4. Design & changes

Minimal-diff, per finding. The two structural moves are (a) inserting a migrate step into the **existing**
entrypoint (F051) and (b) the **explicit two-step Alembic sequencing** the review mandates (F010). Nothing
here adds a subsystem; every change repoints, wires, or scripts what already exists (CLAUDE.md §2/§4/§5).

### 4.1 F009 — repoint the initdb mount (one line)

In `docker-compose.yml:35`, change the bind mount source from
`./orchestrator/database/init_complete_schema.sql` to
`./orchestrator/core/database/init_complete_schema.sql` (confirmed present, 72 KB). No shim, no symlink
(CLAUDE.md §4). This alone lets a fresh Postgres volume initialize a populated schema so the backend health
check passes and the frontend's `depends_on` gate releases.

> **Note on overlap with F010/F051:** the initdb SQL mount and Alembic are **two sources of `CREATE`**. The
> §4.3 sequencing resolves which owns schema creation. The immediate F009 fix restores the *documented*
> current behavior (populated fresh volume); §4.3 then makes the lifecycle single-source so the initdb SQL
> and migrations do not fight. Do not leave both authoritative — pick one in the same wave (CLAUDE.md §5).

### 4.2 F051 — one wait-migrate-seed entrypoint (single lifecycle)

The existing `docker-entrypoint.sh` already does **wait → check → seed** (`wait_for_postgres` →
`check_database` → `load_seed_data`) but **never migrates**. Insert the missing step so the single lifecycle
is **wait → migrate → seed → start**:

- Add a `run_migrations()` step between `check_database` and `load_seed_data` that runs
  `alembic upgrade heads` (post-§4.3: `alembic upgrade head`, singular, once the baseline is single-headed),
  failing closed if the migration errors (drop the current script's "continue anyway" leniency for the
  migrate step — a failed migration must not silently start the app on a half-built schema).
- Keep `load_seed_data` idempotent (it already guards on `credential_types` count).
- This makes the entrypoint the **single owner** of "bring the DB to a correct, migrated, seeded state,"
  collapsing F051's four mechanisms to one lifecycle. The initdb SQL mount's role is decided by §4.3.

### 4.3 F010 — the Alembic two-step (review §7 sequencing, stated as such — NOT a descope)

The roadmap and review §7 are explicit: **"Do not attempt the alembic single-baseline squash first — author
one merge revision now, squash later with a from-zero test."** This PRD follows that order exactly, and
records it as the review's sequencing, not a deferral (CLAUDE.md §12):

- **Step 1 — NOW: one merge revision to collapse the four heads.** Author a single Alembic merge revision
  (`alembic merge -m "prd176 collapse four heads" <the four head revisions>`) so `alembic heads` returns
  **one head**. This is low-risk and immediately makes `upgrade head` deterministic for the entrypoint (§4.2)
  and for prod (`upgrade heads` still works, now trivially single-headed). Re-confirm the four head revision
  IDs on `main` at author time (the count drifted 132→133 since `37fdecc4e`).
- **Step 2 — LATER (this wave, after Step 1 is green): squash to a single baseline with a from-zero replay
  test.** Once the forest is single-headed, author one baseline revision that `CREATE`s the full schema
  from empty (reconciling the tables that today are `ALTER`ed-but-never-`CREATE`d), retire the superseded
  revisions (CLAUDE.md §5 — delete what's squashed, do not keep a parallel history), and **prove it with the
  from-zero CI job** in §5.1. With a from-zero baseline, the **initdb SQL mount (F009) becomes redundant and
  is removed** — Alembic-from-zero is the single source of schema creation (resolving the F009/F051 overlap).

> **Why two steps and not one:** squashing 133 revisions into a baseline while four heads still diverge risks
> silently dropping the `ALTER`-only tables' columns. Collapsing to one head first makes the squash a linear,
> testable replay. This ordering is the review's (§7) and the roadmap's ("author one merge revision now,
> squash later with a from-zero test"), not this PRD's initiative.

### 4.4 F089 — local object storage via MinIO + `S3_ENDPOINT_URL`

- **Add a `minio` service to `docker-compose.yml`** (image `minio/minio`, a `minio_data` volume, a health
  check, on the `automatos` network), plus a one-shot bucket-create init (MinIO `mc mb` or a small init
  container) so the flywheel's target bucket exists on first boot.
- **Wire it through the existing S3 seam:** set `S3_ENDPOINT_URL` (the AWS-SDK/boto endpoint override) in the
  backend service env to the local MinIO endpoint, with the local access/secret keys and a local bucket that
  carries `{workspace_id}` (honoring the `config.py:905-930` workspace-embed guard). No new storage client —
  the flywheel's existing upload path (`DocumentManager.upload_document`) targets S3-compatible storage; MinIO
  *is* S3-compatible.
- **Result:** `services/knowledge_flywheel.py` uploads succeed locally, so the fail-soft `return None` paths
  (`:159,166,223`) stop firing on every output and generated documents persist to durable local object
  storage instead of ephemeral disk. Prod behavior is unchanged (real S3 via the same seam, `S3_ENDPOINT_URL`
  unset).

### 4.5 F068 — local-safe topology defaults; log relay off locally

- **`LOG_RELAY_ENABLED` defaults false** (`config.py:521`): change the default from `"true"` to `"false"` so
  a local instance does not attempt to push to `log-relay.railway.internal` by default; SaaS sets it `true`
  via env. Update the two tests that assert the old default (`tests/test_config_env_centralization.py:174-181`
  — flip `test_log_relay_enabled_default_true` to expect false; the override test stays).
- **`railway.internal` defaults become local-safe** (`config.py:407,408,506,507,519,537,730,753,884`): these
  are `os.getenv(..., "<x>.railway.internal...")` defaults. Make the local default resolve to
  localhost/opt-in (e.g. empty/unset so the dependent feature no-ops locally, or a `localhost` value where a
  local equivalent exists), while SaaS supplies the real Railway host via env. The monitoring handlers already
  tolerate absence (`handlers_monitoring.py:153,207` fall back with `getattr(..., None) or ...`), so unset is
  safe for the observability defaults. Keep all reads in `config.py` (CLAUDE.md §4).
- **`tool_registry.py:1101-1102`** contains `build-server.railway.internal` inside example/seed payloads —
  these are illustrative tool-call examples, not live defaults; leave them unless the F068 re-confirm shows
  they execute. (Flag, do not silently change data.)

### 4.6 F050 — pg_dump backup/restore runbook + DR (tested, with RPO/RTO)

- **Add backup tooling under `scripts/` (or `scripts/dr/`):** a `pg_dump`-based backup script (custom-format
  `pg_dump -Fc` of the primary `orchestrator_db`) and a matching `pg_restore` restore script, both
  parameterized off the canonical `DATABASE_URL` / `POSTGRES_*` config — no hardcoded creds (CLAUDE.md §4,
  security §Secret Management). Include the **separate mem0 instance** as a second backup target (its DB is a
  distinct instance per `MEM0_API_URL`/mem0 deployment — the runbook must cover both, since losing mem0 loses
  durable memory).
- **Author a restore runbook** (`docs/runbooks/DR-postgres.md` or equivalent) covering: backup schedule,
  where dumps are stored, the exact restore procedure for the primary pgvector DB (including the pgvector
  extension and any raw-DDL tables that `create_all()` misses — see memory: init_test_db gap), the mem0
  restore, and a **stated RPO and RTO** (e.g. RPO = last nightly dump; RTO = time to `pg_restore` + re-seed +
  health-green — put concrete target numbers in the runbook and justify them).
- **Test the restore** (§5.6): a CI or scripted test that dumps a populated DB, restores into a fresh volume,
  and asserts row-parity on a sampled set of tables. A backup that has never been restored is not a backup.

> **Scope boundary (CLAUDE.md §12):** this wave delivers **local + single-instance** backup/restore with a
> tested `pg_restore` and a written runbook. Continuous WAL-archiving / PITR and cloud-snapshot automation
> are a **larger, separable** capability — surfaced here as an **open question for Gerard** (§6), not
> silently deferred. If the stated RPO/RTO Gerard wants requires PITR, it is this wave's work; if nightly-dump
> RPO is acceptable, the tested `pg_dump`/`pg_restore` path meets the bar.

---

## 5. Test-first acceptance

Write these **failing first**, then implement to green. The two headline tests are the wave's definition of
done (review §13); the rest are per-finding.

1. **Headline A — from-zero Alembic replay, exactly one head (F010).** A CI job stands up an **empty
   pgvector database**, runs `alembic upgrade heads` (post-squash: `head`), and asserts **exactly one head**
   (`alembic heads` returns a single revision) **and** that the schema is complete (a representative set of
   the tables that are today `ALTER`ed-but-never-`CREATE`d now exist). Fails today (four heads, no from-zero
   path); green after §4.3 Step 2.
2. **Headline B — fresh-clone `docker compose up` smoke test → `/health` 200, no external credentials
   (F009 + F051 + W5).** A CI/scripted smoke test clones clean, provides **only** the required local secrets
   (`POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `API_KEY`) and **no external SaaS credentials** (no Clerk, no real
   S3, no LLM keys), runs `docker compose up`, and asserts the backend `/health` returns **200** and the
   frontend container reaches healthy. This exercises the repointed initdb mount, the wait-migrate-seed
   entrypoint, local MinIO, and the local-safe defaults together. (Requires W5's no-login default; if W5 is
   not yet merged at author time, the test pins the auth-optional flag.)
3. **F009 — initdb path resolves.** A unit/lint assertion that the `docker-compose.yml` initdb mount source
   path **exists on disk** (guards against the exact regression: a compose mount pointing at a missing file).
4. **F051 — the entrypoint migrates.** A test of `docker-entrypoint.sh`'s lifecycle asserting the order is
   **wait → migrate → seed → start** and that a **failed migration aborts startup** (does not fall through to
   `exec "$@"` on a half-built schema).
5. **F089 — local MinIO wired; flywheel persists.** With the local MinIO service up and `S3_ENDPOINT_URL`
   set, a flywheel ingest of a generated output returns a **non-`None` document id** and the object is
   retrievable from the local bucket (the `:159,166,223` fail-soft paths are **not** taken). With MinIO
   absent, the fail-soft still degrades gracefully (no crash).
6. **F068 — defaults are local-safe.** With **no env overrides**, `config.LOG_RELAY_ENABLED is False`, and
   the `railway.internal` defaults resolve to local-safe values (no `*.railway.internal` host is dialed by
   default). The existing centralization tests (`test_config_env_centralization.py`) are updated to the new
   defaults (the SaaS override paths still assert the Railway values when env is set).
7. **F050 — tested `pg_dump` restore with stated RPO/RTO.** A test dumps a populated primary DB with
   `pg_dump -Fc`, `pg_restore`s into a **fresh** volume, and asserts **row-parity** on a sampled table set;
   the runbook is present and states concrete **RPO and RTO** numbers; the mem0 backup target is covered by
   the same tooling.

**Wave-level bar (review §13):** a CI job runs `alembic upgrade heads` from an empty pgvector database and
asserts **exactly one head**, and a fresh-clone `docker compose up` smoke test returns **200** on `/health`
with **no external credentials** — plus the primary DB has a tested restore with a stated RPO/RTO.

---

## 6. Risks & rollback

- **Sequencing risk (F010).** Squashing before collapsing heads can silently drop `ALTER`-only columns.
  **Mitigation:** the mandated two-step — merge revision first (Step 1), squash-with-from-zero-test second
  (Step 2); Headline A guards it permanently.
- **Wave-5 dependency.** Headline B cannot pass no-credential until W5 makes login optional (compose still
  hard-requires Clerk today, `docker-compose.yml:118-121`). **Mitigation:** W6 lands after W5; if authored
  earlier, Headline B pins the auth-optional flag and is marked blocked-on-W5.
- **initdb-vs-migrations double-source (F009/F051).** Leaving both the initdb SQL mount and Alembic
  authoritative reintroduces schema drift. **Mitigation:** §4.3 Step 2 removes the initdb SQL mount once
  Alembic-from-zero is the single source (CLAUDE.md §5).
- **F068 breadth.** Changing nine defaults could hide a host a SaaS path silently relied on. **Mitigation:**
  each default keeps its `os.getenv` override so SaaS is unchanged; only the *default* moves; re-confirm each
  of the nine on `main`.
- **Backup false-confidence (F050).** An untested dump is not a backup, and pgvector/raw-DDL tables can be
  missed by naive restores. **Mitigation:** the restore is CI-tested with row-parity and the runbook calls
  out pgvector + raw-DDL tables explicitly.
- **Rollback:** every finding-fix is an independent commit — the compose mount repoint, the entrypoint
  migrate step, the MinIO service, the config defaults, and the DR scripts revert individually. The Alembic
  merge revision (Step 1) is reversible before the squash (Step 2) is authored.

**Open question for Gerard (§6, not a deferral — CLAUDE.md §12):** DR depth. Does the target RPO/RTO require
**continuous WAL archiving / PITR + cloud-snapshot automation**, or is **nightly-`pg_dump` RPO** acceptable
for this baseline? W6 ships the tested `pg_dump`/`pg_restore` path and runbook either way; PITR is a
separable, larger build — surfaced for decision, built in this wave if that's the call.

---

## 7. References

- Review §13 Wave 6 (the absent PRDs 151–153; the wave's DoD: from-zero `alembic upgrade heads` → one head,
  fresh-clone `docker compose up` → `/health` 200 with no external credentials): `reports/PLATFORM_OS_REVIEW_2026-07-01.md`
- Review §2 — Reliability pillar (replay-from-zero, tested restore w/ RPO/RTO) + Deployability pillar
  (`git clone && docker compose up`, local object store)
- Review §7 "do not do" — the **Alembic two-step** (merge revision now, squash later with a from-zero test)
- Roadmap §3/§4 — W6 placement (Phase B), **depends on W5**; §2 pillar→wave mapping (F009/F010/F068/F089 →
  Deployability/Reliability, closed by W5+W6)
- Findings F009, F010, F051, F089, F068, F050 — `reports/PLATFORM_OS_REVIEW_2026-07-01.md`
- Reuses: existing `docker-entrypoint.sh` lifecycle, the `config.py` S3 seam (`S3_ENDPOINT_URL`,
  `S3_VECTORS_BUCKET`, workspace-embed guard `:905-930`), the flywheel upload path
  (`services/knowledge_flywheel.py`), the centralized config module
- CLAUDE.md §4 (no shims / config-only / no hardcoded values), §5 (delete what you supersede — the initdb
  mount post-squash), §12 (no unilateral descope — the Alembic two-step and DR-depth are the review's
  sequencing and an open owner-decision, stated as such)
