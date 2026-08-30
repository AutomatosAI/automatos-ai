# Automatos AI — orchestrator (backend)

The FastAPI service behind the platform: the REST/SSE API, the agent runtime,
Playbook and Mission execution, RAG, memory, the tool router and the seeds. One
codebase serves both editions — the local edition (`AUTH_EDITION=local`, the
compose stack) and the hosted edition (`saas`) — behind one runtime flag; see
[docs/getting-started/self-hosting.md](../docs/getting-started/self-hosting.md),
section 13.

## Layout

```
orchestrator/
├── main.py              FastAPI app; boot stages (core → trust gate → extensions);
│                        /health, /health/ready, /health/bootstrap
├── config.py            the ONLY module that reads the environment — every dial, every default
├── api/                 routers, one file per surface (the committed route manifest is
│                        checked against the frontend's calls by the route-contract CI lane)
├── core/                database + models; auth (core/auth/hybrid.py — the edition-aware
│                        request context); credentials; the Composio client and its boot
│                        bootstrap; the S3 client factory (core/storage/s3.py); seeds
│                        (core/seeds/) and the seed loader (core/database/load_seed_data.py)
├── modules/             domain modules — agents, tools (router + action registry), rag,
│                        memory, context, documents, …
├── consumers/           the streaming chat service and background consumers
├── services/            cross-cutting services
├── channels/ integrations/ jobs/ contracts/ evals/
├── alembic/             migrations (alembic.ini); exactly one head is a CI gate
├── scripts/             init_fresh_db.py (fresh-database schema), init_test_db.py
│                        (CI schema + seed), seed_*.py
└── tests/               the pytest suite (pytest.ini also collects modules/**/tests
                         and integrations/**/tests)
```

## Development setup — the compose stack is the dev environment

There is no bare-metal setup to maintain. From the repository root:

```bash
cp .env.example .env     # POSTGRES_PASSWORD, REDIS_PASSWORD, API_KEY (+ one LLM key)
docker compose up
```

- The `backend` service builds this directory with the `development` target,
  bind-mounts it at `/app` and runs `uvicorn main:app --reload`, so edits are
  live. Dependency or Dockerfile changes need
  `docker compose up -d --build backend`.
- Configuration is layered: `envs/api.defaults` (committed local topology,
  `AUTH_EDITION=local`), `envs/api.local` (gitignored overrides for any
  `config.py` dial) and the compose `environment:` block (secrets substituted
  from `.env`). `.env` reaches the container only through compose
  substitution.
- Logs: `docker compose logs -f backend`. API reference:
  http://localhost:8000/docs.
- Everything the backend needs — Postgres with pgvector, Redis, MinIO, the
  workspace-worker — is in the same compose file.

## Database lifecycle — what the entrypoint does

`docker-entrypoint.sh` (repository root, mounted into the container) is the
single owner of the schema lifecycle:

1. **Empty database** (no `alembic_version`): `python -m scripts.init_fresh_db`
   builds the full schema — `Base.metadata.create_all` plus the raw-DDL
   extras, then a statement-tolerant replay of the migration forest — and
   stamps Alembic at heads. It refuses a non-empty database that has no
   `alembic_version`. No SQL snapshot is committed; the generator is the
   fresh path.
2. **Every boot**: `alembic upgrade heads`, fail-closed — a failing migration
   stops the container.
3. `python -m core.database.load_seed_data` — idempotent seeds (credential
   types, models, skills, personas, plugin categories, the marketplace
   catalogue; in the local edition also the first-run content: Auto, the
   Researcher/Writer/Analyst roster, the *Two-minute brief* Playbook and a
   welcome Deliverable).
4. In the local edition, the local workspace (`DEFAULT_WORKSPACE_ID`) and the
   operator user are ensured, fail-closed.

Adding a migration: a revision under `alembic/versions/`, keeping **one head**
(the `alembic-from-zero` CI job asserts it and re-runs the fresh-clone path).
The `schema-drift` job (`scripts/ci/schema_drift_check.py`, repository root)
goes red when a migration `ALTER`s a table that no writer — migration, model
or raw-DDL extra — `CREATE`s.

## Tests — CI is the gate

The `test` workflow (`.github/workflows/test.yml`) runs on every push and pull
request:

| Job | What it runs |
|---|---|
| `orchestrator-tests` (required) | `python scripts/init_test_db.py` against an ephemeral Postgres, then `pytest tests` in the local edition (`AUTH_EDITION=local`, `DEFAULT_WORKSPACE_ID` set) with a coverage ratchet (`scripts/check_coverage_baseline.py`). |
| `orchestrator-module-tests` | The `modules/**/tests` and `integrations/**/tests` trees (some need live services). |
| `alembic-from-zero` | Exactly one Alembic head; `init_fresh_db` then `upgrade heads` from an empty database. |
| `schema-drift` | The four-writer drift check. |
| eval lanes | NL2SQL, retrieval-recall, memory-recall and graph-uplift harness self-tests (informational). |

Alongside it: `smoke-fresh-clone` (a real `docker compose up` from an empty
checkout must reach a green `/health` and `/health/ready`), `import-linter`
(module-boundary contracts in `orchestrator/.importlinter`), `dco`,
`gitleaks`, `CodeQL`, `malware-scan` and `check-shopify-isolation`.

Pure tests (no database) can be run inside the compose image:

```bash
docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/orchestrator backend \
  python -m pytest tests/test_prd209_quickstart_honest.py -q -p no:cacheprovider
```

Tests that need Postgres need the schema `init_test_db.py` builds; CI is the
reference run for those.

## Conventions enforced here

- `config.py` is the only environment reader (`os.getenv` nowhere else).
- Every S3 client comes from `core/storage/s3.py` (a source guard fails on a
  stray `boto3.client(`); locally that is MinIO, in the hosted edition AWS S3.
- Composio is consulted through one availability predicate
  (`core/composio/client.py: composio_available()`); without a key the router
  excludes Composio tools instead of failing open — see
  [modules/tools/README.md](modules/tools/README.md).
- Anything hosted-only is gated by `config.AUTH_EDITION`, never by role; the
  local operator is `super_admin` by design.
- Canonical terms: Playbook, Mission, Task, Deliverable, Knowledge Graph,
  Command Center, Auto.

## Module guides

- [modules/agents/README.md](modules/agents/README.md)
- [modules/tools/README.md](modules/tools/README.md)
- [modules/codegraph/README.md](modules/codegraph/README.md)
- [modules/nl2sql/README.md](modules/nl2sql/README.md)
- [consumers/README.md](consumers/README.md)

Contributing: [CONTRIBUTING.md](../CONTRIBUTING.md) — DCO sign-off on every
commit; capability first, core second. Licence: Apache-2.0.
