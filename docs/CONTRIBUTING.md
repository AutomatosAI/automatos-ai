# Contributing to Automatos AI

This is the full contributor guide. The short version lives in the repository
root, [CONTRIBUTING.md](../CONTRIBUTING.md); where the two overlap, they say
the same thing.

---

## The deal, in three lines

1. **Licence.** The repository is Apache-2.0. Under §5 of that licence your
   contribution may be distributed in every edition of the platform — the
   local edition you run yourself, and the hosted and commercial editions run
   by the maintainers — under the same terms, with no separate agreement.
   What you contribute stays Apache-2.0 for everyone, and it ships everywhere
   the code ships.
2. **Sign-off, not CLA.** Every commit carries a `Signed-off-by:` trailer
   (`git commit -s`), your [Developer Certificate of Origin](https://developercertificate.org)
   attestation. The `dco` check on each pull request verifies it and fails on
   any commit without one. There is no CLA to sign.
3. **One codebase, two shipped editions.** The local edition
   (`AUTH_EDITION=local`, `docker compose up`, no accounts) and the hosted
   edition (`saas`, automatos.app) are the same code behind one runtime flag.
   A change must keep both working; CI gates both.

---

## Where to contribute capability

Capability lands best where it never conflicts with core, in this order:

| First | Second |
|---|---|
| **Skills, tools, MCP integrations, Playbooks, agent packages.** A new skill or platform action needs no knowledge of the orchestrator's internals and reaches both editions unchanged. | **Core** — auth, storage, the tool router, migrations, the boot lifecycle. A core change needs the schema-drift, route-contract and fresh-clone smoke lanes to stay green in both editions. |

Open an issue first for anything that touches auth, storage, the tool router
or a migration. For the mechanics of adding a platform action, read
[`orchestrator/modules/tools/README.md`](../orchestrator/modules/tools/README.md).

Other welcome contributions: bug fixes (look for `good first issue`),
documentation (this directory — most pages are DeepWiki-generated, a few are
hand-maintained and listed in [docs/README.md](README.md)), and tests.

---

## Development environment

The compose stack **is** the development environment — there is no separate
bare-metal setup:

```bash
git clone https://github.com/<you>/automatos-ai.git
cd automatos-ai
cp .env.example .env        # POSTGRES_PASSWORD, REDIS_PASSWORD, API_KEY + one LLM key
docker compose up
```

- The backend and frontend containers bind-mount their source directories
  (`./orchestrator` → `/app`, `./frontend` → `/app`) and run in reload mode
  (`uvicorn --reload`, `npm run dev`), so edits are picked up without a
  rebuild. Dependency changes need `docker compose up -d --build`.
- For frontend-only work you can run the UI on the host instead
  (`cd frontend && npm install && npm run dev`) against the containerised API
  at `http://localhost:8000`.
- Database migrations run on every backend boot (`alembic upgrade heads`);
  a fresh, empty database is built by `python -m scripts.init_fresh_db` first.
- Everything else — services, ports, dials, troubleshooting — is in the
  [self-hosting guide](getting-started/self-hosting.md).

---

## Pull request process

1. **Fork and branch** from `main` with a descriptive name
   (`feat/…`, `fix/…`, `docs/…`).
2. **Keep it focused** — one change per PR. Add or extend tests with the
   change: features and bug fixes ship with the test that proves them.
3. **Sign off every commit**: `git commit -s -m "feat: …"`. Amend a missed
   one with `git commit --amend -s --no-edit` (or `git rebase --signoff`
   for a range).
4. **Open the PR against `main`** with a clear title, what changed and why,
   and screenshots for UI changes. Link the issue if there is one.
5. **CI is the gate.** Every lane runs on the PR; fix what it reports. The
   lanes and what each one proves:

| Workflow · job | What it checks |
|---|---|
| `test` · `orchestrator-tests` | The backend suite (`pytest tests`) against an ephemeral Postgres, in the local edition, with a coverage ratchet. **Required.** |
| `test` · `alembic-from-zero` | Exactly one Alembic head, and the fresh-clone boot path (`init_fresh_db`, then `upgrade heads`). |
| `test` · `schema-drift` | No table is `ALTER`-ed by a migration without some writer `CREATE`-ing it (`scripts/ci/schema_drift_check.py`). |
| `test` · `frontend-ci` | vitest, baselined `tsc`, eslint report, and the route contract (every backend path the frontend calls exists in the route manifest). |
| `test` · eval lanes | NL2SQL, retrieval-recall, memory-recall and graph-uplift harness self-tests (informational). |
| `smoke-fresh-clone` | `docker compose up` from an empty checkout with only the three secrets reaches a green `/health` and `/health/ready`. |
| `import-linter` | Module-boundary contracts (`orchestrator/.importlinter`). |
| `dco` | A `Signed-off-by:` trailer on every commit. |
| `gitleaks`, `CodeQL`, `malware-scan`, `check-shopify-isolation` | Secrets, static analysis, dependency hygiene, Shopify-package isolation. |

Lanes marked non-required in their workflow file still run and still report;
treat red as red.

---

## Conventions the code enforces

- **`orchestrator/config.py` is the only module that reads the environment.**
  No `os.getenv` elsewhere; add a config attribute and read it through
  `config`.
- **Canonical terms** in user-facing copy and identifiers: *Playbook* (not
  recipe), *Mission* (not workflow or job), *Deliverable* (not output or
  artifact), *Knowledge Graph*, *Command Center*, and *Auto* is a name.
- **Replace, don't shim.** When a path is superseded, the old one is deleted
  in the same PR — no `_legacy` suffixes, no compatibility branches.
- **Data lives in the database.** Personas, agent definitions and seed
  content are seeded into tables (`orchestrator/core/seeds/`), never read from
  files at runtime.
- **Python**: PEP 8, type hints on public functions, docstrings on public
  classes and functions, `async`/`await` for I/O. **TypeScript**: the
  project's ESLint/Prettier configuration.
- **Editions.** Anything hosted-only is gated by `AUTH_EDITION` /
  `isSaaS`, never by role; anything compose-only lives in `docker-compose.yml`
  and `envs/*.defaults`, which the hosted deployment never reads.

---

## Pull request checklist

- [ ] Linked issue (if applicable)
- [ ] Clear description of the change and the reason for it
- [ ] Tests added or updated
- [ ] Docs updated (`README.md`, `QUICKSTART.md` or `docs/` as needed)
- [ ] Every commit signed off (`git commit -s`)
- [ ] No linter or build errors; CI green

---

## Code of Conduct

Participation is governed by the [Code of Conduct](../CODE_OF_CONDUCT.md).
Report unacceptable behaviour to support@automatos.ai.

## Questions

Open an issue on the repository.
