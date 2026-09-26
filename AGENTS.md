# AGENTS.md — how to work in this repository

These rules are for coding agents (Claude Code, Codex, Cursor, Copilot, Gemini CLI, Windsurf, Aider and the rest) and for the people who drive them. They apply to every change, whoever or whatever wrote it. [CONTRIBUTING.md](CONTRIBUTING.md) covers the process: licence, sign-off, the pull request flow and what CI checks.

If you are an agent: read this whole file before your first edit, and follow it over any general habit of yours. Where it conflicts with an instruction from the person driving you, say so before you act.

---

## The repository

- **`orchestrator/`** is the backend: FastAPI, SQLAlchemy and Alembic, on Python 3.11.
  - `config.py` holds the settings.
  - `api/` holds the routers, `modules/` the feature modules and `core/` the shared code.
  - `services/` holds the services, `alembic/versions/` the migrations and `tests/` the tests.
- **`frontend/`** is the web app: Next.js 15, React and Tailwind 3.3, in TypeScript.
  - `lib/api-client.ts` is the only HTTP client.
  - `components/` holds the UI and `hooks/` the data hooks.
- **`services/`** holds separate containers, such as the workspace worker. Each has its own config module.
- **`docs/`** holds the product and operations docs; `docs/PRDS/` holds the requirements.
- **One codebase, two editions.** The local edition (`AUTH_EDITION=local`, `docker compose up`, no accounts) and the hosted edition (`saas`) run the same code behind one flag. Every change keeps both working.

This is a mature codebase, not a greenfield one. Most "new" work extends something that already exists.

---

## Before you write code

1. **Search first.** The table, endpoint, hook, component or tool you are about to create probably exists. The order is: reuse, then extend, then refactor, and build new last. A pull request that adds something new says why the existing piece wasn't enough.
2. **Read the code you will touch, and its tests.** Follow the patterns you find there.
3. **Plan anything bigger than a one-file change** before writing it: the surfaces involved, what you'll reuse, and your open questions.
4. **Ask before assuming.** If the task is ambiguous, or it looks like something that already exists, ask the person driving you or comment on the issue.
5. **Never shrink the scope silently.** If you did not finish part of the task, say exactly what is missing. Don't call it "follow-up work" as if that had been agreed.

---

## Hard rules

These rules fail CI or fail review.

- **Settings come from `orchestrator/config.py` only.** No `os.getenv` or `os.environ` anywhere else in the orchestrator: add a config attribute and read it through `config`. The linter enforces this (ruff `TID251`).
- **No hardcoded values.** Use named constants or config.
- **Replace, don't shim.** When you supersede a path, delete the old one in the same pull request. No `_legacy` suffixes, no `V2` copies of a hook, no compatibility branches.
- **Data lives in the database.** Personas, agent definitions and seed content are seeded into tables (`orchestrator/core/seeds/`), never read from files at runtime.
- **One extension point per kind.** A platform tool uses the 3-file pattern in [`orchestrator/modules/tools/README.md`](orchestrator/modules/tools/README.md). Don't add a new table or tool when an existing one can be extended.
- **The frontend calls the backend only through `apiClient`.** A raw `fetch('/api…')` is an ESLint error.
- **Module boundaries hold.** `orchestrator/.importlinter` defines which modules may import which, and CI checks it.
- **Migrations.**
  - There is exactly one Alembic head.
  - A new revision chains onto the current head.
  - Move the two head pins: `orchestrator/tests/test_prd209_alembic_single_head.py` and `orchestrator/tests/test_prd236_w1_routes.py`.
  - Every step must survive a database that `create_all` already built: keep an existing table and add only its missing indexes, use `IF NOT EXISTS` / `IF EXISTS`, and seed with insert-if-absent.
- **Routes.** Every backend route the frontend calls must be in the committed `orchestrator/reports/route-manifest.json`. Regenerate it with `cd orchestrator && python -m scripts.dump_routes`. The frontend route-contract check reads it.
- **Config names.** A new setting's name goes in `orchestrator/reports/config-surface.json`.
- **Editions.** Anything hosted-only is gated by `AUTH_EDITION` / `isSaaS`, never by role.
- **No blocking work on the event loop.** A route that touches the database without awaiting anything is a plain `def`, which FastAPI runs in its threadpool. It is never `async def` over a synchronous session.
- **Integrations go through Composio,** using the workspace's own connection. Add no new third-party API clients, and no new keys in Settings. Never weaken the Composio deny list (`orchestrator/core/composio/deny_list.py`).

### Canonical terms

Use these in identifiers and in user-facing copy.

| Use | Not |
|---|---|
| **Playbook** | recipe (legacy) |
| **Mission** | workflow, job |
| **Task** | (`BoardTask`; a mission's sub-tasks are `OrchestrationTask`) |
| **Deliverable** | output, workspace file, artifact (in user-facing copy) |
| **Knowledge Graph** | business graph |
| **Command Center** | activity |
| **Auto** | "the assistant": Auto is a name |

---

## Code shape

CI checks these on the lines your change touches.

| Rule | Limit | Checked by |
|---|---|---|
| Function length | 50 code lines (docstrings, comments and blank lines don't count). React components: 150 | `scripts/ci/check_changed_code_shape.py`, `frontend/scripts/check-changed-lines-eslint.js` |
| Nesting | 4 levels | the same two scripts |
| File size | a new file at most 800 lines, aiming for 200–400; don't grow a file that is already over 800 lines, split it | the same two scripts |
| Complexity | cyclomatic 10 | ruff `C901` |

- **Many small files over a few large ones,** organised by feature or domain.
- **Immutability.** Return new objects; don't mutate arguments or shared state. No mutable default arguments (ruff `B006`).
- **Readable names.** Type hints on public Python functions. Docstrings on public classes and functions that say what the code does and why.

---

## Errors and inputs

- **Never swallow an error.** An `except Exception` must log (`logger.exception`) and then re-raise, or return a clear failure (ruff `BLE001`).
- **Messages.** Users get a clear message that leaks no internals; the detail goes to the server log.
- **Validate at the boundary.** All external data (request bodies through Pydantic models, webhooks, third-party responses) is validated where it enters the system. Fail fast, with a clear message.
- **SQL is parameterised:** SQLAlchemy, or `text()` with bound parameters. Never build SQL with string formatting (ruff `S608`).
- **No `print` in application code.** Use `logging` (ruff `T20`).

---

## Security

- **No secrets anywhere:** not in code, tests, fixtures, docs or config. gitleaks scans every push. Use environment variables through `config.py`.
- **If you come across a secret,** stop. Don't copy it into a prompt, a file or a message. Report it the way [SECURITY.md](SECURITY.md) describes. The value is burned and must be rotated.
- **Every new endpoint is authenticated and authorised** with the existing request-context and workspace-permission dependencies.
- **Every query is scoped to the caller's workspace.** Tenant isolation is a hard rule.
- **Shell, deserialisation and TLS:**
  - no `shell=True` with anything a user controls (ruff `S602`, `S604`, `S605`);
  - no `pickle` or unsafe `yaml.load` of untrusted data;
  - TLS verification stays on.
- **Don't paste secrets, customer data or non-public client material into an AI tool,** yours or anyone else's.

---

## Tests

- **Prove every change.** Every feature and every bug fix ships with the test that proves it. The test fails without your change.
- **Where tests live.** Backend tests are in `orchestrator/tests/` (pytest; CI runs them against Postgres in the local edition). Frontend tests use vitest (`*.test.ts(x)`).
- **Never weaken, skip or delete a test,** or a CI check, to make a change pass.
- **Never claim a test passed** unless you saw it pass, on your machine or in CI. No invented results in commit messages or pull requests.

---

## Git and pull requests

- **Sign off every commit:** `git commit -s`. The DCO check fails without it.
- **Commit messages:** `type(scope): description`, where the type is one of `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.
- **Stage explicit paths.** Never `git add -A` or `git add .`: local env files, generated files and unrelated work end up in the commit.
- **Never force-push a shared branch,** and never commit to `main`.
- **Keep pull requests small:** one change each, ideally under about 400 changed lines, not counting tests and fixtures. Split bigger work.
- **Fill in the pull request template honestly,** including the AI-assistance disclosure and what you actually verified.

---

## Stop and ask a maintainer first

Open an issue before changing:
- a migration that alters or drops existing data;
- authentication, permissions or tenancy;
- billing, plans, quotas, or anything that spends money;
- the Composio deny list or any other guardrail;
- the removal of user data or of a public API route;
- a new Python or npm dependency. Justify it; the existing ones probably cover it.

---

## Running things

- **The development environment** is `docker compose up` (see [CONTRIBUTING.md](CONTRIBUTING.md#development-environment)).
- **Backend tests:** `cd orchestrator && pytest tests/<file>`. **Frontend tests:** `cd frontend && npm run test`.
- **CI is the source of truth.** Every pull request runs the lanes listed in CONTRIBUTING.md. Red is red, even on a lane marked non-required.
