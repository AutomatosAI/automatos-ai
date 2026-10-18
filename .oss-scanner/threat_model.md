# Automatos AI — threat model

Automatos is an AI operations platform: a FastAPI backend (`orchestrator/`), a Next.js
frontend (`frontend/`) and separate service containers (`services/`). Workspaces hold
agents, playbooks, missions, a knowledge base (RAG), documents, social posts and
connections to third-party apps through Composio. LLM agents act on that data with
tools, including tools that read the web, run code in a workspace and post to
connected accounts.

One codebase ships two editions behind one setting:

- **local** (`AUTH_EDITION=local`): `docker compose up` on the owner's own machine,
  no accounts. Every published port, the frontend's included, binds to `127.0.0.1`
  by default (`BIND_ADDRESS`). Whoever can reach the API or the frontend is the
  owner, by design.
- **hosted** (`AUTH_EDITION=saas`): multi-tenant, Clerk sign-in, many workspaces
  on one database.

## Where untrusted input enters

- **HTTP API** (`orchestrator/api/`, ~140 routers). Authentication and workspace
  resolution are in `orchestrator/core/auth/` (`hybrid.py` is the main dependency:
  Clerk JWT, `x-api-key`, widget/SDK keys, the `X-Workspace-ID` header).
- **Webhooks**: GitHub (`api/github_webhooks.py`, `api/codegraph.py`), Composio
  (`api/composio.py`), playbook webhooks (`api/workflow_recipes.py`), Shopify
  (`api/shopify.py`), verticals (`api/verticals.py`), voice (`api/voice_retell.py`).
- **Public widget** (`api/widgets/`, `api/widget_*.py`): anonymous visitors on a
  customer's site chat with an agent under a per-key scope map
  (`core/security/widget_scopes.py`).
- **Uploaded files and fetched documents**: knowledge uploads, attachments, social
  media uploads (`api/socials_media_upload.py`), PDF/OCR parsing, WeasyPrint
  document rendering with a pinned URL fetcher (`modules/documents/`).
- **Content the agents read**: web pages, emails, documents, tool results and
  connected-app data are attacker-controllable. Treat them as prompt injection
  that tries to make an agent call a tool it should not.
- **CLI hosts** (`api/cli_hosts.py`, `services/cli-host/`): a host process on a
  user's machine claims jobs and runs coding-agent CLIs.
- **Workspace worker** (`services/workspace-worker/`): runs a Claude Agent SDK
  session per workspace with file, shell and browser tools, confined to that
  workspace's directory (`canvas_confinement.py`, `workspace_manager.py`).

## What matters most

1. **Tenant isolation (hosted).** One workspace must never read, change or act
   through another workspace's data, files, credentials, connected accounts or
   agents. Check every query's workspace scoping, the `X-Workspace-ID` handling,
   background jobs and schedulers (APScheduler jobs carry workspace ids), Redis
   keys, object-storage keys and the workspace worker's path confinement.
2. **Authentication and authorisation.** Missing auth on a route, role checks
   (`core/auth/workspace_admin.py`, `super_admin.py`), API-key and widget-key
   scopes, and webhook signature checks (constant-time compare, missing-secret
   behaviour, replay).
3. **Agent tool boundaries.** Prompt injection that leads to a tool call outside
   the caller's grant: widget turns reaching owner-only tools, the Composio deny
   list (`core/composio/deny_list.py`) being bypassed, outbound sends or posts
   without the owner's approval, NL2SQL running anything but read-only SQL
   (`modules/nl2sql/`).
4. **SSRF.** Outbound requests from user- or model-supplied URLs must go through
   `core/security/url_validator.py`, `web_access.py` and `pinned_http.py`
   (private, loopback, link-local and metadata ranges; DNS rebinding; redirects).
   This includes WeasyPrint's URL fetcher and any webhook or image URL.
5. **Secrets.** Stored credentials (`core/credentials/`), OAuth tokens and API keys
   must stay encrypted at rest, never logged and never returned to a caller who
   does not own them.
6. **Code and command execution.** Shell commands built from user input, path
   traversal or symlink escapes in the workspace worker and file APIs, unsafe
   deserialisation, template injection in the media and document renderers.

Lower priority: the local edition's own owner attacking themselves; the frontend
port on the local edition (it is the owner's UI).

## How to exercise it

Everything is at `/src`. Python 3.11 and Node 20 are installed with every
dependency; no service is running.

- Start the database: `.oss-scanner/start-db.sh` (PostgreSQL 15 on 127.0.0.1:5432,
  `test`/`test`, database `test_db`, schema already created). Environment variables
  for the tests are set in the image.
- Backend tests: `cd orchestrator && pytest tests/<file> --timeout=60`. The whole
  suite is ~14,000 tests; run the files near the code you are testing.
  `tests/test_prd251w2_media_hosting.py` needs MinIO and fails without it.
- CLI host tests: `cd services/cli-host && pytest -q tests`.
- Frontend tests: `cd frontend && npm run test` (vitest).
- The API app is `orchestrator/main.py` (`uvicorn main:app`). Redis, Qdrant, MinIO,
  Clerk, Composio and every LLM provider are absent: mock them. Use synthetic
  workspaces, users, tokens and content. No real credentials are needed.

## How we rate severity

- **Critical**: unauthenticated access to another tenant's data or credentials;
  remote code execution on the backend or a worker; acting through another
  workspace's connected accounts.
- **High**: an authenticated user of one workspace reading or changing another
  workspace's data; SSRF reaching internal services or cloud metadata; a widget
  visitor reaching owner-only tools or data; secrets in logs or responses.
- **Medium**: privilege escalation inside one workspace (member to admin); prompt
  injection that makes an agent take an unapproved action within the same
  workspace; stored XSS in the app.
- **Low**: issues that need the local edition's owner to attack themselves,
  self-XSS, missing hardening with no demonstrated impact.

Rate on the attacker's real access and demonstrated impact, not the bug class.
Say which edition a finding affects.

## Out of scope

- Volume denial of service, social engineering, and bugs in third-party services.
- Dependency CVEs without a reachable call path in this code.
- `docs/`, `e2e/`, `graphify-out/`, `charts/` and test fixtures.

## Reports

Follow `SECURITY.md`. Include the affected file and route, which edition, an
offline reproduction (a pytest test is ideal) and a focused patch with a
regression test where you can.
