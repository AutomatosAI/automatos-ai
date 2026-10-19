# PRD-262: Container diet: smaller images and a self-host first run that pulls instead of builds

**Status:** Approved to build, 16 Oct 2026 (Gerard: "get it built"). §3 D1–D6 and the §9 questions are taken at their recommendations, and §9 records each answer. Dave confirms D1–D6 in PR review. The answers that shape the waves: US-006 locks `agent-opt-worker` down, because the FutureAGI account could not be checked (§9-Q4); W9 is held until after WebSummit (§9-Q11); TESTER's constraints from §9-Q1 are in §2. §9-Q14, added in review, is open. · **Owner:** Gerard (Dave confirms D1–D6 in PR review) · **Written:** 16 Oct 2026, grounded on `main @ 73f18f175` and on registry and CI measurements taken the same day (`edge` = `sha-73f18f1`). Revised the same day after reviewers fact-checked the draft.
**Source:** the 15 Oct engineering brief *Shrinking the Automatos containers* (claude.ai/artifact/JUEs6PmWZ4H2Fy1ErsEJo3). On 16 Oct, ten survey agents re-checked every claim against main, each checked by a second agent that tried to refute it. Where main or the numbers have moved since the brief, this PRD uses the 16 Oct figure.
**Scope:** automatos-ai only (Gerard, 16 Oct). Container findings in the other repos are listed in Appendix B for a later PRD.
**Type:** Refactor and consolidation, with extensions. It changes the existing Dockerfiles, compose files, Makefile, `images.yml`, the smoke lanes and the boot path. It deletes Gotenberg, the root `Dockerfile.backend`, `docker-compose.images.yml`, `promtail-config.yml` and its phantom profile, and `orchestrator/.railway-watch.json`. It locks `agent-opt-worker` down (US-006) instead of retiring it (§9-Q4; Appendix A stays the alternative). It makes six code swaps (camelot → pdfplumber, the Gemini SDK → Gemini's OpenAI-compatible endpoint, pandas → xlsxwriter and the stdlib, the langchain splitters → vendored code, tree-sitter-language-pack → the installed `tree_sitter_*` grammars, the worker's SQLAlchemy → psycopg2). It replaces the backend's shell entrypoint and healthcheck with Python (US-059) and adds one adapter (Leiden through `graspologic-native`). No new service, datastore or vendor.

## 1. Introduction

**What a self-hoster does today.** The documented default is `make up` (README, QUICKSTART, SETUP), which runs `docker compose up -d --build` (`Makefile:63`). It builds three images from source on the user's machine:

- the backend's *development* stage, with the source bind-mounted and `uvicorn --reload` (`docker-compose.yml:154-161`, `:297-299`; `orchestrator/Dockerfile:139`);
- the frontend;
- the workspace worker.

On a CI runner with no cache, the backend and worker take 2:00–2:30 to build, and a cold frontend build takes another 5–6 minutes (`npm install` 196–246 s, `next build` 90–105 s). SETUP.md tells users to expect 5–15 minutes. The published-images path (`make up-images`) pulls **1,330 MB** (amd64) or **1,284 MB** (arm64) and is promised as "about a minute instead of ten" (QUICKSTART:112). No CI lane has ever booted the published images.

**Why the images are big.** Measured 16 Oct, compressed, amd64:

| Image | Size | Where the bytes go |
|---|---|---|
| `automatos-api:edge` | 520.5 MB | Python packages are 361 MB (69%). About 400 MB installed goes with the unused and test-only packages, opencv (pulled in only so camelot can import) and a deprecated Gemini SDK with 112 MB of discovery documents. Debug info (92.5 MB) and `tests/` directories (108.5 MB) inside site-packages, plus Debian ghostscript (57.7 MB), add roughly 250 MB more. |
| `automatos-workspace-worker:edge` | 506.6 MB | Three JavaScript runtimes: Node 20 from NodeSource, Playwright's bundled Node 22 (120 MB), and the Claude CLI's Bun (253 MB in the published SDK 0.2.165; 268 MB in the current 0.2.168). NodeSource also brings a second Debian Python, C headers and a 31 MB npm cache. The local edition installs Playwright although it has no browser to drive. |
| `automatos-frontend:edge` | 93.8 MB | The standalone build traces in both glibc and musl copies of libvips plus TypeScript. The runtime carries npm, yarn and corepack (36 scanner findings), on a Node 20 base frozen since 15 April. |
| `media-render` (not published) | 1.81 GB on disk | About 200 MB of Mesa/LLVM that headless rendering never loads, 220 Chrome locale files, two copies of the espeak data, an onnxruntime library nothing links, and a 326 MB fp32 voice model. |

**Why boot is slow.** Container start to `/health 200` takes 30–50 s in CI on a fresh database (median 45 s, 9 runs). A warm restart of Gerard's local stack on 16 Oct took **108 s**. Most of that time is avoidable:

- `alembic upgrade heads` runs twice per compose boot: once in the entrypoint (`docker-entrypoint.sh:156`) and again in the CMD (`Dockerfile:139`, `:195`).
- A 38 s seed process rewrites all 418 credential types on every boot (`load_seed_data.py:34-50` always sets `updated_at`).
- Every worker imports cv2, pandas and `google.generativeai` at boot.
- `import main` downloads a tiktoken file from the internet with no timeout (`modules/rag/service.py:24-29`), so an offline host cannot start the API.
- `/health/ready` can never return 503 under uvicorn 0.24, which runs the lifespan before it listens, so the smoke lane measures liveness and readiness at the same instant.

**Why updates are expensive.** A code-only API update re-downloads 14.7 MB, of which 9.3 MB is the `pip uninstall` layer and the `chown -R` copy of `/app`. A frontend code change re-downloads 38.2 MB, including a 9.3 MB `public/` layer whose content did not change. A comment-only edit to `requirements.txt` (`sha-91a33c4`) cost every `:edge` user 375 MB. A re-pushed base tag with byte-identical layers (`sha-0fe19d8`) cost 474 MB.

**Why patching fails.**

- The published images carry fixable Criticals: api 6, worker 8, frontend 7 (Grype, run 38915020775, 16 Oct).
- The scan runs only after the push, only on main, and with `continue-on-error` (`images.yml:170-179`).
- All six API Criticals are Debian curl packages already fixed in `deb13u7`. The GHA layer cache keeps serving the 6 Oct apt layer until Docker republishes `python:3.11-slim`.
- One finding is on the CISA KEV list (starlette 0.41.3), but Grype rates it Medium, so a High cutoff would never block it.
- Node 20 has been out of support since 30 April.
- Nothing updates base images or Actions.

**Why production is a different artefact.** There are three API images. Railway builds its own with the Leiden extras (about 585 MB more). GHCR publishes one without them. Compose builds the development stage. Only the GHCR image is signed and scanned. `railway.json` sets `dockerfileTarget`, a key Railway's schema does not have, so production ships only because `production` is the last stage. `railway.json` has had no healthcheck since 15 April (the Railway UI is unchecked, §10).

This PRD makes the default self-host path pull small, scanned images and boot them in seconds. It measures all of it in CI, because nothing runs on this machine.

## 2. Goals

- `git clone --depth 1 … && make up` gives a working UI with **no local image build** on a clean checkout of any main commit whose images CI has published (FR-1). The three secrets are generated, not typed.
- The default first pull is **≤ 1,000 MB** amd64 (today 1,330 MB), and each image stays under a size budget CI enforces.
- Backend boot: **≤ 30 s fresh** and **≤ 15 s warm** to `/health 200` in CI, with one migration run (none on a warm boot at head) and seeds skipped when nothing changed.
- No published image carries a fixable Critical or a KEV match unless it is allowlisted with a reason and an expiry. PRs that touch an image are scanned before merge. A weekly uncached rebuild brings OS fixes to `edge` and to every `sha-<7>` built after it, and a checkout picks them up with `git pull && make up` (US-011, US-036).
- One API image, from CI to production (D2; W9, after WebSummit).
- Night and retest tooling keeps working (TESTER's answer to §9-Q1). None of it needs `--reload`. The production-stage backend keeps the `/app/logs` volume and its `env_file`, so editing `envs/api.local` and running `docker compose up -d --no-deps backend` still applies the change. `docker compose up --build` still builds a stack from an unmerged ref, and `AUTOMATOS_IMAGE_TAG` gives that build a non-official tag such as `local-<ref>`. Every image, pulled or built locally, carries the `org.opencontainers.image.revision` label, which the workspace's `scripts/bootstrap.sh` (outside this repo) will read instead of the bind-mount path.
- Every bar in §8 is measured by a CI lane. The size, scan and smoke-lane boot bars run on every relevant PR and on main. The update-delta bars (US-001) and the `smoke-images` bars (images built by `make up`, first pull, clone to UI, published-image boot) run on every main publish.

## 3. Decisions for Dave (written at the brief's recommendations)

Gerard took D1–D6 at these recommendations on 16 Oct ("get it built"). Dave confirms them in PR review.

**D1. The vulnerability scan blocks.** PRs that change an image's inputs (its Dockerfile and build context, `images.yml`) or what it is judged by (`.github/image-budgets.json`, `.grype.yaml`) are built and scanned before merge. The scan fails on any fixable Critical, and on any KEV-listed match whatever its severity. Exceptions live in `.grype.yaml`, each with a reason and an expiry of 90 days or less. A weekly uncached rebuild pulls in OS fixes. Dependabot covers Dockerfiles, compose images and Actions, and the pins move to digests in the same change. media-render (§9-Q6) and agent-opt-worker (kept, §9-Q4) are built and scanned the same way, without publishing. *(US-006, US-011, US-012, US-013, US-054)*

**D2. Production runs the image we publish.** Railway deploys the signed, scanned GHCR digest instead of building its own. This needs one API image first, which Leiden through `graspologic-native` (US-027) provides. The worker and the SaaS frontend are published as variants (§9-Q5). The deploy job is gated by a GitHub environment approval. W9, which makes this change, is held until after WebSummit (§9-Q11). *(US-055 to US-057)*

**D3. Trim now, harden later.** Dependencies are trimmed and Node moves to 24 now (Node 22 ends April 2027, Node 24 April 2028). The frontend moves to a hardened or distroless Node base once production deploys the GHCR image. The backend goes last, on Python 3.12. Its start-up script and health check move to Python first, which takes curl and the Postgres client out of the image. A shell-less backend base is a separate question (§9-Q10). *(US-007, US-058, US-059, US-061)*

**D4. The media renderer uses an fp16 voice model after a listening test.** Two candidates: the v1.1 release's `kokoro-v1.0.fp16.onnx` (−162 MB) and the v1.0 release's fp16 (−148 MB).

- The v1.1 file has an upstream quality figure (0.999 spectral correlation against fp32), but it is a different graph (`input_ids` → `waveform` + `duration`). `kokoro-onnx` 0.6.1 adds sentence and clause pauses when it sees a `duration` output, which changes line timing unless we pass `sentence_pause=0, clause_pause=0`.
- The v1.0 file keeps today's graph (`tokens` → `audio`) but has no published quality figure.

*Recommendation: the v1.1 fp16 with explicit zero pauses, if it passes the listening test.* *(US-050)*

**D5. The worker's toolchain is part of the product.** `gcc`, `make`, Node and pnpm stay in the default worker image, because agents build code in their workspaces. What goes: NodeSource and its extras (the official Node 24 replaces them), Playwright's bundled Node, Playwright in editions that have no browser, and SQLAlchemy. The Claude CLI's Bun runtime stays for now: dropping it from the local edition stops the local "Auto session" canvas, so it is §9-Q14. *(US-007, US-043 to US-045)*

**D6. What's dead is retired.** In this repo that means Gotenberg (a 705 MB image whose only client has no callers), the root `Dockerfile.backend`, `promtail-config.yml` with its phantom `observability` profile, and `orchestrator/.railway-watch.json` (no reader). mem0, `automatos-testing` and `automatos-voice` live in other repos (Appendix B). `agent-opt-worker` is live code, so it stays and is locked down (US-006), because the FutureAGI account could not be checked (§9-Q4). Retiring it (Appendix A) stays the alternative. *(US-018; `.railway-watch.json` in US-010)*

## 4. Sequencing

Each wave is one branch, one push, one PR and one CI run. Stories commit inside the wave. Exception: US-004 is a follow-up PR (W0b), opened after W0's lanes have run on five consecutive main pushes. W1 may branch once W0 merges, but it merges only after W0b has recorded the baseline medians, so the "Today" column holds pre-diet numbers.

| Wave | Stories | Depends on | Touches |
|---|---|---|---|
| W0 Measure | US-001 – US-003 | — | `images.yml`, smoke lanes, `scripts/ci/` |
| W0b Baseline | US-004 | W0 + five main pushes | §8, `.github/image-budgets.json` |
| W1 Patch what's exposed | US-005 – US-013 | W0 (merges after W0b) | compose ports, `agent-opt-worker`, Node 24, web stack, frontend lock, CI policy |
| W2 API clean-up, no behaviour change | US-014 – US-022 | W1 | `orchestrator/Dockerfile`, requirements, `.dockerignore`, dead code and config, deletions |
| W3 API code swaps | US-023 – US-028 | W2 | tables, Gemini, pandas, splitters, Leiden, code-graph grammars |
| W4 API boot | US-029 – US-034 | W3 | entrypoint, seeds, imports, lifespan |
| W5 Pull-first default stack | US-035 – US-042 | W4 | `docker-compose*.yml`, `Makefile`, `.env.example`, docs, chart |
| W6 Worker diet | US-043 – US-045 | W1 | `services/workspace-worker/` |
| W7 Frontend diet | US-046 – US-049 | W1 | `frontend/` |
| W8 Media renderer | US-050 – US-054 | W1 | `services/media-render/`, `images.yml` |
| W9 One image to production (D2) | US-055 – US-057 | W3, W4, W6, W7; after WebSummit (§9-Q11) | `images.yml`, Railway (operator) |
| W10 Bases and budgets | US-058 – US-062 | W2–W9 | base images, `images.yml` |

- W6, W7 and W8 touch separate directories and can run as parallel lanes after W1, alongside the W2–W5 chain. They share `images.yml`, `.github/image-budgets.json` and `test.yml`'s `prod-image-parity` job (US-027, US-043 and US-047), so each lane gets a seam review before its push (the 15 Oct parallel-lanes lesson). The lanes merge in any order, so the first-pull bar, which the API, worker and frontend diets all feed, gates from the last of W3, W6 and W7 to merge (§8).
- W9 changes how production deploys. It is held until after WebSummit (10–13 Nov) (§9-Q11, Gerard, 16 Oct). W10 depends on W9, so it lands after WebSummit too.
- BuildKit cache mounts: a cache mount fails a Railway build unless it carries a hard-coded `id=s/<service-id>-<path>`. The service ids are unknown (§10), so no Railway-built Dockerfile (orchestrator, frontend, workspace-worker, media-render, agent-opt-worker) gets a cache mount until they are known. The stories that propose one (US-044, US-047) say so.
- Stories cite `file:line` as of `697ca7d6e` (the frontend's `next.config.js` and `app/layout.tsx` as of `73f18f175`). A story whose file an earlier wave rewrites names the stage, function or key instead.

## 5. User stories

### Part 1: Measure (W0)

#### US-001: Every image build reports its size, its layers and its update cost
**Description:** As Gerard, I want each image build to report what it weighs and what it costs an existing user to update, so that growth is visible on the PR that causes it.

**Acceptance criteria:**
- [ ] PR builds in `images.yml`, and branch builds started by `workflow_dispatch`, push by digest to an ephemeral local registry (a `registry:2` service on `localhost:5000`, buildx `driver-opts: network=host`), so they are measured without writing to GHCR. Main builds keep pushing to GHCR as today.
- [ ] The `pull_request` paths (`images.yml:41-50`) add `frontend/next.config.js`, `frontend/csp-sources.js`, `frontend/.npmrc` and `frontend/public/**`, which change the frontend image or its dependency tree, and `.github/image-budgets.json`. A changes step limits the PR build matrix to the images whose inputs changed, so a frontend-only PR builds only the frontend (both arches).
- [ ] After each build, a step reads the compressed size per image and arch. Because `images.yml` builds with `provenance: mode=max` and `sbom: true` (`:144-145`), `docker buildx imagetools inspect --raw "$REF@$DIGEST"` returns an index, not a manifest. The step selects the `.manifests[]` entry whose `.platform.architecture` is the build's arch, skipping the `unknown/unknown` attestation manifest, then inspects that manifest and sums its `[.layers[].size]`. It writes a table to `$GITHUB_STEP_SUMMARY` with per-layer sizes and the Dockerfile instruction that made each layer.
- [ ] On every build, PR and main, a step diffs the new layer digests against the current `edge` manifest (public on GHCR) and reports the **update delta**: the MB an existing user downloads. A requirements change is reported separately from a code-only change. Images with no `edge` (agent-opt-worker, media-render) report size only.
- [ ] `.github/image-budgets.json` holds one budget per image and arch. While this story is report-only, an over-budget build prints a warning. US-004 switches it to fail. A budget is raised only in the same PR, with a `reason` field.
- [ ] Each build job uploads its layer digests and sizes as an artifact, because each image and arch builds in its own matrix job (`:66-99`) and a PR's local registry ends with its job. A `report` job (`needs: build`) computes which layers the API and worker share (today only the 29.8 MB Debian layer) and adds it to the summary. When a PR builds only one of the two, the other side is its current `edge` manifest. US-058's check reads this job.
- [ ] CI: a unit test feeds the delta script two recorded manifests (`edge`, and `edge` plus a 10 MB `public/` layer) and asserts +10 MB, and the budget check warns on the larger one. A static test asserts the `pull_request` paths include `frontend/public/**`. A throwaway fixture PR is not used: it would be a second PR outside the wave (§4).

#### US-002: The fresh-clone smoke measures the first run properly
**Description:** As Gerard, I want the smoke lane to time a fresh boot and a warm restart to the second, so that the boot bars in §8 have honest numbers behind them.

**Acceptance criteria:**
- [ ] `scripts/ci/smoke-fresh-clone.sh` polls every 1 s, not 5 s (`:40`). It times from `docker inspect -f '{{.State.StartedAt}}' automatos_backend`, not from its own sleeps.
- [ ] A warm phase runs `docker compose restart backend` (database kept) and times it the same way. The median of two warm restarts is reported.
- [ ] The lane also starts the frontend. It asserts `:3000` answers 200 and records the address `docker compose port frontend 3000` reports. In W0 that is still `0.0.0.0:3000` (`docker-compose.yml:375` has no host IP). US-005 (W1) turns it into the `127.0.0.1` assertion.
- [ ] The step summary always shows, not only on failure (`:113-116`): Started→Healthy per service from compose's timestamps, time to `/health 200` and to `/health/ready 200` (separately), the `/health/bootstrap` stage table, and the count of alembic `Context impl PostgresqlImpl` lines per boot.
- [ ] `smoke-fresh-clone.yml`'s path filters add `services/workspace-worker/**`, `docker-compose.dev.yml`, `Makefile`, `frontend/Dockerfile`, `orchestrator/requirements*.txt`, the lock files, `orchestrator/main.py`, `orchestrator/core/seeds/**`, `orchestrator/core/database/load_seed_data.py`, `orchestrator/alembic/**` and `envs/*.defaults`. A nightly schedule is added. Actions are SHA-pinned (`actions/checkout@v4` at `:56` is not today).

#### US-003: A lane that boots what we publish
**Description:** As Gerard, I want CI to pull and boot the published images the way a self-hoster does, so that the first-run promise is a measured number.

**Acceptance criteria:**
- [ ] A `smoke-images` job runs after `images.yml` publishes on main (`workflow_run`) and on `workflow_dispatch` (with a tag input), on `ubuntu-24.04` and `ubuntu-24.04-arm`, starting from an empty Docker cache.
- [ ] Until US-036 lands, it runs today's published path (`docker compose -f docker-compose.yml -f docker-compose.images.yml up -d --no-build` with `AUTOMATOS_IMAGE_TAG=sha-<7>`). After US-036 and US-037 land, it runs `make up` on a clean checkout with no secrets exported.
- [ ] It records: bytes pulled, pull seconds, fresh boot (container start to `/health 200`), warm restart, and time to `:3000 200`. It also asserts that no image was built:
  - backend, frontend and workspace-worker each run a `ghcr.io/automatosai/automatos-*` image whose `RepoDigests` holds the index digest the tag resolves to on GHCR (a locally built image has no `RepoDigests`);
  - no image tag matches `local-*`, and `docker image ls` lists no compose-built `automatos-ai-*` image.
  The datastores come from their own registries (Docker Hub, `cgr.dev`), so they are not part of the check.
- [ ] Each app image carries `org.opencontainers.image.revision` equal to the commit it was built from. `images.yml`'s metadata labels set it today. The workspace-root `scripts/bootstrap.sh`, outside this repo, reads that label instead of the bind-mount path (§9-Q1).
- [ ] On pull requests that touch compose, the Makefile, `envs/` or an image input (US-001's paths), a PR variant builds the three amd64 images in the same job, with the `images.yml` GHA cache and the same labels. It tags them with the GHCR names and `AUTOMATOS_IMAGE_TAG=pr-<sha7>` and runs the same boot steps with `--no-build`. It measures the boot bars. The pull bars (bytes pulled, pull seconds, no local build) can only be measured after a publish, so they come from the main runs alone.
- [ ] This is the first CI boot of the **production** stage. It asserts the backend runs without `--reload` and with the expected worker count.
- [ ] `edge` moves only after this lane passes. The merge job pushes `sha-<7>` but no longer `edge`. A final `promote-edge` job, after smoke-images passes on both arches for that sha, runs `docker buildx imagetools create -t …:edge …:sha-<7>`, which keeps the signed digest. So an image that cannot boot never reaches `edge`, which plain `docker compose up` and the Helm chart pull by default. Schedule runs (US-011) promote their `weekly-<yyyymmdd>` tag the same way.
- [ ] Report-only until US-004. A failing boot fails the job from day one.

#### US-004: Baseline, then the bars
**Description:** As Gerard, I want today's numbers recorded before any diet work, so that every later story is judged against them.

**Acceptance criteria:**
- [ ] US-004 runs after W0 merges, on normal main traffic. `images.yml` (US-001) and `smoke-images` (US-003) run on 5 consecutive main pushes. `smoke-fresh-clone` (US-002) runs on 5 nightly or dispatched runs, because its path filter skips most main pushes. The medians fill the "Today" column in §8 of this file.
- [ ] Gerard confirms or moves each bar in §8. `.github/image-budgets.json` is set to today's size plus 2%, and its check flips from warn to fail.
- [ ] The budget values, the §8 medians and the warn-to-fail flip ship as the first commit of W1. W1 may branch once W0 merges, but it merges only after Gerard has confirmed the bars, so the "Today" column holds pre-diet numbers.
- [ ] The boot and first-run bars stay report-only and flip to fail in the wave that delivers each one (named in §8).

### Part 2: Patch what's exposed (W1)

#### US-005: Every compose port listens on loopback unless the owner opens it
**Description:** As a self-hoster, I want every published port bound to `BIND_ADDRESS`, so that nobody on my network can chat with my backend as the local owner.

**Acceptance criteria:**
- [ ] The frontend (`docker-compose.yml:375`), Adminer (`:531`) and Gotenberg (`:548`) ports gain the `${BIND_ADDRESS:-127.0.0.1}:` prefix. Gotenberg's line then goes with the service in US-018.
- [ ] `orchestrator/tests/test_prd234_s1a_compose_bind.py` checks **every** `ports:` entry in every compose file, not five named variables (`:31-37`).
- [ ] CI: the smoke lane asserts `docker compose port frontend 3000` returns `127.0.0.1:3000` (the address US-002 records in W0).

#### US-006: agent-opt-worker answers only the platform
**Description:** As Gerard, I want the optimisation worker to accept only calls from our API, so that nobody else can spend our evaluation and model quota. (§9-Q4: the FutureAGI account could not be checked, so this story is built. Retirement, Appendix A, stays the alternative.)

**Acceptance criteria:**
- [ ] Every route except `/health` and `/metrics` requires `X-Internal-Token` equal to an env secret. The check is a router-level FastAPI dependency using `hmac.compare_digest`, as at `services/workspace-worker/worker_http.py:69`. It does not copy that middleware's path exemption (`:62`) or its skip when the token is empty (`:66`). It must not be a path comparison, because starlette 0.41.3 lets a forged Host header change `request.url.path`. The worker refuses to start when the token is unset or empty.
- [ ] `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)`. `GET /test` is deleted.
- [ ] `/metrics` stays outside the token check. Once the public domain is removed (operator step below), it is reachable only on Railway's private network. The automatos-monitoring scrape (`prometheus-railway.yml:42-51`, Prometheus v2.51.0) sends no header, and a 401 there would fire its `ServiceDown` rule. If `/metrics` must carry a credential, it accepts `Authorization: Bearer <token>`, the only scrape credential Prometheus 2.51 can send (custom `http_headers` needs 2.55+), and the scrape's `authorization` block is set in the same deploy window.
- [ ] Inputs are capped:
  - `metrics` must be known template names, at most 11 (today `ThreadPoolExecutor(max_workers=len(metrics))`, `main.py:232`, `:310`).
  - `teacher_model` is allow-listed.
  - `dataset` holds at most 10 items (what `futureagi_service.py:171` sends).
  - Concurrent `/optimize` jobs are bounded.
- [ ] `orchestrator/core/services/futureagi_service.py` sends the header. The per-turn live-eval hook (`consumers/chatbot/service.py:1299-1310`) reads a cached "any prompt has scoring on" flag instead of querying the database on the event loop each turn. It closes its session before awaiting the worker, so no pooled connection sits idle for up to 120 s.
- [ ] `run_assessment` (`futureagi_service.py:350`) likewise commits and closes its session before awaiting the worker (assess, safety, or `optimize_prompt`, which polls for up to 1,500 s, `:193`). It opens a new session to write the result. Today it holds one pooled connection (pool 10 + overflow 20) and one threadpool token for up to 25 minutes. A test asserts that no session is checked out while `optimize_prompt` polls.
- [ ] The Prometheus middleware labels requests by route template, not raw path, so unmatched paths fall into one `unmatched` label. The fix goes into `services/shared` and every vendored copy, and a byte-identity test keeps the copies in sync.
- [ ] **Dependencies:**
  - `services/agent-opt-worker/requirements.txt` becomes `requirements.in`. `requirements.lock` is generated with `uv pip compile --generate-hashes` and installed with `--require-hashes`.
  - The PyPI package `fi` is kept out, by a uv override or by installing `agent-opt` with `--no-deps` and listing its real dependencies. `agent-opt` 0.0.1 pulls `fi>=0.0.2`, an unrelated 2019 sdist that runs `setup.py` at build. A CI grep fails on `^fi==` in the lock. An in-image smoke runs `python -c 'import fi.evals, fi.opt.optimizers'`, the modules ai-evaluation and agent-opt provide.
  - `uvicorn[standard]` becomes `uvicorn` (17.8 MB installed).
  - fastapi and starlette move past GHSA-86qp-5c8j-p5mr as in US-008, if the resolver allows. If it does not, the KEV row is allowlisted (US-013) with the resolver conflict as its reason.
  - The Dockerfile gets no BuildKit cache mount. Railway builds it, and a cache mount fails a Railway build unless it carries a hard-coded `id=s/<service-id>-<path>`. The service ids are not known yet (§10).
- [ ] **Scan (D1):** `images.yml` builds `services/agent-opt-worker` for amd64 on PRs that touch it and on main, and pushes it to the local registry (US-001). Its PR paths add `services/agent-opt-worker/**`. It applies the US-013 scan policy and a size budget in `.github/image-budgets.json`. It is not published to GHCR. litellm 1.80.0's three Criticals cannot clear while ai-evaluation 1.1.0 requires `openai<2` (litellm 1.80.5 and later need `openai>=2.8.0`), so they go into `.grype.yaml` with that reason and an expiry of 90 days or less.
- [ ] A new test lane (path filter `services/agent-opt-worker/**`) builds the image and runs TestClient tests:
  - every route but `/health` and `/metrics` returns 401 with no header, with an empty or a wrong `X-Internal-Token`, and with a forged Host header;
  - start-up fails when the token is empty;
  - `/docs` and `/openapi.json` return 404;
  - more than 11 or unknown `metrics` returns 422;
  - a non-allow-listed `teacher_model` returns 422.
- [ ] §9-Q4's data question is answered before W1 ships. Live scoring sends every tenant's chat turn and the system prompt to FutureAGI (`consumers/chatbot/service.py:1299-1310` → `futureagi_service.py:235-268`). `_collect_optimization_dataset` (`futureagi_service.py:308-330`) sends the newest user/assistant pairs from all workspaces, with no workspace filter, to FutureAGI and OpenAI. Gerard's answer becomes an acceptance criterion here (for example a scoped or opt-in dataset). If the answer is that this transfer is not acceptable, Appendix A replaces this story.
- [ ] Operator: set the token on both Railway services in one deploy window, the worker's before its new image deploys (it refuses to start without one). Point `AGENT_OPT_WORKER_URL` at `*.railway.internal` and remove the public domain. CI proof afterwards: Railway's commit status for `Automatos AI - agent-opt-worker` no longer names an `*.up.railway.app` domain.
- [ ] If PRD-241 US-005 (on the unmerged `feat/prd-241-w1-secret-disclosure`) merges first, this story checks it against the criteria above and closes the gaps.

#### US-007: Node 24 in every image and CI lane, from the official images
**Description:** As Dave, I want every Node in our images on a supported release and installed one way, so that we stop shipping an end-of-life runtime and NodeSource's extras.

**Acceptance criteria:**
- [ ] **Frontend:**
  - The `base` stage, which the development and builder stages build on, moves to `node:24-alpine3.24`, pinned as `name:tag@sha256:<digest>` (US-012).
  - The runtime stage is `alpine:3.24` plus `libstdc++`, plus only `/usr/local/bin/node` copied from the `base` stage, so the Node pin sits on a FROM line that Dependabot updates. It carries no npm, yarn, corepack, C headers or curl.
  - The HEALTHCHECK uses busybox `wget` on `/favicon.ico`, which skips middleware and SSR.
  - The `base` stage drops `apk add python3 make g++ curl` (`frontend/Dockerfile:19-23`): no install script compiles on Linux.
- [ ] **Worker** (`services/workspace-worker/Dockerfile:12-16`) and **media-render** (`services/media-render/Dockerfile:41-51`):
  - Node comes from a named stage, `FROM node:24-trixie-slim@sha256:… AS node`, and `COPY --from=node /usr/local/bin/node …` plus npm and corepack, not `/usr/local/include`. The pin sits on a FROM line that Dependabot (US-012) updates; a bare `COPY --from=<image>` is never read by it.
  - NodeSource, its extra apt updates, Debian `python3.13` and gnupg are gone.
  - In the worker, pnpm is pinned exactly to the current 12.x (12.14.0 on 16 Oct; §9-Q9), and `npm cache clean --force` runs in the same RUN, so the 31 MB `/root/.npm` cache is gone.
- [ ] CI's `setup-node` moves to 24 (`test.yml:1025`, the only `node-version` in the workflows today).
- [ ] Tests that pin the old Node move with it:
  - `orchestrator/tests/test_prd251w1_media_render_image.py:68-71` (`test_hyperframes_is_pinned_exactly_on_node_22`, which asserts the NodeSource `setup_22.x` string at `:70`) is updated and renamed for Node 24.
  - `services/media-render/tests/test_image.py:38-41` (`test_hyperframes_is_pinned_exactly_and_runs_on_node_22`, `startswith("v22.")`), which the media-render job runs inside the image (`media-render.yml:94-95`), moves to `v24.` and is renamed.
  - `services/media-render/media_render/versions.py:39` (`npm root -g`) is unchanged here, because npm and the global Hyperframes install stay. It changes in US-053, when Hyperframes moves to the `js` stage's `npm ci`.
- [ ] CI: in each image `node -v` prints `v24.`. On the Debian images, `dpkg -s python3.13`, `dpkg -s nodejs` and `dpkg -s gnupg` each fail (one package per check). In the frontend image, `command -v npm`, `command -v curl`, `command -v yarn` and `command -v corepack` each fail (one name per check). Grype no longer reports the npm-bundled findings (36, including `tar` 6.2.1) or the Node 20 binary findings.
- [ ] All renders and the media-render pre-warm pass in a `media-render.yml` run dispatched on the wave branch (the workflow has no PR trigger, `media-render.yml:7-8`), and the PR records its run id.

#### US-008: Web stack on supported versions (FastAPI, starlette, uvicorn)
**Description:** As Dave, I want the KEV-listed starlette replaced without repeating the F092 cold-boot router loss, so that the API is patched and still serves every route.

**Acceptance criteria:**
- [ ] `fastapi` moves to the current release and `starlette` to ≥ 1.0.1, which fixes GHSA-86qp-5c8j-p5mr. The pins and the F092 comment at `orchestrator/requirements.txt:2-8` are updated.
- [ ] `uvicorn[standard]` moves from 0.24.0 to the newest release that still accepts `websockets==12.0`: 0.50.1 per PyPI metadata on 16 Oct (0.50.0–0.50.1 need `websockets>=11.0`; 0.50.2 onward need `>=13.0`). It brings the multiprocess manager that restarts dead workers (0.30.0).
- [ ] `reports/route-manifest.json` is unchanged by the bump, and the cold-boot route test passes: every router in `router_manifest.py` answers on a fresh process.
- [ ] CI: full `test.yml`, the smoke lane, and a kill-a-worker step in smoke-images' PR variant (US-003). That variant runs on this wave's PR, because `orchestrator/requirements.txt` changes, and boots the production-stage image through compose with no `/app` bind mount. The production CMD still runs `--workers 4` (`Dockerfile:195`). The step kills one uvicorn worker process inside the container and sees a replacement `Started server process` and `/health 200` within 30 s, with no container restart. Once US-029 takes the worker count from `WEB_CONCURRENCY`, the step sets `WEB_CONCURRENCY=2`, because a single uvicorn process has no manager to restart it. `prod-image-parity` cannot host this step: it has no Postgres, and the production CMD runs alembic before uvicorn.

#### US-009: Frontend lockfile refreshed: sharp patched and dead dependencies gone
**Description:** As Dave, I want the frontend lockfile regenerated once, so that it records the right native builds and stops installing what nothing imports.

**Acceptance criteria:**
- [ ] `sharp` resolves to 0.35.5 (allowed by `next@15.5.27`'s range). The lock is regenerated with a current npm, so the `@img/*` entries record `libc` and the glibc libvips is no longer installed on Alpine.
- [ ] Prisma is removed: `frontend/lib/db.ts` (nothing imports it), `frontend/prisma/`, and the `prisma` block in `package.json:15-17`, whose `scripts/seed.ts` does not exist. With it goes the 14.8 MB engine download on every cold install.
- [ ] These unused dependencies are removed (none is imported, required or configured; together with `prisma` and `@prisma/client` above, the lockfile reachability walk frees 85 packages): `@anthropic-ai/sdk`, `@floating-ui/react`, `@headlessui/react`, `@hookform/resolvers`, `@tiptap/react`, `@tiptap/starter-kit`, `dotenv`, `lodash`, `react-grid-layout`, `@types/react-grid-layout`, `tailwind-scrollbar-hide`.
- [ ] CI: frontend-ci (tsc baseline, Vitest, ESLint) and the images build pass. Grype reports no `sharp` finding.

#### US-010: CI proves production is the last stage
**Description:** As Dave, I want a test that fails when a stage is added after `production`, because Railway builds the last stage and ignores `dockerfileTarget`.

**Acceptance criteria:**
- [ ] A test parses the `FROM … AS` lines of `orchestrator/Dockerfile`, `frontend/Dockerfile` and `services/media-render/Dockerfile` and asserts the last stage is `production`. It fails on a fixture with a stage after `production`.
- [ ] `dockerfileTarget` is removed from `railway.json:6`, and the assertion of it at `orchestrator/tests/test_dockerfile_prod_parity.py:85-88` is removed.
- [ ] `orchestrator/.railway-watch.json` (no reader; not a Railway file) is deleted.

#### US-011: A weekly uncached rebuild brings in OS fixes
**Description:** As Dave, I want the images rebuilt from fresh packages every week, so that Debian and Alpine security fixes reach `edge`, and every `sha-<7>` built after it (the uncached run refreshes the layer cache), without a code change.

**Acceptance criteria:**
- [ ] `images.yml` gains `schedule: cron '0 5 * * 1'`. On schedule runs, `no-cache-filters` names the stages that install OS packages. The worker's single stage gets a name so it can be filtered.
- [ ] The frontend runtime stage runs `apk upgrade --no-cache` in the same RUN as `apk add libstdc++` (US-007), and that stage is in `no-cache-filters`. Without it, a digest-pinned Alpine base (US-012) keeps its OpenSSL packages until Dependabot moves the digest.
- [ ] `images.yml`'s `workflow_dispatch` gains a `no_cache` input that applies the schedule's `no-cache-filters`. On a branch it builds, pushes to the local registry and scans like a PR (US-001, US-013), and never publishes.
- [ ] Schedule runs tag `weekly-<yyyymmdd>`. They never re-push a `sha-<7>` tag (metadata-action's `type=sha` is disabled for schedule events), so sha tags stay immutable. On schedule-triggered runs, `smoke-images` (US-003) boots the `weekly-<yyyymmdd>` tag, and `promote-edge` moves `edge` to it only after that passes.
- [ ] CI: a `no_cache` dispatch on the wave branch shows zero fixable Critical `deb` matches for the API and worker, with the curl rows (`deb13u5`, fixed in `deb13u7`) gone, and zero fixable Critical `apk` matches for the frontend. The first schedule run on main confirms it.
- [ ] Note: Railway gets these fixes only through D2 (W9, held until after WebSummit, §9-Q11). Until then its layer cache is separate from ours.

#### US-012: Pinned bases, pinned actions, and something that updates them
**Description:** As Dave, I want every base image and Action pinned and kept current by a bot, so that pinning doesn't freeze security fixes.

**Acceptance criteria:**
- [ ] `.github/dependabot.yml` covers:
  - `docker` for `/orchestrator`, `/frontend`, `/services/workspace-worker`, `/services/media-render` and `/services/agent-opt-worker` (kept, §9-Q4);
  - the compose files' third-party images;
  - `github-actions`;
  - the Python lock files as they land (US-022, US-044, and agent-opt-worker's lock in US-006).
  Updates are weekly and grouped.
- [ ] Every `FROM` line and every third-party compose image (pgvector, redis, minio, searxng, qdrant, adminer) is pinned as `name:tag@sha256:<digest>`. Tag and digest both stay in the reference, so Dependabot updates them together along the tag's line. A digest-only pin would be moved to the `latest` tag's digest. Node copied into another image comes through a named `FROM … AS node` stage (US-007), never a bare `COPY --from=<image>`, which Dependabot does not read.
- [ ] The 57 unpinned `uses:` lines are SHA-pinned. `gitleaks/gitleaks-action@v2` moves to v3.0.0 by SHA (it runs on Node 24).
- [ ] A workflow lint step fails on any `uses:` without a 40-hex SHA. It also resolves each pinned action's `action.yml` at that SHA and fails if a composite action's own `uses:` is not SHA-pinned. Exceptions are listed with a reason.

#### US-013: The scan blocks
**Description:** As Dave, I want a PR that adds a fixable Critical or a KEV to fail before merge, so that we stop publishing known holes.

**Acceptance criteria:**
- [ ] PRs that change an image's inputs scan the image pushed to the local registry (US-001), using `registry:` and `GRYPE_REGISTRY_INSECURE_USE_HTTP=true`. Main builds scan as today, before the merge job, so a failing scan leaves the digest untagged and unsigned.
- [ ] The scan step drops `continue-on-error` and `fail-build: false`. A small policy step reads Grype's JSON and fails on any fixable Critical, or on any KEV-flagged match of any severity, that is not in `.grype.yaml`. The KEV field is confirmed against Grype 0.118's JSON schema.
- [ ] Every `.grype.yaml` entry has `reason` and `expires` (90 days or less). A daily job fails when an entry has expired.
- [ ] The Grype database download starts in the background at job start, overlapping the build (DB load is 73–120 s today).
- [ ] The policy starts in report-only mode for this wave. It flips to fail as the wave's last commit, once US-006 to US-012 have cleared or allowlisted every current finding. PR builds still get the 6 Oct apt layer from the GHA cache, so rows that only US-011's uncached rebuild clears (the curl `deb13u5` rows) are allowlisted with that reason and a 30-day expiry, and come out after the first schedule run.
- [ ] CI: a policy unit test feeds a recorded Grype JSON with a fixable Critical and a KEV-flagged Medium and expects exit 1, and one whose only such rows are allowlisted and expects exit 0. A throwaway fixture PR is not used (§4).

### Part 3: API clean-up, no behaviour change (W2)

#### US-014: Test tools leave the production image
**Description:** As a self-hoster, I want the API image to carry no test tooling, so that I don't download it.

**Acceptance criteria:**
- [ ] `pytest`, `pytest-asyncio`, `pytest-timeout`, `pytest-cov`, `black`, `isort` and `import-linter` (`requirements.txt:66-76`) move to `orchestrator/requirements-dev.txt`.
- [ ] Every CI install of `requirements.txt` also installs the dev file: `test.yml:241`, `:487`, `:771` and `:1130`, and `nl2sql-eval-scheduled.yml:86`. So does `.oss-scanner/Dockerfile:34`, whose build runs pytest (`:61`). The `cache-dependency-path` lines (`test.yml:202`, `:470`, `:754`, `:1113`; `nl2sql-eval-scheduled.yml:72`) list the dev file too. The `images.yml` `pull_request.paths` filter adds the dev file. US-022 later turns each of these installs into one hash-checked install of two locks.
- [ ] The `pip uninstall` RUN (`Dockerfile:162-166`) is deleted. Today it adds a layer instead of removing one, and it misses three of the seven tools. Because it held the image's only stdlib bytecode, the same story compiles the stdlib once as root, excluding site-packages: `python -m compileall -q -x '/site-packages/' --invalidation-mode unchecked-hash /usr/local/lib/python3.11`. Site-packages bytecode comes only from `pybuild`.
- [ ] CI: `docker run --rm img python -c 'import pytest'` fails, and the backend shards pass. In US-001's layer listing, the layer the stdlib compile creates holds stdlib `.pyc` files and no path under `site-packages`.

#### US-015: Unused Python packages go
**Description:** As a self-hoster, I want packages the code never imports removed, so that the image and its scanner report shrink.

**Acceptance criteria:**
- [ ] Removed: `nltk` (and its lines in `orchestrator/start.sh`), `passlib`, `python-jose` (and its `ecdsa`/`rsa` closure), `pgvector` (the Python package; the Postgres extension stays), and `scikit-learn` from the GHCR and local images (Railway keeps it until US-027).
- [ ] The `futureagi==0.6.0 --no-deps` install, a leftover from PRD-58, is removed: the comment at `Dockerfile:47-49` and the `pip install … --no-deps futureagi==0.6.0 && \` line at `:57`. The `--prefix` comment at `:50` and the main install at `:56` stay. The API never imports it (its module is `fi`). It calls agent-opt-worker, which US-006 keeps, over HTTP.
- [ ] `graphifyy` is pinned exactly; it is `>=0.3.28` today and resolves 0.8.31. `psutil` moves to ≥ 6, which has aarch64 wheels (`requirements.txt:42` pins 5.9.6, which arm64 builds from source today).
- [ ] `pybuild` installs with `--only-binary :all: --no-binary pysher,graspologic`, in that order: pip's `:all:` clears any per-package exception given before it. This holds with pip now and with uv after US-022; both let a later per-package `--no-binary` override `:all:`. A source build of any other package fails the build outright. `pysher` 1.0.8 is a pure-Python sdist. `graspologic`, a 3.3.0 sdist used only by Railway's `[leiden]` build, leaves the exception list with US-027.
- [ ] A test asserts that none of the removed names is imported anywhere under `orchestrator/` (AST walk, lazy imports included; `fi` for futureagi, `jose` for python-jose).
- [ ] CI: backend shards and the in-image import smoke (US-020) pass, and the arm64 image builds under the binary-only rule.

#### US-016: The app layer is written once, last
**Description:** As a self-hoster, I want a code-only update to download only the changed code, so that updates take seconds.

**Acceptance criteria:**
- [ ] The production stage creates the user first. Then it creates `/app` and its writable directories (`logs`, `vector_stores`, `projects`, `exports`, `data`) owned by `automatos`, so the `backend_logs` and `backend_data` named volumes start writable. Then come the entrypoint/healthcheck copies and the stdlib compile (US-014). The stage ends with `COPY --chown=1000:1000 . .` as its last filesystem layer. `chown -R /app` (`Dockerfile:169-170`) and the `chown -R` at `:174` are gone. The numeric ID is used because the user exists before the COPY.
- [ ] US-016 changes only `orchestrator/Dockerfile`. The worker's matching change (`COPY --chown=1000:1000 . .` in place of its `COPY . .` and `useradd … chown -R worker:worker /app /workspaces` block) moves to US-043 (W6), which already rewrites that block and runs in parallel with W2. agent-opt-worker's (`services/agent-opt-worker/Dockerfile:8-10`) moves to US-006 (W1), which §9-Q4 keeps; Appendix A deletes the file instead if retirement is chosen later. media-render has no `chown -R` and needs nothing.
- [ ] CI (US-001): a code-only main update has an API update delta of ≤ 6 MB (today 14.7 MB), with exactly one layer over 1 MB changed.

#### US-017: Local builds stop baking uploads and caches into the image
**Description:** As a self-hoster who builds from source, I want the build context to hold only code, so that my uploads never enter an image layer.

**Acceptance criteria:**
- [ ] `orchestrator/.dockerignore` adds `**/__pycache__`, `**/*.py[cod]`, `/documents`, `scripts/eval/tool_routing/.embedding_cache.json`, `scripts/eval/**/results/` and `.import_linter_cache`. It does **not** use `**/*.md`, which would strip the skill seeds the app loads, or `**/documents`, which would strip `modules/documents`.
- [ ] `envs/api.defaults` sets `DOCUMENT_STORAGE_DIR=/app/data/documents`, on the `backend_data` volume. Today the default `documents` resolves to `/app/documents` and, through the bind mount, lands in the host checkout. The persistent copy stays in object storage (`generation_service.py:768-784`).
- [ ] The development stage drops `COPY . .` and `chown -R /app` (`Dockerfile:104-110`). It still creates `/app/logs`, `/app/vector_stores`, `/app/projects`, `/app/exports` and `/app/data` owned by `automatos` (no `-R`), so fresh `backend_logs` and `backend_data` volumes stay writable. Both compose files bind-mount the source over it, so `docker compose up --build` from any ref keeps working, and the guard tests at `test_entrypoint_modes.py:235-240` keep passing.
- [ ] CI: a static test asserts the `.dockerignore` covers the `DOCUMENT_STORAGE_DIR` default and `**/__pycache__`. A smoke step seeds dummy files under `orchestrator/documents` and asserts the build log's `transferring context` stays under a budget.

#### US-018: Delete what's dead (D6)
**Description:** As Dave, I want the services and files nobody uses removed, so that there is less to patch and less to explain.

**Acceptance criteria:**
- [ ] **Gotenberg:**
  - Delete the compose service (`docker-compose.yml:535-550`), its header mention (`:32`) and env (`:270`), `envs/api.defaults:47-48`, `GOTENBERG_URL` (`config.py:1634`), `orchestrator/modules/documents/conversion_service.py` and `gotenberg-client` (`requirements.txt:136`).
  - Correct the docs that still describe it as live: `QUICKSTART.md:190-191` (keeping Adminer), `self-hosting.md:304`, `docs/documentation-prd-corpus/runbooks-operational-docs.md:73`, `docs/knowledge-base-rag/_index.md:33` and `:125`, and `docs/knowledge-base-rag/document-generation-deliverables-rendering.md:23`.
  - Regenerate `orchestrator/reports/config-surface.json`; `test_config_surface_guard.py` blocks otherwise.
- [ ] **Root `Dockerfile.backend`:** delete it, and update the two line-ending tests that list it (`test_818_runtime_files_stay_lf_on_a_windows_checkout.py:37`, `test_checkout_line_endings.py:18`).
- [ ] **`promtail-config.yml`:** delete it. Nothing reads it, and it ships logs to a hard-coded public IP. Correct `envs/observability.defaults`, which documents a `--profile observability` that compose doesn't have. Delete the file if nothing else reads it.
- [ ] **`frontend/tsconfig.tsbuildinfo`:** untrack it (a build artefact, already in `.gitignore`).
- [ ] A test asserts that every `--profile X` mentioned in `envs/` and the self-host docs exists in `docker-compose.yml`.

#### US-019: The API's system packages shrink to what the code calls
**Description:** As a self-hoster, I want the API image to carry only the system tools the code calls.

**Acceptance criteria:**
- [ ] Debian `ghostscript` is removed from `Dockerfile:79`, the comment at `:69`, CI (`test.yml:187`, `:482`, `:766`, `:1125`; `nl2sql-eval-scheduled.yml:80`) and `.oss-scanner/Dockerfile:19`, which mirrors CI. Nothing calls it: camelot's lattice mode fails without the Python binding, and pdfplumber and WeasyPrint don't use it.
- [ ] `fonts-urw-base35` is added to the apt list by name, and to the same CI lists, since it no longer arrives through `libgs10-common`. Its fontconfig aliases map Helvetica, Times, Courier, Palatino and Bookman to metric-compatible faces (Nimbus, C059, P052), so a brand kit naming one of them with no font file does not fall back to DejaVu. This keeps about 15.8 MB installed, plus its `xfonts-utils` dependency, out of the 57.7 MB saving.
- [ ] `osd.traineddata` is deleted in the same apt RUN. The only OCR call (`processors.py:427`) uses the default page mode.
- [ ] `libgdk-pixbuf-2.0-0` is removed, with its mention in the comment at `:70`; WeasyPrint 69 doesn't load it.
- [ ] `libpangoft2-1.0-0` and `libharfbuzz-subset0` are named explicitly. WeasyPrint dlopens them, and today they arrive only through tesseract.
- [ ] git's server-side binaries (`git-daemon`, `git-http-backend`, `git-http-push`, `git-imap-send`, `git-shell`, `scalar`, both copies) are deleted, and `/usr/lib/git-core/git` becomes a symlink to `/usr/bin/git`.
- [ ] **Test image.** pytest is gone from the production image (US-014), so in-image pytest runs use a throwaway image built from the image under test: `FROM <img>`, then `RUN pip install --user` of US-014's dev requirements (with `--require-hashes` from `requirements-dev.lock` once US-022 lands), with the tests mounted read-only. It is never published, and no stage is added to the Dockerfile, so `production` stays last (US-010). This story's codegraph clone tests run in it, and so do US-028's per-language parser tests, with `--network none`.
- [ ] CI, inside the built image:
  - `! command -v gs`;
  - an OCR fixture returns its text;
  - a PDF whose brand font is `Helvetica` renders with the same page count and line breaks as before;
  - `git clone --depth 1` of a public repo works;
  - the codegraph clone tests pass in the test image.

#### US-020: site-packages without debug info and test suites
**Description:** As a self-hoster, I want the Python layer to hold only what runs.

**Acceptance criteria:**
- [ ] In the `pybuild` stage, after the install: `find /install -name '*.so*' -not -path '*.libs/*' -exec strip --strip-debug {} +`. The auditwheel `*.libs` directories are excluded, and `.dynsym` is untouched.
- [ ] Directories named exactly `tests` under site-packages are deleted. Directories named `testing` (`numpy.testing`, `sqlalchemy.testing`, `alembic.testing`) are not touched.
- [ ] CI: a script inside the image finds every `.so` under site-packages outside `*.libs`.
  - Each one that exports `PyInit_<name>` (read from `.dynsym` with a Python ELF parser, since the runtime has no binutils) is imported.
  - Every other one is loaded with `ctypes.CDLL`. An example is `pypdfium2_raw/libpdfium.so`, which pypdfium2 loads through ctypes.
  - The script then runs `import main`.
  - The set of failures must equal a committed allow-list, each entry with its reason. Today the list is `PIL._imagingtk`: it imports tkinter, whose Tk libraries python:slim does not ship. Any other ImportError or OSError fails the step.
  - The same ELF parser's section-header check finds no `.debug_*` sections outside `*.libs`.

#### US-021: botocore ships only the services we call
**Description:** As a self-hoster, I want botocore's service models cut to the ones the code uses.

**Acceptance criteria:**
- [ ] In `pybuild`, `botocore/data/*` is deleted except `s3`, `s3vectors`, `bedrock-runtime`, `sts`, `sso`, `sso-oidc` and the top-level data files.
- [ ] An AST test fails if any `boto3.client(...)` or `boto3.resource(...)` in `orchestrator/` names a service outside that list. This is the guard for future code and for botocore bumps.
- [ ] CI: inside the image, a client is constructed for each kept service. The fresh-clone smoke (`scripts/ci/smoke-fresh-clone.sh`) builds this Dockerfile and starts MinIO with the backend. Its development stage shares `base`, and so the pruned botocore. The smoke uploads a document and fetches its presigned URL. The backend shards prove nothing here: they install the full botocore on the runner.

#### US-022: The API's dependency set is locked
**Description:** As Dave, I want CI, GHCR and Railway to install the same package versions, so that a rebuild never changes what we ship by surprise.

**Acceptance criteria:**
- [ ] **Locks.** `orchestrator/requirements.txt` becomes `requirements.in`. `requirements.lock` is generated with `uv pip compile --universal --python-version 3.11 --generate-hashes`. `requirements-dev.lock` is compiled the same way with `-c requirements.lock requirements-dev.txt`, so dev tools never move a locked version. Until US-027, a second lock covers the Railway `[leiden]` variant. `INSTALL_GRAPH_EXTRAS` picks which lock pybuild installs, in place of today's `sed` (`Dockerfile:51-55`), and keeps its production default `true`.
- [ ] **Install.** `pybuild` installs with `uv pip install --python /usr/local/bin/python --prefix /install --no-deps --require-hashes --no-compile-bytecode -r requirements.lock`, keeping US-015's `--only-binary`/`--no-binary` flags. It uses the uv binary copied from a digest-pinned `ghcr.io/astral-sh/uv` image.
  - The install stays under `/install`. That is the tree US-020 strips, US-021 prunes and the base stage copies (`COPY --from=pybuild /install /usr/local`, `Dockerfile:87`). `--system` would install into pybuild's own `/usr/local` and leave `/install` empty.
  - After the US-020 and US-021 pruning, pybuild runs `python -m compileall -q -s /install -p /usr/local --invalidation-mode unchecked-hash /install/lib/python3.11/site-packages`, because uv has no invalidation-mode option. A shipped `.py` that does not compile (a template or test data) is excluded by name with `-x` and a comment. The step never ends in `|| true`.
  - No BuildKit cache mount. Railway builds this Dockerfile, and a cache mount there fails the build unless it carries a hard-coded `id=s/<service-id>-<path>`. The service ids are unknown (§10).
- [ ] **Two layers.** pybuild installs the one lock in two layers, split by package name. First comes a stable set (numpy, scipy, grpcio, cryptography, SQLAlchemy, Pillow, lxml, fonttools, uvloop, asyncpg, pydantic-core), then the rest. Each half runs the install command above into its own prefix (`/install/stable`, `/install/rest`). The runtime copies each as its own layer, stable first. US-020, US-021 and the compile step (with `-s` set to each prefix) cover both trees. The saving is an estimate until measured: the first grouped Dependabot bump after W2 records it in §8.
- [ ] **CI installs.** `test.yml` (`:241`, `:487`, `:771`, `:1130`), `nl2sql-eval-scheduled.yml:86` and `.oss-scanner/Dockerfile:34` each run one `pip install --require-hashes -r requirements.lock -r requirements-dev.lock`. A hashed lock puts pip in hash-checking mode for the whole command, so the unhashed `requirements-dev.txt` cannot go in the same command. A separate dev install could move locked versions.
  - Every `cache-dependency-path` (`test.yml:202`, `:470`, `:754`, `:1113`; `nl2sql-eval-scheduled.yml:72`) lists both locks. setup-python fails a step whose path matches no file.
  - The `images.yml` `pull_request.paths` filter and the `test.yml:89` `PROD_IMAGE_PATHS` regex (`requirements[^/]*\.txt` today) name `requirements.in` and the lock files.
  - `orchestrator/start.sh:33` installs from `requirements.lock`.
- [ ] **Readers of the old name.** `orchestrator/tests/test_f112_telegram_polling_dependency.py:15` reads `requirements.in`. The GPL guard at `test_prd251w1_media_render_image.py:52` globs `requirements*.txt`, `requirements*.in` and `requirements*.lock`. It asserts that `requirements.in` and `requirements.lock` are among the files it scans, so it still checks the API's real dependency set (AGENTS.md:121: never weaken a test). Any other reader `git grep` finds is updated. The `scripts/ralph/acceptance-*.sh` records of past waves stay as they are.
- [ ] A CI check fails when a lock does not satisfy its input: `requirements.lock` and the `[leiden]` lock against `requirements.in`, and `requirements-dev.lock` against `requirements-dev.txt`. It does not re-resolve against PyPI, so it does not go red on every upstream release.
- [ ] **Reproducible pip layers.** For the API matrix entry only, `images.yml` sets `SOURCE_DATE_EPOCH` to a fixed value (set once, for example the W2 merge time) and adds `rewrite-timestamp=true` to `outputs` (BuildKit ≥ 0.13), so a rebuilt pip layer is byte-identical.
  - The value is fixed, not the lock's commit time. A lock change outside the stable set would otherwise re-stamp the stable layer.
  - Other images are left alone, because a constant epoch on the frontend would clamp `public/` mtimes, which carries an ETag risk.
  - The unchecked-hash `.pyc` keep the clamped `.py` mtimes from marking bytecode stale. In the image, `python -v -c 'import main' 2>&1 | grep -c 'bytecode is stale'` is 0.
- [ ] CI on push-to-main (schedule runs excluded):
  - When neither the lock nor the base digest changed, both pip layer digests equal the previous `edge`'s. Today 14 digests appeared in 17 days.
  - When the lock changed only outside the stable set, the stable layer digest equals the previous `edge`'s, and US-001 reports the second layer's update delta.
  - If the pybuild apt layer itself changed (for example, a binutils update that alters strip output), the check compares the installed dist-info set (name==version) instead.

### Part 4: API code swaps (W3)

#### US-023: Tables come from pdfplumber, not camelot
**Description:** As a self-hoster, I want PDF tables extracted without opencv, which ships only so camelot can import.

**Acceptance criteria:**
- [ ] `modules/rag/ingestion/multimodal/processors.py` extracts tables with `pdfplumber`, which the same file already imports. `TableProcessor` works on pdfplumber's lists of rows. `import camelot` and `import pandas as pd` (`:36-37`) go, and so does the lattice-then-stream loop (`:124-169`). The `pd.DataFrame` helpers `_process_camelot_table` and `_detect_header_row` (`:171-236`) are replaced by versions that take lists of rows, so no `pd.` remains in the file.
- [ ] `camelot-py` and `opencv-python-headless` leave `requirements.in` (US-022), and the lock is recompiled. About 164 MB installed goes, including opencv's bundled FFmpeg and OpenSSL 1.1.1k (13 Grype rows).
- [ ] `openpyxl` is added to `requirements.in` explicitly. Today it arrives only through camelot, and `attachments/extract.py:147`, `rag/ingestion/manager.py:260` and `:600`, `documents/thumbnails/html_sources.py:104` and `services/spreadsheet_workspace.py:54` import it.
- [ ] Today lattice mode fails on every PDF (camelot's ghostscript backend has no Python binding in the image), so production has only ever used stream mode. The swap is judged against stream output.
- [ ] **Reference grids.** Before the swap commit, a committed script records camelot's stream cell grids for the table fixtures as golden JSON, using `camelot-py==0.11.0` and the opencv version from the W2 lock. In the W3 CI run, a step installs those pins in a throwaway venv, regenerates the goldens and diffs them against the committed copy, so the reference is proven in the same run as the swap.
- [ ] After the swap, pdfplumber (`table_settings` with the text strategy) must reproduce each golden grid. Otherwise it must match the grid within a cell-level tolerance that Gerard sets and that is recorded here. The permanent test needs only the committed goldens, not camelot.
- [ ] Every camelot reference in `orchestrator` goes:
  - the `sys.modules.setdefault("camelot", …)` stubs in 54 test files, among them `tests/test_f081_search_multimodal_extracted_only.py:16-19`, `tests/security/test_tenancy_matrix.py`, `tests/security/test_w2s7_fail_closed_authz_gate.py` and `tests/security/test_nl2sql_tenancy.py`;
  - the comments at `tests/conftest.py:113`, `tests/test_board_task_handlers.py:11`, `tests/test_mission_retry_feeds_critique.py:74` and `tests/test_us009_limit_reporting.py:39`;
  - the comments at `modules/rag/ingestion/manager.py:864` and `integrations/shopify/tests/fixtures/regenerate.py:10`.

  Afterwards `git grep -i camelot -- orchestrator` is empty. The US-015 AST test adds `camelot` and `cv2` to its forbidden names.

#### US-024: Gemini through the OpenAI-compatible endpoint
**Description:** As a self-hoster, I want Gemini calls to go through the OpenAI-compatible adapter we already have, so that the deprecated `google-generativeai` and its 112 MB of discovery documents leave the image.

**Acceptance criteria:**
- [ ] Gemini moves onto the existing OpenAI-compatible adapter (PRD-236). The `google` spec in `core/llm/providers.py:135` gets `adapter=ADAPTER_OPENAI_COMPATIBLE` and `base_url_key="GOOGLE_BASE_URL"`. `OpenAICompatibleProvider` (`core/llm/clients/openai_compatible_client.py:282`) then serves Gemini chat, streaming (`stream_response`, `:554`) and tool calls (`generate_response`, `:430`). Today's `GoogleProvider` has neither streaming nor tool support.
- [ ] `GOOGLE_BASE_URL` is a new `config.py` setting next to `OPENROUTER_BASE_URL` and `DEEPSEEK_BASE_URL` (`:597-600`). It defaults to `https://generativelanguage.googleapis.com/v1beta/openai/` and is listed in `reports/config-surface.json` (FR-12).
- [ ] The `LLMProvider.GOOGLE` branch in `manager._create_provider` (`core/llm/manager.py:629-630`) is removed, so Gemini falls through to `OpenAICompatibleProvider` (`:640-642`). The Google credential branch (`:545-546`) goes too, so the key is read the same way as for the other OpenAI-compatible providers (`:567-571`). These are deleted:
  - `core/llm/clients/google_client.py`;
  - its imports (`manager.py:21`, `core/llm/clients/__init__.py:12`) and the `GoogleProvider` export (`:33`);
  - the `ADAPTER_GOOGLE` constant (`providers.py:36`), which nothing else uses.
- [ ] `'google'` leaves `_LIVE_CHECKS` (`api/user_api_keys.py:213`), so the saved-key check falls through to `_check_openai_compatible` (`:172`), a `models.list()` on the same base URL. `_check_google` (`api/user_api_keys.py:165-169`) is deleted.
- [ ] `google-generativeai` (135 MB installed) leaves `requirements.in`. `grpcio` and `protobuf` stay, because qdrant-client and OpenTelemetry need them. The US-015 AST test adds `google.generativeai` to its forbidden names.
- [ ] Tests with `httpx.MockTransport` run `OpenAICompatibleProvider` with the google spec. They cover a completion, a stream, a tool call and the key check, and assert URL, headers and body shape. The stream test uses a `SyncByteStream` body, because `httpx.Response(content=…)` is read in advance.

#### US-025: pandas leaves the image
**Description:** As a self-hoster, I want pandas gone, because no working feature needs it.

**Acceptance criteria:**
- [ ] `create_xlsx` in `modules/tools/execution/exec_document.py` (`:137-170`) writes with `xlsxwriter` (already a requirement) and reads CSV/JSON with the stdlib.
- [ ] `processors.py:37` (`import pandas as pd`) and the `pd.DataFrame` helpers (`_process_camelot_table`, `_detect_header_row`) go with the US-023 rewrite, which works on pdfplumber's lists of rows. That import is on the boot path (`main.py:102` → `api/knowledge_multimodal.py` → `modules.rag` → `ingestion` → `multimodal`). If pandas is uninstalled while it stays, `import main` fails.
- [ ] `modules/tools/services/pandas_ai_service.py` imports pandas inside the methods that use it (module top at `:6` today, used at `:90-113`). This is required, because `modules/tools/services/__init__.py:9` and `modules/nl2sql/service.py:39` import the module at boot. `pandasai` is not installed, so the service already reports itself unavailable.
- [ ] The `pandas` line leaves `requirements.in` and the GHCR lock. Railway's `[leiden]` lock drops it with US-027.
- [ ] The US-015 AST test adds `pandas` to its forbidden names for `orchestrator/`, except inside `pandas_ai_service.py`'s methods.
- [ ] Whether to delete the PandasAI service and its two routes, rather than keep them unable to run, is §9-Q16 (PandasAI charts). If the answer is "delete", this story deletes them instead of making the import lazy:
  - `pandas_ai_service.py` and its exports (`modules/tools/services/__init__.py:9`, `:17-18`);
  - `api/analytics_charts.py`, its router include (`main.py:117`, `:1227`) and its `reports/route-manifest.json` entries (`:338`, `:342`);
  - `NL2SQLService.analytics_engine` (`nl2sql/service.py:39`, `:80`). `analyze_database` (`:743`) keeps returning its "PandasAI service is not enabled or configured." error (`:776-780`).
- [ ] Tests: CSV and JSON into xlsx, read back with openpyxl in the test environment. `POST /api/analytics/charts/generate` keeps returning its existing 404 "PandasAI service is not available…" response (`api/analytics_charts.py:246-251`), never a 500 or an ImportError.

#### US-026: The text splitters we use are vendored
**Description:** As a self-hoster, I want the two splitters we use without langchain-core and langsmith.

**Acceptance criteria:**
- [ ] `RecursiveCharacterTextSplitter` and `PythonCodeTextSplitter` move into an orchestrator module with the MIT notice. They are the only splitters `modules/rag/ingestion/manager.py` reads (built at `:166` and `:175`, used at `:450-452`). A minimal chunk type with `page_content` and `metadata`, which is what `create_documents` returns, moves with them.
- [ ] The vendored code meets AGENTS.md's code-shape limits: functions of at most 50 code lines, nesting of at most 4 and complexity of at most 10 (`scripts/ci/check_changed_code_shape.py`, ruff `C901`). Only the Python separator list is copied, as a constant. langchain's `get_separators_for_language` (484 code lines) is not copied. No shape or ruff exemption is added. The golden test proves the restructure changed no output.
- [ ] Unused code is deleted, not vendored:
  - the `MarkdownHeaderTextSplitter` and `langchain_core.documents.Document` imports (`manager.py:34`, `:37`);
  - the never-read `self.markdown_splitter` (`:171-174`) and its `MarkdownTextSplitter` import (`:33`).
- [ ] `langchain-text-splitters` and `langchain-core` leave `requirements.in`, and with them `langsmith` and a 24 MB `zstandard` (2 Grype rows). The US-015 AST test adds `langchain_text_splitters` and `langchain_core` to its forbidden names.
- [ ] **Golden chunks.** Before the swap commit, a committed script records langchain's chunk lists (`page_content` and `metadata`) for fixture documents as golden JSON. The fixtures are Markdown with headers, Python and plain text. The script uses `langchain-text-splitters==1.0.0` and the `langchain-core` from the W2 lock. In the W3 CI run, a step installs those pins in a throwaway venv, regenerates the goldens and diffs them against the committed copy. A permanent test asserts that the vendored splitters reproduce the goldens exactly.
- [ ] A second test drives `DocumentProcessor.chunk_document` (`manager.py:381`) through its basic-splitter path (`:449-452`, with `SEMANTIC_CHUNKER_AVAILABLE` off) on the Python and plain-text fixtures. The retrieval eval lane does not prove this swap, because it never imports the splitters.

#### US-027: Leiden through graspologic-native: one API image
**Description:** As Dave, I want Railway, GHCR and compose to install the same API dependencies, so that production can run the published image (D2) without self-hosters downloading 585 MB of plotting and numba.

**Acceptance criteria:**
- [ ] `modules/knowledge/graph_service.py` gets its Leiden partition from `graspologic_native.leiden` through a small adapter. Today it imports `graphify.cluster.cluster` at `:46` and calls it at `:302`, `:495` and `:1451`.
- [ ] The adapter reproduces `graspologic.partition.leiden` as `graphify.cluster._partition` calls it. It serves both partition calls that `graphify.cluster.cluster` makes: `cluster.py:133` and the re-split at `:198` (graphifyy 0.8.31). It uses:
  - the same sorted node and edge order as `_partition` (`:35-45`);
  - `random_seed=42`, `trials=1` and the caller's `resolution`;
  - graspologic 3.3.0's defaults for randomness (0.001), iterations, `use_modularity` and edge weights.
- [ ] `graphifyy` is installed without `[leiden]`, and `graspologic-native` is pinned. These are removed:
  - the `INSTALL_GRAPH_EXTRAS` ARG in `pybuild`, the `[leiden]` variant lock it selects in place of the `sed` (US-022), and that lock's entry in US-022's lock check. On 697ca7d6e these are the ARG with its comment (`Dockerfile:32-45`) and the `sed` (`:51-55`);
  - the build args: the backend's `build.args` in `docker-compose.yml` (`:157-160` on 697ca7d6e), the api matrix entry's `build-args` in `images.yml` (`:80-81`), and `deploy/kind/e2e.sh:82`;
  - the parity guards (`test_dockerfile_prod_parity.py:34`, `:43`);
  - `graspologic` from US-015's `--no-binary pysher,graspologic` exception list, which becomes `--no-binary pysher`.
- [ ] The docs stop offering the opt-in. The Leiden opt-in in `docs/getting-started/self-hosting.md:922-924` goes, and `:916` says one thing is opt-in, not two. The pages that describe the ARG are corrected: `docs/deployment-infrastructure/docker-compose-setup.md:215`, `docker-containerization.md:82` and `:88`, `production-deployment-cicd.md:114`, and `docs/getting-started/installation-setup.md:183`. Afterwards `git grep INSTALL_GRAPH_EXTRAS -- . ':!docs/PRDS'` is empty.
- [ ] `prod-image-parity`'s "Backend — graph extras present with no args" step (`test.yml:930-933`) asserts that `import graspologic_native` works and that `import graspologic` fails. US-043 (W6) and US-047 (W7) edit the same job, so this edit is in §4's seam review.
- [ ] **Reference partitions.** Before the swap commit, a committed script records graspologic's partitions on the fixture graphs as golden JSON, using `graspologic==3.3.0` and graphify's fixed seed. In the W3 CI run, a step installs it in a throwaway Python 3.11 venv (graspologic 3.3.0 requires Python below 3.12), regenerates the goldens and diffs them against the committed copy.
- [ ] A permanent test, which needs no graspologic, asserts that the adapter reproduces the goldens. The bar is identical partitions; if that fails, modularity within 1% (§9-Q8, taken at its recommendation). If neither holds, graphify's Louvain fallback is accepted everywhere (§9-Q8) and the adapter is not shipped.
- [ ] Railway's image loses the ~585 MB of Leiden extras, plus pandas (~75 MB) and scikit-learn (~54 MB with narwhals and threadpoolctl). After US-015 and US-025, only the extras still pull those two in: statsmodels and seaborn require pandas, and graspologic requires scikit-learn. In total, about 713 MB installed goes.
- [ ] Railway builds this Dockerfile with its defaults, so production moves to the adapter when W3 deploys. It does not wait for W9, which is held until after WebSummit (§9-Q11).
- [ ] The API can move to Python 3.12 (US-058), because graspologic 3.3.0 requires Python below 3.12.

#### US-028: Code-graph parsing never downloads grammars at runtime
**Description:** As a self-hoster on a restricted network, I want code-graph parsing to use grammars that are already installed.

**Acceptance criteria:**
- [ ] `modules/codegraph/parsers/treesitter_parser.py` loads every grammar in `SUPPORTED_LANGUAGES` (`:75-90`) from its own `tree_sitter_*` package. This replaces today's three-language fallback (`:158-180`). Most packages export `language()`, with two exceptions: TypeScript and TSX come from `tree_sitter_typescript.language_typescript()` and `language_tsx()`, and PHP comes from `tree_sitter_php.language_php()`. Today's fallback calls `language()` on the TypeScript package, which does not have it.
- [ ] Every `tree_sitter_*` package the parser imports is named in `requirements.in`: java, go, rust, c, cpp, ruby, php, swift, kotlin and c-sharp, next to the python, javascript and typescript lines already there (`requirements.txt:112-114` on 697ca7d6e, before US-022 renames the file). Today the ten new ones arrive only through graphifyy, so a graphifyy bump that dropped one would remove it without warning.
- [ ] `tree-sitter-language-pack` leaves `requirements.in`. Its guarded import (`:18-24`) and the `_HAS_LANGUAGE_PACK` branch (`:146-153`) go. Since 1.0.0 it ships no grammars and downloads them on first use, and `_get_parser` (`:141-156`) returns None when that download fails. The US-015 AST test adds `tree_sitter_language_pack` to its forbidden names.
- [ ] A new test parses one fixture file per `SUPPORTED_LANGUAGES` extension and asserts that symbols come back. No per-language parser test exists today.
- [ ] CI: these tests run, with `--network none`, in the throwaway test image that US-019 defines for in-image tests, not inside the production image:
  - it is built `FROM` the image under test, then `RUN pip install --user` of US-014's dev requirements, with `--require-hashes` from US-022's `requirements-dev.lock`;
  - the tests are mounted read-only;
  - it is never published.

  The production image itself still fails `import pytest` (US-014).

### Part 5: API boot (W4)

#### US-029: One migration run per boot, and a production CMD that execs uvicorn
**Description:** As a self-hoster, I want the backend to migrate once and then start, so that boot isn't paying for alembic twice.

**Acceptance criteria:**
- [ ] The entrypoint is the only owner of migrations. Before a **server** command it runs the migration step: the database wait, `init_fresh_db` for an empty or interrupted database (the migration forest cannot replay from empty, `docker-entrypoint.sh:117-125`), the US-030 head check and `alembic upgrade heads`. The flags below set what else runs. Nothing runs before a one-off command: the `prod-image-parity` backend import check in `test.yml` (`docker run --rm parity-be python -c …`) and `docker compose run backend <cmd>` neither migrate nor seed.
- [ ] Each flag has one meaning. The entrypoint header (`docker-entrypoint.sh:12-32`) and `test_entrypoint_modes.py` document both:
  - `AUTOMATOS_SKIP_MIGRATIONS=true` (the Helm api pods): the entrypoint runs none of the boot steps.
  - `AUTOMATOS_MIGRATE_ON_BOOT=true` (compose, `envs/api.defaults:39`): after the migration step, the seed loader and the local workspace/operator SQL run, as today.
  - Neither set (Railway, per the April manifest; §10 checks the live service before W4): only the migration step runs. The entrypoint never runs `core.database.load_seed_data` there, because the loader's `_seed_packages()` is a full refresh that would overwrite curated production packages (`load_seed_data.py:224-232` against `main.py:356-370`).
- [ ] The development and production CMDs, and the comments above them, lose `alembic upgrade heads &&`. The production CMD becomes `exec uvicorn …` without `--workers`, so worker count comes from `WEB_CONCURRENCY`: the image sets 4 (Railway unchanged) and `envs/api.defaults` sets 1.
- [ ] `envs/api.defaults` also sets `UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=5`; uvicorn reads every CLI option from a `UVICORN_*` variable (FR-12). The browser's SSE streams (notifications, board events) never close, which is why the development CMD carries `--timeout-graceful-shutdown 5`. With the variable set, they no longer hold a compose stop or restart of the production stage until Docker's 10 s kill. US-035's check that `docker compose stop backend` finishes in under 8 s with an SSE client connected depends on it. The production CMD does not carry the flag, so Railway's shutdown is unchanged.
- [ ] US-008's worker-kill step in smoke-images' PR variant sets `WEB_CONCURRENCY=2` or more for the container it tests. With `envs/api.defaults`'s value of 1, uvicorn runs without a supervisor, so a killed worker ends the container.
- [ ] Railway keeps migrating, through the entrypoint's migration step, until US-057 (W9, held until after WebSummit) moves it to a pre-deploy step. The Helm api deployment (`charts/automatos/templates/api-deployment.yaml`) sets `AUTOMATOS_SKIP_MIGRATIONS=true`, because the migrate Job owns the schema there. Its comment that the image CMD migrates is corrected, and `charts/automatos/tests/api_test.yaml` asserts the variable.
- [ ] Tests: `test_entrypoint_modes.py` (the production-CMD assertion, `:243-249`) and `test_prd256_fix_dead_backend_restarts.py:279-286` are updated. The latter now asserts that both CMDs start uvicorn with `exec` and contain no `alembic`. A new entrypoint test runs with `AUTOMATOS_MIGRATE_ON_BOOT` unset and a `uvicorn` command, and asserts that alembic ran and `core.database.load_seed_data` did not.
- [ ] CI counts alembic runs by §8's rule: the `Context impl PostgresqlImpl` lines after `init_fresh_db: done`, or all of them on a boot where `init_fresh_db` did not run.
  - After the entrypoint's `🚀 Starting Backend Application` banner, the backend log has no `Context impl` line, so the CMD runs no alembic.
  - A fresh boot counts 0. The replay inside `init_fresh_db` (about 200 lines, one per revision) is its only alembic output, followed by `schema at head — alembic skipped`.
  - On a database one revision behind head (a seeded fixture), the count is 1, and the entrypoint's `Running database migrations` line appears exactly once.

#### US-030: A warm boot at head runs no alembic at all; the per-boot ALTERs become a revision
**Description:** As a self-hoster, I want a restart with no new migrations to skip alembic entirely.

**Acceptance criteria:**
- [ ] Before invoking alembic, the entrypoint computes the head revision(s) by AST-parsing `revision` / `down_revision` from `alembic/versions/*.py`, with no imports. It handles tuple `down_revision` merges. It compares the result with `SELECT version_num FROM alembic_version`. When they are equal it logs `schema at head — alembic skipped`. A missing table still goes to `init_fresh_db`, and a database ahead of the code still goes to alembic and fails loudly.
- [ ] The per-boot `ALTER TABLE … ADD COLUMN IF NOT EXISTS` statements (`main.py:237-251`: `system_prompts.futureagi_eval_enabled`, `skills.content_hash`) move into an alembic revision. Today they take an ACCESS EXCLUSIVE lock on every boot of every worker.
- [ ] The revision chains onto the current single head (`prd258_voice_timings` at 697ca7d6e) and uses `ADD COLUMN IF NOT EXISTS`, so it survives a database that `create_all` or `init_fresh_db` already built. The same commit deletes the `main.py` ALTER block and moves the two head pins, as AGENTS.md requires: `EXPECTED_HEAD` in `orchestrator/tests/test_prd209_alembic_single_head.py` (`:104`), and the chain assertion and pinned head string in `orchestrator/tests/test_prd236_w1_routes.py` (`:266-270`).
- [ ] Unit test: the AST head equals `alembic heads` (the alembic-from-zero job has Postgres). The new revision runs against a pre-seeded schema that lacks both columns, and against one that already has them.
- [ ] CI: the smoke's warm phase logs `schema at head — alembic skipped` and no `Context impl` line.

#### US-031: Seeds run when they change, not on every boot
**Description:** As a self-hoster, I want a restart to skip seed work that can't have changed, so that warm boots take seconds.

**Acceptance criteria:**
- [ ] Two fingerprints are stored in `system_settings`, one per seeding stage, so no migration is needed: `seed_fingerprint.loader` (the entrypoint's seed loader) and `seed_fingerprint.leader` (the lifespan leader block plus the starters pass). Each hashes the seed inputs:
  - every file under `core/seeds/`, plus `core/database/credential_types_seed.json`, `core/database/load_seed_data.py` and `scripts/seed_*.py`;
  - the orchestrator modules the seeders import, today `core/builtin_skills.py`, `core/llm/defaults.py`, `core/llm/list_prices.py`, `core/services/anthropic_catalog_sync.py` and `modules/documents/{seed_templates.py,presets.py,social_starters.py,template_summary.py,templates/**}`;
  - the config values the seeders read, today `AUTH_EDITION`, `DEFAULT_WORKSPACE_ID`, `LOCAL_OPERATOR_EMAIL`, `NEXT_PUBLIC_API_URL` and `CLI_RUNTIME_ENABLED` (`seed_brand_designer.py:119-120`).
- [ ] The LinkedIn credential-schema UPDATE (`main.py:286-303`) moves into `core/seeds/`, so the fingerprint covers it.
- [ ] A stage writes its own fingerprint only after every one of its sections finished without logging an error. Otherwise the stored value stays as it was and the next boot runs that stage again, which keeps today's "the next boot tries again" (`main.py:183`, `:202`).
- [ ] When a stored fingerprint matches, its stage is skipped. The entrypoint checks `seed_fingerprint.loader` without importing the app and does not start the seed loader (`docker-entrypoint.sh:183`). The lifespan checks `seed_fingerprint.leader` and skips the leader seed block (`main.py:257-372`) and the starters pass (`_then_starters`, `main.py:205-217`). These run on every boot whatever the fingerprint: the orphaned-run reaper (`main.py:374-387`, which stays under the same lock), `ensure_local_workspace`, the local first-run seed and the Composio rebind.
- [ ] Credential types and system settings are seeded once per boot, in the lifespan leader block (`main.py:269-284`, the only copy Railway runs), and leave the entrypoint's boot-time loader. The credential-types upsert writes only rows that changed. Packages keep only the lifespan's `create_only=True` path (`main.py:356-370`), and the loader's full refresh (`load_seed_data.py:224-232`) leaves the boot path. Local installs, like production, then stop receiving edits to existing packages on boot. A full refresh stays a manual run of `core/seeds/seed_packages.py`. The `migrate` mode keeps the full loader.
- [ ] `_seed_local_edition_first_run` moves from `load_seed_data._load_platform_defaults` (`load_seed_data.py:248-279`) to the end of the lifespan leader block, after the gated seeds, where it runs on every boot. A warm boot with an unchanged fingerprint then starts no seed-loader interpreter. `migrate` keeps calling it.
- [ ] `load_seed_data`'s `__main__` flushes and calls `os._exit(rc)`, which removes the 6 s interpreter teardown measured on the Mac. `core/llm/__init__.py` exports lazily (PEP 562), so seeding no longer imports the whole LLM stack.
- [ ] Tests:
  - editing a seed fixture, a listed import or a listed config value changes the fingerprint;
  - a failed section leaves its stage's stored fingerprint unchanged;
  - across a warm restart, `credential_types.updated_at` is unchanged;
  - an AST walk of `core/seeds/`, `load_seed_data.py` and `modules/documents/seed_templates.py` fails when a seeder imports an orchestrator module outside `core.models`, `core.database` and `config` that is not in the input list, or reads a `config` attribute that is not hashed.
- [ ] CI: the warm phase logs `seeds unchanged (fingerprint …) — skipped` for both stages, and no seed-loader process starts.

#### US-032: Heavy modules load when used, not at import
**Description:** As a self-hoster, I want each API process to import only what boot needs.

**Acceptance criteria:**
- [ ] The module-level PDF/OCR imports left at the top of `modules/rag/ingestion/multimodal/processors.py` after US-023 (`pdfplumber`, `PIL.Image`, `pytesseract`) move into the methods that use them. The Composio probe in `core/composio/client.py:1534-1541` uses `importlib.util.find_spec`.
- [ ] `test.yml`'s backend job asserts that after `import main` none of `camelot`, `cv2`, `matplotlib`, `pandas`, `nltk`, `google.generativeai` or `googleapiclient` is in `sys.modules`. It also writes the top 25 entries of `python -X importtime -c 'import main'` to the step summary, with a wall-time budget set from the first week's numbers.
- [ ] **Bytecode (§9-Q7):** a stage built `FROM base`, so the bytecode matches the runtime interpreter, copies the app (`COPY . .`) and runs `python -m compileall -q --invalidation-mode unchecked-hash` over the boot module set. The production stage's last filesystem layer becomes `COPY --from=<that stage> --chown=1000:1000 /app /app`, so code and bytecode stay one final layer, as US-016 requires. App `.pyc` are no longer shipped since US-014 and US-017: US-014 deleted the RUN that removed them, and US-017's `.dockerignore` keeps host `.pyc` out of the context. This stage is the image's only source of app bytecode.
- [ ] **Measurement:** the W4 PR's CI run includes a one-off experiment job. No other lane can measure it in W4, because compose still boots the development stage with the source mounted until US-035. The job builds the production stage twice: as committed, and with the compile step removed in the job's workspace. For each build it starts a new container against the same seeded Postgres with `WEB_CONCURRENCY=1`, using `docker compose up -d --force-recreate --no-deps backend`. It does not use `restart`, because a restarted container keeps the `.pyc` its first boot wrote. It reports the median time to `/health 200` over two recreated containers, the compressed size of the final app layer, and the code-only update delta. The image cost counts the app boot set only, because US-014 already compiles the stdlib.
- [ ] **Decision (§9-Q7):** app bytecode stays only if a new container's warm boot reaches `/health 200` at least 2 s sooner and the code-only API update delta still meets US-016's bar: ≤ 6 MB, with exactly one changed layer over 1 MB. App bytecode changes with every code edit, so it is downloaded again on every update. If the gain is under 2 s, W5's push removes the step. If the gain holds but the delta misses, Gerard chooses in that push between removing the step and raising the update bar in US-016 and §8. That push records the numbers here.

#### US-033: The API boots with no internet
**Description:** As a self-hoster on a locked-down network, I want the API to start without reaching the internet.

**Acceptance criteria:**
- [ ] The image sets `TIKTOKEN_CACHE_DIR=/opt/tiktoken` and fetches, at build time, every encoding the code can request (at least `cl100k_base`; `o200k_base` if any model maps to it).
- [ ] The module-level encoders at `modules/rag/service.py:24-29`, `modules/search/optimization/context_optimizer.py:94-98` and `core/context_guard.py:41-46` become a cached getter. Today the fallback `get_encoding` sits outside any `try` and can kill `import main`. tiktoken's `requests.get` has no timeout, so dropped egress hangs about 2 minutes per attempt.
- [ ] An AST test forbids module-level `tiktoken.get_encoding` / `encoding_for_model`.
- [ ] CI: a smoke variant boots the backend in two ways:
  - with egress refused: a proxy to `127.0.0.1:9`;
  - with egress blackholed: the backend stays on the normal compose network. After `docker compose up -d postgres redis minio` creates that network, the runner inserts `sudo iptables -I DOCKER-USER -s <automatos network subnet> ! -d <automatos network subnet> -p tcp -j DROP`, then starts the backend. DNS still answers and the host still reaches the published port, but TCP connects to the internet are silently dropped. That is the case where tiktoken's timeout-less `requests.get` hangs. An internal-only Docker network would not test it: connects there fail at once, and the runner could not reach the published port.

  Both reach `/health 200` within the §8 fresh-boot bar, so a hang shows up as a timeout.

#### US-034: Ready after the core; the rest runs in the background, off the loop
**Description:** As a self-hoster, I want the API to answer once its core is up, and `/health/ready` to mean something.

**Acceptance criteria:**
- [ ] The lifespan reaches its `yield` after phase-1 core and the trust gate, so uvicorn listens and `/health` answers. `_boot_phase_2_extensions` (scheduler, channels), the starters and `_bootstrap_composio_catalog` then run as one background task. That task records their stages in the bootstrap report and sets `app.state.ready = True` once all of them have finished. As today, it does so whether or not a stage failed, and failures show in `/health/bootstrap`. Board dispatch keeps its current position, or a test shows it is safe to move.
- [ ] The synchronous parts of the warm-ups (`startup_tasks.py:27-93`, `semantic_indexer.py:190-198`, ActionRegistry init) run in `asyncio.to_thread`. Today they block the loop for about 5.7 s, and APScheduler logs a missed job.
- [ ] `embed_all_agents_on_startup` runs only in the boot-lock leader, because it writes shared agent embeddings to the database. Today every worker starts it (`main.py:408`), so it runs in each of Railway's four workers. `warm_action_index_on_startup` (`:410`) stays per worker only if its index is process-local, and a comment says so.
- [ ] `/health` answers once uvicorn listens. `/health/ready` returns 503 until the background task finishes, then 200. `QUICKSTART.md:120-123`, `main.py:1296-1298` and the `READY_URL` comment in `scripts/ci/smoke-fresh-clone.sh` describe this.
- [ ] CI: in `/health/bootstrap`, the stages up to and including TRUST_GATE sum to 3 s or less. No `Run time of job … missed` warning appears in the first 60 s. The smoke records first `/health 200` and first `/health/ready 200` separately.

### Part 6: Pull-first default stack (W5)

**TESTER's answer to §9-Q1 (16 Oct).** No night or test tooling needs `--reload`. TESTER's other needs land in these stories:
- The production-stage backend keeps the `backend_logs` named volume at `/app/logs` and its `env_file`. `next-cycle.sh` edits `envs/api.local` and then runs `docker compose up -d --no-deps backend` (US-035, US-036).
- `docker compose up --build` keeps working for retest stacks built from an unmerged ref, and `AUTOMATOS_IMAGE_TAG` gives such a build a non-official tag such as `local-<ref>` (US-036).
- Every image, pulled or built locally, carries the `org.opencontainers.image.revision` label. The workspace-root `scripts/bootstrap.sh` (outside this repo) will read it instead of the `/app` bind-mount source it reads today (US-036).

#### US-035: Compose runs the production backend; hot reload moves to `make dev`
**Description:** As a self-hoster, I want the stack I run to be the image CI builds and scans.

**Acceptance criteria:**
- [ ] `docker-compose.yml`'s backend builds `target: production` (`:161`), with no source, entrypoint or healthcheck bind mounts (`:297-299`). Worker count comes from `WEB_CONCURRENCY=1` (`envs/api.defaults`, US-029).
- [ ] The backend keeps `backend_data:/app/data`, the `backend_logs` named volume at `/app/logs` (`:301-302`), both `/workspaces` mounts, and its `env_file`: `envs/api.defaults` plus the optional `envs/api.local` (`:180-186`). The compose guard tests assert all four (TESTER).
- [ ] `docker-compose.dev.yml` keeps `target: development`, the source mount and `--reload` for `make dev`, and takes over the entrypoint and healthcheck mounts.
- [ ] `docker-compose.dev.yml:4` and `self-hosting.md:903-914` say that the default stack runs the published production images, pulled from GHCR (US-036). They also say that `make dev` builds the development stages with the source mounted and `--reload`, for editing the code.
- [ ] `AGENTS.md` "Running things" (`:151`), `CONTRIBUTING.md` "Development environment" (`:45-57`, the Reload mode and Migrations bullets) and its `smoke-fresh-clone` row (`:92`) are rewritten in this wave:
  - `make dev` (`docker compose -f docker-compose.yml -f docker-compose.dev.yml up`) is the development environment.
  - Plain `docker compose up` and `make up` run the published images, not your checkout.
  - Migrations run once, from the entrypoint (US-029, US-030).
  - The `smoke-fresh-clone` row says what the lane runs after US-039.
- [ ] `test_f228_*` and the compose guard tests are updated.
- [ ] CI: the smoke asserts that:
  - the backend has no bind mount at `/app`, and its command line lacks `--reload`;
  - the process the entrypoint watchdog signals (`SERVER_PID`) is uvicorn itself, with no `sh` in between (from `docker top`);
  - `docker compose stop backend` completes in under 8 s with an SSE client connected (US-029 gives the production CMD `--timeout-graceful-shutdown 5`);
  - a key that only `envs/api.local` sets reaches the backend after `docker compose up -d --no-deps backend` (TESTER).

#### US-036: `make up` pulls the published images
**Description:** As a self-hoster, I want `make up` to download ready images, so that my first run takes minutes on any laptop and leaves no build cache behind.

**Acceptance criteria:**
- [ ] backend, frontend and workspace-worker declare `image: ghcr.io/automatosai/automatos-<x>:${AUTOMATOS_IMAGE_TAG:-edge}` next to `build:`. Plain compose pulls first and builds only when the image is missing.
- [ ] `images.yml` also tags each published image `tree-<12>`, from `git rev-parse HEAD:<context>` (`orchestrator`, `frontend`, `services/workspace-worker`). A main commit with no `sha-<7>` of its own (a `[skip ci]` merge, or the 4–8 min before `images.yml` publishes) can then still pull every image whose directory it did not change. Like `sha-<7>`, a `tree-<12>` tag is never re-pushed: schedule runs (US-011) skip both.
- [ ] `make up` picks one tag for the stack and says which one and why. It then starts the stack with `--no-build`, so compose never builds under an official tag. The rule:
  - `AUTOMATOS_IMAGE_TAG` from the shell or the make command line is used as given. An official tag that GHCR does not have stops `make up` with a message. A `local-*` tag that is not present locally is built.
  - On a clean checkout (`git status --porcelain` is empty for `orchestrator`, `frontend` and `services/workspace-worker`), it uses `edge` when `edge`'s `org.opencontainers.image.revision` label equals HEAD, which means a weekly OS rebuild of this commit (US-011). It reads the label from the registry without pulling. Otherwise it uses `sha-$(git rev-parse HEAD | cut -c1-7)` when that tag exists.
  - Otherwise the tag is `local-<sha7>`, or `local-<sha7>-dirty` when a context has changes. Each image whose context is clean and whose `tree-<12>` tag exists is pulled and re-tagged `local-<sha7>`. This works in a `--depth 1` clone, because HEAD's trees are present. The other images are built, and `make up` names them.
  - A `sha-*` or `tree-*` tag already present locally counts as found.
  - `make build` builds all three from source under the `local-*` tag.
- [ ] When the tag is `edge`, `make up` first runs `docker compose pull backend frontend workspace-worker`, because compose does not re-pull a tag it already has. Otherwise a checkout keeps the OS packages its `sha-<7>` was built with. `git pull && make up` picks up the weekly fixes. `AUTOMATOS_IMAGE_TAG=edge make up` follows `edge` and pulls on every call.
- [ ] `make up` and `make build` write the tag they used to `.env` as `AUTOMATOS_IMAGE_TAG=…`. It is the one key `make` owns there, and each run replaces its own earlier line. Later plain compose commands in the checkout then run the same images. Without this line, `next-cycle.sh`'s `docker compose up -d --no-deps backend` would recreate the backend from `edge`. `make up` ignores the line when it chooses again, so a pin goes in the shell or on the command line.
- [ ] Source builds outside `make` (TESTER):
  - `AUTOMATOS_IMAGE_TAG=local-<ref> docker compose up -d --build` builds the production stages from the checkout under that tag. No `local-*` tag is ever published, so a pull never replaces it.
  - Image names are global to the Docker engine. A retest or lab stack that runs next to the main one sets its own `local-<ref>` in its `.env`, not only in the shell.
  - Without a tag, a plain `docker compose up --build` tags its build `:edge` locally. The next `make up` that chooses `edge` pulls over it.
- [ ] Every app image carries `org.opencontainers.image.revision` (TESTER):
  - Published images get it from `images.yml`'s metadata-action.
  - Local builds get it from compose's `build.labels`, set from `AUTOMATOS_REVISION`. `make up`, `make build` and `make dev` export that as `git rev-parse HEAD`, with `-dirty` when a context has changes. A plain compose build without it gets `unknown`.
  - A `tree-*` image that `make up` re-tagged keeps the revision of the commit that built it, whose context tree is the same as HEAD's.
- [ ] `docker-compose.dev.yml` gives backend, frontend and workspace-worker local names (`image: automatos-<x>:dev`) with `pull_policy: build`, so a development build never takes a `ghcr.io/automatosai/…` tag. Without this, `make dev` (`up --build`) would tag its development-stage builds `…:edge`. A later plain `docker compose up` would find `:edge` present, never pull, and run the development stage without the source mount; after US-017 that stage has no app code.
- [ ] A pulled frontend has its build args baked in (`images.yml:85-89`): local edition, `localhost:8000` and `localhost:9000`. When any frontend build arg in `.env` (`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_WS_URL`, `NEXT_PUBLIC_MEDIA_ORIGINS`, `NEXT_PUBLIC_AUTH_EDITION`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`) differs from those values, `make up` builds the frontend locally and says why. A non-default `API_PORT` or `MINIO_PORT` without matching `NEXT_PUBLIC_*` values is reported as a misconfiguration. This lasts until a runtime-configured frontend exists (§9-Q5).
- [ ] `docker-compose.images.yml` and `make up-images` are deleted. The tests that parse them are updated: `test_compose_published_images.py`, `test_prd209_single_compose.py`, the images-overlay case of `test_f220_workspaces_survive_a_rebuild.py` (`:73`), and `test_818_every_make_target_has_its_commands_in_the_guide.py`, whose `COMPOSE_VARS` drops `IMAGES_COMPOSE` (`:17`). `self-hosting.md` §1b drops `make up-images`. It gives the commands behind `make up` and `make build`, with the tag rule in words for a shell without `make`.
- [ ] README, QUICKSTART and SETUP describe `make up` as a pull. They give the time and size from §8, not "about a minute", and say how OS fixes arrive (above).
- [ ] CI:
  - `smoke-images` (US-003) runs `make up` on a clean checkout with no tag exported, and asserts that nothing was built. Its bars are in §8. It records any Docker Hub 429 (§9, Docker Hub on the first run).
  - On the W5 PR, a job runs `make up` on the PR checkout with `AUTOMATOS_IMAGE_TAG=sha-<7 chars of the merge-base>` exported. It asserts that nothing was built (the three app images exist only under that tag, and no `local-*` tag exists), `/health/ready 200` and `:3000 200`. After merge, `smoke-images` repeats this on the new sha with no tag exported.
  - A unit test runs the tag rule against a stubbed registry, one case per branch.
  - A test parses `docker compose -f docker-compose.yml -f docker-compose.dev.yml config` (with the three secrets exported) and asserts that no service image starts with `ghcr.io/`. A second parses `docker compose config` with `AUTOMATOS_IMAGE_TAG=local-x` and finds the three app images on that tag, with their `build:` sections intact.
  - The build-path smoke (`make build`, US-039) reads each app image's revision label and finds HEAD. `smoke-images` finds the published sha.

#### US-037: `make up` writes the three secrets and refuses placeholders
**Description:** As a self-hoster, I want the stack to generate its own passwords, so that `git clone && make up` is the whole journey and no install runs on known passwords.

**Acceptance criteria:**
- [ ] A Makefile `env` target creates `.env` from `.env.example` when it is missing. `up`, `build` and `dev` run `@$(MAKE) --no-print-directory env` as their first recipe line, not as a prerequisite: the #818 guard's target regex (`^([a-z][a-z-]*):\s*$`, `test_818_…:18`) does not match `up: env`, so `up` would drop out of its parse. `self-hosting.md` §1b documents `make env` with its commands.
- [ ] `env` fills `POSTGRES_PASSWORD`, `REDIS_PASSWORD` and `API_KEY` with `openssl rand -hex 24` only when a value is empty or starts with `CHANGE_ME`. The `.env.example` values are `CHANGE_ME_secure_…` (`:15`, `:21`, `:28`), and the rule reuses the `(CHANGE_ME.*)?` match in `SETUP.md:114-128` (`:122`). It fills them only while the `automatos_postgres_data` volume does not exist yet, because Postgres keeps its first password. It never touches any other existing value.
- [ ] On an install that already started with a placeholder, `env` changes nothing. It stops and prints how to rotate:
  - Postgres: `ALTER USER … PASSWORD`, then the new value in `.env`.
  - Redis and `API_KEY`: the new value in `.env` alone, because both are read at every start (`docker-compose.yml:74`, `:277`).
- [ ] The backend entrypoint refuses to start a server command (US-029's distinction), with the same message, if any of the three starts with `CHANGE_ME`. An unset or empty value passes, because Helm and Railway set `DATABASE_URL` and `REDIS_URL` instead of the two passwords, and one-off commands (`docker run --rm img python -c …`, `test.yml:933`) run with no env.
- [ ] CI:
  - The smoke runs `make up` on a clean checkout with no secrets exported and finds three 48-hex values in `.env`.
  - A unit test runs the guard with each of the three `.env.example` values and expects a non-zero exit. With the three unset, it expects success.
  - A step creates `automatos_postgres_data` first, then runs `make env` on an `.env` that holds a placeholder. `.env` is unchanged and the exit is non-zero.

#### US-038: Health gates in seconds, on a stated Docker floor
**Description:** As a self-hoster, I want each service to start as soon as the one before it is ready.

**Acceptance criteria:**
- [ ] `start_interval: 1s` is added to the postgres (`pg_isready -h 127.0.0.1`, so it never passes on initdb's temporary socket-only server), redis, minio, backend and workspace-worker healthchecks. The Dockerfile HEALTHCHECKs gain `--start-interval` with `# syntax=docker/dockerfile:1`.
- [ ] The documented floor becomes Docker Engine 25 and Compose 2.24.4 (§9-Q2). On older engines Compose refuses `start_interval`. `make up` checks `docker version` first and stops with a message that names the floor.
- [ ] The frontend keeps `depends_on: backend: service_healthy` (the F228 guard). With the faster probe it starts within about 2 s of the backend turning healthy.
- [ ] CI, all timings from compose timestamps:
  - redis and minio turn healthy ≤ 2 s after start;
  - postgres is reported, not gated. On the smoke's fresh volume, `-h 127.0.0.1` passes only after initdb's restart, and Gerard sets its bar in §8 from the W5 number;
  - the frontend starts ≤ 3 s after the backend is healthy.

#### US-039: Source builds keep their cache and overlap with startup
**Description:** As a contributor building from source, I want `git pull && make build` to reuse the dependency layers.

**Acceptance criteria:**
- [ ] A CI experiment runs `docker compose build`, then `docker builder prune -f` (what `make clean` does after every `make up`, `Makefile:64`, `:98`), then a one-line source change and a rebuild. It records whether `npm install` (the frontend builder) and the pip install show `CACHED`.
- [ ] If they don't, `make clean` changes to `docker builder prune -f --filter until=168h`, or a storage cap, keeping the builder stages. The experiment then shows `CACHED`.
- [ ] This story adds no BuildKit cache mount. Railway fails a build whose cache mount lacks a hard-coded `id=s/<service-id>-<path>`, and the service ids are unknown (§10). So no Railway-built Dockerfile (orchestrator, frontend, workspace-worker, media-render, agent-opt-worker) gets one until they are known, and the experiment measures the layer cache only.
- [ ] When building, `make build` starts postgres, redis and minio first (`docker compose up -d postgres redis minio`), so their pull, initdb and health gate overlap the image builds.
- [ ] CI: the smoke runs `make build` instead of `docker compose up -d --build …` (`smoke-fresh-clone.sh:70`). The `health_status: healthy` event for `automatos_postgres` (from `docker events --filter event=health_status`) is timestamped before the backend image build finishes. The log line order cannot show this: compose prints `postgres Healthy` only while a later `up` waits on the dependency, which is after `backend Built`.

#### US-040: A five-container core stack
**Description:** As a self-hoster with a small machine, I want a documented way to run without the workspace worker, so that I can choose a smaller stack.

**Acceptance criteria:**
- [ ] `docker-compose.yml` keeps the worker in the default stack with no profile, so the default stays six containers (§9-Q3). A `docker-compose.core.yml` override gives `workspace-worker` `profiles: [workspace]`. `make up CORE=1` uses it, and §1b gives the plain command (`docker compose -f docker-compose.yml -f docker-compose.core.yml up -d`). That runs the core five: postgres, redis, minio, backend, frontend.
- [ ] Note: the core stack is an override, not `COMPOSE_PROFILES=workspace` in `.env.example`, for two reasons:
  - A `--profile X` flag replaces `COMPOSE_PROFILES` instead of adding to it. Every documented `docker compose --profile all|search|memory|media up` (QUICKSTART.md:154, :190, :253; self-hosting.md:299, :306, :537, :568) and `self-hosting.md:571`'s `COMPOSE_PROFILES=media` would then start without the worker.
  - Existing `.env` files lack the key, and `make env` adds only the three secrets, so upgraded installs would lose the worker without an error.
  With the override, existing `.env` files, those commands, `smoke-fresh-clone.sh` and `test_prd233_s1_worker_local_profile.py:238-242` stay valid as they are.
- [ ] The backend's `depends_on: workspace-worker` gains `required: false` (`:173-174`).
- [ ] Both backend stages create `/workspaces` owned by 1000:1000 before `USER` (`RUN mkdir -p /workspaces && chown 1000:1000 /workspaces`). A fresh `workspaces_data` volume is then uid 1000 whichever service mounts it first. Without this, the backend mounts it first in the core stack (`:309`), and Docker creates it root-owned. A worker started later would then re-own everything under `/workspaces` except `*/projects` (`services/workspace-worker/entrypoint.sh:6-12`), including the bind-mounted host folder. That is the failure F220's start order prevents. A static test asserts the `mkdir` and `chown` in both stages.
- [ ] With no worker, the workspace endpoints and agent workspace tools return a clear "workspace worker not running" error, never a 500 (`core/workspace_client.py:54-60`).
- [ ] CI:
  - A smoke variant with `-f docker-compose.core.yml` reaches `/health/ready 200` with five containers, and a workspace endpoint returns that error. It then starts the worker and asserts that a file in `AUTOMATOS_WORKSPACE_DIR` keeps its host owner.
  - `make up` from an `.env` written before W5 starts six containers.

#### US-041: MinIO runs as its own user
**Description:** As a self-hoster, I want object storage not to run as root.

**Acceptance criteria:**
- [ ] `user: "0:0"` is removed (`docker-compose.yml:110-112`). The Chainguard image runs as 65532.
- [ ] A one-shot compose service, `minio-volume-owner`, repairs old volumes:
  - it runs with `user: "0:0"` on the stack's digest-pinned `redis:7-alpine` image, which is already pulled and has busybox `stat` and `chown`, and mounts `minio_data`;
  - it runs `chown -R 65532:65532 /data` only when `/data` is not owned by 65532, logs that it did, and exits 0;
  - `minio` gains `depends_on: minio-volume-owner: condition: service_completed_successfully`.
  So `make up`, plain `docker compose up` and the §1b commands all repair a volume the official image wrote as root. The backend waits on `minio: service_healthy` (`:169-170`), so a repair that lived only in the Makefile would leave plain-compose upgrades with no backend.
- [ ] CI: a fresh volume uploads a document and the host fetches its presigned URL. An upgrade case starts from a volume pre-populated as root, once through `make up` and once through plain `docker compose up`. In both, MinIO is healthy and the old object reads back.

#### US-042: The self-host docs and the chart tell the truth
**Description:** As a self-hoster, I want the docs to describe what actually starts and what it costs.

**Acceptance criteria:**
- [ ] `self-hosting.md` §3, `docker-compose.yml:19` (`minio-init`, deleted long ago), `envs/frontend.defaults:11-13` and the disk table (`self-hosting.md:21`) match the stack, including `minio-volume-owner` and the core override. The disk and time figures come from §8's measured numbers.
- [ ] `self-hosting.md` §10 (Updating) says what W5 changes for an existing install: `make up` pulls instead of building, the source mount moves to `make dev`, `.env` gains the `AUTOMATOS_IMAGE_TAG` line, and MinIO's volume is re-owned once (US-041).
- [ ] The self-host docs use `git clone --depth 1` (79 MB of tree against a 700 MB full history). `docs/assets/automatos-oss-launch-16x9.mp4` (44.4 MB, unreferenced in the repo) moves to a release asset; links to it outside the repo (the openlaunch kit) are updated first. A repo check fails on any newly tracked file over 5 MB.
- [ ] A docs test compares the service and profile names in `self-hosting.md` §3 with `docker compose config --services`, run per profile and with the core override.
- [ ] Helm:
  - `_helpers.tpl:40-41` renders `repository@digest` when `image.digest` is set.
  - The default no longer combines a moving `edge` with `IfNotPresent` (`Chart.yaml:9`, `values.yaml:48-49`, `:86`, `:100`): `appVersion` is stamped at package time, or `pullPolicy: Always` when the tag is `edge`.
  - `deploy/kind/datastores.yaml:19`, `:62` are pinned by digest.
  - helm-unittest covers both.

### Part 7: Worker diet (W6)

#### US-043: A browser only where one runs
**Description:** As a self-hoster, I want the worker without Playwright, because the local edition has no browser for it to drive. Railway's worker should carry the smallest browser that works.

**Not in this story:** moving `workspace_html_to_png` to media-render. That would take Chromium and Playwright out of the Railway worker altogether (about 0.63 GB installed after this story), but it is an architecture change: media-render would need the HTML and its assets over HTTP, because a Railway volume attaches to one service. It is open in §9 (the html_to_png question), and production usage decides it (§10).

**Acceptance criteria:**
- [ ] `playwright` is installed only when `INSTALL_BROWSER=true`. `requirements.txt:27` moves to `services/workspace-worker/requirements-browser.in`, locked as `requirements-browser.lock` (US-044), and the local and GHCR builds drop about 136 MB installed.
- [ ] With the browser:
  - `playwright install --only-shell chromium` replaces the full Chromium. The launch at `executor.py:545` is headless with no channel, so it uses the headless shell; this saves 604 MB installed on Railway.
  - `driver/node` is deleted in the same RUN as the pip install, with `ENV PLAYWRIGHT_NODEJS_PATH=/usr/local/bin/node`.
  - The browser step moves above `COPY . .`, so a code change no longer re-runs it. `useradd` comes first, with `/app`, `/workspaces` and the cache directory created worker-owned (`install -d -o worker -g worker /app /workspaces /home/worker/.cache`), so `HOME` stays worker-owned.
- [ ] The code goes in last, with `COPY --chown=1000:1000 . .`, and the `chown -R worker:worker /app /workspaces` RUN goes. This moved here from US-016, because this story rewrites the same block.
- [ ] The browser apt list drops what the headless shell doesn't link: `libcups2t64`, `libpango-1.0-0`, `libcairo2`, `libxshmfence1`. `fontconfig` is added to the list explicitly, as `services/media-render/Dockerfile:33` already has it. Today it arrives only through libpango and libcairo, and without it the shell's bundled fontconfig finds no `/etc/fonts`. `fonts-unifont` stays as the CJK fallback.
- [ ] The Mesa/LLVM chain is purged as in US-051 if the render smoke passes. `libgbm.so.1` needs only libdrm and libexpat.
- [ ] The Playwright pin moves from `<1.53` (Chromium 136, cap reason unrecorded) to the current release, if the render smoke passes.
- [ ] CI, in `prod-image-parity`:
  - no `chromium-[0-9]*` directory exists, but `chromium_headless_shell-*` does;
  - `ldd` on the shell finds nothing missing;
  - an in-container render runs as uid 1000 (`set_content` then `screenshot`) and produces a PNG over 0 bytes, including a CJK fixture, which proves the fonts resolve.
  - The local variant asserts `! python -c 'import playwright'`.

#### US-044: Worker dependencies locked, with the SDK in its own layer
**Description:** As Dave, I want the worker's Python set fixed by a lock, and an SDK bump to re-download only the SDK.

**Acceptance criteria:**
- [ ] `services/workspace-worker/requirements.lock` is generated with `uv pip compile --universal --python-version 3.12 --generate-hashes` and installed with `--require-hashes`. Without `--universal`, uv resolves for the machine that runs the compile, but the lock installs on linux/amd64 and linux/arm64.
- [ ] `requirements-browser.in` (`playwright`, US-043) is locked the same way to `requirements-browser.lock`, compiled with `-c requirements.lock`, so the packages it shares with the main lock (typing-extensions) keep the main lock's pins. The browser RUN installs it with `--require-hashes` when `INSTALL_BROWSER=true`.
- [ ] uv comes from a named stage, `FROM ghcr.io/astral-sh/uv:<ver>@sha256:… AS uv`, and `COPY --from=uv /uv /uvx /usr/local/bin/`. This replaces the separate `pip install uv` step, and the stage keeps the pin where Dependabot reads it (FR-9). uv stays at the path agents use today (`executor.py:47` allows `uv`; D5), and the installs use that binary.
- [ ] **No cache mount yet.** Railway builds this Dockerfile until W9, which is held until after WebSummit (§9-Q11). Railway's builder refuses a cache mount with no id, or with an id that lacks the hard-coded `s/<service-id>-` prefix, and the id can't use a variable. The worker's service id is unknown (§10), so the installs run without a cache mount. Once the operator supplies the id, each install becomes one `RUN --mount=type=cache,id=s/<worker-service-id>-/root/.cache/uv,target=/root/.cache/uv`. A static test fails on any `--mount=type=cache` in a Railway-built Dockerfile (orchestrator, frontend, workspace-worker, media-render, agent-opt-worker) whose id does not start with `s/`.
- [ ] `claude-agent-sdk` is pinned exactly, in its own RUN placed last among the installs, after the browser step. Its transitive dependencies (`mcp`, starlette, uvicorn, httpx and others) sit in the stable layer, so an SDK bump moves only its own layer. Today it floats from 0.1.0, a range that allows versions without the bundled CLI.
- [ ] SQLAlchemy leaves the worker (27.7 MB). The one UPDATE (`_update_db_execution`, `main.py:406-441`) uses psycopg2 inside `asyncio.to_thread`, with bound `%(name)s` parameters, normalising a `postgresql+psycopg2://` URL first. Today the UPDATE runs synchronously on the event loop that also serves health, canvas and file routes. `greenlet` leaves the main lock with SQLAlchemy and stays only in the browser lock, because Playwright requires it.
- [ ] CI: every line of both locks is `==` with `--hash`; `pip show claude-agent-sdk` in the image equals the lock; `uv --version` in the image equals the pin; `python -c 'import sqlalchemy'` fails in the image; the worker tests pass.

#### US-045: The worker boots safely and reports healthy quickly
**Description:** As a self-hoster, I want the worker's start-up to be quick and to never re-own files outside its volume.

**Acceptance criteria:**
- [ ] `entrypoint.sh:11` becomes `find "$WORKSPACE_VOLUME_PATH" -mindepth 1 -path '*/projects' -prune -o ! -user worker -exec chown -h worker:worker {} + || echo "[entrypoint] some files could not be re-owned" >&2`. It no longer follows symlinks an agent or a cloned repo created, and it skips files already owned. A failed chown stays non-fatal under `set -e`, as today's `2>/dev/null || true` made it, but it is now logged.
- [ ] `COPY --chmod=755 entrypoint.sh` replaces the separate `chmod` RUN, which today publishes an empty layer.
- [ ] `/metrics` returns 200. `metrics_handler` in `automatos_metrics.py` passes prometheus_client's content type, which includes a charset, as aiohttp's `content_type`, and aiohttp raises on that. The same fix goes to `services/shared` and the other copies.
- [ ] The HEALTHCHECK uses `--start-interval=2s`, and this story adds `# syntax=docker/dockerfile:1` as the Dockerfile's first line (the compose side is US-038). W6 can merge before W5, and without the syntax line a build on Engine 24 rejects `--start-interval`.
- [ ] CI: a shell test in a throwaway container makes a root-owned directory with a symlink to a root-owned file, runs the ownership block, and asserts the target is still root-owned while the link is worker-owned. In the same test, a read-only mount makes the block log `some files could not be re-owned`, and the script carries on. The smoke sees the worker healthy within 20 s and `/metrics` returning 200.

### Part 8: Frontend diet (W7)

#### US-046: The standalone build without sharp, libvips or TypeScript
**Description:** As a self-hoster, I want the frontend image to carry only what the Next server runs.

**Acceptance criteria:**
- [ ] `next.config.js` sets `images: { unoptimized: true }` (§9-Q12: yes) and `outputFileTracingExcludes: { '*': ['node_modules/@img/**', 'node_modules/sharp/**', 'node_modules/typescript/**', 'node_modules/next/dist/compiled/amphtml-validator/**'] }`. The standalone layer goes from 26.0 to about 7.8 MB compressed.
- [ ] The only optimised image, the 48 px `public/brand/automatos-mark.png` on the SaaS sign-in pages, is pre-resized (§9-Q12).
- [ ] CI: the image has no `node_modules/@img`, `sharp` or `typescript`, and its logs never mention sharp. `/chat` returns 200 on a local build. `/sign-in` returns 200 on a SaaS build. No CI build passes Clerk keys today, and `clerkMiddleware` and `ClerkProvider` throw without them, so the SaaS build gets format-valid production-instance dummies. The workflow generates them at run time and never commits them, because gitleaks' Stripe rule matches `sk_live_…`:
  - `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_$(printf 'clerk.example.com$' | base64)`, passed as a build arg;
  - `CLERK_SECRET_KEY=sk_live_$(openssl rand -hex 16)`, passed at run time.

  A signed-out request needs no network.

#### US-047: Install once, configure after
**Description:** As a contributor, I want a change to a public setting not to reinstall every frontend dependency.

**Acceptance criteria:**
- [ ] `frontend/.npmrc` sets `legacy-peer-deps=true`, so Docker and CI resolve the same tree. Today they install 1,089 and 1,096 packages. The tsc baseline proves that losing the peer-only `@types/dom-mediacapture-record` changes nothing.
- [ ] frontend-ci's `Install dependencies` step (`test.yml:1030`) becomes `npm ci --no-audit --no-fund`, and the comment that explains `npm install` (`test.yml:1004-1006`) goes. Today CI's `npm install` quietly reconciles a lock that is out of sync, the case where Docker's `npm ci` fails. The frontend-ci and Docker logs print the same `added N packages`.
- [ ] The Dockerfile runs `npm ci --no-audit --no-fund` right after copying `package.json`, `package-lock.json` and `.npmrc`. The builder stage's `NEXT_PUBLIC_*` ARG/ENV block moves below `COPY . .`, just above the build. `NEXT_TELEMETRY_DISABLED=1` is set.
- [ ] **No cache mount yet.** Railway builds this Dockerfile, and it fails a build whose cache mount lacks a hard-coded `id=s/<service-id>-<path>`. The frontend's service id is unknown (§10), so the Dockerfile has no cache mount until the operator supplies it. Then cache mounts for `/root/.npm` and `/app/.next/cache` use that form, and US-044's static test covers them. Until then, the build RUN ends with `rm -rf .next/cache`, which keeps webpack's cache out of the 171 MB builder layer exported to the GHA cache; the `.next/cache` mount replaces it later. `FRONTEND_CACHE_BUST` stays.
- [ ] A PR that touches only `frontend/.npmrc` or `frontend/next.config.js` builds and scans the frontend image and runs `prod-image-parity`. `images.yml`'s `pull_request.paths`, widened by US-001, covers both files; this story adds them if it does not. `PROD_IMAGE_PATHS` (`test.yml:89`) gains `frontend/\.npmrc`.
- [ ] CI: a PR that changes only a build arg shows `npm ci` CACHED in `images.yml`'s frontend build. The parity step's frontend time is reported in the summary against today's 336–370 s, report-only (§7). Moving parity to buildx with the GHA cache is listed in Appendix C (§9-Q13).

#### US-048: Unchanged frontend layers keep their digest
**Description:** As a self-hoster, I want a frontend code update to download only the code that changed.

**Acceptance criteria:**
- [ ] The production stage copies `public/` from the build context before the standalone output. The builder moves `.next/standalone/node_modules` out of the standalone directory, and the runtime copies it as its own earlier layer. Layer order: node_modules, public, static, server.
- [ ] `frontend/.dockerignore` adds `**/__tests__`, `**/*.test.*`, `tsconfig.tsbuildinfo` and `**/*.md`; the existing `*.md` matches only the root.
- [ ] CI (US-001): a frontend code-only main update has an update delta of 8 MB or less (today 38.2 MB). Two consecutive builds that leave `frontend/public` alone keep the same public layer digest.

#### US-049: `public/` holds only what the app serves, at the size it's shown
**Description:** As a self-hoster, I want the frontend's static files sized for how they're shown.

**Acceptance criteria:**
- [ ] The 283 files in `public/logos` are resized to at most 128 px and optimised, keeping names and paths. The backend refers to them by name (`core/credentials/types.py` and the 418 references in `core/database/credential_types_seed.json`), and the frontend builds `/logos/${brand}.png` from tool names (`lib/logo-service.ts:245`). This takes them from 7.71 to about 1.86 MB.
- [ ] Three files are not images, and each is replaced with a real image under the same name:
  - `Microsoft.png` and `Nginx.png` are HTML documents, which `getLogoUrl` reaches by brand name;
  - `Sentry.png` is 6,904 bytes in no image format (it starts `15 2d 3c 00`), and the seed references `/logos/Sentry.png` (`credential_types_seed.json:2542`, `:5770`, `:7672`).

  The 73 ICO files named `.png` are re-encoded.
- [ ] After a production database check for references (§9-Q12), `public/images` (4 blog covers), `public/favicon_io` and `modal-fix.js` are deleted. None of them is referenced in this repo, automatos-skills, the landing repos or the gitbook.
- [ ] The mascot posters in `public/auto` are compressed lossily only if design signs off (§9-Q12).
- [ ] CI: a frontend-ci step fails when `public/` grows past its budget, when any file under `public/logos` is not a real PNG or SVG by its magic bytes, or when a logo PNG is wider than 256 px.

### Part 9: Media renderer (W8)

**Order and proof for this wave:**
- A small PR merges before W8's. It adds the job that publishes the media assets image (US-053), which is dispatched on main once Gerard has picked the D4 model (US-050).
- US-053 lands first inside W8. US-051's strip and its curl and unzip removal use US-053's builder stages.
- `media-render.yml` runs only on `workflow_dispatch` (`media-render.yml:7-8`; Gerard, 7 Oct), so the W8 PR's own CI never renders. The W8 PR is mergeable only with a `media-render.yml` run dispatched on its head commit (`gh workflow run media-render.yml --ref <branch>`), and the PR records its run id. "The proof run" in US-050 to US-053 means that run.

#### US-050: The fp16 voice model (D4)
**Description:** As Gerard, I want the renderer's voice model at half the size with no audible loss.

**Acceptance criteria:**
- [ ] A listening test compares the two candidates in §3-D4 against fp32 on the same script. Gerard picks by ear, and the choice is recorded here.
- [ ] The model in the assets image (US-053), `kokoro/SHA256SUMS`, the `config.py:179` default and `test_the_kokoro_files_are_hash_checked_at_build` (which pins the file names) move to the chosen file. If it is the v1.1 export, `synthesize_line`'s `model.create` call (`kokoro_tts.py:59-64`) passes `sentence_pause=0, clause_pause=0`, keeping today's timing.
- [ ] CI, in the proof run: `test_kokoro.py`, the script-window step (`media-render.yml:191-223`, `--max-tempo 1.25`) and the music LUFS step pass. First-render `voice_seconds` is no more than 1.25 times the run-37612086159 baseline (2.394 s for one line), because the CPU execution provider has few fp16 kernels. The image shrinks by 140 MB or more.

#### US-051: The renderer drops what renders never load
**Description:** As a self-hoster on `--profile media`, I want the renderer without libraries and files it never uses.

**Acceptance criteria:**
- [ ] The apt RUN ends with `dpkg --purge --force-depends` of the packages reachable only through `mesa-libgallium` (`libllvm19`, `mesa-libgallium`, `libz3-4` and the rest of that closure), about 200 MB. `libgbm.so.1` stays. Renders run with the GPU disabled, and the headless shell brings its own SwiftShader. Debian's ffmpeg stays, keeping its security updates.
- [ ] Chrome's `locales/*.pak` are deleted except `en-US.pak` (amd64; −50.8 MB).
- [ ] The espeak data is moved, not copied, to `/opt/espeak` in the same RUN as the pip install, in US-053's `py` stage (−18.4 MB). The `cp -r` regex in `test_the_espeak_data_is_copied_to_a_short_real_path` is updated.
- [ ] The unlinked `onnxruntime/capi/libonnxruntime.so.*` is deleted (−29 MB; the pybind module links none of it). So are numpy's `tests` directories.
- [ ] The arm64 headless shell is stripped (`--strip-unneeded`; about 130 MB of `.symtab`/`.strtab`; the amd64 build already ships stripped). Neither the runtime nor `python:3.12-slim` has `strip`, so this runs in US-053's `chrome` builder stage. That stage runs `apt-get install -y --no-install-recommends binutils` first, and it fails the build if `readelf -S` still lists `.symtab`.
- [ ] curl and unzip leave the runtime; downloads happen in US-053's builder stages. The HEALTHCHECK and media-render's compose `healthcheck` use Python `urllib`. The Dockerfile HEALTHCHECK gains `--start-interval=2s`, and `# syntax=docker/dockerfile:1` becomes the file's first line, above the comment header. The compose probe's `start_interval` comes with US-038 (W5), whose list of healthchecks gains media-render, so it lands with the Engine 25 floor check.
- [ ] CI, in the proof run (the in-image suite, `tests/test_image.py`, gains the image checks):
  - `ldd` on ffmpeg, ffprobe and the headless shell finds nothing missing;
  - `dpkg -s mesa-libgallium` and `dpkg -s libllvm19` each fail (one package per check);
  - `test $(ls /opt/chrome/locales | wc -l) -eq 1` on amd64;
  - `python -c "import onnxruntime as o; print(o.get_available_providers())"` succeeds;
  - every `media-render.yml` render and the template shards' pixel probes pass;
  - `media-render.yml` fails when `docker image ls` reports the amd64 image above 1.30 GB on disk (§8). US-054 checks the same size in `images.yml` on every PR that touches the renderer.

#### US-052: The renderer's language data matches what the product speaks
**Description:** As a self-hoster, I want the renderer to carry the voices and dictionaries the platform offers, not all of upstream's.

**Acceptance criteria:**
- [ ] Only `en_dict` stays among the espeak dictionaries, plus the phoneme, intonation, language and voice data. The orchestrator only ever sends `en-us` or `en-gb` (`kokoro_voices.py:17`, `:93`), and media-render defaults to `en-us` (`config.py:263`). The other 113 dictionaries come to 17.46 MB.
- [ ] `voices-v1.0.bin` is rewritten in a build stage, after the upstream sha256 check, to the 27 catalogue voices (about −14 MB). The media-render build context (`services/media-render`, for compose and Railway alike) cannot read the orchestrator, so the catalogue's voice ids are copied into `services/media-render/media_render/voices.json`, and the build stage reads that file. An orchestrator test asserts it equals the ids in `modules/socials/kokoro_voices.py:19-47`.
- [ ] Validation follows the data. Today `VOICE_NAME` and `LANGUAGE` are patterns (`validate.py:17-18`), so a code such as `fr-fr` would pass and then fail inside espeak once its dictionary is gone.
  - `validate.voice_name` rejects any voice not in the rewritten voices file, whose names are read at boot.
  - `validate.language` accepts only `en-us` and `en-gb`.
  - Each refusal is a 400 that names the field. Boot refuses a `MEDIA_RENDER_KOKORO_LANG` outside that list, with a message that names the variable.
- [ ] CI, in the proof run: `test_kokoro.py`, the script windows, one synthesized line per catalogue voice, and a phoneme snapshot for a fixed English sentence all pass. A voice outside `voices.json` and a language other than `en-us` or `en-gb` each get a 400 that names the field.

#### US-053: Parallel build stages, locked dependencies, stable big layers
**Description:** As a contributor, I want the renderer to build in parallel stages from locked dependencies, so that it builds faster and its big layers keep their digests.

**Acceptance criteria:**
- [ ] This story lands first in W8 (Part 9 note).
- [ ] Stages:
  - `kokoro`: the model and voices from the assets image, checked against `kokoro/SHA256SUMS`;
  - `js`: `npm ci` from a new `services/media-render/js/package.json` and `package-lock.json`, which pin Hyperframes exactly and its caret-range dependencies;
  - `chrome`: the headless shell. It builds FROM `js`, because amd64 runs `hyperframes browser ensure` and arm64 runs the pinned `playwright-core`, and both need Node. The alternative is to pin the download URLs and sha256s (Chrome for Testing at Hyperframes' `CHROME_VERSION`; Playwright's headless shell at the pinned `playwright-core`), with a test that keeps them in step with the pins;
  - `py`: `uv pip install --require-hashes` from a `requirements.lock` generated as US-044 does for the worker;
  - `music`: on `py` plus ffmpeg, which `music_build.py` needs; the tracks come from the assets image and are checked against `music/manifest.json`.

  The runtime stage `COPY --link`s each one. `production` stays last.
- [ ] The `HYPERFRAMES_*` telemetry and update-check ENV block is set in the `js` stage, which `chrome` builds from, before `npm ci` and before any `hyperframes` call. It stays in the runtime stage, which runs the CLI at render.
- [ ] `media_render/versions.py:39` (`npm root -g`) reads Hyperframes' and GSAP's versions from the `js` stage's `node_modules` in the image, and the runtime puts that tree's `.bin` on `PATH`, so `hyperframes --version` (`tests/test_image.py`) still answers.
- [ ] **Assets image.** The Kokoro and music files are mirrored into a small image, `ghcr.io/automatosai/automatos-media-assets`, so builds stop fetching from GitHub releases and `ende.app`. PR builds never push to GHCR, so the image comes from the PR that merges before W8's (Part 9 note):
  - That PR adds a `media-assets` job to `images.yml` that runs only on a manual dispatch on main. It fetches the files, re-checks every sha256 and then publishes. The files are the D4 model Gerard picks (US-050), the voices file and the music tracks.
  - Its `org.opencontainers.image.licenses` label is `Apache-2.0 AND CC-BY-4.0`, not the `Apache-2.0` that `images.yml` stamps on every image. It carries `music/manifest.json`'s attribution lines and the Kokoro licence notice. It holds no GPL code, so publishing it is separate from §9-Q6.
  - Operator: an org owner makes the package public after its first publish (§10). Railway's media-render build and a self-hoster's `--profile media` build pull it with no credentials.
  - The Dockerfile names it in its own stage, `FROM ghcr.io/automatosai/automatos-media-assets@sha256:… AS assets` (FR-9). The `kokoro` and `music` stages still check every file against `kokoro/SHA256SUMS` and `music/manifest.json`, so the bad-hash CI step still fails on a wrong sha256.
  - CI: an anonymous pull of the pinned digest succeeds, and a build of `services/media-render` with no registry login passes (the proof run's build step).
- [ ] Boot: Kokoro loads in a background thread once the listener is up, and `/health` answers at once. `NODE_COMPILE_CACHE` is seeded at build. The build-time "pre-warm" comment above the Chrome launch RUN is corrected: that step is a smoke test.
- [ ] The same PR updates the guard tests the restructure moves:
  - in `orchestrator/tests/test_prd251w1_media_render_image.py`, the Hyperframes pin test (`test_hyperframes_is_pinned_exactly_on_node_22`, renamed for Node 24 by US-007) reads the exact pin, with no caret, from `js/package.json` and `js/package-lock.json`;
  - in the same file, `test_the_image_switches_off_telemetry_skills_and_update_checks`, which today calls `DOCKERFILE.index("npm install -g")`, asserts the `HYPERFRAMES_*` block is set in the `js` stage before `npm ci` and before any `hyperframes` call;
  - `test_chrome_is_preinstalled_at_build_not_fetched_per_render` follows the new `chrome` step, and `test_the_kokoro_files_are_hash_checked_at_build` follows the sha256 check;
  - `orchestrator/tests/test_prd251w1_music.py:189`, which calls `dockerfile.index("pip install --no-cache-dir -r requirements.txt")`, asserts that the `music` stage builds from `py`, before the app copy;
  - the `services/media-render/Dockerfile` entry of `PROD_DEFAULTS` in `test_dockerfile_prod_parity.py` (empty today) registers any new defaulted ARG with its production value.
- [ ] `.github/dependabot.yml` gains `pip` for `/services/media-render` and `npm` for `/services/media-render/js`, weekly and grouped, as US-012 sets for the other locks.
- [ ] CI, in the proof run:
  - `media-render.yml` reports `image build: N s` against 93 s today, report-only (§7, §9-Q13).
  - Two builds that differ only in the Hyperframes pin give the same Kokoro layer digest.
  - The job log shows that the background Kokoro load finished before the first POST, and the first render's `voice_seconds` is below 2.394 s, the run-37612086159 figure, which included the model load.
  - The image's `NODE_COMPILE_CACHE` directory is non-empty and readable by uid 1000. `check_seconds` is reported against the 5.009 s baseline without failing, because the expected 0.1–0.3 s gain is within run-to-run noise.

#### US-054: The renderer is built and scanned in `images.yml`
**Description:** As Dave, I want the renderer under the same scan and size gate as the other images, although we don't publish it.

**Acceptance criteria:**
- [ ] `images.yml` builds `services/media-render` (target `production`) for amd64 and arm64 on native runners. On PRs and on main it pushes only to the local registry (US-001) and scans with the US-013 policy. It is **not** published to GHCR (§9-Q6: built and scanned, not published), because the image carries GPL-3.0 espeak-ng/phonemizer and GPL x264.
- [ ] It checks its budget: the on-disk size (`docker image inspect -f '{{.Size}}'` after pulling from the local registry) against §8's 1.30 GB amd64 bar. A compressed-size entry in `.github/image-budgets.json` also tracks growth (FR-2).
- [ ] media-render has never been scanned, so its first scan in this wave is report-only. Each fixable Critical or KEV match it shows is fixed (US-051, US-053) or allowlisted in `.grype.yaml` with a reason and an expiry of 90 days or less. Switching the policy to block on this image is the wave's last commit.
- [ ] The cache is `type=registry` on a private GHCR buildcache ref (§9-Q13). Main writes it and PRs only read it, so a 1.8 GB × 2-arch image does not evict the other images' GHA cache.
- [ ] `images.yml`'s `pull_request.paths` adds `services/media-render/**`.

### Part 10: One image to production (W9, D2)

W9 is held until after WebSummit (§9-Q11). Besides W3, W6 and W7 it needs W4, because US-057 builds on US-029's migration flags and US-030's head check (§4).

#### US-055: Railway variants are published, signed and scanned
**Description:** As Dave, I want every image production runs to come out of `images.yml`.

**Acceptance criteria:**
- [ ] The API needs no variant after US-027. The worker publishes a browser variant (`INSTALL_BROWSER=true`, tag suffix `-browser`).
- [ ] The frontend publishes a SaaS variant (tag suffix `-saas`; §9-Q5 leaves runtime configuration for later). It is built with every `NEXT_PUBLIC_*` ARG the Railway frontend service sets today, taken from repo variables: `NEXT_PUBLIC_AUTH_EDITION=saas`, the API/WS URLs, the Clerk publishable key, the four `NEXT_PUBLIC_CLERK_*_URL` values, and `NEXT_PUBLIC_MEDIA_ORIGINS` if set. A value left out falls back to the Dockerfile default (`/dashboard`, `https://*.amazonaws.com`), which can change sign-in redirects and the CSP. The operator copies the current values from the Railway service before the switch, and CI greps the built chunks for each one.
- [ ] Every variant goes through the scan policy, the budget check and cosign signing. It carries the same OCI labels as the other images, `org.opencontainers.image.revision` included (§2). `cosign verify` passes on each index.

#### US-056: Railway deploys the digest CI built
**Description:** As Dave, I want production to run exactly the digest that was scanned and signed.

**Acceptance criteria:**
- [ ] A separate `.github/workflows/deploy.yml` does the deploy. `images.yml` never waits on an approval, so its concurrency group, its `sha-<7>` tags and US-003's `workflow_run` smoke are unaffected. deploy.yml runs:
  - on `workflow_run` (workflows: images, types: completed, branches: main) when that run concluded `success` and its event was `push` or `schedule`, so the US-011 weekly rebuild's OS fixes reach production too;
  - on `workflow_dispatch`, with one digest input per service.
- [ ] Each `images.yml` `merge` leg uploads its index digest as an artifact (`index-digest-<image>`, one per image and variant), because a matrix job's outputs cannot carry three digests. A first deploy.yml job, outside the environment, downloads them using the triggering run's id and passes the digests to the deploy job, so a long approval wait cannot outlive the artifacts.
- [ ] The deploy job has `concurrency: {group: deploy-production, cancel-in-progress: false}` and the GitHub environment `production` with required reviewers. Reviewers approve once that run's `smoke-images` (US-003) has passed. Before it points Railway at a digest, the job runs `cosign verify` against `images.yml`'s signing identity, so a dispatch input cannot deploy an unsigned digest. It then points each image-sourced Railway service (the API, the `-browser` worker and the `-saas` frontend) at `ghcr.io/automatosai/automatos-<x>@sha256:<index digest>`, and triggers a deploy through Railway's API with a scoped `RAILWAY_TOKEN`. The mutation names are confirmed against Railway's schema.
- [ ] A service is redeployed only when its image changed. Every main build makes a new index digest (provenance and SBOM attestations, the build timestamp), and the worker has a volume, so each of its redeploys takes it down. The job compares the new platform manifest's layer digests and the image config's runtime fields (Env, Entrypoint, Cmd, User, WorkingDir, ExposedPorts, Healthcheck; labels and timestamps ignored) with those of the digest Railway runs. It skips services where both are equal and lists them in the step summary. A `workflow_dispatch` input forces all three.
- [ ] After the deploy, the job reads the deployed image back from Railway and asserts it equals the index digest. `/health/ready` must answer 200 within the healthcheck timeout. The step summary records the digest each service now runs.
- [ ] `prod-image-parity` builds the `-saas` frontend and the `-browser` worker with the same build args `images.yml` publishes them with (one shared definition, so the two cannot drift). It runs its edition, Leiden and browser assertions on those builds instead of on default-argument builds, and its frontend `cache-from` (US-047) reads the `-saas` variant's cache. The comments that say Railway builds these files are corrected: `images.yml`'s header, the `NEXT_PUBLIC_AUTH_EDITION` ARG comment in `frontend/Dockerfile`, the `INSTALL_BROWSER` ARG comment in `services/workspace-worker/Dockerfile`, and the `prod-image-parity` header in `test.yml` (`test.yml:903-909` at 697ca7d6e).
- [ ] Operator: disconnect those services' GitHub autodeploy. Set a healthcheck with a 300 s timeout, plus overlap and draining, in the service settings: `/health` on the API and the worker, and `/favicon.ico` on the frontend. The SaaS middleware protects every other non-public path, and its matcher skips files with an extension. An image-sourced service reads no `railway.json`, and a Dockerfile-built one reads it only from a path set in the dashboard, because Railway's config file does not follow the root directory. Store `RAILWAY_TOKEN` as a `production` environment secret.
- [ ] Rollback is a `workflow_dispatch` of deploy.yml with the previous digests, taken from the last good deploy's step summary. The previous image's `migrate-schema` fails on a database ahead of its code (US-030). So a rollback dispatch clears the API service's pre-deploy command through Railway's API for that deploy, and every forward deploy sets it back. Migrations stay backward-compatible with the previous release (expand now, contract one release later), so the previous image serves on the newer schema. The runbook says so.
- [ ] CI: `smoke-images` also starts the previous main run's published API image, with `AUTOMATOS_SKIP_MIGRATIONS=true`, against the schema the new image migrated, and asserts `/health/ready` 200.

#### US-057: Migrations run before the deploy, never inside the serving container
**Description:** As Dave, I want a failed migration to stop a deploy, not take down a serving container.

**Acceptance criteria:**
- [ ] The entrypoint gains a `migrate-schema` mode. It runs only US-029's migration step: the database wait, `init_fresh_db` for an empty or interrupted database, then the US-030 head check and `alembic upgrade heads`. It never runs `load_seed_data`, the leader seeds or `seed_packages`. The existing `migrate` mode runs the full seed loader, whose `_seed_packages` refreshes packages without `create_only` and would overwrite curated production packages (the lifespan calls `seed_packages` with `create_only=True`).
- [ ] The Railway API service's pre-deploy command is `docker-entrypoint.sh migrate-schema`, set in service settings per US-056. It names the entrypoint because Railway runs a Dockerfile or image service's start command in place of the ENTRYPOINT. Its pre-deploy docs are silent on this, and naming the script works either way.
- [ ] The pre-deploy container gets the service's variables, so `migrate-schema` ignores `AUTOMATOS_SKIP_MIGRATIONS`. The service sets `AUTOMATOS_SKIP_MIGRATIONS=true`, so the serving container runs no boot steps. US-059 changes the pre-deploy command to the Python module's command in the same deploy window.
- [ ] Operator: the Railway-built services (media-render, which is built and scanned but not published, §9-Q6; and agent-opt-worker, kept per §9-Q4) get per-service watch paths. In the last 30 days agent-opt-worker redeployed 199 times for 2 commits that touched it.
- [ ] CI: the production-image boot lane starts from an empty database, runs `migrate-schema`, then serves with `AUTOMATOS_SKIP_MIGRATIONS=true`. All alembic output comes from `migrate-schema`, and the serving container logs none. A test asserts `migrate-schema` runs none of `load_seed_data`, the leader seeds or `seed_packages`. A test runs it with `AUTOMATOS_SKIP_MIGRATIONS=true` set and asserts alembic ran.

### Part 11: Bases and budgets (W10)

#### US-058: The API on Python 3.12, sharing the worker's base
**Description:** As a self-hoster, I want the API and the worker to share their Python base layers.

**Acceptance criteria:**
- [ ] The `pybuild` and `base` stages of `orchestrator/Dockerfile` (`:20`, `:64` at 697ca7d6e) move to `python:3.12-slim`, on the same digest as the worker and media-render. This needs US-027 (graspologic 3.3.0 blocks 3.12).
- [ ] Every other 3.11 pin moves in the same change:
  - `requirements.lock` and `requirements-dev.lock` (US-022) are regenerated with `--python-version 3.12`;
  - the stdlib precompile (US-014) and pybuild's site-packages compile (US-022) target the `python3.12` paths. A build check fails if either compiles no files, because the 3.11 paths no longer exist;
  - every backend `setup-python` in `test.yml` (14 at 697ca7d6e), `nl2sql-eval-scheduled.yml:70` and `import-linter.yml:36` run 3.12. `checkout-line-endings.yml:61` stays on 3.11, because it runs only line-ending tests;
  - ruff's `target-version` becomes `py312` (`ruff.toml:7`);
  - `AGENTS.md:13` says Python 3.12.
- [ ] The 3.11 base's `setuptools`, `wheel` and `jaraco-context` findings drop out of the scan.
- [ ] CI (US-001): the API and worker manifests share at least 3 layers. The first pull drops by about 16 MB.

#### US-059: The backend's start-up and health check in Python
**Description:** As Dave, I want the backend's boot path free of shell tools, so that curl and the Postgres client leave the image and a hardened base becomes possible.

**Acceptance criteria:**
- [ ] `docker-entrypoint.sh` is replaced by a Python module in `orchestrator/scripts/`, which ruff exempts from TID251, run from an exec-form ENTRYPOINT. In one interpreter it does the database wait (a psycopg2 connect loop), `init_fresh_db`, the alembic head check and upgrade, the fingerprint-gated seeds, and the local workspace/operator SQL. Running them inside the lifespan leader instead is §9-Q16.
- [ ] With `AUTOMATOS_LIVENESS_WATCHDOG` off, the module `os.execvp`s into uvicorn with `PORT`. With it on (the local edition's default in `envs/api.defaults`), it starts uvicorn as a child and supervises it as `run_supervised` does today: it forwards SIGTERM and SIGINT, probes with `scripts.healthcheck`, sends TERM then KILL after the grace period, and exits non-zero after the configured failed probes.
- [ ] It keeps the `migrate` and `migrate-schema` CLI contracts (`charts/automatos/templates/migrate-job.yaml:43`) and fail-closed exits. It keeps the US-037 `CHANGE_ME` refusal and the US-029 rules: it migrates only before a server command, honours `AUTOMATOS_SKIP_MIGRATIONS` and `AUTOMATOS_MIGRATE_ON_BOOT` as US-029 defines them, and passes one-off commands straight through. `migrate-schema` still ignores `AUTOMATOS_SKIP_MIGRATIONS` (US-057). The ported tests cover each rule.
- [ ] `docker-healthcheck.sh` becomes `python -m scripts.healthcheck` (urllib, no app imports). Every caller moves with it:
  - the compose backend healthcheck (`healthcheck.test`), the HEALTHCHECKs of the Dockerfile's `development` and `production` stages, and the watchdog's default probe use exec-form `python -m scripts.healthcheck`;
  - the smoke lane's path filters in `smoke-fresh-clone.yml` (both `orchestrator/docker-entrypoint.sh` entries) name the new modules, and `images.yml`'s PR path filter covers them;
  - the chart's README (`:18`), the `values.yaml` migrate comment, the `migrate-job.yaml` header comment and `tests/migrate_job_test.yaml:7` describe the Python entrypoint. The chart's `args: ["migrate"]` still works.
- [ ] Operator: in the same deploy window, the Railway API service's pre-deploy command (US-057) changes to the module's `migrate-schema` command.
- [ ] `curl` and `postgresql-client` (with `pg_isready`) leave the `base` stage's apt list. Perl stays, because Debian's git depends on it (see §9-Q20).
- [ ] The test files that pin the shell behaviour are ported to the Python module: the 12 at 697ca7d6e, plus those W4 and W5 add (US-029's flag tests, US-037's guard test).
- [ ] CI: `for b in curl psql pg_isready; do ! command -v "$b" || exit 1; done` passes in the production image. The ported tests and every smoke lane pass.

#### US-060: zstd layers for the published images (§9-Q2)
**Description:** As a self-hoster, I want the images to download in fewer bytes.

**Acceptance criteria:**
- [ ] `images.yml` outputs gain `compression=zstd,compression-level=<n>,oci-mediatypes=true`.
- [ ] The image cache moves from `type=gha` to `type=registry,ref=ghcr.io/automatosai/automatos-<x>:buildcache-<variant>-<arch>,mode=max,compression=zstd,compression-level=<n>,oci-mediatypes=true`, one ref per matrix entry, because the gha backend cannot store zstd. Main writes it; PR builds only read it. Cache-hit layers are then reused as zstd without `force-compression` recompressing 1.2 GB per build. `prod-image-parity`'s frontend `cache-from` (US-047, US-056) moves to the matching ref. This is a cache exporter in CI, not a BuildKit cache mount, so the Railway service-id rule (§4) does not apply.
- [ ] The level is chosen from one measured publish. BuildKit maps levels to four encoder presets, so CLI figures are only a guide: −11% at level 3, −22% at level 9 on the API's pip layer.
- [ ] Grype reads the zstd layers. A deliberate known finding is still reported.
- [ ] Before production runs a zstd digest, Railway is shown to pull one (§10). deploy.yml (US-056) deploys the first zstd publish on main to a non-production Railway environment the operator sets up, before anyone approves its production deploy. The API, the `-browser` worker and the `-saas` frontend each pull there and reach their healthcheck. If Railway cannot pull zstd, the images Railway runs (the API, `-browser`, `-saas`) keep gzip outputs, only the other published images move to zstd, and the zstd first-pull bar in §8 is re-set from the measured publish.
- [ ] CI: every layer above the pinned base image's layers has a `mediaType` ending in `tar+zstd`. The base layers, which stay gzip and keep the digests of the FROM image, are listed separately in the summary. A compatibility job pulls and boots the stack inside `docker:25-dind` (the floor). The first-pull bar moves to its zstd value in §8.

#### US-061: The frontend on a hardened Node base (D3)
**Description:** As Dave, I want the frontend runtime to carry nothing but Node and our server.

**Acceptance criteria:**
- [ ] After W9, the frontend runtime moves to `gcr.io/distroless/nodejs24-debian13:nonroot`, or to Docker's hardened Node image, whichever scans cleaner at decision time. The hardened image is mirrored to a public GHCR package by CI, because builds with no Docker login cannot pull from `dhi.io`: self-hosters' `make build`, US-036 dirty-checkout builds, and the PR lanes (`prod-image-parity` and `images.yml` PR builds).
- [ ] The distroless image has no shell and no `node` on PATH (its ENTRYPOINT is `/nodejs/bin/node`), and it runs as uid 65532 with no `adduser`. So:
  - the HEALTHCHECK keeps its options and becomes exec form: `CMD ["/nodejs/bin/node","-e","fetch('http://127.0.0.1:3000/favicon.ico').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"]`;
  - CMD becomes `["server.js"]`;
  - the parity check runs as a node script with `--entrypoint /nodejs/bin/node`, in place of `sh -c grep`;
  - files are copied `--chown=1001:1001` with `USER 1001`, so the chart's `runAsUser: 1001` (`charts/automatos/templates/frontend.yaml:23`) still owns them;
  - for the hardened-image option, the same steps use that image's node path.
- [ ] CI: zero fixable findings in the runtime layers, `/chat` boots, and the image stays within its budget.

#### US-062: Final budgets, and dive in CI
**Description:** As Gerard, I want the diet locked in, so that the images can't quietly grow back.

**Acceptance criteria:**
- [ ] `.github/image-budgets.json` is set to each image's post-diet size plus 2%. Growth past it fails unless the same PR raises the budget with a reason.
- [ ] `dive --ci` runs on the API and worker with `lowestEfficiency` and `highestWastedBytes` (or `highestUserWastedPercent`) set at the post-diet baseline in a committed `.dive-ci`. This is the check that would have caught the `chown -R` copy and the uninstall layer.
- [ ] The bars in §8 that are still report-only flip to fail.

## 6. Functional requirements

- **FR-1:** On a clean checkout whose `sha-<7>` tag is published and whose `.env` keeps the default `API_PORT`, `MINIO_PORT` and host, `make up` builds no image. If `.env` changes one of those, it builds only the frontend (US-036).
- **FR-2:** Every image and arch built by `images.yml` has a compressed-size budget. A build over budget fails unless the same PR raises the budget with a reason.
- **FR-3:** No image is published with a fixable Critical, or a KEV match of any severity, that is not allowlisted with a reason and an expiry of 90 days or less.
- **FR-4:** A backend container runs `alembic upgrade heads` at most once per start, and not at all when the database is at head. The one exception is the per-revision replay inside `init_fresh_db` on an empty database.
- **FR-5:** The boot seeds run only when their fingerprint changes: the seed files, the modules they import and the config values they read (US-031). The always-run list in US-031 is the only exception.
- **FR-6:** `import main` loads none of `camelot`, `cv2`, `matplotlib`, `pandas`, `nltk`, `google.generativeai` or `googleapiclient`.
- **FR-7:** The API reaches `/health 200` with no internet egress.
- **FR-8:** `production` is the last stage of every multi-stage Dockerfile.
- **FR-9:** Every image reference in a Dockerfile (`FROM`, `COPY --from`, `RUN --mount=from`) and every third-party compose image is pinned as `name:tag@sha256:<digest>`, and an updater opens PRs for each. With Dependabot, each copied image gets its own stage (`FROM node:24-trixie-slim@sha256:… AS node`, then `COPY --from=node …`; the same for the uv image), because its docker parser reads FROM lines only.
- **FR-10:** After W9 (held until after WebSummit, §9-Q11), every Railway service built from this repo's code runs a digest that `images.yml` built, scanned and signed: the API, the workspace worker (`-browser`) and the frontend (`-saas`). media-render (§9-Q6) and agent-opt-worker (kept, §9-Q4) stay Railway-built. `images.yml` builds and scans both without publishing them (US-054, US-006).
- **FR-11:** No image carries a second copy of a runtime it already has: one Node per image, no Debian Python next to the base Python. The Claude CLI's Bun runtime in the worker is §9-Q14.
- **FR-12:** Every new setting the orchestrator application reads is a `config.py` entry listed in `reports/config-surface.json`, with no literals in the modules. Entrypoint and runtime flags read by the shell, uvicorn or a library (`AUTOMATOS_SKIP_MIGRATIONS`, `WEB_CONCURRENCY`, `UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN`, `TIKTOKEN_CACHE_DIR`) and build args (`INSTALL_BROWSER`) are set in `envs/*.defaults` or the Dockerfile and documented there, as `AUTOMATOS_MIGRATE_ON_BOOT` is today. Each separate service (workspace worker, media-render, agent-opt-worker) keeps its settings in its own config.
- **FR-13:** Night and retest tooling keeps working (§9-Q1). The production-stage backend keeps the `/app/logs` volume and its `env_file`. `docker compose up --build` builds a stack from any ref, and `AUTOMATOS_IMAGE_TAG` gives that build a non-official tag such as `local-<ref>`. Every image, pulled or built locally, carries `org.opencontainers.image.revision` set to the commit it was built from.

## 7. Non-goals

- **Other repos.** Monitoring, Qdrant on Railway, mem0, automatos-voice, automatos-testing, academy, the landing sites and markets are in Appendix B, for a later PRD (Gerard, 16 Oct). The mem0 and voice residue inside this repo is §9-Q19.
- **No bars for CI build time or Railway deploy time** (Gerard, 16 Oct). The CI-only speed-ups found on the way are in Appendix C, to pull in if wanted. §9-Q13 pulled in three of them.
- **No custom ffmpeg build.** We would own ffmpeg's security updates, and the Mesa purge (US-051) gets most of the gain while keeping Debian's ffmpeg. A static ffmpeg is larger, and an LGPL build breaks libx264.
- **No `mint` / SlimToolkit.** It deletes whatever a sample run doesn't touch, including modules imported on demand, OCR data and fonts.
- **No Chainguard Python.** The free tier offers only `:latest`.
- **No MinIO replacement.** versitygw, rclone or garage would save 28–52 MB (2–4%) at the cost of migrating `minio_data` and presign/CORS parity.
- **No smaller Postgres.** pgvector publishes no Alpine image, and moving an existing data directory to musl changes text collation.
- **No change to the agent toolchain in the worker** (gcc, make, git, Node, npm/pnpm and uv stay; D5). Node moves from 20 to 24 (US-007). Python packages the worker itself no longer needs (SQLAlchemy, and Playwright in browserless editions) go, so agent scripts, which run the worker's own Python, can no longer import them.
- **No runtime-configured frontend.** The `-saas` variant ships now; runtime configuration is a later decision (§9-Q5).
- **No shell-less backend base.** `execute_command` stays in the API container; moving it to the worker is a later decision (§9-Q10).
- **No layer splitting for parallel downloads.** Pulls are bandwidth-bound: one stream and four ranges measured the same throughput.

## 8. Success metrics (bars set before the baseline; Gerard confirms or moves them in US-004)

The "Today" column holds 16 Oct measurements. US-004 replaces them with five-run medians where a W0 lane measures the row: sizes and update deltas (US-001), fresh and warm boot and datastore timings (US-002), and published-image pull and boot (US-003). The other rows keep their 16 Oct figure: heavy modules (its lane is US-032), egress (US-033), and images built and clone to working UI on the `make up` path, which `smoke-images` measures only from US-036 (before that it runs the `--no-build` published path). All sizes are compressed, except the media-render row, which is on disk because the image is not published; US-054 adds its compressed budget.

| Measure | Where it comes from | Today | Bar | Gates from |
|---|---|---|---|---|
| Images built by the default `make up` on a clean checkout | `smoke-images` | 3 | 0 | W5 |
| Default first pull, amd64 / arm64, unique layers | US-001, `smoke-images` | 1,330 / 1,284 MB | ≤ 1,000 MB each (≤ 850 MB with zstd) | the last of W3, W6 and W7 to merge (zstd: W10) |
| API image, amd64 | US-001 | 520.5 MB | ≤ 320 MB | W3 |
| Worker image, amd64 (local edition) | US-001 | 506.6 MB | ≤ 420 MB | W6 |
| Frontend image, amd64 | US-001 | 93.8 MB | ≤ 75 MB | W7 |
| media-render image, amd64, on disk | `media-render.yml` (`docker image ls`) | 1.81 GB | ≤ 1.30 GB | W8 |
| Update download for a code-only change: API / frontend | US-001 | 14.7 / 38.2 MB | ≤ 6 / ≤ 8 MB | W2 / W7 |
| Clone to working UI: `make up` start to `:3000 200` and `/health/ready 200`, CI runner, empty cache | `smoke-images` | not measured; about 6–7 min estimated on the build path | ≤ 150 s | W5 |
| Backend fresh boot, container start to `/health 200` | smoke, 1 s polls | 30–50 s, median 45 s (5 s polls) | ≤ 30 s | W4 |
| Backend warm restart, database kept | smoke warm phase | not measured in CI (Mac dev stage: 108 s) | ≤ 15 s | W4 |
| Published (production-stage) image, fresh / warm boot | `smoke-images` (PR variant, then main) | never booted in CI | ≤ 30 s / ≤ 15 s | W5 |
| Datastores healthy after start | compose timestamps | 5.5–6.1 s | ≤ 2 s | W5 |
| `alembic upgrade heads` runs per boot: fresh / warm at head | backend log (counted as below) | 2 / 2 | ≤ 1 / 0 | W4 |
| Heavy modules after `import main` | `test.yml` import step | 5 of the FR-6 list (6 on Railway, with matplotlib) | 0 | W4 |
| Boot with egress refused / blackholed | smoke variant | fails or hangs (tiktoken fetch at import) | healthy | W4 |
| Fixable Critical / unallowlisted KEV in images we build | Grype policy | api 6, worker 8, frontend 7 / 1 KEV; agent-opt-worker and media-render not scanned today | 0 / 0 | W1 (media-render: W8) |

**Where each row is measured.** Rows that need a published image come from `smoke-images` on main, right after merge: images built by `make up`, the first pull and clone to working UI. A red result there blocks the next wave. US-003's PR variant measures the published-image boot on the PR (amd64, images built in the job), and `smoke-images` measures it again on main. US-001 measures sizes and update deltas on the PR, against the current `edge`, and again on main. The media-render row comes from a `media-render.yml` run dispatched on the wave branch, because that workflow has no PR trigger. Every other row is measured on the PR.

**How alembic runs are counted.** The count is the `Context impl PostgresqlImpl` lines after `init_fresh_db: done`, or all of them on a boot where `init_fresh_db` did not run. On an empty database, `init_fresh_db` replays the migration forest and logs one such line per revision (about 200) before that line; FR-4 exempts the replay. Today each `alembic upgrade heads` (the entrypoint's and the CMD's) logs one. After US-030, a fresh boot ends at head after the replay, so it counts 0.

**Recorded, not gated.** The update download for a grouped dependency bump, with US-022's two pip layers, is recorded here from the first grouped Dependabot bump after W2.

The projections behind the image bars are sums of measured parts. API: the brief's removals, plus debug strip, `tests/` removal, pandas, ghostscript, the OSD model and the git trim, come to about 300 MB. Worker: Playwright out of the local edition, official Node, SQLAlchemy out and the uv binary come to about 400 MB; this does not count the bundled Claude CLI (about 108 MB compressed, §9-Q14). Frontend: trace exclusions, the `public/` diet and the npm-free Node 24 runtime come to about 70 MB. They are projections until each wave's build reports the real number.

## 9. Open questions for Gerard (each with a recommendation)

Gerard took Q1–Q13 at their recommendations on 16 Oct ("get it built"); each answer is recorded under its question. Q14–Q19 were added in review the same day, after those answers, and are open. So is the data question in Q4.

1. **Pull-first `make up` (US-036).** This commits us to keeping the three GHCR images public. It also changes what plain `docker compose up` runs for contributors, whose hot reload moves to `make dev`. Does any of your night or test tooling (for example in `~/.automatos-analyst`) rely on the default stack's source mount and `--reload`? *Recommendation: yes to pull-first; move any such tooling to `make dev` in the same wave.*

    **Taken, 16 Oct:** yes to pull-first. TESTER, who owns the night and test tooling, answered the same day: no night tooling needs `--reload`, so nothing moves to `make dev`. Four constraints carry into W5 (§2, FR-13):
    - The production-stage backend keeps the `/app/logs` named volume and its `env_file`, because `next-cycle.sh` edits `envs/api.local` and then runs `docker compose up -d --no-deps backend` (US-035).
    - `docker compose up --build` keeps working for retest stacks built from an unmerged ref (US-036).
    - `AUTOMATOS_IMAGE_TAG` lets a local build use a non-official tag such as `local-<ref>` (US-036).
    - Every image, pulled or built locally, carries the `org.opencontainers.image.revision` label, because the workspace-root `scripts/bootstrap.sh` (outside this repo) will read it instead of the bind-mount path (US-003, US-036).

2. **Docker floor.** `start_interval` makes Compose refuse Engines below 25, and zstd layers need 23+. *Recommendation: document Engine 25 and Compose 2.24.4 (Docker Desktop has shipped Engine 25 since early 2024) and enable both.*

    **Taken, 16 Oct:** Engine 25 and Compose 2.24.4 are the documented floor; `start_interval` (US-038) and zstd (US-060) are both enabled. This covers self-hosters. Whether Railway's runtime pulls zstd is an operator check (§10), and the images Railway deploys stay gzip until it passes (US-060).

3. **Is the workspace worker in the default stack?** It is 477 MB of unique bytes, 36% of today's first pull. Without it, agents lose workspace file tools, Code Canvas and the Deliverables explorer. *Recommendation: keep it in the default and ship the five-container `core` option (US-040). The worker diet brings it to about 400 MB.*

    **Taken, 16 Oct:** the worker stays in the default six containers (`COMPOSE_PROFILES=workspace` in `.env.example`), and US-040 ships the five-container core option.

4. **agent-opt-worker: lock down (the brief, US-006) or retire?** It is live code: the admin assess/optimize/safety buttons and a per-turn live-eval hook. The FutureAGI trial was due to end around August 2026 (PRD-58:9). If the account has lapsed, every call already fails. Retirement removes an always-on Railway service, 99 floating packages, a 2019 placeholder package (`fi`) that runs `setup.py` at build, and the per-turn database hook. *Recommendation: check the FutureAGI account and `system_prompts.futureagi_eval_enabled` usage. If it is lapsed or unused, retire (Appendix A) instead of US-006.*

    **Taken, 16 Oct:** the FutureAGI account could not be checked, so US-006 (lock-down) is built in W1. Retirement (Appendix A) stays the alternative.

    **Open: the data question.** When any platform prompt has scoring on, live scoring sends every tenant's chat turn and the system prompt to FutureAGI (`consumers/chatbot/service.py:1299-1310` → `futureagi_service.py:235-268`). `_collect_optimization_dataset` (`futureagi_service.py:308-330`) sends the newest user/assistant pairs from `messages`, across all workspaces and with no workspace filter, to FutureAGI and OpenAI. Is that acceptable under current terms? Gerard answers before W1 ships, and the answer becomes an acceptance criterion in US-006. If the answer is no, that alone settles retirement, and Appendix A replaces US-006.

5. **Frontend editions for D2.** Publish a `-saas` variant (US-055, small), or move the frontend to runtime configuration? Runtime configuration would serve every edition from one image, fix non-default ports for pulled frontends and unblock Helm ingress, but it touches about 17 files and the CSP. *Recommendation: the `-saas` variant now; runtime configuration as its own decision later.*

    **Taken, 16 Oct:** US-055 publishes the `-saas` variant. Runtime configuration is a later decision, outside this PRD (§7).

6. **Publishing media-render.** A public image distributes GPL-3.0 espeak-ng/phonemizer and GPL x264, which brings source-offer obligations, and `images.yml:121` stamps every image Apache-2.0. *Recommendation: build and scan it in CI (US-054) without publishing; Railway keeps building it until a licence decision.*

    **Taken, 16 Oct:** media-render is built and scanned in `images.yml` (US-054) and not published. Railway keeps building it until a licence decision.

7. **Precompiled bytecode** for the app's boot modules ships in the final app layer: US-032 compiles it in a separate stage and copies source and `.pyc` as one layer. Every code update therefore re-downloads it, an estimated 6–8 MB compressed for all app modules and less for the boot set. It speeds only a new container: a restart keeps the `.pyc` the first boot wrote. *Recommendation: let US-032's measurement decide, on a new container. Keep it only if it takes 2 s or more off that boot and the code-only API update delta stays ≤ 6 MB (§8) with US-016's single changed layer. If both cannot hold, Gerard chooses between the boot gain and raising the update bar.*

    **Taken, 16 Oct:** US-032 measures and applies this rule.

8. **Leiden parity tolerance** (US-027). *Recommendation: identical partitions under a fixed seed on the fixture graphs. If that fails, modularity within 1%, or else accept graphify's Louvain fallback everywhere.*

    **Taken, 16 Oct:** identical partitions; if that fails, modularity within 1%. If neither holds, graphify's Louvain fallback is accepted everywhere and the adapter is not shipped (US-027).

9. **pnpm in the worker.** Pin 12.x exactly (58 MB with a 54 MB native binary) or 10.x (20 MB)? Agents' lockfiles care about the major. *Recommendation: pin the current 12.x exactly.*

    **Taken, 16 Oct:** pnpm is pinned exactly to the current 12.x (12.14.0 on 16 Oct; US-007).

10. **A shell-less backend base.** The `execute_command` agent tool runs coreutils (`ls`, `cat`, `grep`, `wc`, `stat`, …) inside the API container (`agent_action_executor.py:45-52`), and GitPython needs git. A no-shell base breaks both. *Recommendation: this PRD lands the Python entrypoint (US-059), which removes curl and psql. Moving `execute_command` to the worker, so the API can go shell-less, is a separate decision for you.*

    **Taken, 16 Oct:** US-059 lands the Python entrypoint. Moving `execute_command` to the worker, and a shell-less base, are a later decision (§7).

11. **W9 timing.** It changes how production deploys, three weeks before WebSummit (10–13 Nov). *Recommendation: run W0–W8 now and schedule W9 for after WebSummit, unless you want the scanned-digest story for the event.*

    **Taken, 16 Oct:** W0–W8 run now. W9 is held until after WebSummit, and W10, which depends on it, follows it.

12. **Content calls.**
    - Turn off Next image optimisation? Its only optimised image is a 48 px logo.
    - Compress the mascot posters lossily? Design should sign off.
    - Delete the 4 blog covers? Only after a database check, because stored posts may reference them.

    *Recommendation: yes, ask design, check first.*

    **Taken, 16 Oct:**
    - Image optimisation is turned off (US-046).
    - The mascot posters get no lossy compression without design sign-off (US-049).
    - No file a database row may reference (the 4 blog covers in `public/images`, `public/favicon_io`, `modal-fix.js`) is deleted until a production check (US-049).

13. **CI speed (Appendix C).** No bars per your answer. Pull any in? *Recommendation: the three cheap ones that keep the GHA cache under its 10 GB cap (cache-to only on publish, `mode=min` for the API, the registry cache for media-render), because evictions turn warm builds cold.*

    **Taken, 16 Oct:** all three are pulled in. `cache-to` only when publishing and `mode=min` for the API go into US-001 (W0). The media-render registry cache is in US-054. The rest of Appendix C stays optional.

14. **Bundled Claude CLI in the local worker.** claude-agent-sdk's wheel carries a Bun-based Claude CLI (268 MB). Dropping it from the local and GHCR builds saves 253.8 MB installed, about 108 MB compressed amd64, about 21% of the worker pull; the §8 worker projection does not count it. Those builds could install the SDK from its sdist, which has no CLI. `services/workspace-worker/canvas_session_service.py:460` would then also have to report `sdk_available=false` when no CLI is found, because today it checks only the credential. The cost: the local "Auto session" canvas (`docs/getting-started/self-hosting.md:425-431`, `:845-847`) stops working for anyone who sets `ANTHROPIC_API_KEY`. Options: drop it from the local edition, publish a `-sdk` worker tag, or fetch the CLI into the volume on first canvas start. *Recommendation: none; it removes a documented feature, so it is your product call. Until you answer, the CLI stays (D5).* **Open** (added in review, 16 Oct).

15. **Should `workspace_html_to_png` move to media-render?** The tool (`exec_workspace.py:457-501` → the worker's `executor.py:422`) is the only reason the Railway worker carries Chromium and Playwright: about 0.63 GB installed after US-043 (about 1.23 GB today). media-render already ships a headless shell, but it would need the HTML and its assets over HTTP, because a Railway volume attaches to one service. It is an architecture change. *Recommendation: decide from production usage (§10: the `html_to_png ok` count over 30 days).* **Open** (added in review, 16 Oct).

16. **PandasAI charts** (`modules/tools/services/pandas_ai_service.py`, `api/analytics_charts.py`) cannot run. `pandasai` was never installed, US-025 removes pandas, and no frontend code calls `/api/analytics/charts/*` (`reports/route-manifest.json:338`, `:342`). Delete the service, the two routes and their route-manifest entries, or finish the feature? AGENTS.md requires a maintainer's yes before a public route is removed. *Recommendation: delete in W3 (US-025 says what goes).* **Open** (added in review, 16 Oct).

17. **Docker Hub on the first run.** After US-036, `make up` still pulls `pgvector/pgvector:pg16` and `redis:7-alpine` anonymously from Docker Hub (`docker-compose.yml:42`, `:67`). The anonymous limit measured on 16 Oct is 100 pulls per hour per IP, and the research session hit HTTP 429. A first run makes two such pulls, so only a shared IP (office, CGNAT, a CI fleet) is at risk. Should both be mirrored to `ghcr.io/automatosai` by a scheduled, digest-pinned, scanned CI job? *Recommendation: `smoke-images` records any Docker Hub 429; mirror if one appears.* **Open** (added in review, 16 Oct).

18. **The Appendix D items.** Fold the `get_usage_bytes` fix (run it in `asyncio.to_thread` and skip the projects bind mount) into US-045, and the Helm chart's object-store values into US-042, or leave them for a separate PRD? *Recommendation: fold both in; they touch files those stories already change.* **Open** (added in review, 16 Oct).

19. **mem0 and automatos-voice residue in this repo.** No code reads any of it:
    - `MEM0_API_URL` (`envs/api.defaults:150`);
    - `'mem0-server'` in the `platform_query_loki_logs` tool schema (`modules/tools/discovery/actions_monitoring.py:30`);
    - the Mem0 and Voice Services blocks in `infrastructure/.env.example` (`:87-94`, `:96-115`).

    Remove them in US-018, and extend `test_no_mem0_residue.py`, which today checks only `orchestrator/`, to scan `envs/`, `infrastructure/.env.example` and `modules/tools/`? *Recommendation: yes. Leave the mem0 and voice entries in `infrastructure/railway-manifest.json` and the `MEM0_DATABASE_URL` backup in `scripts/dr/backup.sh` until those services are retired (Appendix B), because they describe and back up live deployments.* **Open** (added in review, 16 Oct).

20. **Git and Perl out of the API image.** Replacing the git CLI and GitPython with dulwich (`codegraph_service.py:446-479`, `core/security/git_sanitizer.py:162`, `skill_loader.py:607-670`, `scripts/harvest_plugins.py:155`) would remove about 86 MB installed, about 30 MB compressed, once US-059 has removed the Postgres client. Copying a Perl-free git-core from a builder stage instead would remove Perl only (about 50 MB installed). *Recommendation: not in this PRD; US-019's git trim takes the cheap part.* **Open** (added in review, 16 Oct).

21. **Migrations and seeds inside the lifespan leader.** Should alembic and the fingerprint-gated seeds run in the app's lifespan leader, under the boot advisory lock and before `create_tables`, instead of in a separate runner process? `migrate` and `migrate-schema` would stay for Helm and Railway. This reuses the imports the app has already paid for. *Recommendation: measure after W4, and adopt it if fresh boot misses ≤ 30 s.* **Open** (added in review, 16 Oct).

## 10. Operator checks (Railway was unreachable from this session)

The Railway MCP server failed to connect on 16 Oct, so none of the following was checked live:

- Which config file each Railway service reads. Every service has a root directory, and Railway's config file does not follow it, so `/railway.json` may not be loaded at all.
- Whether a healthcheck is set in the UI. The repo has none since `e5aa6b350` (15 Apr).
- Whether `INSTALL_BROWSER` is unset on the worker, so that full Chromium ships (the Dockerfile default is `true`, `services/workspace-worker/Dockerfile:50`).
- Watch paths per service.
- Build durations.
- `AGENT_OPT_WORKER_URL` on the API: internal or public.
- Whether media-render is live.
- The service id of every Railway service that builds a Dockerfile from this repo: the API (`orchestrator`), frontend, workspace worker, media-render and agent-opt-worker. A BuildKit cache mount fails a Railway build unless it carries a hard-coded `id=s/<service-id>-<path>`, so none of those Dockerfiles gets a cache mount until its id is known (US-044, US-047, Appendix C).
- The API service's `AUTOMATOS_MIGRATE_ON_BOOT`, `AUTOMATOS_LIVENESS_WATCHDOG` and `WEB_CONCURRENCY` values, and any custom start command, before W4 (US-029). The April manifest (`infrastructure/railway-manifest.json`, generated 9 Apr) lists none, so today the CMD alone migrates and `--workers 4` sets the worker count. Once US-029 drops `--workers`, a `WEB_CONCURRENCY` set on the service takes over. If the service sets `AUTOMATOS_MIGRATE_ON_BOOT=true`, the entrypoint runs its seed loader there too, and US-029's "neither set (Railway)" case does not apply. A custom start command bypasses the CMD change.
- Whether app sleeping is on for the frontend service (the April manifest says `"sleep": true`, `railway-manifest.json:200`). That would put a cold boot on the ui.automatos.app path, and from W9 the frontend's start time is the user's wait.
- Whether Railway's runtime pulls zstd (`tar+zstd`) layers. Until one Railway service has deployed a zstd digest and passed its health check, the images Railway deploys stay gzip (US-060).
- How often production calls `workspace_html_to_png`: count `html_to_png ok` log lines (`services/workspace-worker/executor.py:599`) over 30 days. It decides §9-Q15.
- Production logs: how often "Table extraction with lattice failed" appears (expected: every PDF with tables).

## Appendix A: retiring agent-opt-worker (if §9-Q4 says retire; replaces US-006)

§9-Q4 was taken on 16 Oct as lock-down, so US-006 is built. This appendix stays the alternative. It applies if Gerard's answer to §9-Q4's data question rules the transfer out, or if a later check finds the FutureAGI account lapsed or unused. If it lands after US-006, it also removes what US-006 added: the `images.yml` build, PR paths and budget entry, the litellm entries in `.grype.yaml`, the Dependabot entries, the lock and the test lane.

- **Delete in code:**
  - `services/agent-opt-worker/` and `orchestrator/core/services/futureagi_service.py`;
  - the live-eval hook (`consumers/chatbot/service.py:1299-1310`);
  - the admin routes (`api/admin_prompts.py:345-460`; optionally keep a read-only assessment-runs history);
  - `config.py:1479-1483` and the FutureAGI settings;
  - the FutureAGI controls in `frontend/components/settings/SystemPromptsTab.tsx`;
  - `services/shared/` (only docs read it).
- **Carry over from US-006:** the workspace worker's `automatos_metrics.py`, today byte-identical to the `services/shared` copy, labels requests by route template (`request.match_info.route.resource.canonical`, with `unmatched` for 404s) instead of the raw `request.path` (`:161`), with a cardinality test. Once `services/shared/` is deleted, US-045 applies its `/metrics` fix to `services/workspace-worker/automatos_metrics.py` only, and the byte-identity test goes, because nothing is left to compare.
- **Delete references in:** `envs/api.defaults:163`, `charts/automatos/values.yaml:41`, `deploy/kind/values.yaml:11`, `infrastructure/.env.example`, `infrastructure/railway-manifest.json`, `orchestrator/modules/tools/discovery/actions_monitoring.py:29`, `:62`, `orchestrator/core/monitoring/automatos_logging.py:355`, `docs/prompt-optimization/*`, `docs/README.md:111`, `docs/overview/_index.md`, `docs/overview/system-architecture.md` and `docs/deployment-infrastructure/{_index,docker-compose-setup,docker-containerization,production-deployment,production-deployment-cicd}.md`. These docs are synced to the public GitBook.
- **Keep, marked as history:** the `system_prompt_eval_runs` entries in `docs/backend-architecture/{database-models,database-models-migrations}.md` and `docs/deployment-infrastructure/database-setup.md`, because the tables stay. The futureagi ALTER in `docs/backend-architecture/_index.md` is updated with US-030.
- **Update in the same PR:** `reports/config-surface.json`, `reports/route-manifest.json`, `test_config_env_centralization.py:58`, `:366`, `:396`, and `test_p2w2_authz_boundary_sweep.py:138`, `:145`. Keep the database tables and column for history.
- **Operator:**
  - delete the Railway service, which removes its public domain and keys;
  - remove `FUTUREAGI_API_KEY`, `FUTUREAGI_SECRET_KEY` and `FUTUREAGI_ENABLED` from the API service, or the per-turn hook stays live;
  - remove the Prometheus scrape and the Loki alert in automatos-monitoring.
- **CI:** a test asserts no file imports `core.services.futureagi_service` and `git ls-files services/agent-opt-worker` is empty. Afterwards, no `Automatos AI - agent-opt-worker` commit status appears on main.

## Appendix B: other repos (for a later PRD)

Re-checked 16 Oct; each item holds unless marked.

- **automatos-monitoring** (live on Railway):
  - Grafana 10.4.0 (end of support June 2025) and log-relay (aiohttp 3.9.3) have public domains. Prometheus 2.51, Loki 2.9 and Alertmanager 0.27 are past support.
  - Loki retention is effectively off on its 50 GB volume, its alert rules are probably never loaded on Railway, and the local compose provisions no dashboards.
  - Railway has no native log drain, so log-relay's `/drain` has no sender.
  - Entrypoints `chown -R` the whole volume on every start.
  - No build or validate workflow, and no Dependabot.
  - promtool and amtool are about 47% and 42% of the Prometheus and Alertmanager images, and nothing at runtime runs them.
  - The planned upgrade grows the stack by about 209 MB, so self-hosting versus a hosted Grafana is open.
- **Qdrant on Railway:** untagged on a stateful volume. Its URL may be the public domain.
- **mem0:** the OpenMemory API and its pgvector database are still deployed. The platform no longer calls them; the residue in this repo is §9-Q19.
- **automatos-voice:** unpinned `torch` resolves to the 3.05 GB CUDA set on CPU-only Railway. PRD-207 moved voice into automatos-ai. Retire it. Its residue in this repo goes with it: the Voice Services block in `infrastructure/.env.example:96-115` (§9-Q19), and in `infrastructure/railway-manifest.json` the voice group (`:24-30`), its two services (`:351-425`) and the API's `VOICE_*` env names (`:175-179`).
- **automatos-testing:** its compose file mounts files that don't exist. PRD-247 already retired it. Archive it.
- **automatos-academy:**
  - `node:20-alpine`;
  - npm cache baked into the image;
  - `.dockerignore` has no `.env` pattern on a single-stage `COPY . .`;
  - migrations run through npm at boot.
- **automatos-ai-landing:**
  - `node:20-alpine`;
  - the runtime reinstalls the whole tree (about 460 MB) for a server that needs three packages;
  - a 10 MB unreferenced `framer-export.zip` sits in the build context;
  - `deploy/studio-proxy` (`caddy:2-alpine` on a floating tag; the front door for studio.automatos.app) is in no image inventory.
- **automatos-ai-landingV2** (automatos.app):
  - `express.static(__dirname)` serves the source directory, internal docs and `uploads/`;
  - the deployed branch is `feat/one-os-family`, not main, so fixes must land there.
- **automatos-markets:**
  - `.dockerignore` misses `.env*`;
  - Next's sharp advisories (it can drop sharp, since both `next/image` uses are unoptimized).
- **All four Node repos:**
  - no CI image build or scan;
  - Railway deploys without waiting for CI;
  - Express runs as PID 1 without a SIGTERM handler in academy and both landing sites (markets' Next standalone server registers its own);
  - floating base tags;
  - academy and landingV2 run `npm install`, not `npm ci`;
  - the academy, landing and landingV2 Dockerfiles contain non-ASCII bytes, which Railway's Metal builder rejects (markets' CLAUDE.md records it).

## Appendix C: CI speed found on the way (no bar; §9-Q13)

§9-Q13 pulled in the three marked *(pulled in)*. The rest stay optional.

Grype:
- Take Grype off the per-arch build path. It is 60% of build-job runner time (633 of 1,064 s per run).
- Scan the SBOM BuildKit already emits, or scan from the registry. Today it does a docker pull plus save (66 s).
- Cache the Grype DB by its build date, not scan-action's fixed key, which can never refresh.

Caches:
- The GHA cache is 11.12 GB against a 10 GB cap. Write `cache-to` only when publishing; branch and PR refs hold 2.7 GB that main can't read. *(pulled in, US-001)*
- Use `mode=min` for the API. Its pybuild `/install` output and compiler stage, which `mode=max` exports and never reuses, are an estimated 0.8–0.9 GB of the cache. In all, 3.19 GB of main's 5.33 GB of BuildKit blobs are not layers of any current image. The frontend keeps `mode=max` for its builder's node_modules. *(pulled in, US-001)*
- Move image caches to a GHCR registry cache. *(media-render pulled in, US-054)*

What gets built:
- Build only images whose inputs changed on main, and re-tag the rest.
- Use per-image PR path filters.
- Publish each image as soon as its own two arches finish.
- Drop gcc/g++ from `pybuild` once psutil ≥ 6 lands (US-015), or install in the runtime stage with uv and remove `pybuild`. Either way removes the 11.2 s `COPY --from`, the 9.6 s apt step on the critical path and a second ~365 MB `mode=max` cache blob per arch. Keep binutils for US-020's strip.
- Set `provenance` only when publishing. Set `sbom` only when publishing, unless PR scans read that SBOM (the Grype item above), in which case keep it on PRs. PR builds then skip the provenance work, and the SBOM layer extraction (about 24.5 s plus a 10 s SBOM step) when `sbom` is off.
- Split the worker into parallel `deps` and `browsers` stages, with the runtime stage last.

Parity and smoke lanes:
- `prod-image-parity` is `test.yml`'s longest job (545–649 s over 8 main runs, 1,093 s once). Make it a 3-way matrix.
- The smoke lane should build through bake with the GHA cache.

media-render.yml (FIXER owns its speed-up):
- Build once instead of five times cold.
- Balance the template shards by measured duration: 545/500/302/253 s today.

Frontend:
- Gate frontend-ci and the parity frontend build on main pushes that touch `frontend/`: 150 of 216 main commits in the last month didn't.
- Shard Vitest, and run DOM-free tests in the node environment.
- Add a `next build` to frontend-ci.
- Turbopack after vendoring the Google fonts.
- Persist the npm and `.next` cache mounts across `images.yml` runs with buildkit-cache-dance (SHA-pinned); without it they are empty on every run. The frontend is Railway-built, so its Dockerfile gets no cache mount until the Railway service id is known (§10, US-047).

gitleaks:
- Use a shallow fetch on push and PR (31 s of 39 s is the full-history checkout).
- Run a real weekly full-history scan; the header claims one, but push runs scan one commit.

## Appendix D: found on the way, not about containers

Whether the first two items fold into this PRD is §9-Q18.

- The worker's `WorkspaceManager.get_usage_bytes` walks the whole workspace root, including the projects bind mount, synchronously on the event loop for every queued task (`workspace_manager.py:132-144`).
- The Helm chart configures no object store (no `S3_ENDPOINT_URL` or AWS variables), so storage features raise `StorageNotConfigured` unless operators add them.
- With the published images, generated documents live in the container's writable layer and are lost on recreate. They are a cache with an S3 copy; US-017 moves them to the data volume.

## Related

- The brief: claude.ai/artifact/JUEs6PmWZ4H2Fy1ErsEJo3 (15 Oct 2026, for Dave).
- PRD-241 US-005 (agent-opt-worker lock-down, unmerged branch). PRD-58 (FutureAGI). PRD-207 (voice moved into automatos-ai). PRD-209 and PRD-234 (single compose, loopback binds). PRD-211 US-002 (`test_no_mem0_residue.py`). PRD-251 W1 (media-render image). F092 (starlette cold-boot router loss), F228 (slow first boot), F416 (liveness watchdog).
- TESTER's answer to §9-Q1 (16 Oct): the constraints in §2 and FR-13.
- CI runs behind the "Today" numbers:
  - images: 38915020775 (16 Oct) and 37948937717 (9 Oct, cold);
  - `test.yml` parity: 38915015762;
  - smoke-fresh-clone: 38399186819 and 8 earlier runs (4–13 Oct);
  - media-render: 37612086159.

