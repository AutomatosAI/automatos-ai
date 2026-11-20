# PRD-280 US-103: every path a component writes at run time

**For:** [PRD-280](PRD-280-CHART-HARDENING-AND-WORKER-ISOLATION.md) US-103 (analysis) and US-104 (a read-only root file system, which mounts what this page lists).
**Read at:** branch `feat/prd280-w1-strict-namespace`, 20 Nov 2026, from the code and the images' Dockerfiles. Nothing was run.
**Labels:** **[T]** read in code; **[I]** inferred from a library's documented default, not read in its source.

Each row is one path a component writes while it runs in the chart, what writes it, and what the path becomes under a read-only root. Paths are as the chart runs each component: the API and the migration Job as uid 1000 in `/app`, the web app as uid 1001 in `/app`, the worker's entrypoint as root, then its process as uid 1000 (`worker`) with `HOME=/home/worker`.

- **Scratch:** nothing there has to outlive the pod. It becomes an `emptyDir`, emptied when the pod is replaced.
- **Volume:** already on a volume the chart mounts. Unchanged.
- **None:** not written in the chart, so nothing is mounted.
- **For Gerard:** today on the container's own disk, lost when the container is replaced; it stays scratch until Gerard decides it needs a home (section 6).

A path an operator sets in `config` (for example `WORKSPACE_EXPORT_TEMP_DIR`, `AUTOMATOS_WORKSPACE`, `DECISION_SHADOW_LOG_PATH`) must be under one of the writable paths below.

## 1. The API

| Path | Written by | Survives a restart? | Becomes |
|---|---|---|---|
| `/app/data/uploads` (`DOCUMENT_UPLOAD_DIR`, set by the chart while `api.data` is on) | `api/document_upload_steps.py:108-111`; `modules/tools/discovery/handlers_documents.py:384-386` **[T]** | Yes | Volume: `api.data` |
| `/app/data/documents` (`DOCUMENT_STORAGE_DIR`): generated documents, templates, thumbnails, brand files, provenance marks | `modules/documents/generation_service.py:751-752`, `:379`, `:421`, `:452`; `api/document_generation.py:394-398`; `modules/documents/thumbnails/store.py:61-63`; `modules/documents/brand_logo.py:193-194`, `:235-236`; `modules/documents/provenance_marks.py:130-135` (a temporary file beside the target, then a rename) **[T]** | Yes | Volume: `api.data` |
| `/app/documents` (`config.py:1877`, `DOCUMENT_STORAGE_DIR`'s default `documents`, under the working directory), only with `api.data.enabled=false` | the same writers as the row above **[T]** | No volume holds it today | Scratch: `emptyDir` at `/app/documents`. For Gerard (section 6) |
| `/tmp/automotas_uploads` (`api/documents.py:153`, `DOCUMENT_UPLOAD_DIR`'s default), only with `api.data.enabled=false` | the upload writers above **[T]** | No volume holds it today | Scratch: `/tmp` |
| `/tmp`, through `tempfile` (`TMPDIR` unset) | `api/wizard.py:887`; `api/documents.py:875`; `api/knowledge_multimodal.py:318`; `api/admin_plugins.py:565`; `api/socials_media_upload.py:174`; `modules/codegraph/codegraph_service.py:461` (repository clones); `modules/knowledge/graph_service.py:1123`; `modules/rag/services/cloud_file_downloader.py:205`; `modules/tools/discovery/handlers_documents.py:296`; `modules/socials/render.py:608`; `modules/socials/publishing.py:143` (with `publish_steps.py:109` inside it); `modules/socials/recipes/footage.py:278`; `modules/socials/recipes/voice.py:498`; `modules/agents/services/agent_action_executor.py:77`; `core/composio/tool_executor.py:110`; `services/knowledge_flywheel.py:398`; `services/owner_knowledge.py:92`; `services/workspace_move/jobs.py:161` and `services/workspace_move/files.py:163` (`WORKSPACE_EXPORT_TEMP_DIR`, `config.py:1106`, empty by default) **[T]** | No | Scratch: `emptyDir` at `/tmp` |
| `/tmp/automatos_mission_uploads` | `api/missions.py:997-1004` **[T]** | No (passed on to the document manager) | Scratch: `/tmp` |
| `/tmp/automatos_mission_outputs` | `services/coordinator_service.py:1446-1449` **[T]** | No (uploaded as a document) | Scratch: `/tmp` |
| `/tmp/automatos_workspace` (`config.py:1875`, `AUTOMATOS_WORKSPACE`) | `modules/agents/services/agent_action_executor.py:426-428`, `:80`, `:183-186`, `:235` **[T]** | No | Scratch: `/tmp` |
| `/tmp/pandasai_charts` (`config.py:1870`) | `modules/tools/services/pandas_ai_service.py:58-59`, reached only after `import pandasai` at `:43`; `pandasai` is not in `orchestrator/requirements.lock` **[T]** | — | None (covered by `/tmp` if it is ever installed) |
| An upload's body over 1 MB | Starlette's form parser spools it to a `SpooledTemporaryFile` in `/tmp` **[I]** | No | Scratch: `/tmp` |
| OCR's input and output images | `pytesseract` writes them to `tempfile.gettempdir()` **[I]** | No | Scratch: `/tmp` |
| `~/.automatos/skills/<source>` (`/home/automatos`) | `modules/agents/services/skill_loader.py:61`, `:260` (created when the loader first starts), `:389-392` (a `git clone`); later reads use the recorded path, `:495`, `:517`, `:1098`, `:1171` **[T]** | Recorded in `SkillSource.local_cache_path` and `Skill.filesystem_path` | Scratch: `emptyDir` at `/home/automatos`. For Gerard (section 6) |
| `~/.automatos/ssh_known_hosts` | `core/security/ssh_host_keys.py:28`, `:51-52`; read at `:64-66` **[T]** | Read by later connections | Scratch: `/home/automatos`. For Gerard (section 6) |
| `~/.composio/files/<toolkit>/<tool>/` | the Composio SDK (`composio` 0.12, `core/composio/client.py:145`): a tool result that carries a file is downloaded under `COMPOSIO_CACHE_DIR`, by default `~/.composio` **[I]** (the SDK's `core/models/_files.py`, read on GitHub, not in the image) | No (the result is passed on) | Scratch: `/home/automatos` |
| `~/.cache/fontconfig` | fontconfig, under WeasyPrint, writes a user cache when the system cache (`/var/cache/fontconfig`, built at `orchestrator/Dockerfile:228`) does not cover a font folder **[I]** | No | Scratch: `/home/automatos` |
| `~/.aws/sso/cache` | botocore, only for an SSO profile; keys from the environment or a projected web-identity token write nothing **[I]** | No | Scratch: `/home/automatos` |
| `/app/logs/decision_shadow.jsonl` (`config.py:1956`, `DECISION_SHADOW_LOG_PATH`'s default `logs/…`) | `core/llm/decisions/engine.py:336-342` (`record_shadow`, called from `consumers/chatbot/auto.py:1014`, `core/llm/decisions/judgements.py:109`, `:219`, `modules/tools/discovery/decision_rerank.py:138`); a failed write is logged at debug **[T]** | Read later by `scripts/eval/decision_shadow` | Scratch: `emptyDir` at `/app/logs`. For Gerard (section 6) |
| `core/.credential_key` under `/app` | `core/credentials/encryption.py:80-82`, `:118`, only when `CREDENTIAL_ENCRYPTION_KEY` is empty; the chart reads that key from the Secret as a required key (`templates/_helpers.tpl:276`) **[T]** | — | None |
| `/var/automatos/documents/reports` | `modules/tools/services/report_generator.py:17`, `:25`, `:69`; nothing constructs `ReportGenerator` **[T]** | — | None |
| `workflow_<id>_<time>.log` in the working directory | `core/utils/logging_utils.py:106`; `EnhancedWorkflowLogger` is constructed only by that module's own `__main__` block (`:505`) **[T]** | — | None |
| `<dest_dir>/<skill>/SKILL.md` | `modules/agents/services/skill_portability.py:350-385`; nothing calls `export_skill_to_folder` **[T]** | — | None |
| `/app/exports` | read by `main.py:1273-1288`; nothing in the image writes there (`pandas_ai_service.py:62` names it as a folder to watch) **[T]** | — | None |
| `__pycache__/*.pyc` beside a module not compiled at build | CPython's import system; the build compiles site-packages, the stdlib and the API's boot modules (`orchestrator/Dockerfile:161-164`, `:335-336`, `:367`). A failed bytecode write is skipped and the module imports as usual **[I]** | No | None |
| tiktoken's encodings | read from `TIKTOKEN_CACHE_DIR=/opt/tiktoken` (`orchestrator/Dockerfile:248-252`); tiktoken writes only an encoding missing there, and the build fails when one is **[T]** | — | None |
| `${TMPDIR:-/tmp}/automatos-liveness.<pid>` | `docker-entrypoint.sh:682`, only with `AUTOMATOS_LIVENESS_WATCHDOG=true`, which the chart does not set **[T]** | — | None (covered by `/tmp`) |
| uvicorn | writes no file; `--workers` above 1 uses `/dev/shm`, the runtime's own memory file system, not the root **[I]** | — | None |
| `/workspaces` | mounted read-only in the API by the chart (`templates/api-deployment.yaml`) | Yes | Volume: unchanged |

## 2. The migration Job

It runs `docker-entrypoint.sh migrate` (`docker-entrypoint.sh:707-713`): `pg_isready` and `psql` (`:138-152`), `python -m scripts.init_fresh_db` (`:215`), `alembic upgrade heads` (`:258`) and `python -m core.database.load_seed_data` (`:318`). None of `scripts/init_fresh_db.py`, `alembic/env.py`, `core/database/load_seed_data.py`, `core/seeds/`, `scripts/seed_starter_agents.py` or `scripts/seed_marketplace_agents_v2.py` opens a file for writing **[T]**.

| Path | Written by | Survives a restart? | Becomes |
|---|---|---|---|
| A here-document's temporary file in `${TMPDIR:-/tmp}` | bash, for the two here-documents at `docker-entrypoint.sh:349` and `:360`; bash 5.2 passes one that fits in a pipe through the pipe and writes a file only when it does not **[I]** | No | Scratch: `emptyDir` at `/tmp` |
| `~/.psql_history` | psql writes it only in an interactive session; the Job's psql reads `-c` or stdin **[I]** | — | None |
| `__pycache__/*.pyc` | as for the API | No | None |

## 3. The web app

The image runs `node server.js`, Next.js's standalone server (`frontend/Dockerfile`, production stage), as uid 1001.

| Path | Written by | Survives a restart? | Becomes |
|---|---|---|---|
| `/app/.next/cache` (images, fetch cache, regenerated pages) | Next.js's image optimiser and incremental cache. Images are not optimised (`frontend/next.config.js:183`), and `app/`, `lib/` and `components/` use no `revalidate`, `revalidatePath`, `revalidateTag`, `unstable_cache` or `force-cache` **[T]**, so neither writes; the build deletes the folder (`frontend/Dockerfile`, the builder's `rm -rf .next/cache`) | — | None |
| Server-side traces (`frontend/instrumentation.ts:12-13`) | sent over the network to the collector **[T]** | — | None |

## 4. The workspace worker

| Path | Written by | Survives a restart? | Becomes |
|---|---|---|---|
| `/workspaces/<workspace>/…`: metadata, task folders, cloned repositories, SSH and git settings, the canvas's state and the coding agent's transcripts (`CLAUDE_CONFIG_DIR`) | `workspace_manager.py:112-122`, `:172`, `:237-270`, `:357`, `:366`; `executor.py:276-277`, `:304-308`, `:364`, `:537`; `canvas_session_service.py:571-574`, `:373-374` **[T]** | Yes | Volume: unchanged |
| The ownership of `/workspaces` | `entrypoint.sh:68-80`, as root, before the drop to `worker` **[T]** | Yes | Volume: unchanged |
| `/etc/passwd`, `/etc/group` | `entrypoint.sh:46`, `:49` (`usermod`, `groupmod`), only with `LOCAL_WORKSPACE_ROOT` set, which compose sets and the chart does not **[T]** | — | None |
| A command an agent runs (`executor.py`) | runs with `HOME` and its caches in the workspace: `executor.py:905` (`HOME`), `:918` (`npm_config_cache`), `:914` (`PYTHONDONTWRITEBYTECODE`) **[T]** | Yes | Volume: unchanged |
| `/home/worker` (`HOME`, `entrypoint.sh:93`) | the coding agent the canvas starts (`canvas_session_service.py:371-378`) inherits it, with only `CLAUDE_CONFIG_DIR` moved to the volume **[T]**; the tools its shell runs keep their caches there: npm in `~/.npm`, pnpm, corepack and uv under `~/.cache` and `~/.local`, pip's `--user` installs in `~/.local` **[I]** | No | Scratch: `emptyDir` at `/home/worker` |
| `/tmp` | the coding agent's shell and its tools' scratch files **[I]**; an upload over 1 MB to the worker's routes (Starlette's spool) **[I]**; with `INSTALL_BROWSER=true` only (the published image is built with `false`, `.github/workflows/images.yml`), Chromium's profile and, under `--disable-dev-shm-usage`, its shared memory (`executor.py:545-551`) **[I]** | No | Scratch: `emptyDir` at `/tmp` |
| `/home/worker/.cache/ms-playwright` | a link made at build (`services/workspace-worker/Dockerfile`, the browser step); Playwright reads `PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers` instead **[T]** | — | Hidden by the `/home/worker` scratch volume; nothing reads it |
| `/app/__pycache__` | CPython, for the worker's own modules, which the image does not compile (`COPY . .` after the installs, which compile only site-packages) **[T]**; a failed write is skipped **[I]** | No | None |

## 5. The memory store

Not in the chart until PRD-280 Wave 6 (US-601). Nothing to list.

## 6. For Gerard: rows on the container's own disk today that are read again later

The chart keeps each on a scratch volume, as it was on the container's disk before. Whether any should move to a volume or to object storage is Gerard's decision.

| Row | Read again by |
|---|---|
| Skill repositories cloned to `~/.automatos/skills/<source>` | the skill loader's sync and its reads of a skill's files (`skill_loader.py:495`, `:517`, `:1098`, `:1171`) |
| SSH host keys in `~/.automatos/ssh_known_hosts` | the next SSH connection (`core/security/ssh_host_keys.py:64-66`) |
| The decision engine's shadow log, `/app/logs/decision_shadow.jsonl` | `scripts/eval/decision_shadow` |
| With `api.data.enabled=false`: generated documents and brand files in `/app/documents` | the document routes; the chart's values already say these are lost on every restart with `api.data` off |
