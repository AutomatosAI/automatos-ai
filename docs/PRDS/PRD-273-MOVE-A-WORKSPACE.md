# PRD-273: Move a workspace — export it from one install, import it on another

**Status:** Draft for review, 6 Nov 2026. Every §8 recommendation is the default unless Gerard changes it.
**Why now:** Gerard runs his business on a self-hosted install (a VPS) and his product work on Automatos Cloud. On
6 Nov: "My VPS is working but it's OLD Auto and he doesn't know me … how is the best way to transfer from the SaaS
workspace … We should add export/import option?" The self-hosted Auto had the seeded default persona, none of his
memories, brand kit, knowledge or agents. A workspace's value is what it has learned; today it cannot leave the
install it was built on.
**The same need, for customers:** a team that starts on Automatos Cloud and moves to its own servers (the
self-hosted enterprise path), a local user who moves to a bigger machine, and the open-core promise that your data
comes with you.
**Scope:** automatos-ai (core and web). Automatos Go and the Design Kit need nothing.
**Depends on:** PRD-181 (the GDPR export, `services/gdpr_service.py`), PRD-270 (the marketplace bundle schema and
resolver, `services/marketplace_bundles/`), the RAG ingestion pipeline (`modules/rag/ingestion/`), durable and field
memory (`modules/memory/durable_store.py`, `modules/context/adapters/vector_field.py`).

## 1. Introduction

An owner or admin clicks **Export workspace** on one install, picks what to bring, and downloads one file. On the
other install they click **Import workspace**, upload the file, see what it holds, and confirm. Auto arrives with
its soul and settings, the brand kit, the business profile, its memories, the documents and knowledge, the agents,
skills and Playbooks, and (by choice) the chat history. Documents and memories are re-indexed with the new
install's own embeddings. Credentials, connections, keys, billing and telemetry never travel: the owner reconnects
apps on the new install.

## 2. Goals

- An owner or admin exports a workspace to one file from Settings, on either edition, and imports it on another
  install of the same or a newer core version.
- After import, Auto on the new install answers with the same soul, communication style and proactive level, follows
  the same brand kit, and recalls the same durable memories (same text, same namespace, same subject).
- Every document arrives with its file, and is searchable on the new install once re-indexing finishes; the page
  shows progress.
- Agents, skills and Playbooks arrive with their prompts and skills; what does not resolve on the target (a plugin it
  lacks, a model it does not offer) is listed, not silently dropped.
- The file never contains a secret: no credential, key, token, connection, phone pairing or password hash.
- A failed import changes nothing: each category commits as a whole or not at all, and the page says which
  categories landed.
- Exporting the same workspace twice and importing both into one install does not duplicate anything.

## 3. Non-goals

- **Live sync between two installs.** This is a one-off move, not replication.
- **Moving telemetry and history that the new install rebuilds or does not need:** metric events, heartbeat
  results, LLM usage, tool logs, notifications, routing decisions, audit logs, code graphs (rebuilt from the repo).
- **Moving secrets of any kind.** Composio connections, channel connections, credentials and API keys are
  reconnected by the owner; the import page lists what to reconnect.
- **Billing and credits.** The credit ledger and balances stay with the install that holds the payment.
- **Importing on Automatos Cloud in v1** (§8-Q8): a hosted import needs size limits, abuse checks and paid
  re-embedding first.
- **A direct "pull from Automatos Cloud" without a file.** See §9.

## 4. What exists today

### 4.1 The GDPR export (PRD-181)

`GET /api/gdpr/export` (`api/gdpr.py`, workspace admins) returns one JSON bundle, format `automatos.gdpr.export/v1`:
- `sql`: every row of every `public` table that has a `workspace_id` column (`workspace_purge._discover_scoped_tables`),
  with only `mobile_devices`' secret columns removed (`EXPORT_REDACTED_COLUMNS`);
- `derived.field_memory` and `derived.durable_memory`: payloads without vectors.

What it does not do, for a move:
- **It misses the workspace row itself.** `workspaces` has no `workspace_id` column, so `workspaces.settings` is not
  exported: the brand kit (`settings['brand_kit']`), the orchestrator block (Auto's soul as the Settings page shows
  it), voice and push settings.
- **It carries secrets.** `credentials`, `sdk_api_keys`, `user_api_keys`, `channel_connections` and
  `composio_entities` rows are exported as stored.
- **It carries no files.** Document rows point at objects in S3 or MinIO; the objects are not in the bundle.
- **It is built in memory.** One real Cloud workspace holds about 800,000 scoped rows across 117 tables; one JSON
  response of that size is not viable.

### 4.2 Where "Auto knows me" lives

| What | Where | In the GDPR export? |
|---|---|---|
| Soul, personality, communication style, proactive and thinking level | the Auto agent row: `custom_persona_prompt`, `use_custom_persona`, `configuration` (slug `auto-{workspace_id}`, `api/workspaces._get_or_seed_auto_agent`); a copy in `workspaces.settings['orchestrator']` | agent row yes; settings no |
| Brand kit | `workspaces.settings['brand_kit']` (`modules/documents/brand_kit`) | no |
| Business profile | `business_profiles` | yes |
| Durable memories | Qdrant, `DurableMemoryStore` (`modules/memory/durable_store.py`) | yes, without vectors |
| Field memory | Qdrant shared collection (`VectorFieldSharedContext`), only when `QDRANT_URL` is set | yes, without vectors |
| Short-term memory | `memory_short_term` | yes |
| Documents | `documents` (+ the file in S3 / MinIO), `document_chunks`; vectors in S3 Vectors on Cloud, pgvector on a local edition (`modules/search/vector_store/backends/pgvector_local_backend.py`) | rows only |
| Knowledge | `knowledge_items`, `kb_entities`, `kb_types`, `kb_formulas`, `entity_relationships`, `workspace_graphs` | yes |
| Agents, skills, Playbooks | `agents`, `skills`, `skill_files`, `workspace_enabled_skills`, `workflow_recipes`, `teams` | yes |
| Chats | `chats`, `messages` | yes |
| Deliverables | `deliverables` (+ files) | rows only |

Sizes from one working Cloud workspace (6 Nov): 1,942 documents and 21,782 chunks; 69,428 knowledge items;
24 agents and 80 skills; 193 chats with 1,566 messages; 2,240 deliverables. Cloud embeds with
`qwen/qwen3-embedding-8b` at 2,048 dimensions.

### 4.3 What to reuse

- **Table discovery:** `workspace_purge._discover_scoped_tables` (every `workspace_id` table, self-maintaining).
- **Agent and Playbook bundles:** `services/marketplace_bundles/` already exports an agent as a portable bundle (its
  prompt, shared persona by slug, model id, global skills by name, Composio apps, plugins by slug) and resolves one
  on another install (`resolve.py`, recording what does not resolve). The workspace file uses the same bundle
  schema; the import writes workspace agents instead of marketplace rows.
- **Re-indexing:** the RAG ingestion manager re-chunks and embeds a document from its file on the target's own vector
  store; durable memories are written through `DurableMemoryStore.add` (namespace, workspace and subject kept), which
embeds them and dedups same-content writes.
- **The Auto settings write path:** `api/workspaces.save_orchestrator_settings` (`_merge_orchestrator`,
  `services/orchestrator_auto_agent.sync_auto_configuration`): the import writes Auto through the same functions, so
  the agent row and the settings block agree.

## 5. Design

### 5.1 The file

One zip, `<workspace-name>-<date>.automatos-workspace`, streamed to disk on export and from disk on import:

```
manifest.json            format "automatos.workspace/v1", source edition, core version, Alembic head,
                         exported_at, exported_by, categories, per-part row counts and SHA-256
workspace.json           the allow-listed settings keys (§5.3)
auto.json                Auto's soul, use_custom_persona, configuration
sql/<table>.jsonl        one row per line, allow-listed tables only, secret columns removed
bundles/agents.jsonl     agent bundles (marketplace_bundles schema); playbooks.jsonl the same
memory/durable.jsonl     text, namespace, subject, metadata; no vectors
memory/field.jsonl       key, value, strength, provenance; no vectors
files/<sha256>           each document and deliverable file once, by content hash
```

Optional passphrase (§8-Q5): every entry but the manifest is encrypted with AES-256-GCM under a key derived from the
passphrase (scrypt), using `cryptography`, already a dependency (`orchestrator/requirements.in`); no new package.
The manifest stays readable so the import page can show what the file holds before the passphrase is typed.

### 5.2 Categories

The owner ticks categories on export; import can untick them again.

| Category | Default | Carries |
|---|---|---|
| **Auto** | on, always | `auto.json`, `workspace.json` (brand kit, orchestrator block), `business_profiles`, `personas` |
| **Memory** | on | durable and field memory, `memory_short_term` |
| **Knowledge** | on | documents and their files, knowledge items and the knowledge graph tables |
| **Agents and skills** | on | agent and Playbook bundles, workspace skills and skill files, enabled skills, teams, routing rules |
| **Chats** | on (§8-Q2) | chats and messages; the Auto thread becomes the target's Auto thread |
| **Deliverables** | on | deliverables and their files |
| **Board** | off (§8-Q3) | board tickets and agent reports |

### 5.3 Allow-list, never deny-list

Every `workspace_id` table is classified in one map in code, `MOVED` or `NOT_MOVED`, with a reason. A guard test
fails when `_discover_scoped_tables` finds a table in neither, so a new table cannot leak into the file by default
and cannot be forgotten either. Settings keys are allow-listed the same way: `brand_kit`, `orchestrator` (without
`llm` and `model`, §8-Q7), `voice_live`, `push`. Never: `byok_overrides`, `integrations`.

`NOT_MOVED` always includes: `credentials`, `sdk_api_keys`, `user_api_keys`, `channel_connections`,
`composio_entities`, `mobile_devices`, `mobile_digest_posts`, `approval_grants`, `credit_ledger`, `credit_balances`,
`workspace_members`, `audit_logs`, and the telemetry tables in §3. Secret-bearing columns of moved tables (a
document source's connection string, for one) are named in the same map and removed.

### 5.4 Import

1. **Validate before writing:** format and version (the target's Alembic head must be the file's or newer), every
   part's SHA-256, entry names (no `..`, no absolute paths), uncompressed size caps, row caps per table, and only
   allow-listed tables and columns. A file that fails writes nothing.
2. **Preview:** the page shows each category's counts and what will not resolve (models, plugins, Composio apps the
   target lacks) before the owner confirms.
3. **Into the target's workspace:** on a local edition, its one workspace. Every `workspace_id` becomes the target's;
   every person reference (`created_by`, `user_id`, owner columns) becomes the importing person.
4. **New ids, mapped:** integer primary keys are re-issued; UUID keys are kept unless taken. An old→new map per
   table rewrites foreign keys (read from the SQLAlchemy metadata) and the JSON references the map names
   (agent ids in a team, a skill's agent). A reference that maps nowhere is dropped and listed.
5. **Idempotent:** each imported row records its source (`import_source`: the file's workspace id and the row's old
   id, in one mapping table). Importing the same file again updates those rows instead of copying them.
6. **One transaction per category**, in dependency order (Auto, agents and skills, knowledge, memory, chats,
   deliverables, board). A category that fails rolls back alone; the result lists each category's outcome.
7. **Auto** is written through the Orchestrator settings write path (§4.3). Same-slug seeded agents follow §8-Q1.
8. **Re-index in the background:** files go to the target's storage; documents are queued for ingestion on the
   target's vector store; memories go through `DurableMemoryStore.add`, which embeds them. The page shows
   "Re-indexing: N of M documents", and Auto's memory and search work as each item lands. Embedding cost is booked to
   the target workspace like any ingestion.

### 5.5 The UI

A **Move workspace** tab in Settings, for owners and admins, on both editions:
- **Export:** category checkboxes, optional passphrase, **Export**; a background job, then a download link that
  expires after `WORKSPACE_EXPORT_LINK_TTL_HOURS`.
- **Import** (local edition in v1): upload, the preview, **Import**; then progress per category and re-indexing, and a
  "Reconnect" list (the Composio apps and channels the source had, as names only).

For a file too big for a browser upload, an operator runs the same import from the server:
`python -m scripts.workspace_move import <file>` inside the backend container (§8-Q6).

### 5.6 Security

- Export and import: workspace owners and admins only (`require_workspace_admin`), audited like the GDPR export.
- The file holds personal data (memories, chats, documents): the download link is single-use and expires; the
  export job's file is deleted after download or expiry; the page says the file is personal data.
- The import never executes anything from the file: rows are inserted through bound parameters, files are stored
  as data, bundles go through `marketplace_bundles/validate.py`.
- Rate limit: one export and one import per workspace at a time.
- **The GDPR export gets the same secret rules:** it stops carrying the `NOT_MOVED` secret tables and columns
  (§5.3), and gains `workspace.json` so a data-subject export includes the brand kit and settings. Its format becomes
  `automatos.gdpr.export/v2`.

## 6. User stories

### Wave 1: export (one PR)

- **US-273-01 — The classification map and its guard.** `MOVED` / `NOT_MOVED` with reasons and secret columns;
  tests: every scoped table classified; a new scoped table in a test schema fails the guard; settings allow-list.
- **US-273-02 — Write the file.** Streamed zip per §5.1, categories per §5.2, files by content hash, manifest with
  counts and SHA-256; tests: no `NOT_MOVED` table or secret column in any part; a workspace with 100k rows exports
  with flat memory use; manifest matches the parts; passphrase round-trip.
- **US-273-03 — The export job and route.** `POST /api/workspace-move/export` (job), `GET …/export/{id}` (status),
  `GET …/export/{id}/file` (single-use link); admin-only; route manifest; tests: a member is refused, the link works
  once, expiry deletes the file.
- **US-273-04 — GDPR export v2.** §5.6; tests: secret tables gone, `workspace.json` present.

### Wave 2: import (one PR, after wave 1 merges)

- **US-273-05 — Validate and preview.** §5.4 steps 1–2; tests: bad hash, zip-slip name, oversize part, unknown
  table, newer Alembic head in the file → refused with nothing written; the preview's counts and unresolved list.
- **US-273-06 — Import the rows.** §5.4 steps 3–6; tests: ids re-issued and foreign keys rewritten; person columns
  become the importer; a second import of the same file changes nothing; a failing category rolls back alone.
- **US-273-07 — Auto, memory and re-indexing.** §5.4 steps 7–8; tests: the Orchestrator GET on the target returns the
  file's soul and settings; durable memories recallable by the same text; documents queued and searchable after
  ingestion; progress counts.
- **US-273-08 — The Move workspace tab.** §5.5; tests: owners and admins only, the import controls only on a local
  edition, preview, progress, the Reconnect list.
- **US-273-09 — The server-side import command and docs.** `scripts/workspace_move`; `docs/getting-started/
  self-hosting.md` gains "Move a workspace from Automatos Cloud".

## 7. Rollout

1. Wave 1 merges and deploys to Automatos Cloud (the export runs there).
2. Wave 2 merges; a self-hosted install updates.
3. Gerard exports his Cloud workspace and imports it on his VPS; the result (counts, unresolved list, Auto's first
   reply) is checked against the Cloud workspace before the PRD is called done.

## 8. Decisions for Gerard (each recommendation stands unless you change it)

1. **Agents that already exist on the target by slug** (the seeded Writer, Engineer …): the file's version updates
   the target's (rec: your customised agents arrive as you left them) vs keep the target's vs import both.
2. **Chats** — on by default (rec: Auto's thread and your history arrive) vs off by default.
3. **Board tickets and reports** — off by default (rec: history, large, links to agents and runs) vs on.
4. **Telemetry** — never moved (rec) vs an optional "everything" category.
5. **Passphrase** — optional, offered on export (rec) vs required vs none.
6. **Big files** — browser upload up to `WORKSPACE_IMPORT_MAX_UPLOAD_MB` plus the server command (rec) vs the
   server command only.
7. **Auto's model** — not carried; the target keeps its own provider and model (rec: providers differ by install)
   vs carried when the target offers the same model.
8. **Directions in v1** — Cloud → self-hosted and self-hosted → self-hosted (rec) vs also import on Cloud.

## 9. Later

- **Pull from Automatos Cloud:** sign in to Cloud from a self-hosted install and import a workspace directly, no file
  (PRD-270's Settings tab is the natural home).
- **Import on Automatos Cloud** (§8-Q8), with size limits, abuse checks and re-embedding billed as credits.
- **Scheduled exports** as an owner-held backup of a self-hosted install.
