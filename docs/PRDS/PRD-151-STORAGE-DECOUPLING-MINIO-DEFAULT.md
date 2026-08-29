# PRD-151: Storage Decoupling — One S3 Code Path, MinIO as the Local Default

**Status:** Draft — **ABSORBED 2026-08-29 into PRD-233 S4** (open-core wave; see `PRD-WAVE-OPEN-CORE.md`). The endpoint seam (`S3_ENDPOINT_URL`) landed via PRD-176 F089; the factory consolidation + deletion of the three bespoke fallbacks builds under PRD-233. Do not build from this doc — the §2 boto3 census remains the reference list (re-verify at build; #625 removed the voice entries).
**Author:** Gerard Kavanagh (with Auto)
**Date:** 2026-06-09
**Type:** Refactor / Consolidation (centralize a scattered dependency; delete bespoke fallbacks)
**Related:** PRD-150 (Auth decoupling — open-core slice 1; this is slice 2), PRD-152 (mem0 & internal services), PRD-153 (One-command local run — provides the MinIO container), PRD-108 (Qdrant field memory — untouched here)

---

## 1. Introduction / Overview

Object storage is the second cloud hard-dependency blocking "clone → `docker compose up` → working platform" (auth is the first, PRD-150). The platform stores documents, generated images, voice audio, marketplace plugin archives, recipe step logs, mission checkpoints, and tool manifests in **AWS S3** — via **12+ independent `boto3.client(...)` instantiations scattered across the codebase, none of which pass `endpoint_url`**. There is no way to point the platform at any S3-compatible store other than AWS.

This PRD does **not** invent a `StorageProvider` interface. S3's API *is* the interface — the industry has standardized on it, and S3-compatible servers (MinIO) implement it byte-for-byte. The decoupling is therefore configuration, not abstraction:

- **One client factory** in core constructs every S3 client, honoring `S3_ENDPOINT_URL`.
- **OSS edition** points the factory at a **MinIO container** (ships in the PRD-153 compose) — full storage features locally, zero AWS.
- **SaaS edition** leaves `S3_ENDPOINT_URL` unset — identical code path against AWS, zero behavioral change.

Per the platform's clean-coding rules this is also a **consolidation with deletions**: the three bespoke local-fallback classes that grew organically (`LocalStorageService`, `LocalImageStore`, the attachments local path) are **deleted** — MinIO serves the "no AWS" case through the *same* code path, so the dual paths go.

### Why now

PRD-150's headline metric (M1: fresh clone boots into a working app) is unreachable while document upload/RAG fails at runtime without AWS credentials. Verified 2026-06-09: `DocumentManager` is constructed lazily (per-request factory, `api/documents.py:77`), so boot is green — but the first document operation crashes. "Boots green, breaks on first use" is not an open-source promise anyone respects.

---

## 2. Current-State Coupling Map (code-verified 2026-06-09)

### The scattered clients (no factory, no `endpoint_url`, ever)

| Call site | Bucket | Data | Failure without AWS creds |
|---|---|---|---|
| `modules/rag/ingestion/manager.py:424` (`DocumentManager.__init__`) | `S3_DOCUMENTS_BUCKET` | KB documents | **Runtime crash on first document op** (client + `_ensure_s3_bucket_exists()` in `__init__`; lazy construction saves boot only) |
| `api/documents.py:296–322` | `S3_DOCUMENTS_BUCKET` | uploads, presigned URLs | Runtime crash |
| `api/document_generation.py:389–402` | `S3_DOCUMENTS_BUCKET` | generated PDFs/HTML | Runtime crash |
| `api/voice_profiles.py:103–113` · `modules/voice/audio.py:28–34` | `S3_DOCUMENTS_BUCKET` | voice audio | Runtime crash |
| `api/recipe_executor.py:795–800` | `RECIPE_LOG_S3_BUCKET` | playbook step logs | Runtime crash on step log |
| `services/checkpoint_service.py:26` | `RECIPE_LOG_S3_BUCKET` | mission checkpoints | Runtime crash |
| `services/tool_manifest_service.py:26` | `RECIPE_LOG_S3_BUCKET` | tool manifest snapshots | Runtime crash |
| `services/workspace_purge.py:29–60` | `S3_DOCUMENTS_BUCKET` | workspace hard-delete prefix wipe | Runtime crash |
| `modules/tools/formatting/result_formatter.py:84` · `modules/tools/discovery/handlers_documents.py` | `S3_DOCUMENTS_BUCKET` | tool result artifacts | Runtime crash |
| `modules/search/vector_store/backends/s3_vectors_backend.py:56` | `S3_VECTORS_BUCKET` | AWS **S3 Vectors** (gated: `S3_VECTORS_ENABLED=false` default) | Only if flag on |
| `api/missions.py:63` | — | **dead import, no usage** | delete |

### The bespoke fallbacks (the dual paths to delete)

| File | Fallback | Why it dies |
|---|---|---|
| `core/services/marketplace_s3.py:268–278` | `LocalStorageService` when no `AWS_ACCESS_KEY_ID` | MinIO serves local through the S3 path |
| `core/services/image_store.py:205–214` | `LocalImageStore` | same |
| `modules/attachments/store.py:82–100` | local filesystem | same (lifecycle rules move to bucket config — see US-008) |

### What is explicitly NOT coupled (verified — do not "fix")

- **Vectors:** the live default is **pgvector in Postgres** (`S3_VECTORS_ENABLED=false`; compose already runs `pgvector/pgvector:pg16`). AWS S3 Vectors is an opt-in backend. **pgvector remains the OSS default — no work needed beyond keeping the S3 Vectors client lazy and flag-gated.**
- **Qdrant** (`field_memory`, PRD-108): self-hostable, boots gracefully when absent, gets a compose service in PRD-153. No code change here.
- **Workspace files:** agent deliverables live on the `workspace_data` volume, written by `services/workspace-worker` (Redis queue, **zero boto3 in `services/`** — verified). The original assumption that workers "need replacing with local file system" is happily false: they already are local. S3 coupling is orchestrator-side only.

---

## 3. Default replacement — recommendation

**MinIO** is the OSS default, running as a compose service.

| Option | Verdict |
|---|---|
| **MinIO** | ✅ S3-API drop-in (`endpoint_url` + path-style), battle-tested, supports presigned URLs + lifecycle (ILM) + prefix ops the code already uses. AGPL is a non-issue: it runs as a separate container, unmodified — no license contamination of the core. |
| SeaweedFS / Garage | S3-compatible but smaller ecosystems; no advantage over MinIO for this use. Rejected. |
| Local-filesystem `StorageProvider` abstraction | ❌ Rejected — creates a permanent second code path + CI matrix for zero gain. The platform already proved this pattern rots (three bespoke fallbacks). One S3 path, two endpoints. |
| AWS S3 Vectors local equivalent | ❌ Does not exist (MinIO does not implement the `s3vectors` API). Irrelevant: pgvector is already the default vector path. |

---

## 4. Goals

- **G1** — Every S3 client in the platform is constructed by **one factory** (`core/storage/s3.py`) honoring `S3_ENDPOINT_URL` / `S3_PUBLIC_ENDPOINT_URL` / `S3_USE_PATH_STYLE` from `config.py`.
- **G2** — With MinIO configured (OSS compose), **all** storage features work locally: document upload + RAG, generated documents/images, voice audio, marketplace archives, playbook step logs, mission checkpoints, tool manifests, workspace purge.
- **G3** — With nothing configured, storage features **fail fast with one clear, actionable error** ("object storage not configured — set S3_ENDPOINT_URL (local/MinIO) or AWS credentials"), not 12 different stack traces.
- **G4** — The three bespoke local fallbacks are **deleted**; `grep -rn "boto3.client" orchestrator/ --include=*.py` returns **only** the factory (plus the gated S3 Vectors backend).
- **G5** — **Zero behavioral change for SaaS** — `S3_ENDPOINT_URL` unset ⇒ byte-identical AWS calls.
- **G6** — Storage integration tests run against a **live MinIO container in CI**, proving the OSS path with real S3 semantics (not mocks).

---

## 5. User Stories

### Phase 0 — The seam

#### US-001: Storage config knobs
- [ ] `config.py` adds `S3_ENDPOINT_URL` (default `""` = AWS), `S3_PUBLIC_ENDPOINT_URL` (default `""` = same as endpoint; used for browser-facing presigned URLs), `S3_USE_PATH_STYLE` (default auto: `true` when `S3_ENDPOINT_URL` set).
- [ ] Existing `AWS_*` / bucket vars unchanged; no `os.getenv()` outside `config.py`.

#### US-002: One client factory
- [ ] New `core/storage/s3.py`: `get_s3_client()` (memoized per process) building `boto3.client("s3", endpoint_url=..., config=BotoConfig(s3={"addressing_style": ...}), ...)`.
- [ ] `is_storage_configured() -> bool` and a single `StorageNotConfigured` exception with the G3 message.
- [ ] `ensure_bucket(name, lifecycle_days: int | None = None)` — idempotent create + optional expiry rule (used by attachments' 7-day lifecycle).
- [ ] Unit tests: endpoint/path-style wiring; unconfigured → `StorageNotConfigured`.

### Phase 1 — Repoint and delete

#### US-003: Repoint all 12 call sites to the factory
- [ ] Every call site in §2 table 1 uses `get_s3_client()`; all module-level/`__init__` client construction becomes lazy (first use), including `DocumentManager` (move `_ensure_s3_bucket_exists` behind first write, via `ensure_bucket`).
- [ ] Delete the dead `boto3` import in `api/missions.py`.
- [ ] `grep -rn "boto3" orchestrator/ --include=*.py` hits only `core/storage/s3.py` + `s3_vectors_backend.py`.

#### US-004: Delete the bespoke fallbacks
- [ ] `LocalStorageService`, `LocalImageStore`, and the attachments local-path branch are **deleted** (no `_legacy`); their callers use the S3 path unconditionally, failing per G3 when unconfigured.
- [ ] Regression tests for marketplace archive fetch, image store, attachment store against MinIO.

#### US-005: S3 Vectors backend stays SaaS-only and lazy
- [ ] `s3_vectors_backend.py` constructs its `s3vectors` client lazily; clear error if `S3_VECTORS_ENABLED=true` without AWS creds (MinIO cannot serve it — document this in the module docstring).
- [ ] pgvector path untouched; parity test asserting default ingestion/search works with zero AWS env (MinIO for documents, pgvector for vectors).

### Phase 2 — Local semantics

#### US-006: Presigned URLs work from a browser in local mode
- [ ] Presigned generation uses `S3_PUBLIC_ENDPOINT_URL` when set (browser reaches `localhost:9000`; backend reaches `minio:9000`).
- [ ] Verified in browser via dev-browser skill: upload + download a document on the compose stack.

#### US-007: Bucket bootstrap
- [ ] On first use per bucket (documents, marketplace, recipe-log), `ensure_bucket` creates it idempotently — a fresh MinIO needs no manual setup.

#### US-008: Attachment lifecycle parity
- [ ] Attachments bucket gets a 7-day expiry lifecycle rule via `ensure_bucket` (MinIO ILM supports this); test asserts the rule exists.

### Phase 3 — Proof

#### US-009: MinIO in CI
- [ ] CI job runs the storage suite against a `minio/minio` service container (documents, marketplace, recipe logs, checkpoints, purge prefix-wipe).
- [ ] AWS path covered by existing mocked tests — both lanes green.

#### US-010: Failure-mode contract test
- [ ] With no storage env at all: each storage feature surfaces the single G3 error (API → 503 + actionable message), and **non-storage features are unaffected** (chat, missions without deliverable upload, auth).

---

## 6. Functional Requirements

- **FR-1** — `core/storage/s3.py` is the only constructor of S3 clients in `orchestrator/`.
- **FR-2** — `S3_ENDPOINT_URL` set ⇒ all object storage targets that endpoint (MinIO); unset ⇒ AWS, unchanged.
- **FR-3** — Presigned URLs are valid for browser use in both modes (`S3_PUBLIC_ENDPOINT_URL`).
- **FR-4** — All client construction is lazy; importing or constructing managers never performs network I/O.
- **FR-5** — Unconfigured storage ⇒ one typed exception, one HTTP error shape, zero crashes elsewhere.
- **FR-6** — `LocalStorageService` / `LocalImageStore` / attachments-local are deleted in the same PR that lands MinIO coverage of their callers.
- **FR-7** — Buckets self-create idempotently; attachments bucket carries the 7-day lifecycle rule.
- **FR-8** — pgvector remains the default vector backend; `S3_VECTORS_ENABLED=true` requires AWS and says so.
- **FR-9** — CI proves the MinIO lane with a live container.

---

## 7. Non-Goals (Out of Scope)

- **A `StorageProvider` interface / pluggable backends** — the endpoint *is* the seam. Pluggable = permanent CI tax for no second implementation we want.
- **Migrating existing AWS data** — SaaS stays on AWS; nothing moves.
- **Qdrant / field-memory changes** — compose wiring lands in PRD-153.
- **A local S3 Vectors emulation** — pgvector already covers OSS.
- **Marketplace content strategy** (central registry vs bundled) — product decision, separate PRD; its *assets* simply ride this factory.
- **Compose file authorship** — the MinIO service definition ships in PRD-153; this PRD makes the code able to use it.

---

## 8. Technical Considerations

- **Reuse over build:** the codebase already contains the target pattern three times (cred-checked fallback factories). This PRD replaces three bespoke patterns + nine crash sites with one factory — net deletion expected in `core/services/` and `modules/attachments/`.
- **Path-style addressing** is required for MinIO (`http://minio:9000/bucket/key`); virtual-host style stays default for AWS. Auto-derive from `S3_ENDPOINT_URL` presence, overridable.
- **Region:** MinIO ignores it; keep `AWS_REGION` defaulting as today so AWS is unaffected.
- **Workspace purge** (`workspace_purge.py`) does prefix deletes — semantics identical on MinIO; covered by US-009.
- **Risk ordering:** Phase 0–1 are mechanical and test-guarded; US-004 (deleting fallbacks) is the only behavior change and lands after MinIO CI (US-009) is green — mirror of PRD-150's parity gate.
- **Sequencing vs PRD-142 Wave 5:** touches `api/recipe_executor.py` (a playbook-consolidation surface). Land this PRD **after** Wave 5 merges, or rebase that one story.

---

## 9. Success Metrics

- **M1:** On the PRD-153 compose with zero AWS env: upload a document, ask a RAG question, install a marketplace plugin, run a playbook (step logs), checkpoint a mission, purge a workspace — all green.
- **M2:** `grep -rn "boto3.client" orchestrator/ --include=*.py` → factory + gated vectors backend only.
- **M3:** Net LOC negative across `core/services/marketplace_s3.py`, `core/services/image_store.py`, `modules/attachments/store.py`.
- **M4:** SaaS regression: existing storage tests pass with `S3_ENDPOINT_URL` unset, unchanged.
- **M5:** CI MinIO lane green and required.

---

## 10. Open Questions

- **Q1 — Bucket topology on MinIO.** Mirror the three AWS buckets (`documents`, `marketplace`, `recipe-log`) or one bucket with prefixes? (Proposed: mirror the three — zero code change to key layouts.)
- **Q2 — Default MinIO credentials.** Compose-generated random root creds in `.env` vs fixed dev defaults? (Proposed: required in `.env` like `POSTGRES_PASSWORD` — PRD-153 owns the answer.)
- **Q3 — `RECIPE_LOG_S3_BUCKET` default is `"automatos-ai"`** (a real prod bucket name). Rename default to `automatos-recipe-logs` while we're here, or leave? (Touches prod env — verify before changing.)
- **Q4 — Voice audio** rides `S3_DOCUMENTS_BUCKET` but the voice *service* is Railway-internal (PRD-152's problem). Confirm voice features are profile-gated in OSS so missing voice-service doesn't dangle audio uploads.

---

## 11. Phase Summary

| Phase | Stories | Character | Gate |
|---|---|---|---|
| 0 — Seam | US-001–002 | Additive | — |
| 1 — Repoint + delete | US-003–005 | Mechanical refactor + **deletions** | US-009 CI green before US-004 merges |
| 2 — Local semantics | US-006–008 | Behavioral (local only) | — |
| 3 — Proof | US-009–010 | Tests/CI | — |

**Estimated blast radius:** ~14 files touched, net-negative LOC, no schema changes, no API contract changes.
