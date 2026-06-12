# Ralph Review Prompt — PRD-151 Storage Decoupling (MinIO default)

You are a fresh-context **adversarial code reviewer**. The build loop claims PRD-151 is complete. Refute it. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-150-auth-decoupling)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD               # file-by-file on anything suspicious
```

Read `scripts/ralph/prd-151.json` (description = binding contract + verifier amendments).

## Hunt list

1. **G5 SaaS invariance (the prod data-loss vector)**: with `S3_ENDPOINT_URL` unset, AWS calls must be byte-identical. `ensure_bucket` must be a hard no-op on AWS. `put_bucket_lifecycle_configuration` must be reachable ONLY for a brand-new locally-created bucket — applied to a pre-existing bucket it REPLACES the AWS rules (data loss). `addressing_style` stays boto-default on AWS (path-style only with an endpoint).
2. **FR-4 lazy construction**: no boto3/factory client construction at module level or in `__init__` (DocumentManager, AttachmentStore, S3VectorsBackend, S3ImageStore) — first-use only. Import the modules and check for network I/O.
3. **Test pollution**: tests must monkeypatch config attributes and call `reset_storage_clients()` in setup/teardown — flag any test relying on ambient `S3_ENDPOINT_URL` env.
4. **StorageNotConfigured semantics per call site**: upload surfaces (documents, attachments, marketplace, images, admin_plugins) → 503; best-effort surfaces (step-log upload, purge wipe, generation_service upload, legacy download fallback, result_formatter) → silent degrade. Hunt broad `except Exception` blocks converting one into the other.
5. **Presigned URLs in browser-facing responses** use `S3_PUBLIC_ENDPOINT_URL` (localhost:9000), never internal compose DNS (minio:9000): documents.py download, document_generation.py redirect, generation_service download_url, voice audio playback, attachments sign_url.
6. **The three PRD-table-missed sites repointed**: api/workflow_recipes.py (step-log fetch), modules/documents/generation_service.py (upload+presign), api/documents.py reprocess download.
7. **Marketplace rename** S3StorageService→MarketplaceS3Service: all 8 caller files construct zero-arg; async signatures unchanged; plugin_cache + plugin_upload_service type hints updated.
8. **Region quirk**: attachments/purge had `config.AWS_REGION or "eu-west-1"`; factory unifies on us-east-1 default — flag if any code now behaves differently when AWS_REGION is set-but-empty.
9. **Secrets**: MinIO creds only as throwaway CI workflow values; no creds in config defaults; G3 error message leaks no config values.
10. **CLAUDE.md compliance**: no _legacy shims; deletions land with their replacement coverage; no `os.getenv` outside config.py; existing env var NAMES (RECIPE_LOG_S3_BUCKET) untouched.
11. **Forbidden surface**: zero compose/infrastructure edits; `recreate_s3_index.py`/`migrate_to_s3_vectors.py` still exist; no alembic migrations anywhere in the diff.

## Verification

Check the branch's latest CI run (`gh run list --branch <branch> --workflow test.yml --limit 1`): a FAILURE that is NEW versus the base branch is a finding; pre-existing reds are noted, not filed.


Run `bash scripts/ralph/acceptance-prd151.sh`. Non-zero exit = automatic CRITICAL finding.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM findings** → reply exactly `REVIEW_PASS` plus a 5-line summary (LOW/nits there).
- **Findings** → append fix stories `P151-RVW-1..n` to `scripts/ralph/prd-151.json` (file:line evidence, mechanical ACs). Commit `chore(prd-151): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only to `origin ralph/prd-151-`* (your fix-story commit may be pushed); never force, never another ref.
