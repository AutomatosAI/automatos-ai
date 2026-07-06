# Ralph Review Prompt — PRD-186 S3 Vectors Relight

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-186 is complete. Your job is to find where it cheated, half-shipped, left a shim, or silently swallowed the very thing this PRD exists to make loud. You fix NOTHING yourself. The bar: a "green" PRD where a placeholder-less-while-enabled bucket could still boot dark, or where the F005 assertion lives in two places, or where a dimension mismatch still only logs — is a finding.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-186.json` (`description` + `acceptanceCriteria` = binding contract). Full spec (reference): `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai-wave1-prds/docs/PRDS/PRD-186-PHASE2-S3-VECTORS-RELIGHT.md`.

**Scope guard (important):** this run is the TWO CODE stories only. If the diff contains any attempt at the OPS work — a real re-embed, a migration run, a probe invocation against a backend, an env/bucket change in code, or an `os.getenv` — that is a finding (out-of-scope / convention break).

## Hunt list — every item is a confirmed-risk class

1. **S1 — the F005 assertion is extracted, not duplicated.** Grep `orchestrator/config.py`: `assert_vector_config_integrity()` must exist as a pure function, and `validate_security` must CALL it. The old inline F005 branch (and its message string) must be **DELETED** — if the check now lives in two places, that is a shim finding. The assertion must raise for `S3_VECTORS_ENABLED=true` + bucket `automatos-ai` and + empty bucket, pass for `automatos-vectors-{workspace_id}`, and be silent when disabled. Find an input class that slips through.
2. **S2 — the failure is un-swallowable.** The integrity check must run **outside/before** the `run_stage` that catches-and-marks-`failed` without re-raising (`core/models/bootstrap.py` ~127-136). Trace the call in `main.py`: if the assertion is invoked *inside* that swallowing try, the plane can still boot dark — that is a CRITICAL finding. Confirm the failure path hard-aborts the process (fail-closed), not just records a `failed` stage.
3. **S3 — dimension mismatch raises, never mutates.** In `s3_vectors_backend.py` `_verify_or_recreate_index`: a confirmed index-vs-configured dimension mismatch must **raise** a typed error, not log-and-continue. And it must **never delete/recreate a populated index** — a delete on the mismatch branch is a data-loss CRITICAL. Find a path that still warns-and-proceeds, or that deletes.
4. **Tests are PURE.** Every new/changed test must run with no DB, no network, no AWS, no real boot — mocked at the boundary. A test that reaches a real service (or is skipped/weakened to go green) is a finding.
5. **Hygiene:** no `os.getenv` outside `config.py`; no hardcoded values; no backward-compat shim anywhere; clean tree; the full orchestrator suite green (no protected test weakened).

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-186-s3-vectors-relight --workflow test.yml --limit 1`: a NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd186.sh`. Non-zero = automatic CRITICAL.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly that the OPS relight (bucket env change + `migrate_to_s3_vectors.py` re-embed + S8-probe re-run) remains **Gerard's to run against prod** — this PRD only made the config-integrity loud.
- **Findings** → append `P186-RVW-1..n` fix stories to `scripts/ralph/prd-186.json` (title, file:line evidence, mechanical ACs, files). Commit `chore(prd-186): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-186-s3-vectors-relight` (never force, never another ref). Do not re-run the build.
