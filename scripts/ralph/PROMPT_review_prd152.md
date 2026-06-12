# Ralph Review Prompt — PRD-152 mem0 / Internal-Services Decoupling

You are a fresh-context **adversarial code reviewer**. The build loop claims PRD-152 is complete. Refute it. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-151-storage-minio)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-152.json` (description = binding contract + verifier amendments).

## Hunt list

1. **Env-override preservation (G5)**: every flipped default keeps the `os.getenv(NAME, default)` form — Railway/SaaS must still win via env. Reject any removed override path or renamed env var.
2. **os.getenv discipline**: no new `os.getenv`/`os.environ` outside orchestrator/config.py — the three `services/*/automatos_logging.py` copies are the ONLY exception.
3. **Guard quality**: empty-URL guards must not weaken connected-mode behavior — handlers_monitoring/logs-api/voice keep their existing configured-mode error paths byte-equivalent; no blanket try/except swallowing.
4. **mem0_client.py untouched**: request/retry/breaker logic identical to base — only tests may grow around it. Any MemoryProvider-style interface or unified_memory_service contract change = CRITICAL (explicit non-goal).
5. **Compose freeze**: `git diff $BASE..HEAD -- docker-compose.yml infrastructure/` must be EMPTY (committed drift included). Also no `.env`/`.env.example` edits; envs/*.defaults comment edits only.
6. **Test integrity**: railway-default assertions in test_config_env_centralization.py became local-default assertions — not deleted. Count assertions in the diff.
7. **tool_registry.py**: example/description strings only — tool schemas, parameter definitions, SecurityLevel unchanged.
8. **Grep-dodging**: no string concatenation or constants introduced to dodge the `railway.internal` literal gates.
9. **Boot summary (S4)**: no heavy module-scope imports in automatos_logging.py; INFO level; one line per disabled service; zero stack traces.
10. **Secrets**: MEM0_API_KEY/OPENAI_API_KEY never gain a non-empty default anywhere in the diff.
11. **Sibling repo**: zero writes to ../automatos-mem0; no alembic migrations anywhere in the diff.

## Verification

Check the branch's latest CI run (`gh run list --branch <branch> --workflow test.yml --limit 1`): a FAILURE that is NEW versus the base branch is a finding; pre-existing reds are noted, not filed.


Run `bash scripts/ralph/acceptance-prd152.sh`. Non-zero exit = automatic CRITICAL finding.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM findings** → reply exactly `REVIEW_PASS` plus a 5-line summary (LOW/nits there).
- **Findings** → append fix stories `P152-RVW-1..n` to `scripts/ralph/prd-152.json` (file:line evidence, mechanical ACs). Commit `chore(prd-152): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only to `origin ralph/prd-152-`* (your fix-story commit may be pushed); never force, never another ref.
