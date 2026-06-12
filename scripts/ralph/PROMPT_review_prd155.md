# Ralph Review Prompt — PRD-155 Route Contract & Mount Honesty

You are a fresh-context **adversarial code reviewer**. The build loop claims PRD-155 is complete. Refute it. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-154-wave0-quick-wins)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-155.json` (`description` = binding contract) so you know what was promised. The whole point of this PRD is a HONEST net — your job is to prove it is honest, not theatrical.

## Hunt list

1. **No hidden suppression (S2)**: there is NO allowlist/skip/`@pytest.mark.skip`/ignore-file letting a real ⊆-manifest violation pass. Grep the contract suite and config for any opt-out. A path the frontend calls that is NOT in the manifest must have been FIXED or the caller DELETED — not excused.
2. **Manifest is real (S1)**: `reports/route-manifest.json` is generated from the actual FastAPI app (not hand-written), deterministic across two runs, and generates with NO Postgres. A stubbed/partial app import that omits routers makes the whole contract a lie — check the dump imports the real app.
3. **Mount honesty (S3)**: zero `try/except ImportError` around router mounts remain in `main.py` (grep gate). Boot RAISES with the router name on failure. `ALLOW_DEGRADED_BOOT` reads through `config.py` only (no `os.getenv` in main.py), defaults OFF. The two previously-silent imports (main.py:115,123) are genuinely fixed or deleted — confirm the diff shows which, and that a "fix" actually imports a real module.
4. **Reachability is real (S4)**: the test enumerates the LIVE registry (not a hand-copied list that can drift); an injected stale tool name fails it. No LLM/network in the test.
5. **CI wiring (S5)**: both jobs added to test.yml use the Postgres-service + per-test-timeout pattern; the non-required rationale + flip plan are documented in the workflow header.
6. **Negative fixtures exist**: S2 fabricated-path fixture and S4 stale-name fixture must actually fail the suites (the net catches violations) — a suite that passes everything including the negative fixture is broken.
7. **Hygiene**: clean tree; no weakened tests; `_sys_guard` on new backend test files.

## Verification

- `gh run list --branch ralph/prd-155-route-contract --workflow test.yml --limit 1`: NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd155.sh`. Non-zero = automatic CRITICAL.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary.
- **Findings** → append `P155-RVW-1..n` fix stories to `scripts/ralph/prd-155.json` (title, file:line evidence, mechanical ACs, files). Commit `chore(prd-155): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-155-route-contract` (never force, never another ref). Do not re-run the build.
