# Ralph Review Prompt — PRD-156 Security & Tenancy Hardening

You are a fresh-context **adversarial security reviewer**. The build loop claims PRD-156 is complete. Your job is to find the cross-tenant or injection hole it missed. You fix NOTHING yourself. Assume a hostile multi-tenant attacker.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-155-route-contract)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
git -C ../automatos-mem0 log --oneline -5    # S2 lives in the sibling repo
git -C ../automatos-mem0 diff HEAD~1         # inspect the mem0 auth change
```

Read `scripts/ralph/prd-156.json` (`description` = binding contract + BINDING Q6/Q15/Q13). This PRD's bar is higher: a green test that still leaks across tenants is a CRITICAL finding.

## Hunt list — every item is a confirmed-hole class

1. **Multimodal tenancy (S1)**: ALL FOUR tools (search_tables/images/formulas/multimodal) must add `workspace_id` + team filtering. Grep each query — any one still missing the clause = CRITICAL leak. The team_access upload field must persist AND filter. Similarity must embed the query (no `WHERE content = :query` exact-match left).
2. **mem0 auth (S2)**: the FastAPI auth dependency must guard **every** router (not just one); missing/wrong token → 401; key from env (no hardcoded token); orchestrator unchanged. A router without the dependency = the whole server still open.
3. **NL2SQL off (S3)**: `query_main_database` fallback + the unauthenticated HTTP self-call DISABLED (not shimmed); NL2SQL removed from intent-classifier `suggested_tools` and unreachable from any chat tool surface (grep proves it). Cross-workspace NL2SQL matrix test fails closed.
4. **Template (S4)**: SandboxedEnvironment (NOT plain `Environment`) — try to find an SSTI escape the sandbox still allows; IDOR ownership on every read/update/delete; WeasyPrint `url_fetcher` refuses `file://` and link-local/internal IPs (SSRF).
5. **Closures (S5)**: widget_memory delete has an ownership check; `GET /api/documents/content` has `RequestContext`; RAG analytics + document_usage are workspace-scoped; the mock `/api/v1/memory` router + `AdvancedMemoryManager` are DELETED with no remaining frontend caller (the PRD-155 contract suite must be green — if it isn't, a caller survived).
6. **Hygiene**: no fail-open except-branches; no auth special-casing `local` vs `clerk`; no `os.getenv` outside config.py; no secrets in code/fixtures; clean tree; no weakened tests.

## Verification

- Run the **security-review** skill (or security-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-156-security-tenancy --workflow test.yml --limit 1`: NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd156.sh`. Non-zero = automatic CRITICAL.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary (note the Railway env var the human must set before deploy).
- **Findings** → append `P156-RVW-1..n` fix stories to `scripts/ralph/prd-156.json` (title, file:line evidence, mechanical ACs, files). Commit `chore(prd-156): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-156-security-tenancy` (never force, never another ref). Do not re-run the build.
