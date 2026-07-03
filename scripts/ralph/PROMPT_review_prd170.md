# Ralph Review Prompt — PRD-170 Code Canvas: Claude Agent SDK Embed

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-170 is complete. Your job is to find what it missed — a session that can escape its workspace mount, a token that reaches a log, a second exec surface that wasn't deleted, an AC faked green that should be DEFERRED. You fix **NOTHING** yourself.

## Scope

```
BASE=$(git merge-base HEAD main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-170.json` (`description` = binding contract + amendments **D3, D11, Q36, Q38, Q41, Q82, Q85**) and `docs/PRDS/PRD-170-CODE-CANVAS-AGENT-SDK.md`. This is a net-new coding surface whose whole safety story is **"nothing applies without approval; the session is confined to its workspace; git is the audit trail."** Tenancy and secret-handling findings are CRITICAL even with every test green.

## Hunt list — every item is a confirmed-failure class

1. **Session confinement (S1, tenancy)** — the session must only touch its own workspace mount. Look for any path that takes a workspace_id from the request and a path from the agent without re-binding to the authenticated workspace mount (path traversal / cross-workspace reach = CRITICAL). One active session per workspace v1 enforced. Resume-after-restart reads state from the volume (not memory). These are CI/Docker ACs — confirm the **contract test exists and is DEFERRED honestly**, not faked green.
2. **Secret handling (S5, security)** — push uses GitHub App installation tokens + platform actor identity; **no token material in logs, errors, or transcripts** (the PRD-154 S12 test class must be present and passing — this is a LOCAL gate, not deferrable). Author/committer correctly attributed. A token in a log line = CRITICAL.
3. **One exec surface (S7, Q85)** — `orchestrator/api/workspace_exec.py` is DELETED (grep — gone, not just unmounted); `workspace_files` POST /exec + the session shell are the only exec paths; the InteractiveTerminal 404 affordance is removed (component targets the session). A surviving second exec surface = CRITICAL.
4. **Reuse, not rebuild (S3/S4)** — the canvas extends the existing `CodingCanvasWidget/` shell and the existing SSE channel shape; a parallel file API or a second streaming channel = finding. The event schema is versioned and a drifted name fails the vitest (confirm the negative fixture). Auto-accept toggle defaults OFF and is session-scoped + visibly indicated.
5. **No heavy-dep smuggling (S4)** — a new heavy frontend dependency (a diff viewer that isn't the in-bundle Monaco) without a memo in the commit body = finding. Approve applies / deny reverts+informs (e2e is morning-human; confirm the proxy + DEFER note).
6. **Honest deferral discipline** — every container/browser AC is either DONE with a real CI/Docker test behind it or `DEFERRED — morning/CI check: …`. An AC marked DONE whose only evidence is "looks right" with no test = CRITICAL (faked green). Conversely, a security AC (path-escape, token-leak) marked DEFERRED that could have been unit-proven = finding.
7. **Hygiene** — no `os.getenv` outside `config.py`; no secrets/tokens in code or fixtures; no multi-user/non-Claude/arbitrary-repo scope creep (non-goals); clean tree; no weakened/skipped tests.

## Verification

- `gh run list --branch ralph/prd-170-code-canvas-sdk --workflow test.yml --limit 1`: a NEW failure vs base = finding (arbitrate new-vs-pre-existing). Note: the Docker-gated job is the authority for the container ACs.
- Run `bash scripts/ralph/acceptance-prd170.sh`. Non-zero = automatic CRITICAL.
- Run the `security-review` skill on `git diff $BASE..HEAD` — this PRD is the highest tenancy/secret surface in the chain; any CRITICAL/HIGH it reports is a finding.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note for the human: the live demo to run in the morning (open canvas on a workspace repo → "add input validation to X and push" → streamed work, diff approvals, branch pushed) and any DEFERRED container ACs the Docker CI job must confirm.
- **Findings** → append `P170-RVW-1..n` fix stories to `scripts/ralph/prd-170.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-170): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-170-code-canvas-sdk` (never force, never another ref). Do not re-run the build.
