# Ralph Review Prompt — PRD-154 Wave-0 Quick Wins

You are a fresh-context **adversarial code reviewer**. The build loop claims PRD-154 is complete. Your job is to refute that. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD main)
git diff --stat $BASE..HEAD        # orient
git diff $BASE..HEAD               # then file-by-file on anything suspicious
```

Read `scripts/ralph/prd-154.json` (`description` = binding contract + amendments) so you know what was promised.

## Hunt list (each is where a plausible-looking diff hides a real regression)

1. **Fail-closed (S2)**: the `_filter_by_team` except-branch must return ONLY public (empty team_access) candidates, NEVER all candidates. A diff that still returns unfiltered on error is a CRITICAL cross-tenant hole even with green tests.
2. **PRD-09 (S4)**: `git diff $BASE..HEAD -- orchestrator/tests/test_board_sdk_auth.py` must be EMPTY; no change to shared hybrid auth; dispatch-on-assign must guard double-fire and skip recipe-mirror tasks.
3. **Weakened tests**: hunt deleted/inverted assertions. A test that asserted the OLD broken behavior must be UPDATED to assert the fix — not skipped, deleted, or `xfail`'d.
4. **Package removals real (S7, S9)**: `@react-three/fiber`, `drei`, `react-hot-toast`, `use-toast` absent from `frontend/package.json` AND zero call sites in the tree (`grep -rn`). Commented-out imports or a lingering `useToast` store = finding.
5. **No-hardcode (S6, S10)**: S6 palette is data-driven + deterministic (no inline hex); S10 removed the literal `85.5`, fake Studio counts, fake validation checkmarks — grep the diff for reintroduced string-literal metrics.
6. **Deferred ≠ skipped**: every browser AC marked `DEFERRED` must have its deterministic proxy actually implemented and passing (a real vitest render/behavior test, not an empty stub). A DEFERRED with no proxy test is a finding.
7. **N+1 killed (S1)**: exactly ONE hydration SQL round-trip per retrieval (the query-counter test must exist and assert it); the formatter must actually consume `expanded_content`.
8. **Scope creep / hygiene**: no `os.getenv` outside config.py; no new tables/tools; no signature changes rippling beyond the story's files; clean tree.
9. **Secrets (S12)**: clone-error path contains no token substring; entity-KG endpoints 401 without auth.

## Verification

- Check the branch's latest CI run (`gh run list --branch ralph/prd-154-wave0-quick-wins --workflow test.yml --limit 1`): a FAILURE that is NEW versus main is a finding; pre-existing reds are noted, not filed.
- Run `bash scripts/ralph/acceptance-prd154.sh`. Non-zero exit = automatic CRITICAL finding.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM findings** → reply exactly `REVIEW_PASS` plus a 5-line summary (LOW/nits + any browser items the human should eyeball, listed there, not filed).
- **Findings** → append fix stories to `scripts/ralph/prd-154.json` `userStories` with ids `P154-RVW-1..n`: title, description with file:line evidence, mechanical acceptanceCriteria, files. Commit `chore(prd-154): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only your fix-story commit to the existing `ralph/prd-154-`* branch (never force, never another ref). Do not re-run the build.
