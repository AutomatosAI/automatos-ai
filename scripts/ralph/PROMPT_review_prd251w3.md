# Ralph Review Prompt — PRD-251 Socials, Wave 3 (schedule and publish)

You are a fresh-context **adversarial reviewer**. The build claims PRD-251 Wave 3 is complete. Find where:
- a post can reach a platform without an approval that matches its content now;
- a post can publish twice;
- an agent can post directly, or something other than the publisher holds the way through;
- a channel's behaviour lives in code instead of the adapter data;
- a scheduled post fires on stale content, never fires, or fires without a leader;
- the UI shows a time, a status or a receipt that is not the post's;
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd251w3.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

The base is the Wave 2 fork point while this branch is stacked on `feat/prd-251-w2-socials-tab`, and the `origin/main` merge-base once Wave 2 is on `main`. Earlier waves' code is the base: never file a finding against a line this diff does not touch, unless this wave's change makes it reachable. Read:
- `scripts/ralph/prd-251w3.json` (binding);
- `docs/PRDS/PRD-251-SOCIALS.md` (D1–D17, the "Waves 2+3 build" block, the Traps);
- `docs/PRDS/prd251-w3-owner-test.md` (what the owner tests by hand; the loop must not claim it).

## Hunt list: every item is a confirmed-risk class

1. **Approval before every way out (D6).**
   - Publish now, retry, a scheduled job, a missed post published on recovery: each calls `assert_publishable` before any executor call, and the claim is a compare-and-set.
   - A target added or changed after approval voids it (Wave 2) and no publish path skips that.
   - Any path around this = CRITICAL.
2. **Publish once.**
   - Two workers, a job fired twice, publish-now racing a scheduled fire, a retry racing a retry: one sequence per target. Proven on the CI Postgres.
   - A `published` target is never run again; the retry runs only failed targets.
   - A double publish reachable = CRITICAL.
3. **One way out (D14).**
   - Only the publisher references `PLATFORM_PUBLISHER`; every agent path (both executor entry points) still refuses every seeded channel's publish actions in a Socials-on workspace; the deny list stays first.
   - No tool, route or parameter lets an agent schedule, approve or publish.
   - A bypass or a weakened gate = CRITICAL.
4. **Channels are data.**
   - No `if toolkit == …` (or equivalent) in the publisher; slugs only in `channel_adapters.py`; `UPLOAD_ACTIONS` not widened; no stale or URL-pull slug called.
   - Each channel's test asserts the documented sequence, not just "some call happened".
   - A violation = HIGH.
5. **Media (D9).** Files go through the resolver's per-call upload spec; URL-only params get presigned inline links from public storage, and are skipped (optional) or fail clearly without it; no `/api/generated-images`. A violation = HIGH.
6. **Retries (D8).** Transient only, bounded by `SOCIALS_MAX_TARGET_ATTEMPTS`, with backoff; a 4xx keeps the platform's message and never retries. Unbounded or blind retries = HIGH.
7. **Scheduling (D10).**
   - Jobs are `social-publish-<post_id>` with picklable args, on the leader only; the reconcile pass registers missing jobs and removes orphans; unschedule, a voiding edit and reject remove the job.
   - A slot missed beyond `SOCIALS_MISFIRE_GRACE_SECONDS` ends `missed` with a notification and publishes nothing; within it, it publishes.
   - Reschedule keeps the approval; PATCH stays the content edit.
   - A fire on stale content or a lost job = CRITICAL; a missing notification = MEDIUM.
8. **The calendar and the controls tell the truth.**
   - Social items only for the caller's workspace, only while Socials is on; times in the post's own timezone; drag and Reschedule call `POST /schedule`.
   - The post view's receipts and statuses come from the targets; Retry offers only failed targets; a 409 reloads.
   - Showing one thing while doing another = CRITICAL.
9. **Background work and recovery.** Publishing runs through `launch_guarded`; a post stuck in `publishing` is failed by the boot reaper with its published receipts kept; nothing publishes from a closed session or blocks the event loop.
10. **Scope and conventions.**
    - New database-touching routes are plain `def` (F105); manifest (with methods and `route_count`) and config-surface updated; `apiClient` uses the right verbs.
    - No migration, or exactly one justified, create_all-first safe revision with both head pins moved.
    - No `os.getenv` outside `config.py`; every commit DCO-signed; no `node_modules`; no new dependency (Python or npm); compact CSS only in the compact region; no Phase 2 code; no skill content or generated-seed edits; pushes only to this branch.
11. **Claims.** Nothing in the diff, the commit messages or the DONE marks may claim a live publish, a browser check or a real platform's behaviour: those are the owner's. A claimed check = CRITICAL.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff. Any CRITICAL or HIGH it reports is a finding.
- Run `bash scripts/ralph/acceptance-prd251w3.sh` yourself. A green build with a red gate is a finding.
- Check CI: `gh run list --branch feat/prd-251-w3-publish --limit 5`.
- Spot-check three `DONE` acceptance criteria at random against the code. Evidence that does not exist = CRITICAL.
- **Nothing runs on the owner's machine:** no server, docker, browser or database. CI is the evidence.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary noting:
  - (a) the approval check on each way out, traced through publish now, a scheduled fire and a retry;
  - (b) the publish-once proof (the claim and its CI concurrency test);
  - (c) the one-way-out proof (who references `PLATFORM_PUBLISHER`);
  - (d) a missed slot, traced;
  - (e) the owner's next step: run `docs/PRDS/prd251-w3-owner-test.md` on the socials stack.

  Final line: `REVIEW_PASS`
- **Findings:**
  1. Append `P251W3-RVW-n` stories to `scripts/ralph/prd-251w3.json`, each with a title, `file:line` evidence and mechanical ACs.
  2. Commit with `git commit -s -m 'chore(prd-251): Wave 3 review findings → fix stories'`, then push to `origin feat/prd-251-w3-publish`.

  Final line: `REVIEW_FINDINGS`
