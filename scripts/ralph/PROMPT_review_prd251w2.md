# Ralph Review Prompt — PRD-251 Socials, Wave 2 (the Socials tab)

You are a fresh-context **adversarial reviewer**. The build claims PRD-251 Wave 2 is complete. Find where:
- a channel can be added to an approved post without voiding the approval;
- the UI shows something other than what an approver is approving;
- an agent can post directly;
- the composer lets the model invent a template, a source or a variable;
- publishing code slipped into this wave;
- a Wave 1 review fix (`P251W1-RVW-2..7`) leaves its failure scenario reachable;
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd251w2.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

The base is the Wave 1 fork point while this branch is stacked on `feat/prd-251-w1-video-engine`, and the `origin/main` merge-base once Wave 1 is on `main`. Wave 0 and Wave 1 code is the base, not this run's work: never file a finding against a line this diff does not touch. **The one exception is the `P251W1-RVW-2..7` stories:** they fix Wave 1 findings. Judge each fix against its own failure scenario on HEAD. If the scenario is still reachable, that is a finding, even where the diff left the line alone. Read:
- `scripts/ralph/prd-251w2.json` (binding);
- `docs/PRDS/PRD-251-SOCIALS.md` (D1–D17 and the "Waves 2+3 build" block);
- `docs/PRDS/prd251-w2-owner-test.md` (what the owner tests by hand; the loop must not claim it).

## Hunt list: every item is a confirmed-risk class

1. **Channels are approved content (D6, US-204).**
   - The content hash covers the target set.
   - Adding, removing or changing a target on an approved or scheduled post voids the approval.
   - `assert_publishable` refuses the old approval.
   - Series approval goes post by post through `service.approve`, hash-bound; a post added or edited later is not covered.
   - Any path around this = CRITICAL.
2. **The UI tells the truth (US-205..US-210).**
   - The approval view shows the exact media (via `FilePreview` and the `/media` route) and the per-channel copy that will publish.
   - Approve sends the shown hash, and a 409 reloads.
   - Unsourced claims need the second confirmation.
   - The board is read-only and matches the list.
   - The composer offers only available channel kinds and blocks over-limit copy.
   - The model's proposal is validated: workspace templates only, candidate sources only, schema-valid variables.
   - Showing one thing and approving another = CRITICAL.
3. **One way out (D14, US-203).**
   - An agent calling a registry-classified publish action in a Socials-on workspace is refused on every agent path, with the deny list still first.
   - The platform publisher's way through is untouched.
   - A new bypass or a weakened deny list/gate = CRITICAL.
4. **Slugs are data (US-203).**
   - Channel slugs appear only in the adapter data.
   - Seeded adapters work with an empty cached `parameters`.
   - The stale and URL-pull slugs are never offered as usable.
   - `UPLOAD_ACTIONS` is not widened.
   - A violation = HIGH.
5. **Media and hosting (D9, US-202).**
   - Presigned URLs are inline, with the right content type and the config TTL, through the one storage factory.
   - The MinIO check actually ran in CI: read the `orchestrator-tests` log for `test_prd251w2_media_hosting` and its HEAD and Range assertions. A test that skipped = HIGH.
   - No `/api/generated-images`.
6. **The composer's model call (US-207).**
   - It goes through `create_llm_manager` (usage tracked, request type `socials_compose`), with a timeout from config.
   - Invalid JSON retries once, then 502.
   - The brand voice comes from the brand kit and the built-in skills, never hardcoded.
7. **The migration (US-201).**
   - ONE revision, chained onto the base's `EXPECTED_HEAD`, with both head pins moved.
   - It is create_all-first safe, and a test runs `create_all` first, then the upgrade twice.
   - A second head or a create_all-unsafe step = HIGH.
8. **Nothing publishes yet.**
   - `publisher.py` is untouched, and publish-now still answers 501.
   - No `social-publish-` jobs, no `DateTrigger`, no channel publishers, no calendar source.
   - Publishing code in this wave = HIGH (it belongs to Wave 3's run and test).
9. **Scope and conventions.**
   - New database-touching routes are plain `def` (F105).
   - Route manifest (with methods) and config-surface updated; `apiClient` uses the right verbs.
   - No `os.getenv` outside `config.py`.
   - Every commit is DCO-signed.
   - No `node_modules`, no new npm dependency.
   - Compact CSS only in the compact region.
   - No Phase 2 engagement code.
   - No skill content or generated-seed edits.
   - Pushes only to this branch.
10. **Claims.** Nothing in the diff, the commit messages or the DONE marks may claim a browser check or a real render time: those are the owner's. A claimed check = CRITICAL (evidence that does not exist).
11. **Wave 1's review fixes (`P251W1-RVW-2..7`).** Replay each story's failure scenario, as written in its description, against HEAD:
    - **RVW-2:** a voice render past either cap speaks nothing and books nothing. Voice is priced and checked in the per-workspace window it shares with footage, and its booking lands before `speak()` returns.
    - **RVW-3:** concurrent renders cannot pass the quota together, on either path (`render_post` and `generate_social`). A reservation is released on every ending, and render seconds are booked before `run_render` returns.
    - **RVW-4:** one workspace cannot hold every admission slot, and `workspace_busy` is handled by the client.
    - **RVW-5:** a process killed after a submit leaves the spend booked, and exactly one net amount stays per job.
    - **RVW-6:** no Automatos copy reaches an agent through `platform_get_template_schema`, and the brand rule flags `rgb()`/`hsl()`/named colours.
    - **RVW-7:** both settings are in config-surface, and a starter's `css` is carried or refused.

    Severity when a scenario is still reachable: the original finding's (RVW-2 CRITICAL; RVW-3 and RVW-4 HIGH; RVW-5 and RVW-6 MEDIUM; RVW-7 LOW). A money fix without its concurrency test on the CI Postgres = HIGH.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff. Any CRITICAL or HIGH it reports is a finding.
- Run `bash scripts/ralph/acceptance-prd251w2.sh` yourself. A green build with a red gate is a finding.
- Check CI: `gh run list --branch feat/prd-251-w2-socials-tab --limit 5`, and read the `orchestrator-tests` log for the MinIO test.
- Spot-check three `DONE` acceptance criteria at random against the code. Evidence that does not exist = CRITICAL.
- **Nothing runs on this machine:** no server, docker, browser or database. CI is the evidence.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary noting:
  - (a) the hash covering targets, traced through a target edit;
  - (b) the approval view's media and copy sources;
  - (c) the MinIO HEAD/Range evidence from CI;
  - (d) the post gate's publish-action refusal;
  - (e) each Wave 1 fix's failure scenario, traced on HEAD (one line each);
  - (f) the owner's next step: run `docs/PRDS/prd251-w2-owner-test.md` on the socials stack.

  Final line: `REVIEW_PASS`
- **Findings:**
  1. Append `P251W2-RVW-n` stories to `scripts/ralph/prd-251w2.json`, each with a title, `file:line` evidence and mechanical ACs.
  2. Commit with `git commit -s -m 'chore(prd-251): Wave 2 review findings → fix stories'`, then push to `origin feat/prd-251-w2-socials-tab`.

  Final line: `REVIEW_FINDINGS`
