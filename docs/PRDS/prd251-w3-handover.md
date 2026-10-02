# PRD-251 Wave 3: handover (2026-09-30)

Wave 3 finishes PRD-251: it makes approved posts actually publish, on time, to LinkedIn, X, Instagram, TikTok and YouTube. This note is for the session that builds it.

## Where things stand

- **Waves 0 and 1** are on `main`.
- **Wave 2 is complete** on `feat/prd-251-w2-socials-tab` (tip `fbfe9ec`). Its 16 stories are DONE, and CI is green on every required job (run 36651236634). The owner tests it by hand in his own test cycle; it is not on `main` yet.
- **Wave 3's branch is `feat/prd-251-w3-publish`**, cut from that tip. Nothing is built on it yet. It holds the kit:
  - `scripts/ralph/prd-251w3.json`: the binding contract, nine stories US-301..US-309 with checked `file:line` anchors;
  - `scripts/ralph/PROMPT_build_prd251w3.md`: how to build (rules, the per-story protocol);
  - `scripts/ralph/PROMPT_review_prd251w3.md`: the adversarial review at the end;
  - `scripts/ralph/acceptance-prd251w3.sh`: the final gate (`--print-base` prints the diff base, `fbfe9ec`);
  - `docs/PRDS/prd251-w3-owner-test.md`: the owner's live checklist (not the build's to run or claim).

## The stories, in order

1. **US-301, the publisher, proven with LinkedIn.** One engine that runs a target's steps from the adapter data through `ComposioToolExecutor.execute(..., way_through=PLATFORM_PUBLISHER)`. It covers:
   - the approval check first, and a compare-and-set claim so a post publishes once;
   - file-first uploads through a per-call upload spec, never by widening `UPLOAD_ACTIONS`;
   - transient-only retries, and receipts;
   - `published` / `partially_published` / `failed`, the `publish-now` and `retry` routes, and notifications.
2. **US-302 X, US-303 Instagram, US-304 TikTok, US-305 YouTube and the generic adapter.** Each is data plus tests on the same engine; never a branch on the channel's name.
3. **US-306, scheduling.** One-shot `social-publish-<post_id>` jobs, a reconcile pass, missed slots.
4. **US-307, the calendar.** The `social` kind in the Command Center calendar, and drag to reschedule.
5. **US-308, the publish controls and receipts** in the post view.
6. **US-309, one way out, end to end** (agents still can't post), and the docs.

## Decisions already made (owner, 2026-09-30)

- **Build Wave 3 now and complete PRD-251.** Development first; the owner tests by hand in his planned test cycle.
- **Instagram images:** Instagram takes JPEG and our stills render as PNG. Make the JPEG in media-render (it has ffmpeg), or refuse the Instagram image target clearly. Add no image library to the orchestrator (US-303).
- **Tests:** each story writes its tests and needs its CI run green before its ACs are marked DONE. In a cloud session you may install the dependencies and run the tests locally before pushing; CI stays the only evidence.
- **Live publishing, real platforms and browser checks are the owner's.** Never claim them.

## Open items the owner has not decided (don't build them unasked)

- Picking a Deliverable for an image or footage slot (the backend can only generate slot footage from a prompt).
- Previews spend no media money (template graphics and Kokoro).
- Listing LinkedIn organisations as the post's author.
- Landing Wave 2 on `main` needs re-chaining `prd251_wave2` onto main's new head (`document_chunks_ingestion_columns`) and resolving the two head-pin tests. That is the owner's integration step, not this wave's.

## Things that will bite

- **Big files.** `orchestrator/api/socials.py` (970 lines), `modules/socials/service.py` (992), `services/activity_service.py` (1,327) and `frontend/components/command-center/calendar-tab.tsx` (1,019) are over 800. New code goes in new modules, and these files get only the lines a route, a transition or a hook needs.
- **`scripts/ralph` is gitignored.** `git add` the kit JSON by its path (it is tracked); a new file there needs `git add -f`.
- **The route manifest** carries a `route_count` that tests check against `len(routes)`: bump it with every route you add.
- **`test_prd251_api.py`** pins the full socials route list: add new routes to it.
- **A push cancels the branch's running CI.** Wait for a story's run before pushing the next story.
- **The adapter data does not yet say** which result field a `$steps.<id>` reads, or where a publish step's remote id and permalink are. US-301 adds that to the data (e.g. a step's `returns`).
- **Local test setup in a cloud container:**
  - the tests need the Postgres env vars set (port 1 fails fast; SQLite-backed tests still run);
  - tiktoken can't download its files there: stub `tiktoken.get_encoding` in a `sitecustomize.py` outside the repo;
  - never commit either.

## How the last session ran (it worked)

- **The loop, per story:**
  1. read the story and grep its anchors;
  2. build it with its tests;
  3. run the tests and the changed-line checks locally;
  4. commit with `-s` and push;
  5. wait for CI green;
  6. commit the DONE marks with the run id.
- **The changed-line checks CI runs:**
  - ruff through `diff-quality --violations=ruff.check --compare-branch=origin/main --fail-under=100`;
  - `python scripts/ci/check_changed_code_shape.py --base origin/main`;
  - in `frontend/`: `node scripts/tsc-baseline-check.js`, `node scripts/check-changed-lines-eslint.js origin/main` and `node scripts/check-route-contract.js`;
  - `lint-imports --config orchestrator/.importlinter`.
- **Review before landing.** Stories that touch the post gate, the executor, approval or scheduling (US-301, US-306 and US-309 always) get a code review before their commit.
