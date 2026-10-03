# PRD-251C: Socials plans that run themselves. A week made at once and approved in one sitting, research from the first plan, no repeated posts, stories, and results that teach the plan

| | |
|---|---|
| Owner | Gerard Kavanagh |
| Written | 2026-10-03 |
| Status | Draft for owner review |
| Builds on | PRD-251B Waves 1–3 (on `main`), and `feat/socials-flex` (3 Oct: post Delete, Approve & post now, a size per channel, plan Delete) |
| Prerequisite | `feat/socials-flex` merged. History (C5) relies on its rule that deleting a plan keeps the plan's posts |
| Target | Automatos runs its own Web Summit countdown (Lisbon, 9 Nov 2026) on a weekly plan |

## Framing

PRD-251B made plans that make each post on its own day, so token use is spread out (owner, 2026-10-02). The owner's test on 3 Oct showed the cost of that design:

- **Daily approvals.** A plan with a 06:00 slot needs a person every day.
- **Research fails.** Research fails with a 409 in every workspace that never installed the Socials package.
- **Repeats.** Nothing stops a plan from posting "What is a Mission?" again.
- **No stories.** Stories cannot be planned.
- **No learning.** Nothing learns from how posts did.

This PRD changes when posts are made and approved, what research knows, and what a plan learns. **It keeps every PRD-251 and 251B rule**:
- agents draft and never publish;
- every post is approved against the exact content that goes out (D6);
- publishing goes only through the workspace's own Composio connections;
- paid tools are reached through Composio only (D15);
- Socials can be switched off.

It reverses two 251B entries, by the owner's decision of 3 Oct:
- **"No month of content is generated ahead"** (251B Goal 6, B7, "Not in this PRD"). A plan may now make its posts a week or a month ahead, as the owner chooses. Daily stays available.
- **"Metrics"** (251B "Not in this PRD"). Each post's results are now read back (Wave 4). Comments and replies stay out.

## What the owner asked for (2026-10-03)

- "When does a plan start, when is the content designed, when can I approve it... should we plan it weekly, monthly and render live each day or weekly or monthly, as I can't be around to approve every morning at 6am."
- "If we choose weekly then we build all on a Sunday (user's choice) and then we agree and approve?"
- "I'd also like to mix it with, for example, 7 images, one every day, one video a week with Higgsfield or another partner, stories and reels."
- "We need to make it world class and use Automatos to learn your business and suggest and guide you."
- On research: "Creating a playbook for the research is great, and yes I agree it's automatically created on first plan creation; the user can adapt, edit, change agents and so on."
- On repeats: "We need to save past posts and they must be included in the research, as we don't want to be posting the same posts over and over, like What is a Mission and so on."

## What exists today (verified on main ffae58a4a and `feat/socials-flex`, 2026-10-03)

Paths are under `orchestrator/` or `frontend/`.

- **When posts are made** (`modules/socials/plans.py:36-40, 372-379`). Each slot's post is made at the plan's make time (default 07:00, in the plan's timezone) on the slot's day. Videos are made 1 day early and images 0 days early. Both are capped at 3 days early (`MAX_VIDEO_DAYS_EARLY`, enforced for both by `_days_early`, :194-198). A post is made at least 2 hours before its slot (`MIN_MAKE_LEAD`), else a day earlier, so a 06:00 slot is made at 07:00 the day before. The leader-only tick (`services/socials_plan_maker.py`):
  - makes the due slots, at most `SOCIALS_PLAN_MAX_SLOTS_PER_TICK` per tick (`collect_due`, :135);
  - submits each post for approval;
  - sends "Today's posts are ready" once per plan per day (`modules/socials/plan_notify.py:25`).
- **Approval.**
  - Posts are approved one at a time in the Queue.
  - **Series approval** exists (`modules/socials/campaigns.py:340`). It approves each shown post by its content hash and leaves (and lists) any post that changed. It needs the workspace's `series_approval` switch (default off) and the campaign in series mode (`assert_series_allowed`, :196-201).
  - A slot that passes unapproved follows the plan's `late_policy`, `skip` or `next_slot` (`modules/socials/plan_late.py`).
- **Research** (`services/socials_plan_research.py`).
  - It runs the workspace's own copy of the marketplace playbook **Content bank research** (`core/seeds/seed_socials_package.py:376-391`, prompt :257-271), found by `cloned_from_id` (:54-66). Without a copy it answers 409 `NEEDS_PACKAGE` (:44-47).
  - A copy appears only when the whole Socials package (2 agents, 5 playbooks) is installed. Workspace c1 never installed it, so every **Research again** failed (the owner's screenshot, 3 Oct).
  - The weekly run covers active plans. A trial workspace on the hosted edition gets none (PRD-222).
- **Installing one playbook** (`services/package_installer.py:304-345`). `_install_playbook` clones one marketplace playbook with the agents it needs. It is idempotent, and a re-install leaves the workspace's edited copy as it is.
- **Repeats.** A topic is refused only when its title matches one in **the same plan's** bank, lowercased with spaces collapsed (`modules/socials/topics.py:106-114`). That leaves three gaps:
  - Nothing compares a topic with other plans or with posts that went out. "What is a Mission?" and "What is a mission" are different titles.
  - The research prompt reads only its own plan's bank (step 1).
  - Research lists Deliverables, and every post's rendered files are image Deliverables, so an old post can come back as new material.

  `next_topic` takes the oldest unused topic of the plan (:190-201).
- **History.**
  - A post that went out keeps its receipts: `remote_id`, `permalink` and `published_at` on each target (`core/models/socials.py:422-425`).
  - Deleting such a post is refused (`api/socials_delete.py`, on `feat/socials-flex`).
  - Deleting a plan keeps its posts, unlinked (`plan_store.delete_plan`, on `feat/socials-flex`). Its topics go with it (FK CASCADE, `core/models/socials.py:358`).
- **Formats.**
  - A cadence row is `image`, `carousel`, `video` or `text` (`FORMAT_KINDS`, `services/socials_plan_maker.py:62-68`). A video becomes the channel's video, reel or short.
  - `story` is a valid target kind (`core/models/socials.py:73`) with a 9:16 size (`modules/socials/channel_sizes.py`), but no channel adapter can post one. Instagram's adapter has image, reel and carousel only (`modules/socials/channel_adapters.py:202-343`).
- **Visuals** (`modules/socials/plan_visuals.py`). A plan has one mix of `templates`, `library`, `ai_images` and `ai_footage`, drawn slot by slot. A row cannot say "this weekly video always uses Higgsfield footage". AI media come from fal.ai, Kie.ai or Higgsfield MCP through Composio, priced and capped per post and per month (`modules/socials/media_caps.py`).
- **Results.** No code reads a post's reach, likes, comments or clicks back.
- **Voice.** The composer gets the brand kit's `voice` text. A person's edits before approving are not kept.
- **Notifications.** Plan and approval notices go through `NotificationDispatcher`: in-app, Telegram, Slack or webhook, per the user's preferences (`modules/socials/notify.py`).
- **Uploads.** An uploaded picture or video is the whole post, and it publishes as uploaded: the adapters send the stored file. No crop exists for each channel, although the Upload tab says "It is cropped for each channel" (`studio/editor-look-sources.tsx`, `DROP_HINT`).

## Goals

1. **A rhythm per plan.** Each plan has a rhythm the owner picks: daily (as today), weekly or monthly. On a weekly plan:
   - the coming week's posts are made at the owner's day and time;
   - they are reviewed on one screen and approved in one sitting;
   - each post is still approved as the exact version shown.
2. **Research from the first plan.** Research works from the first plan in every workspace, with a playbook the owner can open, edit and give to another agent.
3. **No accidental repeats.** Every post the workspace made is history, which research, the content bank and the composer check against.
4. **A mix per row.** A plan mixes formats row by row: daily images, a weekly video with AI footage from a chosen partner, reels, stories and carousels.
5. **Results come back.** Each post's numbers are read after it goes out, and a weekly note says what worked. Auto proposes changes, which the owner applies or ignores.
6. **Auto learns and guides.** Auto learns the owner's voice from their edits, and guides the plan: a low bank, a channel not connected, an event coming.
7. **A person approves everything.** Nothing publishes without a person approving the exact content (PRD-251 D6). There is no auto-approval and no approval by silence.

## Decisions (proposed 2026-10-03; each reversible before its wave starts)

### C1 · A plan's rhythm: daily, weekly or monthly
- **New fields.** `make` gains:
  - `rhythm` (`daily|weekly|monthly`, default `weekly` for new plans; existing plans keep `daily`, O1);
  - `batch_day` (weekly: `mon`–`sun`, default `sun`, O3);
  - `batch_date` (monthly: 1–28, default 25).

  `make.time` stays as it is, and its default for a weekly plan is 17:00.
- **Weekly.** At `batch_day` and `make.time`, every slot of the 7 days from the next midnight (in the plan's timezone) is made.
- **Monthly.** At `batch_date` and `make.time`, every slot of the next calendar month is made (O4).
- **Daily** keeps today's timing (`make_at`).
- **Batch identity.** A post made in a batch carries the batch's key, e.g. `2026-W42` or `2026-11`. A slot that appears after its batch was made, through a new row or a moved slot, is made at the next tick, as a daily slot is.
- **Spend.** Spend is booked at batch time, under the same render quota and media caps. Stills cost no render minutes; a video's minutes and its AI media are reserved when it is made. A slot over a cap is skipped and notified, never half-made (as 251B B7).
- **Large batches.** A large batch is made over several ticks (the per-tick cap stays). **"Your week is ready"** goes out once every slot of the batch is made or skipped.

### C2 · The week's review, approved in one sitting
- **One screen.** The Queue groups a batch as one section, e.g. "Week of 12 Oct · Automatos launch · 8 posts", in slot order. Each post shows its channel previews, copy and claims, with today's actions: Edit, Make another take, Request changes, Reject and Delete.
- **Approve the week.** One action approves every shown post by its content hash, through the series-approval core (`campaigns.approve_series`). A post that changed since it was shown is left and listed, so it is approved separately after a second look.
- **The workspace switch.** Approving a batch does not need the workspace's `series_approval` switch (O2). The switch stays for campaigns that are not plans.
- **After approval.** An approved post can still be edited, which voids its approval and puts it back in the Queue, as today. It can also be moved (that keeps the approval) or deleted.

### C3 · Reminders, and approving from anywhere
- **The evening before.** At 20:00 (plan timezone, configurable) the evening before, one reminder lists that day's unapproved posts. After its slot passes, a post follows `late_policy` as today.
- **Links.** Every batch notice and reminder links to the week's review. Through Telegram, Slack or in-app it opens on a phone. Approving inside a Telegram or Slack message is O8.

### C4 · Research comes with the first plan
- **Installed with the first plan.** When a workspace saves its first plan (from the form or from Plan with Auto), the server installs the **Content bank research** playbook and the agent it runs, the Social Media Director, through the package installer.
  - It installs that playbook only, not the rest of the Socials package.
  - It is idempotent, and a failure never fails the plan's save: it is logged, and the bank says research is not set up.
- **The workspace's own playbook.** The owner edits its prompt and steps, or gives it to another agent, in Playbooks. Research always runs the workspace's copy.
- **If it was deleted.** **Research again** reinstalls the playbook from the marketplace and runs it: no more 409 for a missing package. The weekly run never reinstalls on its own. It tells the owner instead: "Research is off: its playbook was removed. Research again restores it."
- **Timing.** On a weekly plan, research runs the day before the batch by default (`research.day` = `batch_day` − 1), so each batch is made from fresh topics.

### C5 · History: every post is remembered, and none repeats by accident
- **What history holds.** Every post of the workspace that went out, is approved or scheduled, or waits for approval. It spans every plan and the posts people made by hand, and it survives plan deletion, because posts stay.
- **How research reads it.**
  - A new read tool, `platform_get_social_history(days, limit)`, gives each post's title, topic and angle, format, channels, first line of copy, date and state, newest first. The window and limit are config.
  - The same history also comes in `platform_get_social_plan`'s answer, so the workspace copies made before this PRD see it without a prompt change.
- **The guard.** `add_topics` refuses a topic too close to:
  - any topic in any of the workspace's banks; or
  - a post in history within the plan's repeat window (`research.repeat_after_days`, default 60; O5).

  The reason names the earlier topic or post and its date. "Too close" means the same title once case, punctuation and contractions are folded ("What's" and "What is"), or word sets that overlap at or above a config threshold. The check is deterministic and cheap: no model or embedding call per topic.
- **A deliberate repeat.** A person adding a topic in the bank gets a warning instead of a refusal ("Posted 5 Oct as 'What is a Mission?'"). They may repeat on purpose.
- **The composer.** When making a post, it gets the opening lines of the last posts (count in config), so a new post does not reuse a hook.
- **Deliverables.** Research's Deliverables source leaves out Socials' own post files. Those are history, not new material.

### C6 · Formats and visuals per row
- **Stories.** A cadence row may be a `story` (image or video, 9:16). It is published through Instagram's story publishing in Composio, if its toolkit takes `media_type` STORIES. That is verified before the wave (US-C301 spike). Reels stay as today: a video row on Instagram.
- **A row's visual.** A row may set its own visual, which overrides the plan's mix for that row: `templates`, `library`, `ai_images`, or `ai_footage` with a named toolkit (e.g. Higgsfield MCP). AI media are priced, capped and booked as today (D13). The Plan page shows each row's estimated monthly spend.

### C7 · Results come back
- **Reading.** At 1 day and at 7 days after a target goes out, its numbers are read through that channel's read actions in Composio, only when the toolkit is connected, allowlisted and not deny-listed (the capability-registry rule). Each read is stored per target. A number the platform does not give stays empty.
- **The weekly note.** It comes with the weekly batch, or on Mondays for daily plans. It shows:
  - the week's posts with their numbers;
  - the best and the worst;
  - what the best share: format, time, topic.
- **Proposals.** Auto proposes plan changes, such as a time, a format's share or an angle. Each is one click to apply and never applied by itself.
- **Research weighs results.** Topics like the best performers go first in the bank.

### C8 · Auto learns the owner's voice
- **Examples.** When a person approves a post whose copy differs from what Auto drafted, the pair (draft, approved) is kept as a voice example. The workspace keeps the newest N (config).
- **Use.** The composer gets the newest examples together with the brand kit's voice text.
- **Control.** Brand kit → Voice lists the examples, and the owner removes any of them.

### C9 · Auto guides the plan
The Plan page and the weekly note carry the plan's health, each item with one click to act:
- the bank has fewer unused topics than the next two batches need;
- a cadence channel is not connected;
- no video this week;
- a cap is close.

Research may also add **dated topics**, pinned to a day (`pinned_on` exists), for events it finds in the plan's goal, notes or knowledge, such as "Web Summit in 5 weeks: a countdown".

## Stories

Every wave: **one** Alembic revision where a wave needs one, chained onto `EXPECTED_HEAD`, safe when `create_all` ran first, with both head pins moved (`tests/test_prd209_alembic_single_head.py`, `tests/test_prd236_w1_routes.py`). Routes that touch the database are plain `def` (F105). The route manifest and config surface are updated. Commits carry DCO sign-off. CI on Postgres is the evidence; nothing runs on the build machine.

### Wave 1 — research from the first plan, and history (no repeats)

**US-C101 · Research installs with the first plan (M)**
- **Install.** Saving a workspace's first plan installs the research playbook and its agent (C4) through `package_installer._install_playbook`, made public for this.
  - It runs after the plan's commit, through `launch_guarded`: the plan route is a plain `def`, and the installer is async.
  - A failure is logged and leaves the plan saved.
- **Tests.**
  - The first plan installs once, and a second plan installs nothing.
  - An edited copy is left as it is.
  - A failing installer does not fail the save.
  - The installed copy is what `installed_playbook` finds.

**US-C102 · Research again restores a missing playbook (S)**
- `POST /plans/{id}/research` installs the playbook when it is missing, then runs it. The 409 stays only for a marketplace playbook that is itself missing (its seed did not run).
- The weekly run notifies instead of reinstalling (C4).
- **Tests:** both paths; a hosted trial still gets no weekly run.

**US-C103 · History (M)**
- A `history(db, workspace_id, days, limit)` read in `modules/socials`, behind:
  - the agent tool `platform_get_social_history`, registered by the 3-file pattern;
  - `GET /api/socials/history`.
- `platform_get_social_plan` adds a `history` field to its answer.
- **Tests:**
  - what history holds (C5): states, plans, hand-made posts, after a plan's deletion;
  - newest first, bounded;
  - another workspace sees nothing.

**US-C104 · No repeats (M)**
- The C5 guard in `topics.add_topics`, research only: across the workspace's banks and history within `research.repeat_after_days`.
- Its refusal names the earlier topic or post and its date.
- A person's topic gets the warning (C5).
- **Tests:**
  - "What is a Mission?" against "what's a mission";
  - another plan's topic;
  - a post 30 days and 90 days old, with a 60-day window;
  - the threshold in config;
  - a person's override.

**US-C105 · Research and the composer read history (S)**
- **Research prompt.** The seed's research prompt reads history first and picks only what neither history nor the bank covers. The seed changes the marketplace row; workspace copies get history through `platform_get_social_plan` (C5).
- **Composer.** It gets the last posts' opening lines.
- **Deliverables.** Research's Deliverables source leaves out Socials' post files.
- **Tests:** what the tools return; what the composer receives.

**US-C106 · The bank shows near-repeats (S)**
- A topic card says "Posted 5 Oct as 'What is a Mission?'" when history holds a close post.
- **Tests:** the card and its link to the post.

### Wave 2 — the week made at once, approved in one sitting

**US-C201 · The wave's migration (S)**
- **Migration.** `prd251c_wave2` adds `social_posts.batch_key` (string, nullable, indexed with `campaign_id`).
- **Validation.** `make.rhythm`, `make.batch_day` and `make.batch_date` are validated in `plans.validate_make`. `make` is JSON, so this needs no column.
- **Tests:** the usual migration tests; validation refusals.

**US-C202 · Rhythm on the Plan page (M)**
- Step 4, Making and approving, gains **Rhythm** (Daily · Weekly · Monthly) with its day and time, and says when the next batch is made.
- Plan with Auto drafts a weekly plan.
- **Tests:** the controls; what is saved; the next-batch line.

**US-C203 · Making a batch (L)**
- The tick makes every slot of a due batch (C1), keyed by `batch_key`, idempotent per `slot_key`, within the per-tick cap.
- It sends **"Your week is ready"** once the batch is made or skipped.
- Daily plans are unchanged.
- **Tests on Postgres:**
  - a week made across ticks;
  - two ticks racing;
  - a slot over a cap skipped and notified;
  - a slot added after its batch made at the next tick;
  - DST weeks.

**US-C204 · The week's review in the Queue (L)**
- The Queue groups each batch (C2), showing its posts' previews and actions, and **Approve the week**.
- **Tests:**
  - grouping;
  - ordering;
  - a changed post left out and listed;
  - an editor's view (`REVIEW_ROLES`).

**US-C205 · Approve the week (M)**
- `POST /api/socials/plans/{id}/batches/{batch_key}/approve` with the shown posts and their hashes. It goes through the series-approval core without the workspace switch (O2).
- **Tests:**
  - all approved;
  - a stale hash left;
  - another plan's post refused;
  - another workspace's 404.

**US-C206 · Reminders (S)**
- The evening-before reminder (C3), linked to the week's review.
- **Tests:** one per plan per day; none when everything is approved.

**US-C207 · Research the day before the batch (S)**
- A weekly plan's research defaults to the day before `batch_day`. Its last run shows on the bank.
- **Tests:** the default; the owner's change kept.

### Wave 3 — formats and visuals per row

**US-C301 · Stories (L)**
- **Spike first.** Confirm that Composio's Instagram toolkit takes `media_type` STORIES. If it does not, this story stops and the owner is told.
- **Then:**
  - an Instagram `story` recipe in `channel_adapters.py`;
  - `story` cadence rows;
  - the editor's Story kind;
  - 9:16 renders with the story safe zone.
- **Tests:** the recipe's steps; a story row made and published (mocked Composio).

**US-C302 · A row's visual (M)**
- `cadence[].visual` (C6) overrides the plan's mix for its row, with an optional `toolkit` for AI media. The Plan page row shows it and the row's estimated monthly spend.
- **Tests:**
  - the override;
  - a toolkit not connected refused;
  - caps unchanged.

**US-C303 · Uploads cropped for each channel (M; only if O6 says crop)**
- A still upload is rendered once per channel shape through media-render (crop to fill, centred), as template sizes are. The preview shows those exact files.
- **Tests:** files per shape; publishing gives each channel its own (`publish_sources.media_for`).

### Wave 4 — results, voice and guidance

**US-C401 · The wave's migration (S)**
- `prd251c_wave4` creates:
  - `social_post_stats`: target, read at, numbers as JSON, source action;
  - `social_voice_examples`: workspace, post, draft, approved, created at.
- **Tests:** the usual migration tests.

**US-C402 · Reading results (L)**
- A `read` recipe per channel in `channel_adapters.py`, offered through the capability registry.
- A leader-only job reads each target at 1 and 7 days.
- **Tests:**
  - mocked Composio answers;
  - a toolkit not connected skipped;
  - numbers the platform does not give left empty;
  - no read for a target that never went out.

**US-C403 · The weekly note (M)**
- What C7 lists, through `NotificationDispatcher`, linked to the Socials history view.
- **Tests:** its contents from fixtures; one per plan per week.

**US-C404 · Auto's proposals (M)**
- Proposals from results (C7), each a plan change applied by one click through the plan's `PUT`.
- **Tests:** a proposal applied; a proposal ignored changes nothing.

**US-C405 · Research weighs results (S)**
- The history tool carries each post's numbers.
- The prompt and `next_topic` prefer topics like the best performers.
- **Tests:** the bank's order with results.

**US-C406 · Voice from edits (M)**
- C8: examples kept on approval, given to the composer, and listed and removable in Brand kit → Voice.
- **Tests:**
  - kept only when the copy differs;
  - the newest N;
  - removed ones are not used.

**US-C407 · Plan health (M)**
- C9's checks on the Plan page and in the weekly note, plus dated topics from research.
- **Tests:** each check from fixtures.

**US-C408 · The History view (M)**
- Socials gains **Posted**: posts that went out, newest first, each with its receipts, numbers and topic, and a filter by plan, channel and format.
- **Tests:** the list, the numbers and the filters.

## Not in this PRD

- **No approval without a person.** Approving by silence, auto-approval, or any approval without a person (D6).
- **No engagement.** Comments, replies and other engagement.
- **No ads.** Paid ads and boosting.
- **No own partner clients.** Our own API clients or keys for partners: Composio only (D15).

## Editions

Both editions get every story. The hosted trial keeps PRD-222's rule: no weekly research and no reading of results in the background. **Research again** and the owner's own batches still work.

## Delivery plan (to Web Summit, 9 Nov)

1. `feat/socials-flex` merges; this PRD is reviewed.
2. Wave 1 (research and history), then the owner's test.
3. Wave 2 (weekly batches), then the owner's test. Automatos' Web Summit countdown then runs as a weekly plan.
4. Waves 3 and 4: order and timing are the owner's call (O9).

Each wave's owner test (`docs/PRDS/prd251c-wN-owner-test.md`) covers the browser checks.

## Open questions (owner)

1. **Default rhythm for new plans.** Weekly (recommended) or daily? Existing plans keep daily either way.
2. **Approve the week without the series-approval switch?** Recommended yes: each post is still approved by its hash. The switch stays for campaigns that are not plans.
3. **Batch default.** Sunday 17:00?
4. **Monthly.** Keep it as an option (allowed, not the default), or drop it? A month ahead goes stale, the review is long, and the spend lands at once.
5. **Repeats.** Is a 60-day repeat window right? Should a person's topic only warn, or be refused too?
6. **Uploads (from 3 Oct).** Real crops for each channel (US-C303, recommended), or correct the Upload tab's wording?
7. **Results.** Which numbers matter most: reach, engagement rate or clicks? Where do they show: Socials' Posted view only, or also Analytics?
8. **Approving from Telegram or Slack.** Approve buttons inside the message, later? (Today the notices link to the review.)
9. **Order after Wave 2.** Stories and row visuals (Wave 3) first, or results and learning (Wave 4)?

## Traps (carried from PRD-251 and 251B, and new)

- **Workspace copies keep their prompt.** Workspace copies of the research playbook never receive a seed prompt change, because the installer leaves edited copies alone. Put what research must know (history, results) in the tools' answers, not only in the prompt.
- **No per-topic model calls.** The repeat guard is deterministic and cheap; no embedding or model call per topic (the 15 Sep embedding storm).
- **Async in a plain `def` route.** The installer is async, so run it through `launch_guarded` or `anyio.from_thread`, never `asyncio.run` in a threadpool thread.
- **Batch timestamps.** A batch creates many posts in one transaction, and `now()` is the transaction's start (F209): set `created_at` from the clock.
- **Migrations.** Two heads break nine tests at once. Chain onto `EXPECTED_HEAD` and move both pins in the same commit.
- **Shape ratchet.** A touched function stays ≤ 50 code lines with nesting ≤ 4. `api/socials.py` is over 800 lines: add new routes in their own modules, as `socials_delete.py` and `socials_plans.py` do.
- **Config.** No `os.getenv` outside `config.py`. New numbers (windows, thresholds, counts, times) are config.
- **Merging.** Never merge mid-run. Trial-merge sibling branches before calling a wave done.
