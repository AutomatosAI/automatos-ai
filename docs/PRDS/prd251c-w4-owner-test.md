# PRD-251C Wave 4: the owner's test (results, voice and guidance)

Wave 4 was written directly on `feat/prd-251c-w4-results-voice`, stacked on Wave 3. Run this on the local stack once the branch is merged into local main, the backend is restarted (the `prd251c_wave4` migration runs at boot) and the frontend is rebuilt. Record what failed in the PR.

The default this wave took while O7 is open: a post's **engagement** is the sum of its likes, comments, shares, saves, replies, reposts, quotes and reactions. Every number a platform gives is kept and shown. Results show in Socials (Posted, the weekly note and the Plan page), not in Analytics.

What each channel gives (Composio's read actions, checked 2026-10-04):

| Channel | What it gives |
|---|---|
| X | views, likes, reposts, replies, quotes |
| Instagram | views, reach, likes, comments, shares, saves (a story: views, reach, replies, shares) |
| LinkedIn | reactions only; a member's post has no read for reach |
| YouTube | views, likes, comments |
| TikTok | nothing: no statistics action, and the publish returns no video id |

## 0. Set up

1. **Restart the backend** and check the boot log for `prd251c_wave4`, the results job ("posts' numbers are read every 3600s") and the plan tick.
2. **To see numbers without waiting a day:** publish a post. Then either wait an hour past the 1-day mark, or put a reading in by hand (psql on the local database, with that target's ids):
   ```sql
   INSERT INTO social_post_stats (id, workspace_id, post_id, target_id, reading, read_at, numbers, source_action)
   VALUES (gen_random_uuid(), '<workspace id>', '<post id>', '<target id>', 1, now(), '{"likes": 12, "views": 300}', 'MANUAL');
   ```

## 1. Reading results (US-C401, US-C402)

- [ ] **A post published on X more than a day ago** gets its 1-day numbers at the next hourly read. It gets them once, and its 7-day numbers come after a week.
- [ ] **Disconnect a channel** before its read: nothing is kept, and the read comes once the channel is back (within two days).
- [ ] **A TikTok post** never gets numbers, and nothing fails.

## 2. Posted (US-C408)

- [ ] **Socials → Posted** lists what went out, newest first. Each post shows its plan, its topic, when it went out, a link to each channel's post, and its numbers ("300 views · 12 likes (after a day)"). A post not read yet says when it will be.
- [ ] **The filters:** plan, channel and format each narrow the list. The plan filter is in the address (`&plan=`).

## 3. The weekly note (US-C403)

- [ ] **A weekly plan:** when its batch is made, a notice "Your week on socials: …" arrives, alongside "Your week is ready". It lists the week's posts with their numbers, the best and the worst, what the best share, the plan's health and Auto's proposals.
- [ ] **A daily plan:** the note comes on Monday at the plan's make time.
- [ ] **Once a week:** a second tick sends nothing.
- [ ] **The bell:** the note opens the plan's Posted, "Your week is ready" opens the Queue, and a plan's notice opens the plan.

## 4. Auto's proposals (US-C404)

- [ ] **With at least three read posts at two times (or in two formats),** where one does 1.5 times better, the Plan page shows "Auto proposes" with the change and why. Press Apply: the plan changes (the cadence step shows it).
- [ ] **Leave one unapplied:** the plan does not change.
- [ ] **The best post's topic** is proposed as a research note ("More like …"). Applied, it shows in What to research → notes.

## 5. Research weighs results (US-C405)

- [ ] **Research's run** (Playbooks → the run's transcript): `platform_get_social_plan`'s history has each post's numbers and engagement.
- [ ] **The bank's order:** with a best performer read, the next post is made from the bank's topic most like it, not simply the oldest.
- [ ] **Dated topics:** put "Web Summit, 9 Nov" in the plan's goal and run research. The bank gets topics pinned to days before 9 Nov (a countdown), each within the plan's dates.

## 6. The owner's voice (US-C406)

- [ ] **A plan-made post:** rewrite its copy, then approve it. Brand kit → "Voice: how you rewrite Auto" shows Auto's draft beside your version.
- [ ] **Approve one unchanged:** nothing is added.
- [ ] **Remove one** (owner): it goes, and a viewer sees no Remove.
- [ ] **Redraft with Auto** on a new post writes closer to your approved versions.

## 7. Plan health (US-C407)

- [ ] **The Plan page** shows "Plan health" when something needs you, and it shows nothing when all is well:
  - a bank short of the next two batches, with Research again;
  - a cadence channel not connected, linking to Tools & Integrations;
  - no video this week, with Add a video row (opens Cadence);
  - a cap at 80% or more, linking to the Brand kit's AI tools.
- [ ] **The weekly note** names the same items.
