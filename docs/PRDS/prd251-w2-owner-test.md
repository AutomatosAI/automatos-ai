# PRD-251 Wave 2: the owner's test (the Socials tab)

What the Ralph loop can't prove: the tab seen in a browser, and real render times. Run this on the Socials test stack before merging Wave 2 and before Wave 3 launches. Record what failed in the PR.

Nothing publishes in Wave 2. "Publish now" answers "Channel publishing arrives in Wave 3", and that's expected.

## 0. Set up

1. **Bring up the stack with this branch and the media profile.** On first use, run `~/.automatos/socials-stack/socials-stack.sh init`, which writes `stack.env`. If the Wave 1 test hasn't done it yet, give `media-render` a `socials_` container name in `~/.automatos/socials-stack/override.yml`; the guard refuses to start the stack otherwise. Then:

   ```
   SOCIALS_WT=<workspace>/.worktrees/automatos-ai/feat+prd-251-w2-socials-tab \
   SOCIALS_BRANCH=feat/prd-251-w2-socials-tab \
   COMPOSE_PROFILES=media ~/.automatos/socials-stack/socials-stack.sh up
   ```

   Frontend: http://localhost:23000. API: http://localhost:28000.
2. **Composio key:** the channel list reads your Composio connections. Add your key to `~/.automatos/socials-stack/stack.env` yourself, then `socials-stack.sh rebuild`.
3. **Skills:** sync the Socials skills from your local `automatos-skills` checkout (`python3 scripts/sync-skills.py <the Socials skill names>`). Never sync `platform-management` this way.
4. **Turn Socials on:** Deliverables → Socials → "Turn on Socials". The platform switch is already on for this stack only.
5. **Brand kit:** set it, or install the Socials package and run "Brand kit from your website".
6. **Connect some channels** in Composio (LinkedIn is enough to start), so the composer has channels to offer.
7. **Refresh the action cache** if the channels show "missing action": `POST /api/tools/sync` (workspace admin), or Settings → Tools → Sync.

## 1. The tab (S2.1)

- [ ] **List and board:** the board and the list show the same posts and the same count per status. The board has no drag.
- [ ] **Search:** global search finds the Socials page, and finds a post by its title.
- [ ] **Browser checks (the PRD's acceptance):**
  - at **390 px** the list and the detail stack, with a back control, and the board scrolls sideways;
  - at **1440 px** the two-column layout holds.
  - Check both shells (Studio and Classic).

## 2. The channel list (S3.2, pulled into Wave 2)

- [ ] **Connected channels appear** with the post kinds they support.
- [ ] **X's setup note shows:** Composio removed its managed X credentials in February 2026, so X needs your own X API app.
- [ ] **Unavailable kinds say why:** a missing action, or needs public storage.

## 3. The composer (S2.2)

- [ ] **Brief:** "Announce our Harvest Club launch" with your connected channels selected. You get copy per channel, a template, variables and proposed sources.
- [ ] **Unsourced claims:** a claim without a source shows the red **Unsourced** chip.
- [ ] **Render time:** an image renders within **90 s**, and a 40 s video within **6 min**. Write down both times.
- [ ] **Preview:** editing a variable after the preview says "Preview out of date".
- [ ] **Limits:** character counts turn red over the limit and block submit.

## 4. Approval (S2.3, S2.4)

- [ ] **Notification:** submitting sends an approval notification that opens the post.
- [ ] **What you approve:** the approval view shows the **exact** media (the video plays) and each channel's copy.
- [ ] **Edits reset approval:**
  - editing after approval shows **"Approval reset: content changed"**;
  - adding or removing a channel after approval resets it too.
- [ ] **Unsourced claims:** approving with them asks a second time and names the claims.
- [ ] **Request changes and reject** both work, and both show in the history.
- [ ] **Series approval** (turn it on for the workspace):
  - approving a campaign approves its posts;
  - a post added afterwards still needs its own approval.

## 5. Agents (D14)

- [ ] **Drafting:** ask the Social Media Director, in chat, to draft a post. It appears as **needs approval**.
- [ ] **Direct posting:** ask an agent to post straight to a connected channel. It is refused and pointed at the draft tool.

## 6. Media links (S3.4)

- [ ] **Headers:** `curl -I "<a media URL from the approval view>"` returns `Content-Type: video/mp4` and `Content-Disposition: inline`.
- [ ] **Range:** `curl -r 0-99 -o /dev/null -w '%{http_code}' "<url>"` returns `206`.

## 7. Wave 1's review fixes (the run's first stories after the migration)

CI proves these with concurrency tests. Two can be seen by hand:
- [ ] **Voice respects the cap:**
  - set the workspace's monthly media cap to `0`;
  - render a post with a Fish Audio voice;
  - it fails with `voice_refused`, names the cap, and nothing is spoken or booked.
- [ ] **Brand-neutral templates:** open a starter's variables through the composer, or ask the Director for the template's schema. The examples say `@yourbrand` / `yourbrand.com`, never `automatos`.

## Then

Merge Wave 2 when satisfied. Wave 3 launches after this test.
