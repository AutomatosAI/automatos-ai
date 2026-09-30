# PRD-251 Wave 3: the owner's test (scheduling and publishing)

What the Ralph loop can't prove: real posts on real platforms, and the calendar and controls seen in a browser. CI proves every channel against a mocked Composio executor only. Run this on the Socials test stack before merging Wave 3. Record what failed, and each channel's live post link, in the PR.

**Use test accounts you own.** Every step here posts publicly unless the step says otherwise. Delete the test posts afterwards.

## 0. Set up

1. **Bring up the stack with this branch and the media profile:**

   ```
   SOCIALS_WT=<workspace>/.worktrees/automatos-ai/feat+prd-251-w3-publish \
   SOCIALS_BRANCH=feat/prd-251-w3-publish \
   COMPOSE_PROFILES=media ~/.automatos/socials-stack/socials-stack.sh up
   ```

   Frontend: http://localhost:23000. API: http://localhost:28000.
2. **Composio key** in `~/.automatos/socials-stack/stack.env`, then `socials-stack.sh rebuild`.
3. **Connect your test accounts** in Composio: LinkedIn, X (with **your own X API app**: Composio removed its managed X credentials in February 2026), Instagram (a professional account), TikTok and YouTube.
4. **Refresh the action cache:** Settings → Tools → Sync, so every channel's actions are present.
5. **LinkedIn images** need the workspace's own "LinkedIn Community Management OAuth2 API" credential (the platform's credential store): Composio cannot upload LinkedIn images, so the workaround posts them.
6. **Public storage (for the YouTube thumbnail only):** set `SOCIALS_PUBLIC_MEDIA_BUCKET` in `stack.env` if you want to test the thumbnail; without it the thumbnail step is skipped and the receipt says so.
7. Wave 2's setup still applies: Socials on, the brand kit set, a few rendered posts.

## 1. Publish now, one channel at a time (S3.3)

For each channel, compose a post for that channel only, render it, approve it, then **Publish now** and confirm (the confirmation names the channels). The post shows **Publishing**, then each channel's receipt under **Publish status**. Record the live link from the receipt.

The result fields each step reads (the post's id and link) could not be checked against Composio's docs from the build: if a channel publishes but its receipt has no id or link, record the channel and kind, and the tool's answer from the backend log (`[Socials]` lines), so the adapter data can be corrected.

- [ ] **LinkedIn:** a text post, an image post (through the image workaround) and a video post (`LINKEDIN_UPLOAD_VIDEO` → `LINKEDIN_CREATE_VIDEO_POST`). The PRD leaves open whether video needs the workaround's approach too: note what happened.
- [ ] **X:** a text post, an image and a video (the chunked upload).
- [ ] **Instagram:** an image (it must arrive as JPEG: media-render converts the PNG still, so the media profile must be up), a Reel and a 3-slide carousel. The receipt's link comes from a read after publishing.
- [ ] **TikTok:** a video. It uploads the file; the privacy level is the most private one your account allows unless you chose another, and the AI label is on for generated footage. Check both on TikTok.
- [ ] **YouTube:** a video, with the privacy and category chosen in the composer (start with **private**). With public storage, check the custom thumbnail.
- [ ] **Receipts:** each channel's receipt in the post view shows **Published** and a link that opens the live post.

## 2. Failure and retry

- [ ] **Partial failure:** post to two channels where one is broken on purpose (for example disconnect X in Composio after approving). The post ends **Partially published**: the good channel's receipt links to its post, the broken one shows the platform's message, and a notification arrives.
- [ ] **Retry:** reconnect, then **Retry the failed channels**. Only the failed channel is posted; the other is not posted twice.
- [ ] **Stale content never posts:** approve a post, edit its copy, then try **Publish now**. It refuses (the approval was reset), and nothing reaches the platform.

## 3. Scheduling (S3.1)

- [ ] **On time:** schedule a post **2 minutes ahead**. It publishes once, on time, and its receipt appears.
- [ ] **In your timezone:** the post view and the calendar show the slot in your own timezone.
- [ ] **Reschedule keeps the approval:** reschedule a scheduled post from the post view (the date and time are in your browser's timezone). It stays **Scheduled**, still approved, at the new time.
- [ ] **Unschedule:** it goes back to **Approved**, and nothing publishes at the old time.
- [ ] **Missed slot:** schedule a post a few minutes ahead, stop the backend (`socials-stack.sh stop api`, or your stack's equivalent) until well past the slot plus the grace (`SOCIALS_MISFIRE_GRACE_SECONDS`, 30 min by default; lower it in `stack.env` for the test), then start it. The post ends **Missed**, you get a notification, and nothing was posted. Reschedule it or publish it now from the post view.
- [ ] **Within the grace:** the same with a short outage inside the grace: the post publishes on recovery.

## 4. The calendar (S3.1)

- [ ] **Social posts appear** in the Command Center calendar with their own kind and colour, at their slot in your timezone.
- [ ] **Click** opens the item's menu, like every calendar item; **Open post** opens it in the Socials tab.
- [ ] **Drag** a social post to another day or hour (Day and Week views: the quarter hour under the pointer; Month view: the same time on the new day): its slot moves, in the post's timezone, and it stays approved. **Reschedule…** from the item's menu does the same, with a date and time in the post's timezone.
- [ ] **Browser checks:** the calendar and the post view's publish controls at **390 px** and **1440 px**, in both shells.

## 5. A restart mid-publish

- [ ] **Stop the backend while a post is publishing** (a long video upload), and start it again more than `BOOT_REAPER_STALE_MINUTES` later. The post ends **Failed** or **Partially published** by its channels; a channel that was uploading says the platform may have taken it (check it before you retry), and a published channel keeps its link.

## 6. One way out (D14)

- [ ] **Agents still can't post:** ask an agent to post straight to a connected channel. It is refused and pointed at the draft tool, even now that the platform itself publishes.
- [ ] **Agents can't schedule or publish** through Socials either: ask the Social Media Director to publish a draft. It drafts, and says a person approves and publishes it in the tab.

## Then

Merge Wave 3 when satisfied. That completes PRD-251 (Phase 2, engagement, is a separate PRD).
