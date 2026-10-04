# PRD-251C Wave 3: the owner's test (stories, a row's own visual, uploads cropped)

Wave 3 was written directly on `feat/prd-251c-w3-formats-visuals`, stacked on Wave 2. It needs no migration. Run this on the local stack once the branch is merged into local main, the backend is restarted and the frontend is rebuilt. Record what failed in the PR.

- **O6:** this wave takes the PRD's recommendation, so an upload is cropped for each channel.
- **The spike:** Composio's `INSTAGRAM_POST_IG_USER_MEDIA` takes `media_type` STORIES (checked 2026-10-04).

## 0. Set up

- **Workspace c1:** Instagram connected through Composio (a Business or Creator account), plus LinkedIn and X.
- **Public storage** is set (`SOCIALS_PUBLIC_MEDIA_BUCKET`), as Instagram needs.

## 1. Stories (US-C301)

- [ ] **Editor, an image post with Instagram ticked:** "As a story" shows under Instagram. Tick it, and Instagram's size reads 1080 × 1920. X has no such switch.
- [ ] **Switch the format to Video:** the story stays a story. Switch to Carousel: the switch goes, and Instagram posts a carousel.
- [ ] **Render preview (image):** the 9:16 file keeps its words clear of Instagram's top bar and reply box (the page sits 250 px from the top and 340 px from the bottom at 1080 × 1920).
- [ ] **Approve and publish to Instagram:** the post appears as a story on the account. Its receipt shows in the editor.
- [ ] **A video story** (a 9:16 video template) publishes as a story too.
- [ ] **A plan's cadence row:** the format "Story (image)" on a row with Instagram and LinkedIn says that LinkedIn gets no posts from the row. Save it: the plan makes Instagram stories for that row.

## 2. A row's own visual (US-C302)

- [ ] **Cadence row → Visual:** "The plan's mix" by default. Choose "AI images", and an AI tool select appears with the workspace's default and the toolkits that make stills now.
- [ ] **The row says its AI spend:** "AI media: about N shots a month (about $X)". The summary gives the plan's total and says the monthly cap still holds.
- [ ] **A toolkit not connected:** pick it on a saved plan (it shows as "not available now"), then Save. The plan refuses, naming the toolkit and why, and nothing is saved.
- [ ] **The maker uses it:** a slot of that row is made with AI images. The post's footage asks name the chosen toolkit, and the render makes them with that toolkit only. The monthly and per-post caps still hold.
- [ ] **"Template's own" on a row** makes plain template posts even when the plan's mix is AI first.

## 3. Uploads cropped for each channel (US-C303)

- [ ] **New post → Look → Upload a landscape photo** as the whole post. Tick LinkedIn, X and Instagram (as a story).
- [ ] **The editor says** "Render preview crops it to each one's shape", and Render preview is on.
- [ ] **Render preview:** the post ends in Needs approval. Each channel's preview shows its own crop (LinkedIn square, X 16:9, the story 9:16), centred and with nothing on it. The render uses no render minutes.
- [ ] **Publish:** each channel gets its own crop.
- [ ] **Change the channels and render again:** it crops again from the original.
- [ ] **A video upload** still has nothing to render: Render preview is off, and Submit sends it as it is.
- [ ] **A plan with the Library visual** crops the Library picture it picks before approval.
