# PRD-251B Wave 1: the owner's test (the Socials Studio)

Wave 1 was written directly on `feat/prd-251b-w1-studio` (no Ralph loop) and tested on the local stack. Run this on the socials stack before merging Wave 1 and before Wave 2 starts. Record what failed in the PR.

**Not in this build:** the 15 s and 30 s cuts of UI story promo and App promo (US-B104 part 2: the length mechanism is there, the two templates still declare only their original length), and the CI screenshot lane (US-B112, skipped on the owner's "less CI" call). Section 0a and the 0:15 / 0:30 checks below are skipped until then.

The mockup is the visual spec (PRD-251B B1): `docs/PRDS/prd251b-reference/` (`Main.dc.html` the calendar, `Editor.dc.html` the post editor, `Queue.dc.html` the Queue, `Nav.dc.html` the sub-navigation). Compare layout, hierarchy, controls and copy. Plans (`Plan.dc.html`) and the Brand kit tab (`Brand.dc.html`) are Waves 2 and 3: in Wave 1, **Plans** lists today's campaigns and **Brand kit** opens the existing brand-kit dialog.

## 0a. Before the stack: the pictures (skipped: US-B112 is not built)

When built, every push on this branch runs the **socials-studio-screens** workflow, which builds the frontend against a mocked API (the mockup's sample data, the clock on Wed 14 Oct 07:20) and uploads PNGs of every Studio screen at 1440 px and 390 px.

- [ ] Open the branch's latest run of that workflow on GitHub (Actions → socials-studio-screens), download the `socials-studio-screens-<sha>` artifact, and lay each PNG beside its mockup screen (`Main.dc.html`, `Editor.dc.html`, `Queue.dc.html`, `Nav.dc.html`). Note every structural difference (a missing control, a wrong order, wrong copy) and every token miss (`TOKENS.md` says which colour, type and radius each role must use). Those notes become fix stories before the stack test.

## 0. Set up

1. **Bring up the stack with this branch and the media profile:**

   ```
   SOCIALS_WT=<workspace>/.worktrees/automatos-ai/feat+prd-251b-w1-studio \
   SOCIALS_BRANCH=feat/prd-251b-w1-studio \
   COMPOSE_PROFILES=media ~/.automatos/socials-stack/socials-stack.sh up
   ```

   Frontend: http://localhost:23000. API: http://localhost:28000. The stack's database must be at ONE alembic head after boot (`alembic heads` inside the backend container prints one line: `prd251b_wave1`).
2. **Composio key** in `~/.automatos/socials-stack/stack.env`, then `socials-stack.sh rebuild`. Connect LinkedIn and X at least (Instagram, TikTok and YouTube if you have them).
3. **Skills:** sync the Socials skills from your local `automatos-skills` checkout (`python3 scripts/sync-skills.py <the Socials skill names>`). Never sync `platform-management` this way.
4. **Turn Socials on:** Deliverables → Socials → "Turn on Socials". The platform switch is already on for this stack only.
5. **Brand kit:** set it, or install the Socials package and run "Brand kit from your website".
6. **Refresh the action cache** if the channels show "missing action": Settings → Tools → Sync.

## 1. The shell (US-B107)

- [ ] **Sub-navigation:** Calendar · Queue (with the count of posts needing approval) · Plans · Brand kit, in that order, matching `Nav.dc.html`. **New plan** and **New post** sit in the header.
- [ ] **The URL holds the view:** `?view=queue` reloads on the Queue; `?post=<id>` reloads on that post; the browser back button returns to the previous view.
- [ ] **Brand kit** opens the existing brand-kit dialog (the tab arrives in Wave 3); so does `?view=brand`. **Plans** lists the campaigns; **New plan** opens the plan (campaign) form.
- [ ] **A read-only role** (viewer) sees Calendar · Queue · Plans, no Brand kit, no New plan or New post.
- [ ] **Browser checks:** at **390 px** the sub-navigation wraps, the calendar scrolls sideways, the editor's sections stack with the preview below them; at **1440 px** the two-column layouts hold. Check both shells (Studio and Classic).

## 2. The calendar (US-B108)

- [ ] **Month · Week · List:** the three views show the same posts; List is the old list/board (it has its own List | Board toggle inside).
- [ ] **Click a chip:** a post waiting for approval opens in the Queue; any other opens in the editor.
- [ ] **Chips** read like the mockup: `time · format` (a video shows its length, e.g. `17:00 · Video 0:30`), the title, the channel badges, and a status word (Planned, Making, Needs you, Scheduled, Posted, Skipped). The word is there in every state, not colour alone.
- [ ] **Channel filter:** All channels · X · LinkedIn · Instagram · Video (plus any other connected channel); each hides the others' chips.
- [ ] **Drag before approval** moves the post to the dropped day and keeps its time; the editor's **When** shows the new slot and the post is still a draft or still waiting for approval (no "Approval reset").
- [ ] **Drag after approval** (a scheduled post) reschedules it: the chip moves and the post's history shows the reschedule (both drags write PUT /slot, which keeps the planned slot and the schedule together).
- [ ] **Posted, Making and Failed chips do not drag.**
- [ ] **Today rail:** the day's posts and the status key; **Review N posts** opens the Queue.
- [ ] **Command Center → Calendar** still shows scheduled posts as before, and PRD-252's board is unchanged.

## 3. The post editor (US-B109, US-B103, US-B104, US-B102)

- [ ] **New post** opens the editor matching `Editor.dc.html`: Brief (with **Redraft with Auto**), Format, Channels and sizes, Look, Claims and sources, When, and the base Copy and the Preview beside them. Header: **Back to calendar**, the title (editable), its status, **Save draft**, **Render preview**, **Submit for approval**. The first **Save draft** creates the post and the URL becomes its `?post=<id>`.
- [ ] **"Blank draft" is gone:** the editor is the only way to start a post. A post that is rendering, publishing, posted or archived opens in its read view (receipts, history), as does any post for a viewer.
- [ ] **Format Image:** every connected channel row shows its post kind and size (e.g. `1600×900 · 16:9`); the "Renders …" line lists the distinct ratios; unticking a channel removes its ratio.
- [ ] **Format Video:** the **Length** chips are exactly the lengths the chosen template declares (in this build every template declares one length; 0:15 and 0:30 arrive with US-B104 part 2). Changing the template changes the chips. The voice picker shows once the post is saved; Music is a note (the template's own track: there is no per-post music choice in the API yet); the AI footage switch names the connected toolkit, or says why it is off.
- [ ] **Format Carousel:** the Slides stepper runs 2–10.
- [ ] **Format Text only:** only X and LinkedIn stay enabled, the others say "Needs an image or video"; the Look section is hidden; the preview says "Text only: nothing to render."
- [ ] **Look → Template:** a thumbnail gallery of the workspace's social templates of that format, **Let Auto pick** first. Each card shows a rendered thumbnail (not a blank). Edit a template in Deliverables → Templates, come back: its thumbnail has been re-rendered.
- [ ] **Look → Upload:** a PNG and an MP4 upload; each appears as the post's media and in the preview at every chosen channel's aspect; a 50 MB PNG is refused (413, the limit is 10 MB for images and 200 MB for MP4) and a renamed text file is refused (415).
- [ ] **Look → Library:** the workspace's image and video Deliverables; picking one makes it the post's media.
- [ ] **Redraft with Auto** keeps your format, channels, template and length and rewrites the copy and the variables.
- [ ] **Submit for approval** sends the post to the Queue; the exact chosen template was used (open the post: its template id is the one you chose).

## 4. The preview (US-B110)

- [ ] **One tab per chosen channel.** Each shows that channel's copy with a counter against its limit (X 280, LinkedIn 3000, Instagram 2200, TikTok 2200, YouTube 100) and the media at that channel's aspect.
- [ ] **Render times:** an image renders within **90 s**; a video at its template's own length: write the time down (the 0:15 and 0:30 targets apply once US-B104 part 2 lands).
- [ ] **Stale:** editing the brief or a variable after a render marks the preview "Preview out of date".
- [ ] **The note:** images say they use no render minutes; a video estimates `length × sizes` render minutes; text says nothing to render.

## 5. Planned slots (US-B105)

- [ ] **Approve with a future slot:** set When to 20 minutes ahead, submit, approve. The post is **Scheduled** for that slot (the chip moves to Scheduled; the Command Center calendar shows it).
- [ ] **A slot that passes unapproved:** set When to 3 minutes ahead, submit, do not approve. Within a few minutes of the slot it becomes **Skipped**, you get the missed notification, and nothing was published.
- [ ] **Editing the slot keeps the approval:** on a scheduled post, change When: the post is rescheduled, no "Approval reset" banner.
- [ ] **Approve with no slot** behaves as before (approved, not scheduled).

## 6. The Queue (US-B111)

- [ ] **The heading** counts today's posts ("2 posts need you today" / "1 post needs you today" / "All caught up for today").
- [ ] **Each post** shows the exact media (the video plays), each channel's copy with its counter, the claims with their sources, and "No unsourced claims." or the red Unsourced chips.
- [ ] **Approve** schedules it into its slot ("Approve · publishes 12:00").
- [ ] **Request changes** opens "What should change?"; **Send back to Auto** records your comment and a new take comes back to the Queue with a new hash.
- [ ] **Make another take** re-composes and re-renders; the new take replaces the old one in the Queue.
- [ ] **Reject** works and shows in the history.
- [ ] **Approve all shown** appears only with the workspace's series-approval switch on and more than one post shown; with it off, the button is absent. A post that changed since it was shown is reported, not approved.
- [ ] **Stale:** open the same post in two tabs, approve in one, approve in the other: the second says the post changed and reloads.

## 7. Off means invisible (US-B106)

- [ ] **Workspace Settings** has a Socials on/off toggle (owner or admin). Turn it **off**: the Socials tab is gone, the Command Center calendar shows no social items, the marketplace shows no Socials package, Auto's tool list in chat has no `socials` actions (ask Auto to draft a post: it says Socials is off), and the brand kit shows no social handles.
- [ ] **Turn it on again:** everything is back, with no restart.
- [ ] **The master switch** (System Settings → Socials, super-admin) off: the same five surfaces disappear for every workspace.

## Then

Merge Wave 1 when satisfied (single-head check after; `orchestrator/reports/route-manifest.json` is also edited by #861, so whichever merges second regenerates it with `cd orchestrator && python -m scripts.dump_routes`). Wave 2 (plans and the content bank) starts after this test.
