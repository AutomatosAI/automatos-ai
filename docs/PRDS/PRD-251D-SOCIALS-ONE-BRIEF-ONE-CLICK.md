# PRD-251D: Socials, one brief and one click. Auto does the typing, the editor does the approving, and a post looks the way you chose

| | |
|---|---|
| Owner | Gerard Kavanagh |
| Written | 2026-11-16 |
| Status | Wave 1 in build; Waves 2 and 3 specified, with the owner calls listed at the end |
| Builds on | PRD-251, 251B and 251C (on `main`), and the base branch `feat/prd-251d-socials-base` (16 Nov: Auto writes a chosen template's empty fields, the fields show in the order the image shows them, the two calendars show the same coming posts) |
| Target | A person writes a brief, picks a look, and has a post waiting for their approval without typing a field or leaving the editor |

## Framing

The owner, 16 Nov 2026: "its still too complicated and when I select a template i am forced to fill in loads of fields, i might as well write the post myself. I do like the template model but it needs to be easy to just give a brief and select a template then the agent should fill in the required fields for me." And: "Review the full social model and see how we can make it slicker, better and higher quality, more flexible options."

The base branch removed the forced fields. This PRD removes the rest of the work the product hands back to the person: the trip to the Queue to approve their own post, the two-step render and submit, the channels that are never ticked, a day and time Auto cannot set, a preview that only exists after a render, and a look that can only change by changing template.

**It keeps every PRD-251, 251B and 251C rule:**
- agents draft and never approve, schedule or publish (D6, D14);
- every post is approved against the exact content that goes out (D6);
- publishing goes only through the workspace's own Composio connections;
- Auto uses only the facts it is given, and asks for the rest (F378);
- Socials can be switched off.

## What exists today (verified on `main` 8cfc0fde5 and the base branch, 16 Nov 2026)

Paths are under `orchestrator/` or `frontend/`.

- **Approving.** The approve controls (`components/deliverables/socials/socials-post-review.tsx`) are mounted only by the Queue's pane (`studio/queue-pane.tsx`) and the older detail pane (`socials-post-detail.tsx`). The editor (`studio/socials-editor.tsx`) has none: a post that waits for approval, opened from the calendar, can be read and edited and not approved. The editor's header hides Submit for it (`studio/socials-editor-header.tsx`, `SUBMITTABLE_STATUSES`).
- **Rendering and submitting.**
  - "Submit for approval" saves the post and calls `POST /posts/{id}/submit` (`hooks/use-socials-editor.ts`, `useSubmitEditor`; `api/socials.py`, `submit_social_post`). It renders nothing, so a post can wait for approval with no file.
  - A render ends the post in `needs_approval` by itself (`modules/socials/service.py`, `TRANSITIONS[ACTION_RENDER_DONE]`). For an image, "Render preview" is the real render.
  - For a video, "Render preview" is a half-resolution preview that keeps the status, and the note under it says "The full render is made when you submit" (`studio/preview-model.ts`, `renderNote`). Nothing makes it.
  - A render writes a chosen template's empty fields first (base branch, `api/socials_compose.py`, `write_missing_fields`). It does not write an empty caption.
- **Channels.** A new post starts with no channel ticked (`studio/editor-model.ts`, `newDraft`: `kinds: {}`). Approve is disabled for a post with no channel. With no channel connected, the card says "No social channel is connected. Connect one in Composio to publish." as plain text (`studio/editor-channels-card.tsx`). A connect button exists (`connect-toolkit.tsx`) and is used only by the voice picker and the brand kit's AI tools.
- **Auto in chat.** `platform_create_social_post` and `platform_update_social_post` (`modules/tools/discovery/actions_socials.py`, `handlers_socials.py`, parameters in `core/social_post_schema.py`) take no day or time. The weekly Playbook writes "Proposed for: <day and time>" into the brief as text (`core/seeds/seed_socials_package.py`). The slot itself is set by `service.set_planned_for`, which is not content: it never changes the hash or an approval. Approving a post with a slot still ahead schedules it into that slot (`service.approve`).
- **Template gallery.** A grid ordered by name: a thumbnail, the name and a Photo badge (`studio/editor-look-card.tsx`). Each entry already carries the template's description, which says what it is for ("For a review, a testimonial or a thank-you.") and is not shown. There is no search.
- **Preview.** "Not rendered yet" until a render (`studio/socials-preview.tsx`). An image template is a full HTML document with its own layout script (it fits the text, breaks the headline on `|`, removes empty blocks). The renderer (`services/media-render`) fills the words on the server with HTML escaping, injects the brand tokens as `:root{--brand-*}`, stages the fonts and GSAP, and takes a still at 0 seconds. Nothing in an image template starts hidden, so the page looks finished the moment it loads. No HTML-only preview path exists for social templates.
- **Look.** Every template field is text or a number. Colours, fonts and the logo come only from the brand kit's tokens. A different look means a different template.
- **Views.** The Calendar's List layout renders the older post list, with its own Board, status list and detail pane. Plans, Campaigns and Series are one object (`social_campaigns`) with several screens.

## Decisions

Each carries the recommendation it was built on. The owner calls that can still change one are at the end.

| # | Decision |
|---|---|
| E1 | **Submit makes the file.** "Submit for approval" on a post with a visual starts the full render through the render route, which ends the post in `needs_approval`. It never sends a post for approval whose file does not show its current words. A text post, and a post that is the person's own video file, submit as they do today. As built: Submit renders every time, because a post does not record what its file was made from (US-D108 adds that). Since Wave 2 (US-D108) it renders only when the file does not show the post yet: the post has no file, its file is older than its words, the editor holds an unsaved change to a field, the template, the length or the format, or the visual is a still of the person's own, which is cropped every time. |
| E2 | **An empty caption is Auto's to write, like an empty field.** When a render starts with the caption empty and the post has a brief, Auto writes it from the brief for the post's channels. A caption that holds words is never rewritten, and nothing is written into a post that holds an approval. |
| E3 | **Approve where you are.** The editor shows the Queue's own approve controls for a post that waits for approval, to a person who may approve. Approval binds to the stored content (D6), so the controls are off while the editor holds unsaved edits, and after a saved change to what the file shows, until the file is made again. |
| E4 | **Channels start ticked.** A new post starts with the channels the workspace's most recent post used that can take the format. With no earlier post and exactly one connected channel, that one. Otherwise none, as today. |
| E5 | **Connect from the editor.** Each supported channel that is not connected has a Connect button in the Channels card (the existing Composio connect flow). |
| E6 | **Auto sets the slot, never the schedule.** The create and update tools take a planned day and time with its timezone. It is written with `set_planned_for` only, and only while the post is a draft, waits for approval or was sent back. The tools still never approve, schedule or publish: an approved or scheduled post's slot is refused, saying a person moves it. |
| E7 | **The gallery says what a template is for.** Each card shows the template's description. A search box filters by name and description; chips filter to templates with a photo or with none. "Let Auto pick" stays first. |
| E8 | **An instant preview of an image post** (Wave 2). The server returns the template as one self-contained page, filled and escaped by the same rules as the renderer, with the brand tokens, the font faces and a stub for the timeline library. The browser shows it in a frame with `sandbox="allow-scripts"` (no same-origin) and its own content security policy: inline script and style and `data:` files, and no request the policy covers. A policy does not cover WebRTC, so the page's first script takes that API off its window. As built: an image of one still, and only a starter the platform ships, unchanged; a carousel and a workspace's own template keep their rendered preview. It is labelled a preview: the approved file is always the render. |
| E9 | **Fewer views** (Wave 2). The list of what is removed is an owner call, below. As built: US-D202 and US-D203. The campaign screens (US-D204) stay. |
| E10 | **Look options on a post** (Wave 3). Design step first: it touches the template contract and what an approval covers. |
| E11 | **Video templates explain their fields** (Wave 3, rescoped). The three software videos have 66 to 69 required fields. Giving them defaults would print the sample's words in every business's video, because Auto only writes a field that has no default. So every field gets a label and a description in plain words, and only text that is right for any business gets a default. |

## Wave 1: the editor finishes the job

### US-D101: Approve from the editor
**As** an owner who opened a post that waits for approval, **I want** to approve it there, **so that** I do not have to find it again in the Queue.

- [ ] The editor shows the approve controls of `socials-post-review.tsx` (Approve with the Queue's "publishes HH:MM" wording, Request changes, Reject, the unsourced-claims confirmation) for a post in `needs_approval`, to a person who may approve (the role rule the Queue uses).
- [ ] While the editor holds unsaved edits the controls are off and say to save first (the component's `dirty`).
- [ ] After a saved change to what the file shows (the brief, a field, the template, the length or the format) on a post that already has a file, the controls stay off and say why, and Submit is offered to make the file again. The editor holds this for the sitting; US-D108 makes the server hold it.
- [ ] A viewer, and a post in any other status, see no approve controls.
- [ ] After approving, the editor shows the post as approved or scheduled, with its Activity card.
- [ ] Tests: the controls show for `needs_approval` and an approver; not for a viewer; not for a draft; off while dirty; Approve calls the approve route with the post's content hash.

### US-D102: Submit for approval makes the file
**As** a person with a brief and a look, **I want** one button to make the post and send it for approval.

- [ ] On a post with a visual whose file is not the render of its current content, "Submit for approval" saves, then starts the full render (`POST /posts/{id}/render`, no `preview`). The post is `rendering`, then `needs_approval`.
- [ ] On a text post, and on a post that is the person's own video file, it saves and submits as today.
- [ ] Every post with a visual renders on Submit, one that was rendered before included. The post does not record which content its file was made from (an edit keeps the old file, and a person's saved edit leaves no history entry), so "this file still shows these words" cannot be told apart from "the words changed since". US-D108 records it; Submit can then skip a render that would change nothing.
- [ ] A person's own still picture renders its crops (today's `OWN_STILL` path) and ends in `needs_approval`.
- [ ] The button says what it will do: its label or the line under it names the render, and for a video the render minutes (the existing `renderNote` wording, now true).
- [ ] A refusal (422 naming what Auto could not write, 429 no render minutes) shows as it does for Render preview, and the post is left a draft.
- [ ] "Render preview" stays for a look before submitting.
- [ ] Tests: an unrendered image post submits through the render route and not the submit route; a post rendered and sent back renders again; a text post submits; a video calls the full render, not the preview.

### US-D103: A render writes an empty caption
**As** a person who gave a brief, **I want** the caption written with the image's words.

- [ ] When the person's render or preview route starts and the post's caption is empty (no base text and no channel text) and the post has a brief, Auto writes the caption from the brief for the post's channels (every connected channel when it has none), in the same composer call that writes the empty fields, and saves it as part of the same edit. The history line names it. With no brief Auto would have only the title, so the caption stays the person's to write.
- [ ] A post that holds an approval is never written into, fields or caption: a preview is allowed for it and keeps its status, so Auto's words there would void the approval.
- [ ] A caption that holds any words is never changed. A post with every field filled and an empty caption still gets its caption.
- [ ] "Let Auto pick" (a post with no template) gets the same: the template, its fields and an empty caption.
- [ ] A plan's or an agent's render is unchanged (`render_post` itself asks nothing).
- [ ] Tests fail without the change: an empty caption is written at render and kept in the saved post; a caption with words is untouched; nothing is asked when fields and caption are both filled.

### US-D104: A new post starts with its channels ticked
- [ ] A new post starts with the channels of the workspace's most recent post that can take the post's format, each with its default post kind.
- [ ] With no earlier post and exactly one connected channel that can take the format, that channel is ticked.
- [ ] Changing the format keeps the existing rule (`withFormat`): a channel that cannot take it is left out.
- [ ] The person's own ticks are never overridden once they change one; an existing post opens with its own channels.
- [ ] Tests: one connected channel is ticked; the last post's channels are ticked; two channels and no earlier post tick none; an existing post is unchanged.

### US-D105: Connect a channel from the editor
- [ ] In the Channels card, each supported channel that is not connected shows a Connect button (`connect-toolkit.tsx`); the list refreshes when the connect window reports back.
- [ ] With no channel connected, the card shows the buttons in place of the plain sentence.
- [ ] No new route and no new API client: the existing Composio connect flow only.
- [ ] Tests: the buttons show for unconnected channels and not for connected ones.

### US-D106: Auto sets the day and time a post is planned for
**As** a person talking to Auto, **I want** "post it Friday at nine" to put the post on Friday at nine, waiting for my approval.

- [ ] `platform_create_social_post` and `platform_update_social_post` take `planned_for` (an ISO date and time) and `timezone` (an IANA zone). The parameter descriptions say it is the slot the post is planned for, that the owner still approves it, and that approving it schedules it into that slot.
- [ ] The slot is written with `service.set_planned_for` only. On update it is allowed only while the post is `draft`, `needs_approval` or `changes_requested`; any other status is refused in plain words ("it is approved: the owner moves it in Socials"). A past time and an unknown timezone are refused with the reason.
- [ ] Setting the slot never changes the content hash, the status or an approval.
- [ ] The tools' answers, and `platform_get_social_post` and `platform_list_social_posts`, show the planned slot in the post's timezone.
- [ ] Nothing approves, schedules or publishes: no call to `schedule`, `apply_slot` or the publisher from a tool.
- [ ] The weekly Playbook's prompt sets `planned_for` in place of writing "Proposed for:" in the brief. Skill text is not edited here (the skills repo is its source).
- [ ] Tests fail without the change: a created post carries the slot; an update moves it on a draft; an approved post's slot is refused and unchanged; the hash and status are untouched; a bad timezone and a past time are refused.

### US-D107: The gallery says what each template is for
- [ ] Each template card shows the template's description under its name, cut to two lines.
- [ ] A search box filters the gallery by name and description as the person types; no match says so.
- [ ] Chips filter to All, With a photo (a template with an image slot) and No photo. "Let Auto pick" is always shown first.
- [ ] The chosen template stays visible when a filter would hide it.
- [ ] Tests: the description shows; search narrows the list; each chip filters; "Let Auto pick" stays.

## Wave 2

### US-D201: Instant preview of an image post

What the research found (16 Nov): an image template needs its own inline script to look right (it fits the text, breaks the headline on `|`, colours the accent words, removes empty blocks), and that script calls the timeline library once after the layout is done. Nothing starts hidden and the still is taken at 0 seconds, so a browser that runs the script shows the finished card. The renderer's page assembly (`services/media-render/media_render/composition.py`: `fill_placeholders`, `build_document`) lives in another container and cannot be imported, so the orchestrator assembles the page itself and one test holds the two equal. The kit's fonts can be 0.5 to 2 MB as data URIs, so they are fetched once and not with every keystroke.

- [ ] `POST /api/socials/templates/{template_id}/preview-page` takes the fields' values, a size of the template's and whether it is a story, and returns one HTML page: the words filled on the server with HTML escaping by the renderer's rules (a value never lands inside a script or an event attribute), the brand tokens as `:root{--brand-*}`, the size variables, the story safe zone for a story, an empty slot's element removed, the logos as `data:` URIs, a stub in place of the timeline library, and a marker where the font faces go. The page names no address to fetch: every file it shows is inline.
- [ ] `GET /api/socials/brand-kit/preview-fonts` returns the kit's `@font-face` rules with the fonts as `data:` URIs. The editor fetches it once per kit and puts it at the marker.
- [ ] One test holds the page's filled composition equal to what the renderer's own fill makes for the same template and values, so the two cannot drift.
- [ ] The editor shows the page in a frame with `sandbox="allow-scripts"` and no `allow-same-origin`, and the page carries its own content security policy: inline script and style, `data:` images and fonts, nothing else. It refreshes a moment after the person stops typing and is labelled a preview; the approved file is always the render.
- [ ] A photo the post already has shows in its slot (the editor reads the file through the post's media route and hands it to the page as a `data:` URI). With none, the slot shows what the renderer shows for an empty one.
- [ ] A video, a text post and a post that is the person's own file keep today's preview. Once a render exists and is current, the frame shows the rendered file.
- [ ] Both routes are authenticated, scoped to the caller's workspace and in the route manifest, and the change passes the security review. A workspace's own template runs its script only inside the sandboxed frame.

As built (Wave 2):
- Only an image template of one still previews. A carousel's slides are shown one at a time by the renderer's runtime, which a browser does not have, so the route refuses it (422) and it keeps its rendered preview.
- A field left empty with no default shows as empty text, so the template's own script takes its block out (0 for a number and false for a true/false, the only kinds a script may hold). The note under the frame says Auto writes an empty field when the post is made.
- A value the template refuses (a field over its limit, say) keeps the last page and says why under the frame, also when no page has shown yet. Any other failure keeps today's frame.
- The page reads the kit's logos and tokens and none of its fonts. The editor fetches the faces once per kit (`fonts_version`) and puts them at the marker; a save does not ask for the page or the fonts again.
- Only a starter the platform ships, unchanged, previews as you type. A template runs its own script to lay itself out. The frame's sandbox (no same origin, forms, popups or top navigation) keeps a script from the app, and the policy and the removed WebRTC API narrow what it can reach, but none of them keeps a script from sending out what its own page holds: the words on screen and the kit's logos. So the route answers only when the template's markup is a starter's, byte for byte. A workspace's own template, or a starter someone edited, is refused with a sentence the editor shows under the frame, and keeps its rendered preview (owner call 5).

### US-D108: A file older than its words is never approved

Found while building Wave 1 (16 Nov): a post that waits for approval, edited and saved, keeps its old file, and Approve accepts that file with the new words. `finish_render` records no content with the render, `update_post` keeps `media` after an edit, and a person's saved edit leaves no history entry, so nothing can tell. The Queue has had this since PRD-251B; a retake is safe, because it renders again. Wave 1's editor guards its own sitting only. This closes it on the server, which D6 ("approved against the exact content that goes out") needs.

- [ ] A render records what the file was made from (a hash of what a render reads: the template, the fields, the format and the length), on its `render_done` entry.
- [ ] A post says whether its file is current (`file_current` in its JSON): its last render was made from what it holds now. A post with no template, a text post and a person's own file are always current.
- [ ] Approve refuses a post whose file is not current, saying to render it again. The Queue and the editor show the same sentence in place of Approve, with the button that renders.
- [ ] The update tool's answer says when an edit left the file older than the words, so an agent renders again.
- [ ] Submit renders only when the file is not current.
- [ ] Tests: an edit after a render makes the file not current and Approve is refused; a render makes it current again; a slot move, a caption change and a channel change do not.

As built (Wave 2):
- The record covers the template, each field that has a value, the format, the length, and each photo or footage spot's prompt and file; for a video also its voice and music. A claim mark, the caption, the channels, the slot and the brief are not on the file.
- Asking for AI options changes a spot's prompt, and picking one changes its file, so either leaves the file older than the post until it renders again.
- A post rendered before this record existed has none and counts as current, so nothing that exists today is refused.
- The approve route, a plan's "Approve the week" and a series all hold to it: the route answers 409 with the sentence, the week and the series leave the post out and say why.
- An agent's read, a list row and a ticket's note say the post needs a new render, in place of the owner's approval it cannot have yet.
- The Activity card's own render is off while the editor holds unsaved edits, as Render again in the approve controls is.

### US-D202: The Calendar's List is one plain list
The research (16 Nov): the List layout renders the older `socials-post-list.tsx`, which brings a second list, a Board and a second detail pane. The Board and the status list are today the only views of an archived post and of a post with no slot, so the list stays and the rest goes.

- [ ] The List layout shows every post grouped by status, archived and unslotted ones included, in the studio's own style. A post opens in the post page, as it does from the calendar.
- [ ] `socials-board.tsx`, `socials-view-toggle.tsx` and the detail pane inside `socials-post-list.tsx` are deleted with their tests' pins moved or removed; nothing else imports them.
- [ ] Tests: the list shows an archived post and a post with no slot; a click opens the post page.

As built (Wave 2): `socials-post-list.tsx`, `socials-board.tsx`, `socials-view-toggle.tsx`, `socials-status-list.tsx` and `socials-post-card.tsx` are deleted; the List is `studio/socials-calendar-list.tsx`. This month's render minutes show under its hint. A refetch that fails keeps the posts already listed.

### US-D203 (owner call 2): one post page for every status and role
`socials-post-detail.tsx` is still the only surface for a viewer, for a post that is rendering, publishing, published, partly published or archived, and for a person in the Queue who may not approve. Moving those into the post page lets the older pane go.

As built (Wave 2), on the recommendation: the post page shows every status to every role. A post a person may edit opens in the editor; any other opens read-only (`studio/socials-post-view.tsx`) with its channels, receipts, history and the actions its status allows. The lines about a render (since when it renders, what a waiting draft waits for, why the last render failed) and the full render of a failed or waiting post moved to the Activity card (`studio/post-notices.tsx`). `socials-post-detail.tsx` is deleted. Two things changed with it, both the owner's to keep or reverse: "Request changes" in the Queue always sends the post back to Auto with the comment (the plain send-back to the author was offered only in the deleted pane), and the hint "Saving a change sends this post back for approval" is no longer shown.

### US-D204 (owner call 2): campaigns that are not plans
Their screens are the only place to make an ad-hoc campaign, add or remove its posts by hand, switch series approval on and approve a series. A plan's "Approve the week" does not use them. Removing them removes those four things. Not built: the screens stay, on the recommendation.

## Wave 3

### US-D301: Look options on a post

What the research found (16 Nov): every starter reads its colours and its logo only through the brand tokens and the `brand.*` variables, so a look can be applied to the tokens of one render and no template file changes. Every kit already yields a light palette (paper) and a dark one (stage); nine starters are built on paper and ten on the dark stage or a photograph. The accent is already switched for the whole kit by `accent_use`.

- [ ] A post carries a look: light or dark, the accent strong or sparing, the logo on or off. It is a render setting like the voice and the music (`RENDER_FIELDS`): stored on the post in a new nullable column, validated, and left out when it is the kit's own.
- [ ] `bundle_for` applies it to one render's tokens and variables: the accent through the kit's `accent_use`, the logo by the empty-logo value the templates already hide, light or dark by mapping the paper tokens to the stage's (with one new derived token for a card on the dark ground).
- [ ] Light or dark is offered only for a template built on paper; the gallery entry says which looks a template takes. The accent and the logo are offered for every template.
- [ ] The editor's Look card has the three controls under the gallery; they save with the draft and mark the preview out of date.
- [ ] An approved post's look is the look of its approved file: a render is refused while a post holds an approval, as today, so the look changes only with an edit that sends it back.
- [ ] The renderer's contrast check passes for every starter in every look it is offered in, shown by the CI job that renders the starters.
- [ ] Migration: one head, chained onto the current one, safe on a `create_all`-built database, both head pins moved.

### US-D302: Video templates explain their fields
- [ ] Every field of `app-promo`, `cinematic-product-promo` and `ui-story-promo` has a label and a description that say what it is and where it shows, in plain words.
- [ ] A field whose text is right for any business has a default. A field that names a product, a number, a person or a claim has none.
- [ ] The count of required fields per template is in the PR, before and after.

## Not in this PRD

- Auto approving, scheduling or publishing. D6 and D14 stand.
- An auto-approve setting for a person's own posts.
- One feed for both calendars. The base branch made them show the same coming posts and refresh each other.
- Comments, replies and paid promotion.

## Owner calls

None of these blocks Wave 1.

1. **Channels (E4).** Recommended: the last post's channels. The other choice is every connected channel that takes the format.
2. **Fewer views (E9).** US-D202 is built as specified (the Board and the second detail pane inside the list go; the list stays, because it is the only view of an archived or unslotted post). Two larger removals were left as an owner call: US-D203, one post page for every status and role, which lets the older detail pane go (built in Wave 2 on the recommendation); and US-D204, removing the screens for campaigns that are not plans, which removes ad-hoc campaigns, adding or removing a campaign's posts by hand, the series approval switch and "Approve series". Recommended: yes to US-D203, and keep the campaign screens until a plan can do what they do.
3. **Look options (E10).** Recommended: all three (light or dark, the accent strong or sparing, the logo on or off), with light or dark only on the nine templates built on paper. A look cannot change under an approval, because a render is refused while a post holds one.
4. **Video templates (E11).** Recommended as rescoped above: explain the fields, default only what is right for anyone.
5. **Instant preview and custom templates (E8).** The first recommendation was to let a workspace's own template run its script in the sandboxed frame. Building it showed the sandbox and the policy cannot keep a script from sending out what its page holds, so Wave 2 is built the other way: only the platform's own starters preview as you type, and a workspace's own template keeps its rendered preview. Recommended now: keep it so. Opening it to custom templates safely means serving the page from an origin of its own with a network-level policy, or drawing it without the template's script; either is new work.
