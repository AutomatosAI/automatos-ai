# PRD-260 Play: the RUN checklist

> **Historical.** This checklist describes builtin Play, the theme compiled into core, before PRD-261 A-19 (wave W-A8) took it out of core; Play now ships as an installable package from the theme kit. It moved here from `docs/testing/` with the other owner-test checklists.

For Gerard's local test of the Play wave (branch `feat/prd-260-play-theme`), on the local stack with the frontend built from the branch tip. CI proves the look engine, Auto's state machine and the screens against test data; it can't see a browser, a video's alpha channel, a real backend or a production build. These steps can.

How the theme works, with file paths: `docs/frontend-architecture/play-theme.md`. The spec: `docs/PRDS/PRD-260-PLAY-THEME.md`. The 15 Oct review and its fixes: `scripts/ralph/prd260-review-fix-plan.md`. The owner's test list these steps cover: `ownerTest` in `scripts/ralph/prd-260.json` (the map is at the end).

**How to record:** tick the boxes in a copy of this file or in the wave's PR. Anything that fails goes in the PR as a finding: the step number, what you saw, the browser, and any console error verbatim.

**Before step 1:** the branch tip has every story built: T-001 to T-010 and the review fixes P260-FX-01 to P260-FX-31. A step that checks a review fix only a browser can show names it, as (FX-nn), or the founder decision it follows, as (Fn). Sections A to N and P run on the local stack (the local edition); section O needs a hosted-edition build.

---

## A. Build and start (TESTER first)

1. [ ] **Check nothing else is building.** One Docker build at a time on this Mac: ask TESTER, and look at `uptime` and free disk first.
   - *Expected:* no other `docker build` or `compose up --build` running; memory headroom for a Next.js build.

2. [ ] **Build the frontend image from the branch tip.** In the worktree checked out at the tip (`git log -1 --oneline` matches the PR's head commit), rebuild only the frontend container of your running stack, with that stack's project name and env file so it replaces the running frontend instead of starting a second stack:
   ```sh
   docker compose -p <your stack's project> --env-file <your stack's .env> up -d --build --no-deps frontend
   ```
   - *Expected:* the image's `npm run build` (`next build`) finishes, its route list includes `/conversations`, `/looks`, `/looks/style`, and the container starts; http://localhost:3000 serves.
   - This is the only `next build` proof for this PR: no CI lane runs it for a frontend-only change. It proves the bundle (imports, CSS Modules, `next/font/local`, server and client boundaries, the lazy slots Play's screens load through). It does not prove types or lint: `frontend/next.config.js` sets `ignoreBuildErrors` and `ignoreDuringBuilds`, and CI's `tsc` and vitest cover those. A failed build is a finding: paste its output verbatim.

3. [ ] **The backend.** The wave changes no file under `orchestrator/` or `services/`, so the running backend serves this test as it is.
   - *Expected:* `curl -s http://localhost:8000/health` answers JSON, and http://localhost:3000 opens in your current style, unchanged.

4. [ ] **Test data and a clean slate.** Use a workspace holding test data only (c1 on the local stack): the steps send messages, make Deliverables and may start a Mission (real tokens). Make the browser window at least 1024 px wide (1440 is good). In the DevTools console on localhost:3000:
   ```js
   Object.keys(localStorage).filter((k) => k.startsWith('automatos-play')).forEach((k) => localStorage.removeItem(k)); sessionStorage.removeItem('automatos-play.hello-played')
   ```
   - *Expected:* no error. Play will open in Ember and Auto will say hello.

## B. Switch to Play, no flash

5. [ ] **The theme menu before Play.** Open the theme menu in the header.
   - *Expected:* Style lists **Classic · Studio · Play**; no "Customise look…" item; the Tone items (Light, Dark, System) are live.

6. [ ] **Choose Play.**
   - *Expected:* the whole app turns into Play at once, with no reload: Play's sidebar ("Automatos Play" wordmark, New chat, the nav with Conversations right after Chat and Look last, Recent, Docs and Settings at the foot, the connection pill "Connected · localhost"), the page in Ember (graphite ground, molten accent, the serif display type). In DevTools: `<html>` has the classes `studio play` and `dark`, and its inline style holds the look (for example `--pl-ground-bg: #131110` and Studio tokens such as `--background`). The theme menu now shows "Customise look…", and the Tone items rest under the note "Set by your look": dimmed, Dark selected, still reachable with the arrow keys, and choosing one changes nothing (FX-06).

7. [ ] **Reload, no flash.** Reload normally, then with DevTools > Network > Disable cache and Slow 4G throttling.
   - *Expected:* the first frame is already in Ember's colours: never Studio's paper, the Classic chrome or a light frame first. Play's frame and the page may follow Ember's empty ground by a moment: under Play nothing of them renders until the browser knows the window's width (FX-25), so a phone never mounts them. The cookies `automatos-style=play` and `automatos-play-tone=dark` are set (DevTools > Application > Cookies).

## C. Chat with Auto

8. [ ] **Hello.** Open `/chat` in a new tab (the session flag was cleared in step 4; sessionStorage is per tab).
   - *Expected:* Auto plays hello once, then idles ("Ready when you are"). Chrome loads `/auto/hello.webm` (DevTools > Network, filter `auto/`); its response carries `Cache-Control: public, max-age=86400, stale-while-revalidate=604800`, so a state Auto has shown once plays again from the cache (FX-15). No black box around Auto: the ground shows through around his outline.

9. [ ] **The first open.** Press New chat in the sidebar.
   - *Expected:* Auto large, "Hi, I'm Auto.", "Your Automatos, the fun way", and three chips: "Plan my revision week", "Write captions for our bake sale", "Budget a weekend in Lisbon". A chip fills the message box with its words (don't send it).

10. [ ] **The first open on a laptop (FX-14).** Set the window to 1366 × 657 (DevTools device toolbar, Responsive, 1366 × 657) and press New chat; then, back at your usual window size, zoom to 125% (Cmd +), press New chat again and zoom back to 100%.
    - *Expected:* on both first opens Auto, "Hi, I'm Auto." and "Your Automatos, the fun way" never overlap the page header, the example chips or the message box: the main column scrolls instead (scroll down to reach the chips and the box). On a tall window they still sit centred above the chips.

11. [ ] **Listening.** Click into the message box, then click outside it.
    - *Expected:* Auto listens while the box has focus, and goes back to idle when it loses it.

12. [ ] **Thinking.** Send `hello`.
    - *Expected:* your message shows at once; Auto thinks ("Thinking", his ring in the accent, the page glow brighter) until the reply's first words appear, then goes to idle ("Ready when you are") while the rest streams in word by word, and stays idle when the reply finishes. The conversation appears under Recent.

13. [ ] **Searching.** Send `Search the web for the latest news about electric bikes`.
    - *Expected:* a tool line "Searching the web" with the query and a spinner; Auto searching, caption "Searching the web" with the query under it. The line turns ✓ with its time in seconds to one decimal (for example "Searched the web · 5 results · 2.1 s"; Studio's chat rounds to whole seconds: F12) and Auto goes back to idle. With a model that streams its reasoning, Auto goes back to Thinking while it reasons between tool calls (FX-16). "What Auto did" lists the receipts once; the web search has no tag. If a receipt says done, Auto celebrates when the turn ends.

14. [ ] **Reading.** With at least one document in the workspace, send `Search my documents for <a word from that document>`.
    - *Expected:* a tool line for the document search ("Looking through your files"); Auto reading ("Reading") while it runs, then idle.

15. [ ] **Working, a Deliverable, celebrate (FX-04, FX-14).** Send `Write a one-page report about electric bikes as a document`.
    - *Expected:* a tool line for the document; Auto working ("Working on it"). When the Deliverable lands, Auto celebrates once ("Your <kind> is ready" with the type in words, such as "Your document is ready", never a raw id like `blog_post`; "It's on your shelf too.") and a Deliverable card shows in the reply with **Open**. The turn's done receipt doesn't make him celebrate a second time, and after the one celebration he goes back to idle. "What Auto did" lists the done write with an accent "Done" tag; a refused call, when a turn has one, keeps its words in the quiet line colour with a red ✕ and a red "Not done" tag, a skipped one a muted "Skipped". Send `thanks` straight after: Auto thinks again.

16. [ ] **The Deliverable card opens Play's Deliverable page.** Press Open.
    - *Expected:* `/deliverables?deliverable=<id>` shows Play's page: the Shelf back link, the title, type, date and "Made by"; the side panel with Ask Auto about this, Download, and About this Deliverable. The core's Deliverable sheet does not open over it. `generate_document` makes a PDF or a DOCX, which Play doesn't preview: the content is the file card, "Play can't preview this document here." with Download the file (whether Play previews it, as Studio's sheet does, is decision F6, open).

17. [ ] **The printed sheet (FX-01, FX-13).** Open a markdown Deliverable: on the shelf (step 30), the Report chip (step 23's Mission files its report there), or any Deliverable whose file is `.md`.
    - *Expected:* the text is on the printed sheet: headings in the look's display type, lists and tables rendered, and no raw HTML (a `<br>` or `<script>` in the text prints nothing). Where the text has them: a link is the sheet's accent (dark molten on Ember's cream sheet, never Studio's light navy), `inline code` sits on a faint line-coloured chip and reads clearly, a fenced code block is a plain block on the sheet with no Copy button, and Tab to the link draws its focus ring in the same sheet accent, plainly visible on the cream. "Made by" names who made it: Auto (with the agent, if one); "Uploaded", "Made in Templates" or "Made in Socials" for what Auto didn't make. One made from sources lists "Sources · N" in the side panel, each a site's name.

18. [ ] **Ask Auto about this, Download (FX-10, FX-12).** On step 16's page press Download, then Ask Auto about this. Then open `/chat?draft=Line%20one%0A%0ALine%20two`.
    - *Expected:* Download saves the file and it opens; a markdown or text Deliverable's Download saves its own text, never the API's JSON envelope (Studio's sheet too). Ask Auto about this opens `/chat` on a new chat with words about the Deliverable waiting in the message box (not sent); the `draft` parameter leaves the URL; a later New chat starts empty. The hand-made link puts "Line one Line two" in the box on one line: a link can't hide a second line there.

19. [ ] **Oops on a snag.** Send `Fetch https://example.invalid/ and summarise it`.
    - *Expected:* when the fetch tool line ends ✕ (failed), Auto plays oops at once ("Something went wrong", the coral glow). If Auto answers without a failed call, note it and check oops with the mid-turn snag in step 51.

20. [ ] **Stop (F4, FX-30).** Send `Write a 2,000-word story about a lighthouse` and press Stop while words are streaming.
    - *Expected:* Send shows as Stop while Auto answers; the words stop and stay in Auto's reply, with "You stopped this answer." under them (Studio's chat drops the reply instead); Auto leaves thinking, and doesn't celebrate the stopped turn; a tool call still running on that reply reads "Stopped: …" (for example "Stopped: searching the web"). Send another message straight away: it streams normally, under the stopped reply, which keeps its words and its note. In the backend log the first turn was cancelled, not finished.

21. [ ] **Reload mid-turn (FX-02).** Send `Write a 2,000-word story about a lighthouse` and reload the page while words stream.
    - *Expected:* the conversation reopens with Stop showing and Auto thinking while the server finishes the reply; the reply lands on its own, with no second reload, then Send comes back and Auto goes idle (Play's chat holds the core's `chat_changed` lane open, as Studio's chat does). Don't press Stop while it waits: Stop during an awaited reply leaves Stop and a thinking Auto for up to 5 minutes, on `main` too (M48, the review's section C).

22. [ ] **New chat and back (FX-03).** In a conversation with a few messages, open Deliverables and press New chat in the sidebar; then pick that conversation under Recent. Then, in a new chat, set DevTools > Network to Offline, send `hello`, set it back to No throttling within a few seconds and press New chat.
    - *Expected:* New chat stays disabled until the chat session has loaded on that page, then opens an empty first open; the conversation picked under Recent opens with its messages, never as an empty first open. After the failed first send, New chat gives a fresh first open.

23. [ ] **A Mission suggestion.** Send `Research three electric bike brands, compare their prices and write a one-page report` (this spends tokens).
    - *Expected:* when the reply finishes, the core's suggestion card shows under it ("This task could benefit from a Multi-Agent Mission…", the goal, Launch Mission and No thanks), as in Studio's chat. Launch Mission: the toast "Mission created — review the plan", and the card becomes "Mission Created" with the way to `/missions/<id>`, which opens Studio's Mission page inside Play's frame. If no card comes, the core judged the message simple: ask for more steps.

24. [ ] **An owner-approval ask (FX-08).** Ask for an owner-only change on a test agent, for example `Rename the agent <a test agent> to <a new name>`.
    - *Expected:* the tool line reads "Waiting for you", and the "Action approval needed" card shows inside Auto's reply, as in Studio's chat: the action, the oversight note, the parameters, Approve & run and Deny, in the look's colours. Auto listens ("Listening", "Waiting for your answer on the card.") through the end of the turn and doesn't nap. Press Approve & run: the card says what happened ("Approved — the action ran."), Auto goes back to idle, and the agent has its new name in Studio's Agents page. Ask again for another change and press Deny: "Denied — the action will not run.", nothing changes.
    - If a turn creates a Mission that waits for approval, the "Mission plan — approval needed" card shows the goal and its steps; Approve & run starts it ("Mission approved — execution started") and the card leaves the reply, as in Studio.
    - Ask for another owner-only change and press **Stop** as soon as the card shows: the reply stays, with "You stopped this answer." (step 20), and the card stays inside it; Auto keeps listening, and Approve & run or Deny still works; then Auto goes back to idle. Closing it with its X also returns him to idle. (In Studio the reply goes and the card stays on its canvas.)
    - If a turn fails after its card showed and you press Try again (as in step 51), the failed reply goes, and the card stays under the conversation ("Approvals still open") until it is answered or closed; once answered it moves under "Approvals".
    - Raise another card and, while it waits, open Conversations (or another conversation under Recent), then come back to this one: the card is still there, under "Approvals still open" after the conversation (the reply reloaded from the server carries the server's id), Auto listens again, and Approve & run works. A reload drops the card, as Studio's canvas does.

25. [ ] **The withdraw offer and tickets (F1, FX-29).** With an approval card waiting (step 24), answer in the chat `No, don't do that`. Then send `Add a task to the board to check the tyre pressure on Friday`.
    - *Expected:* Auto's reply to the "no" carries the core's withdraw card, in the look's colours: the request, Withdraw and Keep it. Withdraw: "Withdrawn." and the toast "Request withdrawn"; the request no longer waits on a yes. The task's reply carries the core's ticket card, as in Studio's chat; "Open on the board" opens `/command-center?tab=board&task_id=<id>`.

26. [ ] **The nap.** Leave the chat with no input for 3 minutes.
    - *Expected:* Auto sleeps, "Having a little nap" and "Move or type to wake me."; moving the mouse wakes him to idle.

27. [ ] **Reduced motion.** Turn on Reduce motion (macOS: System Settings > Accessibility > Display) and reload. Run one short turn. Turn it off again afterwards.
    - *Expected:* Auto is a still picture (the state's PNG), and the caption still changes through the turn; the glow doesn't crossfade.

28. [ ] **Safari (FX-15).** In Safari, choose Play from the theme menu first (Safari has its own cookies and storage), then repeat steps 8 and 13. Then set Safari > Settings > Websites > Auto-Play to "Never Auto-Play" for localhost and reload `/chat`; set it back afterwards.
    - *Expected:* Safari loads `/auto/<state>.mp4` (Web Inspector > Network); no black box, no dark fringe; the rest as in Chrome. With autoplay blocked, Auto shows his stills instead of a frozen clip, and hello still ends after its 5 s.

## D. Conversations

29. [ ] **Conversations (FX-10, FX-14).** Click Conversations in the sidebar. Type a word from one conversation's title in the search box, then a word no title has.
    - *Expected:* `/conversations` shows Play's list grouped Today, This week and Earlier, with this run's conversations under Today and the newest at the top without a reload (the list asks again each time it opens and after every turn). The "Conversations" title and New chat sit in the list's own column, New chat's right edge in line with the search box's. The search box filters by title; with no match it says "No conversation titles match “<word>”." A row opens `/chat?chatId=<id>` with the same messages; New chat opens an empty chat. Auto idles in the mascot column. Recent in the sidebar lists the same newest conversations.

## E. The shelf and Load more

30. [ ] **The shelf (FX-12).** Click Deliverables. Then make one more Deliverable in the chat (`Write a short note about tyre pressure as a document`), come back to the shelf, and reload it. Then leave the shelf open in this tab and make another Deliverable from a chat in a second tab.
    - *Expected:* `/deliverables` shows "Your shelf" with the count; the newest card in the spotlight ("Newest"); the type chips (All and the types) filter the grid; a card opens Play's Deliverable page. The first visit in this browser only remembers what is on the shelf (step 4 cleared the record), so Auto doesn't celebrate then. After the new Deliverable, Auto in the header celebrates once for it, and not again on the reload. The open shelf asks again every 30 s while the tab is visible: within 30 s of the second tab's Deliverable, with no reload, it lands in the spotlight and Auto celebrates once.

31. [ ] **Load more.** In a workspace with more than 24 Deliverables, scroll to the end.
    - *Expected:* Load more shows and appends the next page. With 24 or fewer it doesn't show.

32. [ ] **The core's sections stay in reach.** Follow each link in "More in Deliverables" (Blogs, Templates, Brand kit, Socials where offered, Explorer).
    - *Expected:* each opens the core's section inside Play's frame (Studio's version, in the look's colours) and works.

33. [ ] **Bad, missing and Outputs links (FX-11).** Open `/deliverables?deliverable=not.an.id`, then `/deliverables?deliverable=does-not-exist-123`. Then open `/deliverables?tab=outputs`, `/deliverables?tab=outputs&deliverable=<step 16's id>` and `/deliverables?tab=brand`.
    - *Expected:* the first shows the shelf with "That link doesn't name a Deliverable, so here's your shelf." and makes no request for it; the second, on the backend's 404, says "This Deliverable isn't on your shelf. It may have been deleted." `?tab=outputs` is Play's shelf (the Outputs tab is the shelf); `?tab=outputs&deliverable=<id>`, the link the backend writes, is Play's Deliverable page with no core sheet over it; `?tab=brand` is the core's Brand kit inside Play's frame.

## F. Make it yours

34. [ ] **Open it (FX-01, F12).** Click Look in the sidebar (or "Customise look…" in the theme menu). In DevTools select the "Make it yours." heading (an `h1`); then pick Workshop and look again.
    - *Expected:* `/looks`, Make it yours, as in your screenshots: the five presets (Ember, Obsidian, Aurora, Workshop, Daylight) and any saved looks in a row, a live preview with Auto, and the editor: background, accent, type, Auto's shell, "Auto's light · ring" (his eyes are part of the clips, so the light colours his ring and the connection pill's dot, not his eyes), the glow. The heading's computed `font-family` is the look's display face: Instrument Serif's next/font family (`__Instrument_Serif_…`), italic, under Ember; Fredoka's (`__Fredoka_…`) under Workshop; never Newsreader at Studio's weight 500.

35. [ ] **Three looks.** Pick Ember, then Aurora, then Daylight.
    - *Expected:* each changes the preview and the whole app at once, with no reload: the sidebar, the page, the preview's accent and type, the tone, and Auto's ring, which is Mint in Ember and Daylight and Ice in Aurora (Obsidian and Workshop use the Mint light too; the light itself is changed in step 37). Daylight turns the tone Light (the Tone items stay under "Set by your look", Light selected, the `automatos-play-tone` cookie says `light`); Ember and Aurora are Dark.

36. [ ] **Obsidian with a cobalt Auto.** Pick Obsidian and set Auto's shell to Cobalt.
    - *Expected:* Auto's body and eyes take the blue tint in the preview and, on `/chat`, in the mascot column, in the clip and in the still; his ring keeps the light's colour.

37. [ ] **Change every option (FX-18).** Change the accent, the background, the type, Auto's shell, Auto's light and the glow (drag it to 0, then all the way up).
    - *Expected:* each change applies to the whole app at once; at 0 the glow reads "Off" and the page glow is gone. The slider stops at the look's safe glow, the most the page's text still reads at AA over: 25% for Ember's Graphite and Molten, 9% for Daylight (whose own 10 paints as 9).

38. [ ] **Make it yours on Paper (FX-19).** Pick Daylight, then press each accent in turn, and pick a background, a type, a shell and a light.
    - *Expected:* every accent swatch has a thin grey edge, so Cream, Mint and Gold show plainly on the Paper ground, and the chosen one has a dark ring round it. The chosen background, type, shell and light, and the Daylight card in the Looks row, have a 2px edge in the dark molten of the "See the style sheet" link, never the light orange, and the glow slider is the same dark molten. Pick Ember: the edges and the slider are molten, as before.

39. [ ] **Save, Save as, Rename, Delete (F12).** From an edited preset use Save as; rename the new look; save an edit to it; delete it.
    - *Expected:* Save as adds "My look 1" to the row, with "Your look" under its name (never a "Saved just now" stamp), and picks it; Rename and Save change it; Delete removes it, and deleting the active look returns the app to Ember. A preset can't be renamed, deleted or saved over.

40. [ ] **Export (FX-20).** Save a look again and Export it.
    - *Expected:* the notice says "Your browser is saving `<name>.look.json`. If it didn't appear, use Copy JSON." and the browser downloads `<name>.look.json`. Opened, it holds exactly eleven fields (`schema`, `id`, `name`, `sub`, `interface`, `ground`, `accent`, `font`, `shell`, `light`, `glow`): catalogue keys and a number, no colours.

41. [ ] **Import.** Delete that saved look, then import the file; then paste the same JSON and import it; then paste it with `"accent": "#ff0000"`; then with `"name": "<b>hi</b>"`.
    - *Expected:* the file comes back as a saved look under a new id (`look-N`); the paste adds a second copy; the bad accent shows `"accent" must be one of the catalogue's accents: molten, cream, mint, gold, periwinkle, rose.` and adds nothing; the `<b>` name shows as plain text, never bold. A file of 1 MB or more is refused with "That's 1 MB or more. A Look file is well under 1 KB: check it's a .look.json file."

42. [ ] **Reload keeps the active look (FX-20).** With a saved Daylight-based look active, reload with the cache off and Slow 4G; then make an unsaved edit and reload again. Then open Make it yours in two tabs and pick another look in one.
    - *Expected:* the page paints straight in the look, never a frame of Ember's dark ground first; the unsaved edit survives the reload too. Make it yours shows only its heading until the look is read, never Ember's card pressed first. The second tab follows the pick at once, its tone too, and Save as new look in either tab keeps both tabs' looks.

43. [ ] **A long look name (FX-19).** Make the window 1024 px wide. On `/looks` use Save as, rename the new look to `abcdefghijklmnopqrstuvwxyz0123456789abcd` (40 characters, no space), then change one option so the status reads `<name> · edited`.
    - *Expected:* "Make it yours." keeps its full size and reads on one or two lines, never a sliver one word wide; the status beside it wraps, right-aligned, within at most 40% of the row; the look's card in the Looks row shows the whole name, broken over lines, never cut off at its edge.

44. [ ] **No storage (optional; FX-06, FX-20).** In a private window with site data blocked, switch to Play.
    - *Expected:* Play stays the style (it never flips back to Studio), opens in Ember, and Make it yours still re-themes the app; Save, Duplicate, Rename and Import add "This browser isn't keeping looks, so it lasts until you close this tab." to their notice; nothing is kept after a reload.

## G. The style sheet

45. [ ] **The live style sheet (FX-14, FX-15, FX-18, FX-20).** Open `/looks/style` (from Make it yours, or directly). Press "Play Auto's states", then Copy tokens.
    - *Expected:* the active look's colour, type, scale, motion and components, and Auto's 12 states as stills until "Play Auto's states" plays their clips; with Reduce motion on, that button is disabled and each state stays its PNG. Pick another look on `/looks` and come back: the swatches follow. Where the sheet shows the look's computed values, they are the engine's: in Ember text on bg 16.36:1, secondary text on bg 8.54:1, ink on molten 6.74:1; in Daylight the bg `#F4EEE8`, accent as text `#93441B` (darker than flat Paper alone needs, so it also reads over molten's own tint), error as text `#A54B49`. The receipts sample under Components shows an accent "Done" tag beside "Made a report: Laptop shortlist". The copied `:root` block names the display face itself (`--pl-display-family: 'Instrument Serif',serif` under Ember), never a next/font variable.

## H. A Studio page in Play's frame

46. [ ] **Pages without a Play design (F9, FX-17).** From the sidebar open the menu items "Command Centre" (`/command-center`), "Agent Management", "Tools & Integrations", "Knowledge Base", "Marketplace" and "Analytics", in Ember, then at least one again in Aurora and in Daylight.
    - *Expected:* each is Studio's page inside Play's frame (Play's sidebar and header row), recoloured by the look: the page in the ground's background, cards in its surface, borders and secondary text from the ground, primary buttons in the look's accent made readable on Studio's fills (a lighter molten `#FB9F6F` with the ground's dark bg as ink in Ember; a dark molten `#793A1A` with the ground's bg as ink in Daylight). Errors and destructive actions (Studio's `--accent` and `--destructive`) are Play's coral, readable on the ground, never the look's accent: under Aurora a failed item is coral, not mint, and under Ember a destructive button is coral, distinct from the primary. Everything works: open a card, a tab, a dialog, a dropdown. Success, warning and info colours follow the tone. The theme menu's Tone items say "Set by your look".

47. [ ] **Discuss, ⌘K and the floating Auto (FX-26).** Press ⌘K and type `conv`, then `make`, then `style`. On Command Centre's board press Discuss on a ticket that has an agent. Open `/looks` and the floating Auto widget, then `/conversations` and the widget.
    - *Expected:* ⌘K lists Conversations, Make it yours and Style sheet among the pages (only under Play). Discuss opens Play's chat with the bar "You are talking to Auto in Play's chat. To ask <agent>, switch to Studio from the theme menu." (Studio's bar says to pick them in the message box, which Play's box can't). The widget says "Ask me anything about Make it yours" on `/looks` and "…about Chat" on `/conversations`, never "Dashboard".

## I. The 404

48. [ ] **Under Play.** Open `/nope`.
    - *Expected:* Play's lost screen: Auto lost, "This page wandered off", and a way back to `/chat` that works. The tab's title is "404: This page could not be found." (FX-21), as it is for the plain 404 in steps 54 and 55.

## J. Automatos is asleep

49. [ ] **Stop the backend (FX-21).** With Play open on any page: `docker compose -p <your stack's project> stop backend`.
    - *Expected:* not on the first failed `GET /health`: the frame's poll (every 30 s) fails, it is asked again three times 2.5 s apart (DevTools > Network: four failed `/health` requests), and only then does the whole window become "Automatos is asleep" (no sidebar, Ember mock 07): "Not connected" at the top, Auto sleeping, the heading with keyboard focus, the start command (local edition), Check again, the countdown and "Automatos didn't answer at localhost." Until then the page stays as it was, with anything typed in it. Each ask gives up after 5 s, so a backend that takes the connection and never answers (`docker compose -p <your stack's project> pause backend`, then `unpause`) shows the asleep screen too, at worst 27.5 s after the poll (four asks of 5 s, three waits of 2.5 s; DevTools shows each `/health` cancelled at 5 s). In the local edition the screen offers `make up` with Copy (it copies exactly `make up`) and the line "No make? docker compose up -d"; the hosted edition's line is step 67.
    - *Then:* if you opened another page in Play before stopping the backend, press the browser's Back button while the screen shows. The asleep screen is there at once; the earlier page never flashes in before it.

50. [ ] **Check again, then start it (FX-21).** Press Check again while it is still stopped; then `docker compose -p <your stack's project> start backend`.
    - *Expected:* Check again makes a new `GET /health` (DevTools > Network), followed by its retries while the backend stays stopped, and the screen stays; while they run the countdown reads "Checking…", and once they have failed it reads "Checking again in 10 seconds." and counts down again. Once the backend answers, the page mounts again (on the countdown or Check again) with keyboard focus on it, the pill says "Connected · localhost" and Auto wakes to idle.
    - *Offline:* with the screen showing, set DevTools > Network to Offline. The line under the title reads "You're offline. Check your internet connection: this page comes back as soon as Automatos answers.", with no start command and no "Nothing's broken on your side"; back to No throttling, the edition's own line returns.

51. [ ] **A snag mid-turn (FX-07, FX-21).** Send `Write a 2,000-word story about a lighthouse` and stop the backend while words stream. Press Try again while it is still stopped. When the asleep screen shows, start the backend.
    - *Expected:* the reply goes, and "I hit a snag" with "Auto couldn't answer that just now." and Try again shows under your message, with a toast, and Auto plays oops. Try again while the backend is stopped brings the same snag back (it re-sends). Within the next poll plus about 7.5 s (up to 27.5 s for a backend that never answers) the asleep screen takes the window. Once the backend is back the same conversation shows, loaded again from the server, with no snag box: the page mounted again, and a failed turn isn't kept. Send the message again by hand.

## K. Back to Studio and Classic

52. [ ] **Studio.** Theme menu > Studio.
    - *Expected:* Studio exactly as before Play: the Studio shell and sidebar, Studio's own colours. In DevTools `<html>` has `studio` but not `play`, and `document.documentElement.style.cssText` holds no `--pl-` property and none of the look's Studio tokens (`--background`, `--primary`, …); the `automatos-play-tone` cookie is gone; "Customise look…" is gone; the Tone items are live and show your own stored tone (if you had Light before Play, it is Light again).

53. [ ] **Reload in Studio.**
    - *Expected:* no frame of Play or of the look's colours.

54. [ ] **Play's routes under Studio.** Open `/looks`, `/looks/style`, `/conversations` and `/nope`.
    - *Expected:* `/looks` shows "Make it yours is part of the Play theme." and `/looks/style` "The style sheet is part of the Play theme.", each with a Switch to Play button and no redirect loop; `/conversations` sends you to `/chat` before anything paints; `/nope` is the plain 404 ("404", "This page could not be found.", a link home). Switch to Play on `/looks` turns the app into Play there, with keyboard focus on the "Make it yours." heading (FX-22).

55. [ ] **Classic.** Theme menu > Classic; reload; open `/nope`.
    - *Expected:* Classic exactly as before: `<html>` has neither `studio` nor `play`, no look properties, your own tone; the same plain 404.

## L. Below 1024 px

56. [ ] **Classic and Studio narrow.** Make the window narrower than 1024 px (or DevTools device toolbar at 768 and 390 px) in Classic, then Studio.
    - *Expected:* each looks as today.

57. [ ] **Play narrow (FX-25).** The same in Play: open `/chat`, `/deliverables`, `/conversations` and `/looks`.
    - *Expected:* no Play frame and no Play pages, not even for a moment: the fallback chrome Studio gets, with the core's chat and Deliverables pages; `/conversations` sends you to `/chat`; `/looks` and `/looks/style` say "It opens on a laptop or desktop screen: Play's pages need a wider window." with no Switch to Play. The fallback takes the look's colours and its tone (the look is not width-gated): the theme menu's Tone items stay under "Set by your look", and "Customise look…" is gone. Widening back past 1024 px brings Play's frame back.

## M. The two changes cherry-picked from PRD-259 Wave 1

58. [ ] **Rounded corners (`--radius`).** In Classic and in Studio, open the theme menu and hover its items; look at buttons and cards.
    - *Expected:* the highlighted item and the `rounded-md`/`rounded-lg` controls have rounded corners (`--radius` is now 0.5rem in Classic, 8px in Studio) where they drew square before.

59. [ ] **`dark:` classes apply (FX-05).** In Classic Dark and in Studio Dark, ask Auto in the core chat for a small markdown table.
    - *Expected:* the table sits on its darker panel (`dark:bg-background/40`, header row `dark:bg-background/60`) and its text reads in the theme's foreground, with no grey-on-grey. Where a now-live `dark:` class used to fade a coloured line (a workflow step's error, a memory or workflow badge), the text reads in the theme's foreground over its tint instead.

## N. Forced colours, keyboard and screen readers

60. [ ] **Forced colours (FX-19).** Chrome DevTools > More tools > Rendering > "Emulate CSS media feature forced-colors: active" (on Windows: Settings > Accessibility > Contrast themes). Open `/looks` in Ember, then in Daylight, and Tab through it; then Tab through `/deliverables` (the type chips, a card, the "More in Deliverables" links, Load more), a Deliverable's page (the Shelf link, Download) and `/conversations` (New chat, the search box, a row); then open `/looks/style`.
    - *Expected:* on `/looks` the six accent swatches, the ground squares, the shell and light dots and the tops of the look cards keep their own colours; the chosen accent, background, type, shell and light and the active look's card each carry a ring in the system's highlight colour (the options and the card a wider border too); every swatch shows a ring when Tab reaches it. Everywhere else every control Tab reaches shows an outline, and a pressed type chip with the focus has a thicker ring than without it. On `/looks/style` the colour chips and the spacing steps keep their colours.

61. [ ] **Keyboard focus (FX-22).** Keyboard only, with DevTools' `document.activeElement` at hand. On `/looks` save a look (step 39) and use its "This look" toolbar: edit it and press Save; Rename, type a name and press Save name; Rename then Cancel; Delete, then Keep it; Delete, then Delete look. Then Try again on a failed turn (step 51), the last Load more on the shelf (step 31), and the asleep screen coming and going (steps 49 and 50).
    - *Expected:* focus is never left on `<body>`: Save, Save name and Cancel land on Rename; Delete opens its confirmation with focus on "Keep it"; Keep it lands on Delete; Delete look lands on Duplicate. Try again moves focus to the message box; when the last Load more goes, focus moves to the first new card's link. The asleep screen's heading takes focus as it opens, and the page takes it when it comes back.

62. [ ] **A screen reader (optional; FX-22, FX-24).** With VoiceOver on (Cmd F5): import the same JSON twice (step 41), and paste a bad one twice; type a query on Conversations, then one that matches nothing; open `/deliverables?deliverable=does-not-exist-123`; open `/looks/style`; stop and start the backend (steps 49 and 50).
    - *Expected:* each notice is read every time, the repeats included. Conversations reads the count ("3 conversations match “…”.") or the no-match line. The missing Deliverable's "This Deliverable isn't on your shelf. It may have been deleted." is read. The style sheet's 12 Autos are never read as statuses ("Something went wrong"); each slot's own name says the state. The nap is read as "Automatos is asleep", the return as "Automatos is back".

## O. The hosted edition

63. [ ] **A hosted-edition build.** Run the frontend built with `NEXT_PUBLIC_AUTH_EDITION=saas` against a Clerk test instance, with a trial workspace holding test data only (TESTER: one Docker build at a time, as in step 1). Switch to Play.
    - *Expected:* sign-in works; Play turns on as in step 6, with the hosted profile control in Play's header row.

64. [ ] **Accounts (F5, FX-10, FX-20).** From Play's header row sign out, then sign in again; save a look (step 39); then sign in as a second account in this browser; then back as the first.
    - *Expected:* Play stays the style through sign-out and sign-in. The second account's Looks row has none of the first account's saved looks, and once the account has loaded Play shows Ember (the very first paint can still be the last account's look: the pre-paint script runs before sign-in and reads the look that account used); its Recent and Conversations list only its own conversations. Back on the first account, its looks and its active look return.

65. [ ] **New chat after a reload (FX-03).** With a few chat tabs open in Studio, switch to Play, reload `/deliverables` and press New chat as soon as it is enabled. Then open the app in Studio and in a second browser signed in to the same account.
    - *Expected:* a new chat opens. Your chat tabs are all still there in Studio and in the second browser: no empty tab list was saved to the server.

66. [ ] **A spent trial (FX-07).** In a workspace whose trial is spent, send a message in Play's chat, then save a key from the banner.
    - *Expected:* the core's trial banner shows above the message box. After the key is saved the conversation on screen stays, with its messages, and the next message sends.

67. [ ] **Asleep, hosted (FX-21).** Stop the backend, as in step 49.
    - *Expected:* the asleep screen says "Automatos isn't answering right now. Nothing's broken on your side: this page comes back as soon as it answers.", with no start command; while the browser is offline it shows the offline line instead (step 50).

68. [ ] **Onboarding stages (F2, open).** With a workspace at each onboarding stage (not started, power-up, completed), open `/chat` in Studio, then in Play.
    - *Expected:* Studio's chat shows its onboarding opener, connect cards, power-up card, setup checklist and trial balance pill as before. Play's chat shows none of them, only the spent trial's banner (step 66): whether it should is decision F2. Write in the PR what a new workspace on Play misses.

## P. The open question

69. [ ] **§7.1: Play as the default?** Open a private window with no cookies.
    - *Expected:* the app opens in Studio Dark, as before this wave (`DEFAULT_UI_STYLE` and `APPEARANCE_DEFAULTS_VERSION` are unchanged). Then decide whether Play becomes the default for new users, and write the decision in the PR. Nothing in this wave changes it.

---

## Map: the owner's test to these steps

| `ownerTest` entry (`scripts/ralph/prd-260.json`) | Steps |
|---|---|
| 1. TESTER first: build the frontend image from the branch tip (one Docker build at a time), then the local stack from the tip | 1–4 |
| 2. Switch to Play from the theme menu; the whole app turns into Play; reload with no flash | 5–7 |
| 3. Chat with Auto and his reactions (hello, listening, thinking, searching/reading/working, celebrate, oops); Stop; a Deliverable card to Play's Deliverable page; a Mission suggestion starts a Mission as in Studio | 8–28 (hello 8, listening 11, thinking 12, searching 13, reading 14, working and celebrate 15, Deliverable card 16–18, oops 19 and 51, Stop 20, Mission 23) |
| 4. Make it yours with three looks; accent, ground, type, shell, light, glow; save, export, import back; reload keeps the active look | 34–44 (three looks 35, a cobalt Auto 36, every option 37, save 39, export 40, import 41, reload 42) |
| 5. A Studio page without a Play design inside Play's frame, recoloured and working; the Tone items say "Set by your look" | 46 (and 6) |
| 6. Back to Studio, then Classic, both as before (no look colours on `<html>`, your tone restored); below 1024 px every style looks as today | 52–57 |
| 7. The visible changes cherry-picked from PRD-259 W1: rounded corners, `dark:` classes in Classic Dark and Studio Dark | 58–59 |
| 8. §7.1: should Play become the default for new users? (not changed here) | 69 |
| Beyond the list (review M59): Play in the hosted edition: accounts and their looks, New chat after a reload, a spent trial, the hosted asleep line, the onboarding stages | 63–68 |

The PRD's other pages, beyond the owner's list: Conversations (29), the shelf and Load more (30–33), the style sheet (45), the 404 (48, 54, 55) and "Automatos is asleep" (49–51). The review fixes and decisions only a browser can show are marked in their steps (FX-nn, Fn); forced colours, keyboard focus and screen readers have their own section (60–62).
