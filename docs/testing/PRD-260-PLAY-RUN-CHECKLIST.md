# PRD-260 Play: the RUN checklist

For Gerard's local test of the Play wave (branch `feat/prd-260-play-theme`), on the local stack with the frontend built from the branch tip. CI proves the look engine, Auto's state machine and the screens against test data; it can't see a browser, a video's alpha channel, a real backend or a production build. These steps can.

How the theme works, with file paths: `docs/frontend-architecture/play-theme.md`. The spec: `docs/PRDS/PRD-260-PLAY-THEME.md`. The owner's test list these steps cover: `ownerTest` in `scripts/ralph/prd-260.json` (the map is at the end).

**How to record:** tick the boxes in a copy of this file or in the wave's PR. Anything that fails goes in the PR as a finding: the step number, what you saw, the browser, and any console error verbatim.

**Before step 1:** the branch tip has every story merged in, T-001 to T-010 (T-008a, T-008b and T-009 were built on parallel branches). Without them `/looks`, `/looks/style`, the 404 and the asleep screen are missing.

---

## A. Build and start (TESTER first)

1. [ ] **Check nothing else is building.** One Docker build at a time on this Mac: ask TESTER, and look at `uptime` and free disk first.
   - *Expected:* no other `docker build` or `compose up --build` running; memory headroom for a Next.js build.

2. [ ] **Build the frontend image from the branch tip.** In the worktree checked out at the tip (`git log -1 --oneline` matches the PR's head commit), rebuild only the frontend container of your running stack, with that stack's project name and env file so it replaces the running frontend instead of starting a second stack:
   ```sh
   docker compose -p <your stack's project> --env-file <your stack's .env> up -d --build --no-deps frontend
   ```
   - *Expected:* the image's `npm run build` (`next build`) finishes, its route list includes `/conversations`, `/looks`, `/looks/style`, and the container starts; http://localhost:3000 serves.
   - This is the only `next build` proof for this PR: no CI lane runs it for a frontend-only change. It proves the bundle (imports, CSS Modules, `next/font/local`, server and client boundaries). It does not prove types or lint: `frontend/next.config.js` sets `ignoreBuildErrors` and `ignoreDuringBuilds`, and CI's `tsc` and vitest cover those. A failed build is a finding: paste its output verbatim.

3. [ ] **The backend.** The wave changes no file under `orchestrator/` or `services/`, so the running backend serves this test as it is.
   - *Expected:* `curl -s http://localhost:8000/health` answers JSON, and http://localhost:3000 opens in your current style, unchanged.

4. [ ] **Test data and a clean slate.** Use a workspace holding test data only (c1 on the local stack): the steps send messages, make Deliverables and may start a Mission (real tokens). Make the browser window at least 1024 px wide (1440 is good). In the DevTools console on localhost:3000:
   ```js
   Object.keys(localStorage).filter((k) => k.startsWith('automatos-play')).forEach((k) => localStorage.removeItem(k)); sessionStorage.removeItem('automatos-play.hello-played')
   ```
   - *Expected:* no error. Play will open in Ember and Auto will say hello.

## B. Switch to Play, no flash

5. [ ] **The theme menu before Play.** Open the theme menu in the header.
   - *Expected:* Style lists **Classic · Studio · Play**; no "Customise look…" item; the Tone items (Light, Dark, System) are enabled.

6. [ ] **Choose Play.**
   - *Expected:* the whole app turns into Play at once, with no reload: Play's sidebar ("Automatos Play" wordmark, New chat, the nav with Conversations right after Chat and Look last, Recent, Docs and Settings at the foot, the connection pill "Connected · localhost"), the page in Ember (graphite ground, molten accent, the serif display type). In DevTools: `<html>` has the classes `studio play` and `dark`, and its inline style holds the look (for example `--pl-ground-bg: #131110` and Studio tokens such as `--background`). The theme menu now shows "Customise look…", and the Tone items are disabled under "Set by your look" with Dark selected.

7. [ ] **Reload, no flash.** Reload normally, then with DevTools > Network > Disable cache and Slow 4G throttling.
   - *Expected:* the first frame is already Play in Ember: never Studio's paper, the Classic chrome or a light frame first. The cookies `automatos-style=play` and `automatos-play-tone=dark` are set (DevTools > Application > Cookies).

## C. Chat with Auto

8. [ ] **Hello.** Open `/chat` in a new tab (the session flag was cleared in step 4; sessionStorage is per tab).
   - *Expected:* Auto plays hello once, then idles ("Ready when you are"). Chrome loads `/auto/hello.webm` (DevTools > Network, filter `auto/`). No black box around Auto: the ground shows through around his outline.

9. [ ] **The first open.** Press New chat in the sidebar.
   - *Expected:* Auto large, "Hi, I'm Auto.", "Your Automatos, the fun way", and three chips: "Plan my revision week", "Write captions for our bake sale", "Budget a weekend in Lisbon". A chip fills the message box with its words (don't send it).

10. [ ] **Listening.** Click into the message box, then click outside it.
    - *Expected:* Auto listens while the box has focus, and goes back to idle when it loses it.

11. [ ] **Thinking.** Send `hello`.
    - *Expected:* your message shows at once; Auto thinks ("Thinking", his ring in the accent, the page glow brighter) until the reply's first words appear, then goes to idle ("Ready when you are") while the rest streams in word by word, and stays idle when the reply finishes. The conversation appears under Recent.

12. [ ] **Searching.** Send `Search the web for the latest news about electric bikes`.
    - *Expected:* a tool line "Searching the web" with the query and a spinner; Auto searching, caption "Searching the web" with the query under it. The line turns ✓ with its time and Auto goes back to idle. "What Auto did" lists the receipts once. If a receipt says done, Auto celebrates when the turn ends.

13. [ ] **Reading.** With at least one document in the workspace, send `Search my documents for <a word from that document>`.
    - *Expected:* a tool line for the document search; Auto reading ("Reading") while it runs, then idle.

14. [ ] **Working, a Deliverable, celebrate.** Send `Write a one-page report about electric bikes as a document`.
    - *Expected:* a tool line for the document; Auto working ("Working on it"). When the Deliverable lands, Auto celebrates ("Your <kind> is ready", "It's on your shelf too.") and a Deliverable card shows in the reply with **Open**. "What Auto did" lists a done receipt.

15. [ ] **The Deliverable card opens Play's Deliverable page.** Press Open.
    - *Expected:* `/deliverables?deliverable=<id>` shows Play's page: the Shelf back link, the title, type, date and "Made by"; the content on the printed sheet with headings and lists rendered and no raw HTML; the side panel with Ask Auto about this, Download, and About this Deliverable. The core's Deliverable sheet does not open over it.

16. [ ] **Ask Auto about this, Download.** Press Download, then Ask Auto about this.
    - *Expected:* Download saves the file and it opens. Ask Auto about this opens `/chat` on a new chat with words about the Deliverable waiting in the message box (not sent); the `draft` parameter leaves the URL; a later New chat starts empty.

17. [ ] **Oops on a snag.** Send `Fetch https://example.invalid/ and summarise it`.
    - *Expected:* when the fetch tool line ends ✕ (failed), Auto plays oops at once ("Something went wrong", the coral glow). If Auto answers without a failed call, note it and check oops with the mid-turn snag in step 43.

18. [ ] **Stop.** Send `Write a 2,000-word story about a lighthouse` and press Stop while words are streaming.
    - *Expected:* Send shows as Stop while Auto answers; the words stop and stay in Auto's reply, with "You stopped this answer." under them (Studio's chat drops the reply instead); Auto leaves thinking; a tool call still running on that reply reads "Stopped: …" (for example "Stopped: searching the web"). Send another message straight away: it streams normally, under the stopped reply, which keeps its words and its note. In the backend log the first turn was cancelled, not finished.

19. [ ] **A Mission suggestion.** Send `Research three electric bike brands, compare their prices and write a one-page report` (this spends tokens).
    - *Expected:* when the reply finishes, the core's suggestion card shows under it ("This task could benefit from a Multi-Agent Mission…", the goal, Launch Mission and No thanks), as in Studio's chat. Launch Mission: the toast "Mission created — review the plan", and the card becomes "Mission Created" with the way to `/missions/<id>`, which opens Studio's Mission page inside Play's frame. If no card comes, the core judged the message simple: ask for more steps.

20. [ ] **An owner-approval ask.** Ask for an owner-only change on a test agent, for example `Rename the agent <a test agent> to <a new name>`.
    - *Expected:* the tool line reads "Waiting for you", and the "Action approval needed" card shows inside Auto's reply, as in Studio's chat: the action, the oversight note, the parameters, Approve & run and Deny, in the look's colours. Auto listens ("Listening", "Waiting for your answer on the card.") through the end of the turn and doesn't nap. Press Approve & run: the card says what happened ("Approved — the action ran."), Auto goes back to idle, and the agent has its new name in Studio's Agents page. Ask again for another change and press Deny: "Denied — the action will not run.", nothing changes.
    - If a turn creates a Mission that waits for approval, the "Mission plan — approval needed" card shows the goal and its steps; Approve & run starts it ("Mission approved — execution started") and the card leaves the reply, as in Studio.
    - Ask for another owner-only change and press **Stop** as soon as the card shows: the reply stays, with "You stopped this answer." (step 18), and the card stays inside it; Auto keeps listening, and Approve & run or Deny still works; then Auto goes back to idle. Closing it with its X also returns him to idle. (In Studio the reply goes and the card stays on its canvas.)
    - If a turn fails after its card showed and you press Try again (as in step 43), the failed reply goes, and the card stays under the conversation ("Approvals still open") until it is answered or closed.

21. [ ] **The nap.** Leave the chat with no input for 3 minutes.
    - *Expected:* Auto sleeps, "Having a little nap" and "Move or type to wake me."; moving the mouse wakes him to idle.

22. [ ] **Reduced motion.** Turn on Reduce motion (macOS: System Settings > Accessibility > Display) and reload. Run one short turn. Turn it off again afterwards.
    - *Expected:* Auto is a still picture (the state's PNG), and the caption still changes through the turn; the glow doesn't crossfade.

23. [ ] **Safari.** Repeat steps 8 and 12 in Safari.
    - *Expected:* Safari loads `/auto/<state>.mp4` (Web Inspector > Network); no black box, no dark fringe; the rest as in Chrome.

## D. Conversations

24. [ ] **Conversations.** Click Conversations in the sidebar.
    - *Expected:* `/conversations` shows Play's list grouped Today, This week and Earlier, with this run's conversations under Today; the search box filters by title; a row opens `/chat?chatId=<id>` with the same messages; New chat opens an empty chat. Auto idles in the mascot column.

## E. The shelf and Load more

25. [ ] **The shelf.** Click Deliverables. Then make one more Deliverable in the chat (`Write a short note about tyre pressure as a document`), come back to the shelf, and reload it.
    - *Expected:* `/deliverables` shows "Your shelf" with the count; the newest card in the spotlight ("Newest"); the type chips (All and the types) filter the grid; a card opens Play's Deliverable page. The first visit in this browser only remembers what is on the shelf (step 4 cleared the record), so Auto doesn't celebrate then. After the new Deliverable, Auto in the header celebrates once for it, and not again on the reload.

26. [ ] **Load more.** In a workspace with more than 24 Deliverables, scroll to the end.
    - *Expected:* Load more shows and appends the next page. With 24 or fewer it doesn't show.

27. [ ] **The core's sections stay in reach.** Follow each link in "More in Deliverables" (Blogs, Templates, Brand kit, Socials where offered, Explorer).
    - *Expected:* each opens the core's section inside Play's frame (Studio's version, in the look's colours) and works.

28. [ ] **Bad and missing Deliverable links.** Open `/deliverables?deliverable=not.an.id`, then `/deliverables?deliverable=does-not-exist-123`.
    - *Expected:* the first shows the shelf with "That link doesn't name a Deliverable, so here's your shelf." and makes no request for it; the second, on the backend's 404, says "This Deliverable isn't on your shelf. It may have been deleted."

## F. Make it yours

29. [ ] **Open it.** Click Look in the sidebar (or "Customise look…" in the theme menu).
    - *Expected:* `/looks`, Make it yours, as in your screenshots: the five presets (Ember, Obsidian, Aurora, Workshop, Daylight) and any saved looks in a row, a live preview with Auto, and the editor: background, accent, type, Auto's shell, Auto's light, the glow.

30. [ ] **Three looks.** Pick Ember, then Obsidian, then Daylight.
    - *Expected:* each changes the preview and the whole app at once (the sidebar, the page, Auto's ring), with no reload. Daylight turns the tone Light (the Tone items stay disabled, Light selected, the `automatos-play-tone` cookie says `light`); Ember and Obsidian are Dark.

31. [ ] **Obsidian with a cobalt Auto.** Pick Obsidian and set Auto's shell to Cobalt.
    - *Expected:* Auto's body is tinted blue in the preview and in the mascot column, in the clip and in the still; his eyes and ring keep the light's colour.

32. [ ] **Change every option.** Change the accent, the background, the type, Auto's shell, Auto's light and the glow (drag it to 0, then up to 40).
    - *Expected:* each change applies to the whole app at once; at 0 the glow reads "Off" and the page glow is gone.

33. [ ] **Save, Save as, Rename, Delete.** From an edited preset use Save as; rename the new look; save an edit to it; delete it.
    - *Expected:* Save as adds "My look 1" to the row and picks it; Rename and Save change it; Delete removes it, and deleting the active look returns the app to Ember. A preset can't be renamed, deleted or saved over.

34. [ ] **Export.** Save a look again and Export it.
    - *Expected:* the notice says "Your browser is saving `<name>.look.json`. If it didn't appear, use Copy JSON." and the browser downloads `<name>.look.json`. Opened, it holds exactly eleven fields (`schema`, `id`, `name`, `sub`, `interface`, `ground`, `accent`, `font`, `shell`, `light`, `glow`): catalogue keys and a number, no colours.

35. [ ] **Import.** Delete that saved look, then import the file; then paste the same JSON and import it; then paste it with `"accent": "#ff0000"`; then with `"name": "<b>hi</b>"`.
    - *Expected:* the file comes back as a saved look under a new id (`look-N`); the paste adds a second copy; the bad accent shows `"accent" must be one of the catalogue's accents: molten, cream, mint, gold, periwinkle, rose.` and adds nothing; the `<b>` name shows as plain text, never bold. A file of 1 MB or more is refused with "That's 1 MB or more. A Look file is well under 1 KB: check it's a .look.json file."

36. [ ] **Reload keeps the active look.** With a saved Daylight-based look active, reload with the cache off and Slow 4G; then make an unsaved edit and reload again.
    - *Expected:* the page paints straight in the look, never a frame of Ember's dark ground first; the unsaved edit survives the reload too. Make it yours shows only its heading until the look is read, never Ember's card pressed first.
    - *Also (F5, M25):* with Make it yours open in two tabs, pick another look in one: the other follows at once, its tone too, and Save as new look in either keeps both tabs' looks. Signed in as a second account in this browser (SaaS), the Looks row has none of the first account's looks and the page opens in Ember; back on the first account, its looks return.

37. [ ] **No storage (optional).** In a private window with site data blocked, switch to Play.
    - *Expected:* Play opens in Ember and Make it yours still re-themes the app; Duplicate, Rename and Import add "This browser isn't keeping looks, so it lasts until you close this tab." to their notice; nothing is kept after a reload.

## G. The style sheet

38. [ ] **The live style sheet.** Open `/looks/style` (from Make it yours, or directly).
    - *Expected:* the active look's colour, type, scale, motion and components, and Auto's 12 states; with Reduce motion on, each state is its PNG. Pick another look on `/looks` and come back: the swatches follow. Where the sheet shows the look's computed values, they are the engine's: in Ember text on bg 16.36:1, secondary text on bg 8.54:1, ink on molten 6.74:1; in Daylight the bg `#F4EEE8`, accent as text `#AA4C1C`, error as text `#A54B49`.

## H. A Studio page in Play's frame

39. [ ] **Pages without a Play design.** From the sidebar open the menu items "Command Centre" (`/command-center`), "Agent Management", "Tools & Integrations", "Knowledge Base", "Marketplace" and "Analytics", in Ember, then at least one again in Daylight.
    - *Expected:* each is Studio's page inside Play's frame (Play's sidebar and header row), recoloured by the look: the page in the ground's background, cards in its surface, borders and secondary text from the ground, primary buttons in the accent (molten with dark ink in Ember; darkened molten `#AA4C1C` with the ground's bg as ink in Daylight). Everything works: open a card, a tab, a dialog, a dropdown. Success, warning and error colours follow the tone. The theme menu's Tone items say "Set by your look".

## I. The 404

40. [ ] **Under Play.** Open `/nope`.
    - *Expected:* Play's lost screen: Auto lost, "This page wandered off", and a way back to `/chat` that works. The tab's title is "404: This page could not be found." (P260-FX-21), as it is for the plain 404 in steps 46 and 47.

## J. Automatos is asleep

41. [ ] **Stop the backend.** With Play open on any page: `docker compose -p <your stack's project> stop backend`.
    - *Expected:* not on the first failed `GET /health`: the health poll (up to 30 s) fails, it is asked again three times 2.5 s apart (DevTools > Network: four failed `/health` requests), and only then does the connection pill say "Not running" and Play's frame show the asleep screen in place of the page: Auto sleeping, "Automatos is asleep", the detail, Check again and the recheck countdown. Until then the page stays as it was, with anything typed in it. Each ask gives up after 5 s, so a backend that takes the connection and never answers (`docker compose -p <your stack's project> pause backend`, then `unpause`) shows the asleep screen too, at worst 27.5 s after the poll (four asks of 5 s, three waits of 2.5 s; DevTools shows each `/health` cancelled at 5 s). In the local edition the screen offers `make up` with Copy (it copies exactly `make up`) and the line "Installed with the published images? Run make up-images. No make? docker compose up -d"; in the hosted edition, "Automatos isn't answering right now…".
    - *Then:* if you opened another page in Play before stopping the backend, press the browser's Back button while the screen shows. The asleep screen is there at once; the earlier page never flashes in before it.

42. [ ] **Check again, then start it.** Press Check again while it is still stopped; then `docker compose -p <your stack's project> start backend`.
    - *Expected:* Check again makes a new `GET /health` (DevTools > Network), followed by its retries while the backend stays stopped, and the screen stays; while they run the countdown reads "Checking…", and once they have failed it reads "Checking again in 10 seconds." and counts down again. Once the backend answers, the page mounts again (on the countdown or Check again), the pill says "Connected · localhost" and Auto wakes to idle.
    - *Offline:* with the screen showing, set DevTools > Network to Offline. The line under the title reads "You're offline. Check your internet connection: this page comes back as soon as Automatos answers.", with no start command and no "Nothing's broken on your side"; back to No throttling, the edition's own line returns.

43. [ ] **A snag mid-turn.** Send `Write a 2,000-word story about a lighthouse` and stop the backend while words stream; when the asleep screen shows, start it again; then press Try again.
    - *Expected:* the reply ends in "I hit a snag" with Try again and Auto plays oops; the asleep screen follows once the health check has failed on every retry (the next poll plus about 7.5 s for a stopped backend, up to 27.5 s for one that never answers); after the backend is back, Try again answers the last message again.

## K. Back to Studio and Classic

44. [ ] **Studio.** Theme menu > Studio.
    - *Expected:* Studio exactly as before Play: the Studio shell and sidebar, Studio's own colours. In DevTools `<html>` has `studio` but not `play`, and `document.documentElement.style.cssText` holds no `--pl-` property and none of the look's Studio tokens (`--background`, `--primary`, …); the `automatos-play-tone` cookie is gone; "Customise look…" is gone; the Tone items are enabled and show your own stored tone (if you had Light before Play, it is Light again).

45. [ ] **Reload in Studio.**
    - *Expected:* no frame of Play or of the look's colours.

46. [ ] **Play's routes under Studio.** Open `/looks`, `/looks/style`, `/conversations` and `/nope`.
    - *Expected:* `/looks` and `/looks/style` show "Make it yours is part of the Play theme." with a Switch to Play button, and no redirect loop; `/conversations` sends you to `/chat` before anything paints; `/nope` is the plain 404 ("404", "This page could not be found.", a link home).

47. [ ] **Classic.** Theme menu > Classic; reload; open `/nope`.
    - *Expected:* Classic exactly as before: `<html>` has neither `studio` nor `play`, no look properties, your own tone; the same plain 404.

## L. Below 1024 px

48. [ ] **Classic and Studio narrow.** Make the window narrower than 1024 px (or DevTools device toolbar at 768 and 390 px) in Classic, then Studio.
    - *Expected:* each looks as today.

49. [ ] **Play narrow.** The same in Play: open `/chat`, `/deliverables` and `/conversations`.
    - *Expected:* no Play frame and no Play pages: the fallback chrome Studio gets, with the core's chat and Deliverables pages; `/conversations` sends you to `/chat`. The fallback takes the look's colours (the look is not width-gated). Widening back past 1024 px brings Play's frame back.

## M. The two changes cherry-picked from PRD-259 Wave 1

50. [ ] **Rounded corners (`--radius`).** In Classic and in Studio, open the theme menu and hover its items; look at buttons and cards.
    - *Expected:* the highlighted item and the `rounded-md`/`rounded-lg` controls have rounded corners (`--radius` is now 0.5rem in Classic, 8px in Studio) where they drew square before.

51. [ ] **`dark:` classes apply.** In Classic Dark and in Studio Dark, ask Auto in the core chat for a small markdown table.
    - *Expected:* the table sits on its darker panel (`dark:bg-background/40`, header row `dark:bg-background/60`) and its text reads in the theme's foreground, with no grey-on-grey.

## N. The open question

52. [ ] **§7.1: Play as the default?** Open a private window with no cookies.
    - *Expected:* the app opens in Studio Dark, as before this wave (`DEFAULT_UI_STYLE` and `APPEARANCE_DEFAULTS_VERSION` are unchanged). Then decide whether Play becomes the default for new users, and write the decision in the PR. Nothing in this wave changes it.

## O. The review fixes (P260-FX)

The 15 Oct review's fixes that only a browser can show (`scripts/ralph/prd260-review-fix-plan.md`). Each FX story that needs one adds its step here.

53. [ ] **Play's type wins over Studio's (P260-FX-01).** Under Ember, open `/looks` and in DevTools select the "Make it yours." heading (an `h1`); then pick Workshop and look again. Then open an Ember Deliverable whose text has a link and `inline code` (ask Auto: `Write a short note as a document with a link to https://example.com and the command npm run build in backticks`), and press Tab until the link has focus.
    - *Expected:* the heading's computed `font-family` is the look's display face: Instrument Serif's next/font family (`__Instrument_Serif_…`), italic, under Ember; Fredoka's (`__Fredoka_…`) under Workshop; never Newsreader, at Studio's weight 500. On the cream sheet the link is the sheet's accent (dark molten, not Studio's light navy), the inline code sits on a faint line-coloured chip and reads clearly, and the focus ring is drawn in the same sheet accent, plainly visible on the cream.

54. [ ] **Make it yours on Paper (P260-FX-19).** On `/looks` pick Daylight, then press each accent in turn, and pick a background, a type, a shell and a light.
    - *Expected:* every accent swatch has a thin grey edge, so Cream, Mint and Gold show plainly on the Paper ground, and the chosen one has a dark ring round it. The chosen background, type, shell and light, and the Daylight card in the Looks row, have a 2px edge in the dark molten of the "See the style sheet" link, never the light orange, and the glow slider is the same dark molten. Pick Ember: the edges and the slider are molten, as before.

55. [ ] **Forced colours (P260-FX-19).** Chrome DevTools > More tools > Rendering > "Emulate CSS media feature forced-colors: active" (on Windows: Settings > Accessibility > Contrast themes). Open `/looks` in Ember, then in Daylight, and Tab through it; then Tab through `/deliverables` (the type chips, a card, the "More in Deliverables" links, Load more), a Deliverable's page (the Shelf link, Download) and `/conversations` (New chat, the search box, a row); then open `/looks/style`.
    - *Expected:* on `/looks` the six accent swatches, the ground squares, the shell and light dots and the tops of the look cards keep their own colours; the chosen accent, background, type, shell and light and the active look's card each carry a ring in the system's highlight colour (the options and the card a wider border too); every swatch shows a ring when Tab reaches it. Everywhere else every control Tab reaches shows an outline, and a pressed type chip with the focus has a thicker ring than without it. On `/looks/style` the colour chips and the spacing steps keep their colours.

56. [ ] **A long look name (P260-FX-19).** Make the window 1024 px wide. On `/looks` use Save as, rename the new look to `abcdefghijklmnopqrstuvwxyz0123456789abcd` (40 characters, no space), then change one option so the status reads `<name> · edited`.
    - *Expected:* "Make it yours." keeps its full size and reads on one or two lines, never a sliver one word wide; the status beside it wraps, right-aligned, within at most 40% of the row; the look's card in the Looks row shows the whole name, broken over lines, never cut off at its edge.

57. [ ] **The first open on a laptop, Conversations' title row, the receipts' tags (P260-FX-14).** Under Ember, set the window to 1366 × 657 (DevTools device toolbar, Responsive, 1366 × 657) and press New chat; then, back at your usual window size, zoom to 125% (Cmd +), press New chat again and zoom back to 100%. Click Conversations. Then send `Search the web for tyre pressure advice and write a short note about it as a document` and read "What Auto did"; open `/looks/style` and find the receipts sample under Components.
    - *Expected:* on both first opens Auto, "Hi, I'm Auto." and "Your Automatos, the fun way" never overlap the page header, the example chips or the message box: the main column scrolls instead (scroll down to reach the chips and the box); on a tall window they still sit centred above the chips. On Conversations the "Conversations" title and New chat sit in the list's own column, New chat's right edge in line with the search box's, not across the whole main column. In "What Auto did" the done write (the document) ends with an accent "Done" tag, the web search has no tag; a refused call, when a turn has one, keeps its words in the quiet line colour with a red ✕ and a red "Not done" tag; a skipped one a muted "Skipped". The style sheet's receipts sample shows the same "Done" tag beside "Made a report: Laptop shortlist".

---

## Map: the owner's test to these steps

| `ownerTest` entry (`scripts/ralph/prd-260.json`) | Steps |
|---|---|
| 1. TESTER first: build the frontend image from the branch tip (one Docker build at a time), then the local stack from the tip | 1–4 |
| 2. Switch to Play from the theme menu; the whole app turns into Play; reload with no flash | 5–7 |
| 3. Chat with Auto and his reactions (hello, listening, thinking, searching/reading/working, celebrate, oops); Stop; a Deliverable card to Play's Deliverable page; a Mission suggestion starts a Mission as in Studio | 8–23 (hello 8, listening 10, thinking 11, searching 12, reading 13, working and celebrate 14, Deliverable card 15–16, oops 17 and 43, Stop 18, Mission 19) |
| 4. Make it yours with three looks; accent, ground, type, shell, light, glow; save, export, import back; reload keeps the active look | 29–37 |
| 5. A Studio page without a Play design inside Play's frame, recoloured and working; the Tone items say "Set by your look" | 39 (and 6) |
| 6. Back to Studio, then Classic, both as before (no look colours on `<html>`, your tone restored); below 1024 px every style looks as today | 44–49 |
| 7. The visible changes cherry-picked from PRD-259 W1: rounded corners, `dark:` classes in Classic Dark and Studio Dark | 50–51 |
| 8. §7.1: should Play become the default for new users? (not changed here) | 52 |

The PRD's other pages, beyond the owner's list: Conversations (24), the shelf and Load more (25–28), the style sheet (38), the 404 (40, 46, 47) and "Automatos is asleep" (41–43); the review fixes (53–56).
