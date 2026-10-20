# Play parity: the manual pass (PRD-261 A-17b, Q22)

The browser lane holds builtin Play and the installed Play package to the same automated assertions
(`specs/play-parity/`, see `README.md`). This list is the rest: what no lane can check (how it looks,
real devices, real Safari, real forced colours, a screen reader) and what needs the hosted (SaaS)
edition, which no lane runs. Each item is done twice, once under builtin Play and once under the
package, and the results go in the table at the end.

**Who and where** is decision D4 (Q22): the person, and the environment (a Railway preview or
staging). Items marked **SaaS** need the hosted edition; the rest also work on the local edition.

## Before you start

Write down, in the results table's header:

- the core commit (`git rev-parse HEAD` of the deployed build) and the kit commit (`KIT_REF` in
  `.github/workflows/browser-e2e.yml`);
- the environment's URL, the edition (local or SaaS), the browser and its version, the device;
- your name and the date.

Get each Play:

1. **Builtin Play first.** Theme menu (the Appearance button) › Style › Play. The style cookie
   `automatos-style` reads `theme:builtin:play`. Do every item now, in a workspace that has **not**
   installed Play's package: once it has, the menu hides the builtin entry (D1-c).
2. **Then the package.** Download `automatos.play-1.0.0.autotheme` from the `play-package` artifact
   of a green browser-e2e run (or build it in the kit at `KIT_REF`: `npm ci --ignore-scripts`, then
   `npm run build` and `npm run pack` in `themes/play`). Settings › Appearance › Themes › Install
   theme, upload it, opt in to its private code, choose it. The style cookie reads
   `theme:install:<uuid>` and `<html>` has `data-theme-host="active"`. Do every item again.

Screenshots: name each `<item>-builtin.png` and `<item>-package.png` (for example
`A1-chat-builtin.png`), attach them to the wave's pull request (or the Q22 issue), and put the
link in the table. Don't commit them to this repo. Use the test workspace's own data only: no
customer data in a screenshot.

## A. What automation can't see

| # | What to do | What to see |
|---|---|---|
| A1 | At 1440 px, take the same screenshot under each Play: the chat on first open; a reply with tool lines and "What Auto did"; the shelf; one Deliverable; Conversations; Make it yours in Ember and in Daylight; the style sheet; a 404 (`/nope`); the asleep screen (stop the backend). | The two Plays are the same page: layout, type, colours, the page glow, Auto's ring. Any difference is a finding, with both screenshots. |
| A2 | Open the chat in real Safari (macOS and iOS) and in Chrome; send a message; open the style sheet and press "Play Auto's states". | Auto plays with a transparent background (HEVC alpha in Safari, VP9 alpha in Chrome), never a black box; Pause Auto stops him and he stays paused after a reload; with Reduce motion on (system setting) he is a still and "Play Auto's states" is off. |
| A3 | Make it yours: pick each Type (Editorial, Clean, Characterful, Playful). DevTools › Network › Font. | Each face renders in its own style. Under the package the woff2 files come from `/api/themes/files/<digest>/assets/fonts/`; under builtin Play from the app's own font files (`/_next/static/media/`). |
| A4 | On Windows, turn on a Contrast theme (Settings › Accessibility › Contrast themes). Open `/looks` in Ember, then Daylight, and Tab through it; then the shelf and Conversations. (FX-19, M19) | Every focused control has a visible outline; the chosen option, card, accent and light are marked; the colour swatches keep their own colours. The lane emulates forced colours in Chromium only. |
| A5 | VoiceOver (Cmd F5): import the same look twice; stop and start the backend; open `/looks/style`. (FX-22, FX-24) | Each import is announced, also the second; "Automatos is asleep", then "Automatos is back"; the style sheet's 12 Autos are not announced as live status. |
| A6 | Send a message that runs a web search, then one that makes a Deliverable. | The glow behind the page follows Auto: brighter while he works, brightest when he celebrates, coral on a snag. |
| A7 | On a real phone and a tablet (below 1024 px), under each Play: open `/agents`, `/chat`, the theme menu. | The core's own chrome and pages, in the look's colours; the Tone items rest under "Set by your look"; no Play frame. |
| A8 | With a real model: ask Auto for something that needs a grant (for example "delete the document X"), approve it; ask for a Mission, approve its plan. | The approval cards show in Auto's reply; Approve & run runs the action and the card says so; the Mission starts. The lane checks these cards with recorded turns only. |

## B. SaaS only

### B1. An expired session is `auth`, not a theme failure (A-08b)

The files proxy (`app/api/themes/files/[...path]/route.ts`) answers a signed-out request with 401,
and the theme host reports the reason `auth`, which never counts toward the three-failure revert
(A-04c). No lane runs the SaaS edition, so this is checked here.

1. Sign in, choose the installed package, and check `<html data-theme-host="active">`.
2. End the session without reloading the tab: Clerk Dashboard › Users › your user › Sessions ›
   Revoke (or wait past the instance's session lifetime).
3. In the same tab, reload the page. Then do it twice more within ten minutes.

What to see, each time:

- Studio shows, with the notice "The theme could not start, so Studio is showing instead." and
  "Your session was refused. Sign in again to load it.";
- `<html data-theme-host="failed" data-theme-host-reason="auth">` (DevTools › Elements);
- localStorage `automatos-theme-failures` has no entry for the install (DevTools › Application);
- after the third time there is **no** "This theme failed three times in ten minutes" notice, and
  once you sign in again, `GET /api/themes/active` still names the install as your choice and Play
  loads.

If the page sends you to sign-in before the theme host runs, note it: then the `auth` path is not
reached, and the item needs a different way to expire the token (tell the owner).

### B2. The step-up prompt (A-07c, A-07f)

Sensitive routes need, in SaaS, a sign-in from the last `STEP_UP_MAX_AGE_MINUTES` minutes (10 by
default, `orchestrator/config.py`). Sign in, wait more than that, then, as a workspace admin:

1. Settings › Appearance › Themes › Install theme, upload the package. Clerk's "confirm it's you"
   prompt opens. Confirm it: the upload completes once (one install in the list, not two).
2. Upload again and cancel the prompt: nothing is installed, and the page says it wasn't done.
3. Within ten minutes of confirming, opt in, set the default and save the workspace theme settings:
   no prompt.
4. A-07f: wait again, then change a member's role (Team), invite someone (Team › Invite) and create
   an API key (Settings › API keys). Each opens the same prompt; cancelling changes nothing.
5. As a member without manage rights: Install theme is not offered (A-07d: such members see the
   list and the choice only), and choosing a theme never prompts.

What to see: the prompt only where listed, one request completing after it, nothing done after a
cancel. Record any prompt that loops or never appears (risk R5: the instance's tokens may not
carry `fva`).

### B3. Two devices, one choice (A-02c)

The server holds the theme choice (PA-12); the style cookie only caches it. The lane checks this
with two browser contexts on one machine (`specs/theme-choice.spec.ts`); here it is two devices.

1. Sign in as the same user, in the same workspace, on device A and device B (two machines, or a
   phone and a laptop).
2. On A, choose the installed package in the theme menu. On B, reload.
3. On B, choose Studio. On A, reload.
4. On B, open a private window and sign in.

What to see: in step 2, B reloads once more by itself and lands on Play; in step 3, A lands on
Studio; in step 4, the first load lands on the choice the server holds. No tab reloads more than
once by itself, and none loops (risk R16).

## C. The gaps the automated pass found (PRD-261 A-17b)

These tests keep their assertions and are `test.fixme` for the package only (builtin Play runs
them), each naming its gap (`GAP_*` in `support/play-parity.ts`). Until the owner fixes them, the
package is **expected to fail** these checks by hand too; record what you see, don't count it as a
new finding.

| Gap | Where in core | What the package does differently | Tests marked fixme for the package |
|---|---|---|---|
| G1 | `contexts/ui-style-context.tsx`: `activeTheme` is `findThemeEntry(style)` with no installed entries, so under `theme:install:*` `useActiveTheme()` is null | No look runtime mounts (`components/theme-host/theme-look-runtime.tsx` `declaresLooks(null)`): the SDK's looks area answers `available: false`, so Make it yours can't pick, save, rename or delete a look; no look forces the tone; the theme menu shows no "Set by your look" and no Customise look…; ⌘K lists none of Play's pages. The package does keep the look its boot painted (the default look, Ember): until W-A6 the switched-off LookProvider cleared it on hydration and the package drew with no look colours at all; check that the colours stay after the page settles | FX-19 Daylight; FX-19 long name; FX-22 This look's toolbar; below 1024 px the tone forced |
| G2 | `orchestrator/services/themes/serve.py` `_boot`/`theme_boot`: `lookStorage` and `lookGuard` are null; `lib/theme-host/boot.ts` reads the tone cookie for builtin boots only; `lib/theme-host/boot-sanitize.ts` `STORAGE_KEY` refuses the `/` in `automatos-theme.<publisher>/<name>.*` keys; the look runtime keeps every theme's looks under builtin Play's keys | A stored look and its tone are not painted before first paint: the first frame is the package's default look (Ember) whatever the user chose | below 1024 px a stored look from the first frame |
| G3 | `lib/theme-host/builtin/play-only-route.tsx` `playStyleFor` picks the install; `contexts/ui-style-context.tsx` `setStyle` reloads for an installed style (PA-13) | Switch to Play reloads the page, so focus is not handed to the screen's heading (FX-22, M21) | FX-22 Switch to Play |

## Results

Core commit: ______ · Kit commit: ______ · Environment: ______ · Edition: ______ · Browser and
device: ______ · Person: ______ · Date: ______

| Item | Builtin Play | Package | Screenshot (builtin) | Screenshot (package) | Notes |
|---|---|---|---|---|---|
| A1 | | | | | |
| A2 | | | | | |
| A3 | | | | | |
| A4 | | | | | |
| A5 | | | | | |
| A6 | | | | | |
| A7 | | | | | |
| A8 | | | | | |
| B1 | | | | | |
| B2 | | | | | |
| B3 | | | | | |
| C (G1–G3 by hand) | | | | | |
