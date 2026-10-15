# PRD-260 Play theme — review fix plan

Lead-review merge of the ultra review (`scripts/ralph/prd260-ultra-review-salvage.json`): 166 confirmed findings (`real: true`) and 14 rejected. Code read on `feat/prd-260-play-theme` at 57e5cbe35 in `.worktrees/automatos-ai/feat+prd-260-play-theme`; the findings were made on bbd6254d4, and 57e5cbe35 changes only three test files, so every cited line still holds. Diff base d7affad605. Rules: that worktree's AGENTS.md (functions ≤ 50 code lines, components ≤ 150, no edits inside an over-limit function, apiClient only, named constants, replace don't shim, a test that fails before and passes after) and the PRD-260 gates (N3 frontend only, N8 layout script, N16 globals.css doesn't grow, N19 --pl-* only). Nothing was executed; severities of the highs were re-checked against the code. `#n` is the item's index in the salvage file's `items`; `Mnn` is a merged finding; paths are under `frontend/` unless they start with `docs/` or `scripts/`.

## Summary

| Severity | Confirmed findings | Merged findings | In fix stories | Founder decision (A) | Docs only (B) | Pre-existing on main (C) |
|---|---:|---:|---:|---:|---:|---:|
| critical | 0 | 0 | 0 | 0 | 0 | 0 |
| high | 18 | 6 | 5 | 1 | 0 | 0 |
| medium | 70 | 50 | 37 | 7 | 2 | 4 |
| low | 78 | 51 | 38 | 4 | 7 | 2 |
| **total** | **166** | **107** | **80** | **12** | **9** | **6** |

Fix stories: 28 (P260-FX-01 to P260-FX-28): 4 high, 19 medium, 5 low. Founder decisions: 14 (F1 to F14). Two highs were downgraded on re-reading (M07, M08); the 18 confirmed highs merge into 6 high entries.

**Merge readiness:** not ready — six high findings stand (five fixable now in P260-FX-01 to FX-04, the sixth, the missing withdraw offer, waits on decision F1), and two medium stories change what Classic and Studio users get (P260-FX-05 now-live dark: regressions, P260-FX-23 Play code in every user's bundle), so Gerard should decide which mediums must land before merge.

### Highs, re-checked

| Merged | Final | Verdict | Pre-existing on main? | Why |
|---|---|---|---|---|
| M01 (#17, #32, #44, #48, #59, #102, #49, #50, #101, #58) | high | Confirmed | No. The `.studio` element rules are on main (base globals.css:353, :869, :3043; PRD-260 adds only the two --radius lines); the conflict is new because PRD-260 renders `<html class="studio play">` (lib/ui-style.ts:52) and Play's module rules are single classes. | Re-read globals.css:355-373, :871-877, :3040-3045 and ui-style.ts:49-53; Tailwind 3.3.3 with plain postcss emits `@layer` unlayered. 26 Play h1/h2/kbd/code elements carry single-class module styles, and the sheet's `.prose a`/`code` lose to `.studio .md-view a` and `.studio code:not(pre code)`. The look's Type axis never reaches a heading, sheet links read about 2.1:1 and inline code on dark grounds about 1.1:1, against the PRD's AA goal. |
| M02 (#127) | high | Confirmed | No. Studio's chat.tsx:251 mounts the lane; Play's chat (new) replaces it at /chat. | Only chat.tsx and command-center-shell.tsx call useBoardEventStream; PlayChat (play-chat.tsx) doesn't. A reload or a Deliverable's Open mid-turn leaves the composer blocked for 5 minutes and the reply unmerged. |
| M03 (#20, #128) | high | Confirmed | No (use-play-conversation.ts is new). | useActiveView's `adopt` sets `adopted.current` on every chat-id frame and nothing clears it; after New chat, Recent → X relabels the empty draft as X. The next message goes into a history the user can't see. |
| M04 (#70, #110) | high | Confirmed | No (use-start-chat.ts is new). The store's habit of scheduling a server save before it has a key is on main, but no caller hit it before. | useStartChat calls newDraft() with no hydration; hydrate() then overwrites the draft, and in SaaS apply() has already scheduled a PUT of an empty tab list that wins the merge by updatedAt. |
| M05 (#93, #158) | high | Confirmed | No (lib/mascot is new). The reference app has the same enqueue but never dispatched deliverable-added from its chat. | enqueue() dedupes only against the queue; deliverable-added plus the done receipt queues celebrate behind itself and the finished player never reports clip-ended again. Hit on the flagship 'make me a document' turn. |
| M06 (#25, #10) | high | Kept high (one verifier raised it from medium); scope goes to the founder (F1) | No (auto-reply.tsx is new). | Play's AutoReply never reads message.withdrawOffer or taskCards. With M46 (cards lost on navigation), a pending grant has no way to be denied from Play once its card is gone: the F091-C2 safety fix is defeated for Play users. |
| M07 (#33) | medium | Downgraded from high | Yes, the mechanism: on main a Classic↔Studio switch followed by Command Center's Refresh drops `dark` the same way. PRD-260 adds Play as a third trigger and makes every dark: class depend on it. | Needs a client-side style switch and then the one router.refresh() call site (command-center-shell.tsx:195) in the same page session; Clerk's refresh path is gated off. A reload repairs it. Visible and persistent until then, but narrow. |
| M08 (#15) | medium | Downgraded from high; needs a backend change (F7) | Yes, the root: no server-side retraction exists on main. On main Studio's regenerate mostly sent nothing (a stale reload); PRD-260's retry fix makes it send, and Play's Try again is new. | The new answer arrives and nothing is lost; the cost is a duplicated question (and an error note) in the stored history and in the model's context. The fix needs a new backend route, outside K11's frontend-only wave. |

### Story order

| Story | Severity | Title | Closes | Depends on |
|---|---|---|---|---|
| P260-FX-01 | high | Play's own type, inline code and sheet colours win over Studio's element rules | M01, M16 | — |
| P260-FX-02 | high | Play's chat listens on the chat_changed lane | M02 | — |
| P260-FX-03 | high | New chat and adopted conversations open what the owner picked | M03, M04, M12 | — |
| P260-FX-04 | high | Auto's reducer: one celebration per turn, never stuck on a one-shot | M05, M101, M102, M47 | — |
| P260-FX-05 | medium | Now-live dark: classes stop regressing Classic and Studio | M41, M107 | — |
| P260-FX-06 | medium | Theme plumbing: &lt;html&gt; classes survive root re-renders, Play survives blocked storage, the Tone items behave | M07, M87, M85 | — |
| P260-FX-07 | medium | The conversation on screen survives remounts and races | M09, M10, M11 | P260-FX-02, P260-FX-03 |
| P260-FX-08 | medium | Approval cards outlive the chat screen (T-005c: 'exactly as in Studio') | M46, M77, M80 | P260-FX-07 |
| P260-FX-09 | medium | Stored replies: saved turn errors, background posts, Try again, Deliverable links | M29, M30, M76 | — |
| P260-FX-10 | medium | History caches are per account and fresh; ?draft= is one plain line | M45, M44, M56 | P260-FX-07 |
| P260-FX-11 | medium | Deliverables routing under Play follows K8 | M13, M14 | — |
| P260-FX-12 | medium | Deliverables data stays on screen and Download gets the file | M37, M57, M38, M49, M82 | P260-FX-11 |
| P260-FX-13 | medium | Deliverable content and labels: code blocks, raw HTML, provenance, captions, sources | M36, M39, M104, M95, M81 | P260-FX-01 |
| P260-FX-14 | medium | Chat layout fidelity: first open on laptop heights, the Conversations header, receipts' Done tag | M31, M35, M34 | P260-FX-01 |
| P260-FX-15 | medium | Auto's player: blocked autoplay, the boundary race, the style sheet's 12 clips, clip caching | M26, M27, M23, M75 | P260-FX-04 |
| P260-FX-16 | medium | Auto follows the model's reasoning, turn caps and the right tool kind | M54, M100, M103 | P260-FX-04 |
| P260-FX-17 | medium | Studio tokens under a look keep their meaning and reach AA on Studio's own fills | M52, M53 | —; blocked: F9 for M52 (M53 can land alone). |
| P260-FX-18 | medium | Play's accent tints and page glow are inside the AA proof | M50 | P260-FX-17 |
| P260-FX-19 | medium | Make it yours: selection and colour samples readable on Paper and in forced colours; long names | M18, M19, M73 | — |
| P260-FX-20 | medium | Looks: tabs stay in sync, failed writes are told, the editor waits for the stored look, exports are honest | M25, M98, M72, M71, M99 | —; blocked: If F5 makes looks per user, the key scoping lands in this story (same files). |
| P260-FX-21 | medium | Asleep and 404 edges: /health timeout, first render, countdown, offline, start command, 404 title | M43, M89, M94, M83, M84, M68 | — |
| P260-FX-22 | medium | Keyboard focus survives unmounts and frame swaps | M20, M21 | P260-FX-21 |
| P260-FX-23 | medium | Load Play's code only when Play renders | M17 | P260-FX-11, P260-FX-21 |
| P260-FX-24 | low | Announcements and accessible names | M22, M74 | P260-FX-22 |
| P260-FX-25 | low | One width source for Play | M69 | P260-FX-11, P260-FX-21, P260-FX-23 |
| P260-FX-26 | low | Small wiring: the Discuss hint, the widget's page, ⌘K's Play pages | M67, M70, M90 | — |
| P260-FX-27 | low | Hygiene: one /health poll owner, shared helpers and route constants, dead code | M88, M93, M105 | P260-FX-21 |
| P260-FX-28 | low | Guard and regression tests | M96, M91 | P260-FX-19 |

## Fix stories

Each story is one focused build session. 'Closes' lists the merged finding, then every source finding with its file and its own title (the distinct symptoms). Test names are vitest unless they say pytest or RUN checklist; CI is the gate.

### P260-FX-01 — Play's own type, inline code and sheet colours win over Studio's element rules

**Severity:** high

**Closes**

- **M01** (high) Studio's element rules (`.studio h1/h2`, `.studio code/kbd`, `.studio .md-view a` and `h1-h3`) outrank Play's single-class CSS-module rules, because `<html>` carries `studio` under Play: every Play heading loses the look's type, inline code on the Deliverable sheet is invisible on dark grounds, sheet links read about 2.1:1
  - #17 · high · `frontend/app/globals.css:355` — Studio's `.studio h1, .studio h2` (and `.studio kbd`/`code`) base rules beat every Play heading's CSS-module class under &lt;html class="studio play"&gt;, so Play's titles and labels render in Studio's Newsreader 500, not in the look's display type
  - #32 · high · `frontend/app/globals.css:355` — Studio's element rules (.studio h1/h2, kbd, code) override Play's own heading and label type on every Play screen; the look's Type choice never reaches page headings
  - #44 · high · `frontend/app/globals.css:355` — Studio's element rules in globals.css override Play's own type and link colours on every Play page, because &lt;html&gt; carries 'studio' under Play
  - #48 · high · `frontend/app/globals.css:355` — Studio's `.studio h1, .studio h2` rule outranks every Play heading: titles and section labels render in Newsreader, and the look's Type option never reaches them
  - #59 · high · `frontend/app/globals.css:355` — Studio's `.studio h1, .studio h2` base rule re-types every Play heading and label in Newsreader 500, so the look's Type choice never reaches h1/h2
  - #102 · high · `frontend/components/play/deliverables/shelf.module.css:33` — Headings on Play's Deliverables pages ignore the look's type: Studio's `.studio h1, .studio h2` (0,1,1) outrank Play's single-class heading rules, so the shelf title, card titles, Deliverable title and panel label render in Newsreader 500
  - #49 · high · `frontend/components/play/deliverables/deliverable-content.module.css:69` — Inline code in a Deliverable is invisible on the printed sheet under Graphite, Obsidian and Night: Studio's `.studio code:not(pre code)` paints a dark chip under dark sheet ink
  - #50 · high · `frontend/components/play/deliverables/deliverable-content.module.css:55` — Links in Deliverables and in Auto's replies take Studio's navy (`.studio .md-view a`): steel blue at 2.1:1 on the cream sheet, and report headings switch to Newsreader
  - #101 · high · `frontend/components/play/deliverables/deliverable-content.module.css:55` — Links inside a Deliverable's sheet are light navy on cream (about 2.1:1) under every dark look, which fails WCAG AA: Studio's `.studio .md-view a` outranks Play's `.prose a`
  - #58 · medium (finder: high) · `frontend/components/play/deliverables/deliverable-content.module.css:55` — Links on a Deliverable's printed sheet read at about 2.1:1, and their focus ring at 1.1 to 2.6:1, on every dark ground (Ember, Obsidian, Aurora, Workshop)
- **M16** (medium) Keyboard focus on the Deliverable sheet is drawn in the ground's accent text colour (1.06 to 2.62:1 on the sheet)
  - #39 · medium · `frontend/app/play-theme.css:153` — Keyboard focus on links in a Deliverable's page is drawn in the ground's accent text colour, 1.06 to 2.62:1 on the cream sheet (invisible under Obsidian)

**Files**

- components/play/shell/page-columns.module.css (.pageTitle)
- components/play/chat/first-open.module.css (.title)
- components/play/chat/chat.module.css (.speaker; .markdown a, code, pre, strong, ol, table, td and any heading rule)
- components/play/edges/edges.module.css (.title, .inline, .commandText)
- components/play/deliverables/shelf.module.css (.title), shelf-card.module.css (.cardTitle and its `[data-newest]` variant), deliverable-detail.module.css (.title), detail-panel.module.css (.panelLabel)
- components/play/deliverables/deliverable-content.module.css (.prose and every `.prose <element>` rule; header comment lines 7-9; `.document`)
- components/play/conversations/conversations.module.css (.groupLabel)
- components/play/shell/play-sidebar.module.css (.sectionLabel), play-frame.module.css (.kbd)
- components/looks/looks.module.css (.heading, .sectionLabel), manage.module.css (.label), style/style.module.css (.heading, .label)
- lib/__tests__/prd260-play-guards.test.ts
- docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md (one cascade step)

**Change**

1. Leave app/globals.css untouched: Studio pages inside Play's frame must keep Studio's headings (D2) and the file may not grow (gate N16). The `.studio` rules are on main; the conflict is new because PRD-260 puts `studio` and `play` on `<html>` together, and Tailwind 3.3.3 emits `@layer` content unlayered, so specificity decides.
2. Prefix with `:global(html.play)` every Play module rule whose class sits on an `h1`, `h2`, `kbd` or `code` element (26 elements: 7 h1, 16 h2, 1 kbd, 2 code; grep `<(h1|h2|kbd|code)[^>]*className={styles.` under components/play and components/looks), including any `@media` override of the same class and the newest-card variant. The selector stays pure for CSS Modules and becomes (0,2,1), which beats `.studio h1`/`.studio kbd` (0,1,1) and `.studio code:not(pre code)` (0,1,3).
3. On `kbd` and `code` rules (`.kbd`, `.inline`, `.commandText`, `.markdown code`, `.prose code`) set background, padding, border-radius and font-size explicitly, because Studio's rule sets all four.
4. Markdown surfaces: in deliverable-content.module.css prefix `.prose` itself (it ties with `.md-view` at (0,1,0) today, so the sheet's text colour depends on stylesheet order) and every `.prose <element>` rule (h1-h6, a, strong, blockquote, code, pre, hr, table/th/td, `> :first-child`); in chat.module.css prefix every `.markdown <element>` rule. (0,2,2) beats `.studio .md-view a` and `h1-h3` (0,2,1). The sheet's links then take `--pl-color-sheet-accent` (already held to 4.5:1 for all 24 pairs) and the reply's links `--pl-color-accent-text`. Do not add `--navy` to the look's Studio tokens: that would recolour Studio pages' info states.
5. Correct deliverable-content.module.css's header comment: there are no cascade layers in Tailwind 3 output.
6. M16: add `.document { --pl-focus-outline: 2px solid var(--pl-color-sheet-accent); }`, so the zero-specificity base focus rule draws a sheet-readable ring on everything focusable on the sheet.

**Proving test (fails before, passes after)**

- New guard in lib/__tests__/prd260-play-guards.test.ts (vitest runs with css:false, so the cascade is proven statically): collect `<h1|h2|kbd|code className={styles.X}>` from the TSX under components/play, components/looks and components/mascot (skip __tests__), resolve each `styles` import to its .module.css, and assert every rule whose last compound targets `.X` starts with `:global(html.play)`; assert every `.prose`/`.markdown` rule whose subject is an element starts with it too. A small specificity counter checks each such Play rule beats the matching globals.css rule (`.studio h1`, `.studio code:not(pre code)`, `.studio .md-view a`). Fails today on every rule listed.
- Same file: assert deliverable-content.module.css `.document` defines `--pl-focus-outline` from `--pl-color-sheet-accent`. Fails today.
- Add a RUN-checklist step (jsdom has no cascade): under Ember and Workshop the computed font-family of /looks' h1 is the look's display face; Tab to a link on an Ember Deliverable sheet: cream-sheet accent, visible ring.

**Code shape:** CSS only plus the guard: no TSX and no over-limit function touched; every variable stays --pl-* (gate N19).

### P260-FX-02 — Play's chat listens on the chat_changed lane

**Severity:** high

**Closes**

- **M02** (high) Play's chat never mounts the chat_changed SSE lane (useBoardEventStream): an awaited (detached) reply never merges, the composer is blocked for 5 minutes, background posts never appear live
  - #127 · high · `frontend/components/play/chat/play-chat.tsx:25` — Play's chat never mounts the chat_changed SSE lane, so an awaited (detached) reply never arrives and the composer stays blocked for 5 minutes

**Files**

- components/play/chat/play-chat.tsx
- components/play/chat/__tests__/play-chat.test.tsx
- components/play/chat/__tests__/play-chat-approvals-stream.test.tsx
- every other test that renders &lt;PlayChat/&gt; without a QueryClientProvider (play-chat-turn.test.tsx, play-chat-approvals.test.tsx)

**Change**

1. Import useBoardEventStream from '@/hooks/use-board-event-stream' and call `useBoardEventStream(true)` as the first line of PlayChat, with a one-line comment pointing at chat.tsx:247-251. It goes in PlayChat, not the keyed PlayChatView, so switching conversations keeps one connection. The root providers already supply the QueryClient the hook needs.
2. Tests that render PlayChat without a QueryClientProvider mock the hook (`vi.mock('@/hooks/use-board-event-stream', () => ({ useBoardEventStream: vi.fn() }))`), as the command-center tests do.

**Proving test (fails before, passes after)**

- play-chat.test.tsx: the mocked hook is called with `true`.
- play-chat-approvals-stream.test.tsx, real useChat inside a QueryClientProvider: route the stubbed fetch for `${apiClient.getBaseUrl()}/api/v1/tasks/stream` to a ReadableStream the test controls; getChat resolves `{...STORED_CHAT, turnInFlight: true}`; expect Stop and Auto thinking; make getChatMessages return the list plus an assistant 'Here it is.' and enqueue `event: chat_changed\ndata: {"chat_id":"c1"}\n\n`; expect 'Here it is.', Send back and Auto idle. Fails today: nothing opens the stream.

**Code shape:** PlayChat stays far under 150 lines.

**Note:** Stop pressed during an awaited reply still blocks for 5 minutes: that is M48, pre-existing on main (section C).

### P260-FX-03 — New chat and adopted conversations open what the owner picked

**Severity:** high

**Closes**

- **M03** (high) The chat id adopted mid-turn is never cleared (every turn re-adopts): after New chat, reopening that conversation shows an empty first-open bound to it, and the next message lands in a hidden history
  - #20 · high · `frontend/hooks/use-play-conversation.ts:185` — Reopening a conversation after New chat shows an empty chat bound to it: the adopted chat id is never cleared
  - #128 · high · `frontend/hooks/use-play-conversation.ts:185` — The id adopted mid-turn is never cleared, so after New chat, reopening your last conversation shows an empty first-open screen instead of loading it
- **M04** (high) Play's New chat writes to the chat-session store before it is hydrated on any page but /chat: the previous conversation opens instead, and in SaaS an empty tab list is PUT to the server
  - #70 · high · `frontend/hooks/use-start-chat.ts:12` — Play's 'New chat' writes to an unbound session store on any page other than /chat: it opens the previous conversation, and in SaaS it wipes the user's server-side tab list
  - #110 · high · `frontend/hooks/use-start-chat.ts:14` — New chat (sidebar and Conversations) runs before the session store hydrates: it opens the wrong conversation, and in the hosted edition it saves an empty tab list to the server
- **M12** (medium) New chat does nothing while the current new chat is still unnamed (its first send failed)
  - #24 · medium · `frontend/hooks/use-start-chat.ts:15` — New chat does nothing while the current new chat is still unnamed (e.g. its first send failed)

**Files**

- hooks/use-play-conversation.ts (useActiveView, plus a small callee hook)
- hooks/use-start-chat.ts
- new stores/play-chat-store.ts
- components/play/shell/play-sidebar.tsx (NewChatButton)
- components/play/conversations/play-conversations.tsx
- components/play/deliverables/shelf-body.tsx
- components/play/chat/__tests__/play-chat.test.tsx
- new hooks/__tests__/use-start-chat.test.tsx

**Change**

1. M03: in useActiveView make adoption one-shot and tied to the draft on screen: `adopt` records the id only while a draft is shown (`if (shown.current === null) adopted.current = chatId`), `show()` sets `adopted.current = null`, and the consume branch clears it after use. Any later move to that id goes through loadConversation. Safe for the new-chat path: adopt(X) runs synchronously before chatIdAssigned(X), with no show() between.
2. M04: useStartChat calls `useChatSessionHydration()` (idempotent; the widget and the chat page already do) and returns `{ startChat, ready: hydrated }`; startChat does nothing until hydrated. NewChatButton, the Conversations page and the shelf's empty state render `disabled={!ready}`. Change the return shape and all three callers in the same PR (replace, don't shim).
3. M12: a fresh-draft signal the session store can't express: stores/play-chat-store.ts `{ freshDrafts: number; freshDraft(): void }`. startChat runs newDraft(), freshDraft(), router.push(CHAT_HREF). A small callee hook in use-play-conversation.ts subscribes to freshDrafts and, when it changes while activeChatId is null, shows draftView(++drafts.current): a new key, so a new useChat.

**Proving test (fails before, passes after)**

- play-chat.test.tsx (a): draft → captured onChatIdUpdate('new-1') → newDraft() → getChat/getChatMessages mocked for 'new-1' → openChat('new-1') → getChat('new-1') called, the stored message shows, no HELLO heading.
- play-chat.test.tsx (b): stored c1 open → onChatIdUpdate('c1') (a later turn's frame) → newDraft → openChat('c1') → getChat called a second time, c1's messages render.
- play-chat.test.tsx (c): a draft whose first send failed (status 'error', messages [user]) → New chat → the hero shows and useChatWithReceipts mounts again.
- hooks/__tests__/use-start-chat.test.tsx with the real zustand store: seed localStorage `{activeChatId:'abc', openChatIds:['abc']}` under chatSessionKey(ws,user); render PlaySidebar on '/deliverables'; New chat is disabled before hydration; after hydration a click gives activeChatId null, draftOpen true, openChatIds still ['abc']; with isSaaS mocked true, putChatSession never receives `openChatIds: []`. All four fail today.

**Code shape:** useActiveView is about 42 code lines (limit 50): the two adoption lines go in place, the fresh-draft effect goes in a callee. Do not edit chat-session-store.ts's create() callback (over 50 lines). The three components stay under 150.

### P260-FX-04 — Auto's reducer: one celebration per turn, never stuck on a one-shot

**Severity:** high

**Closes**

- **M05** (high) A one-shot can be queued behind itself (deliverable-added mid-turn plus the done receipt at finish): Auto freezes on 'celebrate' for the rest of the conversation, or celebrates twice for one Deliverable
  - #93 · high · `frontend/lib/mascot/reducer.ts:78` — Auto freezes on 'celebrate' for the rest of the conversation after a turn that makes a Deliverable (the same one-shot gets queued behind itself)
  - #158 · medium (finder: low) · `frontend/hooks/use-play-turn.ts:115` — One Deliverable makes Auto celebrate twice: once when the document tool files it, again at finish from the same call's done receipt
- **M101** (low) Pressing Stop after a 'done' receipt makes Auto celebrate
  - #7 · low · `frontend/lib/mascot/reducer.ts:1` — Pressing Stop after a 'done' receipt makes Auto celebrate
- **M102** (low) Hello on the first Play chat open replaces the thinking of an awaited reply
  - #159 · low · `frontend/lib/mascot/reducer.ts:218` — Hello on the first Play chat open replaces the thinking of an awaited reply; Auto then rests 'Ready when you are' while the reply is still coming
- **M47** (medium) The shelf's Auto stays on 'Something went wrong' with the coral glow after Try again loads the shelf
  - #96 · medium · `frontend/hooks/use-shelf-mascot.ts:42` — The shelf's Auto stays on 'Something went wrong' with the coral page glow after Try again loads the shelf

**Files**

- lib/mascot/reducer.ts
- lib/mascot/types.ts (MascotState, ClientEvent)
- lib/mascot/player.ts (retarget comment)
- components/play/chat/play-chat-view.tsx (Stop)
- hooks/use-shelf-mascot.ts
- lib/mascot/__tests__/reducer.test.ts
- hooks/__tests__/use-auto-mascot.test.tsx
- components/play/deliverables/__tests__/play-shelf.test.tsx

**Change**

1. M05: in enqueue(), `if (ONE_SHOTS.has(next.mascot) && next.mascot === state.mascot) return state` before the existing dedupe; add one line to retarget's comment in player.ts: the player relies on the reducer never re-targeting the one-shot on screen.
2. M05 (#158): add `celebratedThisTurn` to MascotState and INITIAL_MASCOT; the 'deliverable-added' case sets it while turnActive; onFinish proposes CELEBRATE only when `celebrateOnFinish && !celebratedThisTurn`; onSend and endTurn reset it. The shelf's deliverable-added outside a turn is unchanged.
3. M101: a ClientEvent `{ type: 'stop' }` that clears celebrateOnFinish; PlayChatView's onStop becomes a small callback that dispatches it and then calls chat.stop().
4. M102: in 'app-open' keep a pose above idle behind hello: `const pending = pendingPose(state); const greeted = propose(state, HELLO); return greeted !== state && pending.rank > RANK.idle ? enqueue(greeted, pending) : greeted`.
5. M47: a ClientEvent `{ type: 'request-ok' }`; reducePresence: `pending.mascot === 'oops' && !state.turnActive ? settle(state, rest(state)) : state`; use-shelf-mascot dispatches it when view.status becomes 'ready'.

**Proving test (fails before, passes after)**

- reducer.test.ts: `replay([SEND, {type:'deliverable-added'}, receipts('done'), FINISH, ended('celebrate')]).mascot === 'idle'`; `replay([deliverable-added, deliverable-added, ended('celebrate')])` is 'idle', and a following SEND gives 'thinking'; deliverable-added → ended('celebrate') → receipts('done') → FINISH stays 'idle' (one celebration); a done-receipts turn with no Deliverable still celebrates; receipts('done') → stop → FINISH does not celebrate; SEND → app-open shows 'hello' → ended('hello') → 'thinking' → FINISH → 'idle'; request-failed → request-ok → 'idle', and during a turn oops stays.
- use-auto-mascot.test.tsx: the freeze sequence through the hook, onClipEnded('celebrate'), rerender with status 'streaming' → 'thinking'.
- play-shelf.test.tsx: a failed first load, then Try again with data → Auto's caption is the shelf count, not 'Something went wrong'. All fail today.

**Code shape:** Reducer functions stay small. PlayChatView is about 66 lines (limit 150) and takes the onStop callback; usePlayChat is 37 of 50 lines, so add nothing there.

**Note:** The reference app has the same enqueue; it never dispatched deliverable-added from its chat, so the port's wiring made this reachable.

### P260-FX-05 — Now-live dark: classes stop regressing Classic and Studio

**Severity:** medium

**Closes**

- **M41** (medium) dark: classes made live by the darkMode cherry-pick regress dark tones in Classic, Studio and Play: text below AA, a lost focus indicator and hover, overridden borders
  - #45 · medium · `frontend/components/widgets/WorkflowWidget/StepDetail.tsx:157` — Now-live dark: classes with reduced opacity drop error and badge text below WCAG AA in Classic Dark, Studio Dark and Play's dark looks
  - #177 · medium · `frontend/components/widgets/WorkflowWidget/WorkflowSteps.tsx:118` — Now-live dark:bg-blue-950/20 removes the running workflow step's only keyboard-focus indicator, and its hover, in every dark tone
  - #47 · low · `frontend/components/chatbot/sheet-artifact.tsx:96` — In dark tones the now-live dark:bg/dark:border classes cancel the hover state of SheetArtifact's clarification options
  - #178 · low · `frontend/components/chatbot/text-artifact.tsx:75` — text-artifact's model chip drops from AA to about 3:1 in Classic Dark and Studio Dark (the audit kept dark:text-info/70)
  - #179 · low · `frontend/components/ui/alert.tsx:13` — Alert's now-live dark:border-destructive overrides the softer borders that the sign-in, sign-up and reset-password error alerts ask for
- **M107** (low) The dark: guard catches only neutrals and hex, so palette classes the audit rules 'fix' pass
  - #132 · low · `frontend/lib/__tests__/dark-variant-tokens.test.ts:23` — The dark: guard catches only neutrals and hex, so coloured palette classes that the audit itself rules 'fix' pass

**Files**

- components/widgets/WorkflowWidget/StepDetail.tsx (:157, :160)
- components/widgets/WorkflowWidget/index.tsx (:145)
- components/widgets/WorkflowWidget/WorkflowStatus.tsx (:61, :80)
- components/widgets/MemoryWidget/MemoryItem.tsx (:39, :45)
- components/chatbot/text-artifact.tsx (:75)
- components/ui/alert.tsx (:13)
- components/widgets/WorkflowWidget/WorkflowSteps.tsx → new workflow-step-row.tsx
- components/chatbot/sheet-artifact.tsx → new components/chatbot/sheet-clarifications.tsx, rendered from artifact-viewer.tsx
- lib/__tests__/dark-variant-tokens.test.ts
- docs/PRDS/PRD-259-DARK-VARIANT-AUDIT.md (rows 128, 152, 178 and the rows touched)

**Change**

1. Reduced-opacity dark text gets the foreground and lets the tint carry the colour: StepDetail :157/:160 and WorkflowWidget index :145 → `text-destructive dark:text-foreground`; WorkflowStatus :61/:80 and MemoryItem :39/:45 → `dark:text-foreground` in place of `dark:text-info/80`, `dark:text-destructive/80`, `dark:text-agent/80`; text-artifact :75 drops `dark:text-info/70` (text-info on info/10 is 4.54 Classic Dark, 5.70 Studio Dark).
2. WorkflowSteps: the 85-line `steps.map` callback is over the function limit, so move its body into WorkflowStepRow (pure deletion plus one JSX line). In the row, the running state is `bg-info/10` with no dark: variant, and focus is `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring`, hover:bg-muted/50 kept.
3. SheetArtifact is about 236 lines (over 150): pure-delete its clarification block (87-118) into SheetClarifications({ metadata }), rendered by ArtifactViewer above &lt;SheetArtifact&gt; for kind 'sheet'; the option button keeps `border-border/60 bg-card/50 text-foreground hover:border-warning/40 hover:bg-warning/5` with no bare dark: overrides.
4. alert.tsx:13: delete `dark:border-destructive` (the base border-destructive/50 is the token and follows every tone; tailwind-merge can't remove a dark: class from a call site).
5. M107: widen FIXED_DARK_CLASS to every palette hue and arbitrary colour values; fail on `dark:text-<token>/<n>` with n &lt; 90; fail on a class string with a bare `dark:<prop>-` beside `hover:|focus:<prop>-` and no `dark:hover:<prop>-`; add the audit's blocked rows to ALLOWLIST with their verdicts; extend the self-test with `dark:text-amber-500` and `dark:bg-[rgb(0,0,0)]`, keeping `dark:text-warning` passing.
6. Optional, same lines (pre-existing M86, section C): `bg-warning text-warning-foreground` at MemoryItem.tsx:57 and WorkflowStatus.tsx:86, plus a guard against `bg-X text-X` on one semantic token.

**Proving test (fails before, passes after)**

- The widened guards fail today on these classes and pass after.
- workflow-steps test: a running step's row button has `focus-visible:ring-2` and no `dark:bg-` class; SheetClarifications renders the questions and copies one on click; `<Alert variant="destructive" className="border-destructive/20">` keeps border-destructive/20 and has no `dark:border-` class.
- A contrast test that reads the `.dark` and `.studio.dark` triplets from globals.css: text-foreground over destructive/info/agent at 20% on --card, and text-info over info/10, are ≥ 4.5:1.

**Code shape:** SheetArtifact and the WorkflowSteps map callback are over the limits: extraction by pure deletion only. StepDetail (109), WorkflowStatus (105), MemoryItem (61), TextArtifact (121) and ArtifactViewer (116) are under 150 and take line edits.

**Note:** The class strings are on main, but they generated no CSS there; PRD-260's cherry-picked `darkMode: ['class', '.dark']` (D9) makes them render, so these are PRD-260 regressions for Classic and Studio users.

### P260-FX-06 — Theme plumbing: &lt;html&gt; classes survive root re-renders, Play survives blocked storage, the Tone items behave

**Severity:** medium

**Closes**

- **M07** (medium) A root re-render (router.refresh) after a client-side style switch rewrites `<html class>` from the server string and drops the tone class
  - #33 · high (finder: medium) · `frontend/app/layout.tsx:55` — router.refresh() or a server-action re-render after a client-side style switch rewrites &lt;html class&gt; from the server string and drops the tone class
- **M87** (low) With localStorage unreadable, choosing Play flips straight back to Studio
  - #143 · low · `frontend/contexts/ui-style-context.tsx:82` — Choosing Play in a browser whose localStorage is unreadable flips straight back to Studio; step 37 and play-theme.md say Play runs in Ember there
- **M85** (low) Under Play the theme menu's Tone is locked below 1024 px (where /looks can't open) and its items are unreachable by keyboard
  - #75 · low · `frontend/components/ui/theme-toggle.tsx:96` — Below 1024 px under Play the look still forces the tone, and the only place to change it refuses to open, so a phone user can't get Light or Dark without leaving Play
  - #147 · low · `frontend/components/ui/theme-toggle.tsx:95` — Under Play, keyboard and screen-reader users never reach the Tone section of the theme menu or its 'Set by your look' note

**Files**

- app/layout.tsx
- lib/ui-style.ts (STYLE_CLASS_SCRIPT)
- scripts/ralph/acceptance-prd260.sh (gate N8)
- contexts/ui-style-context.tsx
- components/providers.tsx
- components/ui/theme-toggle.tsx
- components/__tests__/prd260-theme-menu.test.tsx
- lib/__tests__/ui-style-play.test.ts
- contexts/__tests__/ui-style-context.test.tsx

**Change**

1. M07: React stops owning the style classes. `<html className>` carries the font variables only, a value identical on every render. A constant `STYLE_CLASS_SCRIPT[uiStyle]` (built at module load from styleHtmlClasses with JSON.stringify; its only input is the parsed enum, never cookie text) adds and removes exactly STYLE_HTML_CLASS_NAMES in `<head>`, before LOOK_PREPAINT_SCRIPT (which needs `play`). A server re-render then changes only a script's text, which never runs again. Gate N8 must allow this second constant. If a second script is refused, the fallback is a small hook in components/theme-provider.tsx that observes documentElement's class and re-adds `forcedTheme ?? resolvedTheme` when React drops it.
2. M87: the once-per-mount defaults effect in UiStyleProvider gets a `useRef` guard, so next-themes rebuilding setTheme when forcedTheme changes can't re-run it and flip a storage-blocked browser back to Studio.
3. M85: the forced tone is desktop-only like the rest of Play (D7): Providers passes `forcedTheme={narrow ? undefined : playTone}`; ThemeToggle computes `toneLocked = isPlay && !narrow` for the note, the item state and 'Customise look…'. Under a lock the Tone items stay focusable but inert: no `disabled`, `aria-disabled`, `onValueChange` undefined, `onSelect` preventDefault, and each item's aria-describedby points at the 'Set by your look' note.

**Proving test (fails before, passes after)**

- Render RootLayout with mocked cookies() for 'studio' and for 'play': the &lt;html&gt; className is identical. Run STYLE_CLASS_SCRIPT['play'] in jsdom on `className = 'dark'`: classList is ['dark','studio','play']. prepaint.test keeps passing.
- prd260-theme-menu.test.tsx with the real next-themes ThemeProvider, UiStyleProvider and LookProvider: localStorage getItem/setItem throw, choose Play → &lt;html&gt; keeps 'play' and the style stays 'play'.
- Same file: useIsTabletOrBelow mocked true under Play → Tone items enabled, no 'Set by your look', no 'Customise look…', choosing Dark sets the stored tone; at desktop width ArrowDown reaches the 'Dark' menuitemradio, which has aria-disabled='true' and the description 'Set by your look', and Enter leaves the stored tone unchanged. All fail today.

**Code shape:** RootLayout about 35 lines; ThemeToggle about 62; Providers about 41: all under 150.

**Note:** M07's mechanism is on main (Classic↔Studio switch plus Command Center's Refresh); PRD-260 adds Play as a third trigger and makes every dark: class depend on the dropped `dark` class.

### P260-FX-07 — The conversation on screen survives remounts and races

**Severity:** medium · **Depends on:** P260-FX-02, P260-FX-03

**Closes**

- **M09** (medium) Play's conversation view is held above the remount boundary: after the asleep screen, or a workspace refresh (saving a key from the trial banner), the chat comes back from a stale snapshot
  - #22 · medium (finder: high) · `frontend/components/play/chat/play-chat.tsx:26` — After 'Automatos is asleep' (any remount of the chat screen), Play's chat comes back from a stale snapshot: a named new chat shows empty, later turns and the snag/Try again vanish
  - #135 · medium · `frontend/components/play/chat/play-chat.tsx:26` — After 'Automatos is asleep' clears, Play's chat comes back showing an old transcript: a new chat reopens as the empty 'Hi, I'm Auto.' screen
  - #71 · medium · `frontend/hooks/use-play-conversation.ts:217` — In SaaS, saving a key from the trial-exhausted banner in Play's chat remounts the chat with stale messages, so a conversation started in this session shows as empty
- **M10** (medium) Play's cold start never cancels: a late history answer replaces a new chat or a discussion opened meanwhile
  - #161 · medium · `frontend/hooks/use-play-conversation.ts:128` — Play's cold start never cancels, so a late answer replaces a new chat or discussion opened in the meantime
- **M11** (low) /chat?chatId= with an empty value leaves Play's chat on 'Opening this conversation…' for good
  - #164 · low · `frontend/hooks/use-play-conversation.ts:79` — /chat?chatId= with an empty value leaves Play's chat on 'Opening this conversation…' for good

**Files**

- components/play/chat/play-chat.tsx (new PlayChatScreen in the same file)
- hooks/use-play-conversation.ts (useColdStart, useDeepLink, useDraftLink)
- hooks/use-chat-session.ts (the readiness line)
- components/play/chat/__tests__/play-chat.test.tsx
- hooks/__tests__/use-chat-session.test.tsx

**Change**

1. M09: PlayChat renders `<MainLayout fullBleed><FirstRunNudge/><PlayChatScreen/></MainLayout>`; PlayChatScreen calls usePlayConversation() and renders PlayChatView or PlayChatOpening. When AsleepGate remounts the frame, the hook remounts with it: its refs reset and the active chat reloads through getChat + getChatMessages (turnInFlight included). The ?chatId=/?draft= params have already left the URL, and cold start doesn't fire because a chat is open. Gate the returned view on `view.chatId === activeChatId` so a Recent link never shows the previous view for one render. useBoardEventStream (FX-02) stays in PlayChat.
2. M09 (#71): hooks/use-chat-session.ts keeps the key while a known workspace refetches: `ready = isLoaded && (workspaceCtx === null || !workspaceCtx.isLoading || workspaceCtx.workspace != null)`. A refresh that returns the same workspace never drops `hydrated`; a different workspace still rehydrates. This blip is on main; the fix also stops the same flicker for Studio and the widget.
3. M10: useColdStart's then/catch act only while `useChatSessionStore.getState().session` still has no open chat and no draft, as app/chat/page.tsx's cancelled guard does.
4. M11: normalise once, `const linked = params?.get(CHAT_ID_PARAM)?.trim() || null`, shared by useDraftLink and useColdStart; drop a blank chatId with router.replace.

**Proving test (fails before, passes after)**

- play-chat.test.tsx with a controllable useBackendHealth mock: adopt 'c-1' with two server messages, connection down (asleep screen) → connected: getChatMessages('c-1') was called, both messages render, no "Hi, I'm Auto.".
- A stored chat whose getChatMessages mock grows between the down/up toggle shows the new messages.
- Workspace provider isLoading true → false with the same workspace: 'Opening this conversation…' never renders and the mocked useChatWithReceipts mounted once.
- getChatHistory returns a deferred promise; newDraft() before it resolves → activeChatId stays null, openChat not called, the first-open screen shows.
- Mount with `?chatId=` → first-open or cold-start conversation, not OPENING. hooks test: hydrated stays true across a same-workspace refresh. All fail today.

**Code shape:** useActiveView stays ≤ 50 code lines (FX-03 already edits it); usePlayConversation stays small; PlayChat under 150.

### P260-FX-08 — Approval cards outlive the chat screen (T-005c: 'exactly as in Studio')

**Severity:** medium · **Depends on:** P260-FX-07

**Closes**

- **M46** (medium) Approval cards live only inside the keyed chat screen: leaving the chat or switching conversation loses a pending card (Studio keeps it)
  - #26 · medium · `frontend/hooks/use-play-turn.ts:146` — Approval cards live only inside the keyed chat screen: leaving the chat or switching conversation loses a pending card (Studio keeps it)
- **M77** (low) An answered loose approval stays under 'Approvals still open'
  - #31 · low · `frontend/components/play/chat/approvals.ts:177` — An answered approval stays under 'Approvals still open'
- **M80** (low) 'Approvals still open' cards sit at the scroller's edge, outside the conversation column
  - #165 · low · `frontend/components/play/chat/play-chat-view.tsx:198` — 'Approvals still open' cards sit at the scroller's left edge, outside the conversation column

**Files**

- hooks/use-play-turn.ts (useLiveApprovals)
- new stores/play-approvals-store.ts
- components/play/chat/play-chat-view.tsx (useApprovalSlots)
- components/play/chat/approvals.ts (looseApprovals)
- components/play/chat/turn-end.tsx
- components/play/chat/approval-cards.tsx
- components/play/chat/__tests__/play-chat-approvals.test.tsx
- components/play/chat/__tests__/play-chat-approvals-stream.test.tsx
- docs/frontend-architecture/play-theme.md:400

**Change**

1. M46: lift the filed approvals out of the keyed screen into an in-memory zustand store (like the workspace store's widgets): entries are `FiledApproval & { chatId }`; useLiveApprovals reads and writes it through the existing pure fileApprovals and withStatus, filtered by the view's chatId; '' is re-keyed to the id on onChatIdUpdate. On remount, a card whose reply isn't in the messages renders as loose. Cleared on reload, as Studio's widgets are.
2. M77: loose cards still waiting render under LOOSE_APPROVALS_LABEL; answered ones under APPROVALS_LABEL (split by isWaitingCard).
3. M80: loose cards render inside TurnEnd's `.after` column (a `before?: ReactNode` slot), so they share the conversation column.
4. Correct play-theme.md:400 ('live only' holds across a reload, not across navigation).

**Proving test (fails before, passes after)**

- play-chat-approvals.test.tsx: raise a tool_approval, unmount PlayChat, remount it on the same conversation → the card is there and Approve & run calls grant(42). Fails today.
- play-chat-approvals-stream.test.tsx: after Stop then Approve & run, the card is no longer inside the region named 'Approvals still open'; the loose region sits inside the turn-end container (roles only).

**Code shape:** PlayChatView about 66 lines; hooks stay ≤ 50 (put store access in a small helper hook if useLiveApprovals grows).

### P260-FX-09 — Stored replies: saved turn errors, background posts, Try again, Deliverable links

**Severity:** medium

**Closes**

- **M29** (medium) Stored turn errors and background posts render as ordinary answers in Play (Studio badges them; the approved app shows a failed reply)
  - #27 · medium · `frontend/components/play/chat/auto-reply.tsx:110` — Stored turn errors and background posts render as ordinary answers in Play (Studio badges them; the approved app shows a failed reply)
- **M30** (medium) Try again / regenerate re-sends an empty message when the last user message was loaded from the server (rows carry parts, not content)
  - #29 · medium (finder: low) · `frontend/lib/chat/retry.ts:25` — Try again / Studio regenerate re-sends an empty message when the last user message was loaded from the server
- **M76** (low) The 'Deliverable card from receipts' path is dead: receipts never link a Deliverable, the port rejects the real absolute open_url, and the test uses a link shape the backend never sends
  - #130 · low · `frontend/components/play/chat/__tests__/play-chat-turn.test.tsx:243` — The receipt→Deliverable-card test uses a link shape the backend never sends; the port also rejects the real absolute open_url that the Play reference's test covered
  - #28 · low · `frontend/components/play/chat/deliverables.ts:126` — A stored reply never shows its Deliverable card: receipts never link a Deliverable, so the 'from receipts' path is dead

**Files**

- components/play/chat/message-text.ts
- components/play/chat/auto-reply.tsx
- lib/chat/retry.ts
- components/play/chat/deliverables.ts (deliverableIdFromLink)
- hooks/use-play-turn.ts (comment at :70)
- docs/frontend-architecture/play-theme.md
- components/play/chat/__tests__/play-chat-turn.test.tsx
- lib/chat/__tests__/retry.test.ts
- the deliverables.ts unit test

**Change**

1. M29: two pure readers in message-text.ts: `savedTurnError(message)` (source.origin === 'turn_error' → its text without the ⚠️ prefix, else null) and `provenanceOf(message)` (metadata.source when it isn't the error label). AutoReply: `failure = message.error?.message ?? savedTurnError(message)`, no Answer for a saved error, a provenance line (styles.notice) for background posts.
2. M30: lastTurn's content is `message.content || <joined text parts>` through a local pure helper (lib/ must not import components/play). The bug is on main in the old reload; PRD-260 moved it into retry.ts and wires it to Play's Try again, which M29 makes reachable on stored rows.
3. M76 (frontend half): deliverableIdFromLink accepts absolute http(s) links (parse with LINK_BASE, keep only the id validated by DELIVERABLE_ID; deliverableHref rebuilds the href relatively); port the reference's absolute open_url case; change the play-chat-turn fixture to the real open_url shape. Correct use-play-turn.ts:70 and play-theme.md: a stored reply shows no Deliverable card until receipts link Deliverables (backend, decision F8).

**Proving test (fails before, passes after)**

- play-chat-turn.test.tsx: a stored newest row `{source:{origin:'turn_error',label:'Error'}, parts:[{type:'text', text:'⚠️ The model is unavailable'}]}` renders 'I hit a snag' with that line and Try again, and no answer markdown; a row with metadata.source 'Auto · background' shows the label.
- retry.test.ts: a stored-shape user message `{id, role:'user', parts:[{type:'text', text:'hello'}]}` with no content gives `lastTurn(...).content === 'hello'`.
- deliverables unit test: `https://app.example/deliverables?tab=outputs&deliverable=d-7` → 'd-7'. All fail today.

**Code shape:** AutoReply is 120 lines (limit 150): if the provenance line pushes it near the limit, put it in a small StoredNote component.

### P260-FX-10 — History caches are per account and fresh; ?draft= is one plain line

**Severity:** medium · **Depends on:** P260-FX-07

**Closes**

- **M45** (medium) Play caches chat history under keys without the user or workspace, so the next account on the browser sees the last one's conversations
  - #21 · medium · `frontend/hooks/use-conversation-list.ts:35` — Play caches chat history under keys without the user or workspace, so the next account on the browser sees the last one's conversations
- **M44** (medium) Conversations and Recent stay stale after a turn (no invalidation; a 60 s staleTime; Recent keyed by the active chat)
  - #12 · medium (finder: low) · `frontend/hooks/use-conversation-list.ts:34` — Play's Conversations list (and Recent) stay stale after a turn: no invalidation, and the root QueryClient's 60 s staleTime keeps the old list
  - #111 · medium · `frontend/hooks/use-recent-chats.ts:24` — Recent and Conversations show a stale history: a conversation just started drops out of Recent, and Conversations can miss it for up to a minute
- **M56** (medium) ?draft= lets any link put hidden multi-line (or bidi) text in the owner's message box
  - #19 · medium · `frontend/lib/play/draft.ts:49` — ?draft= lets any link put hidden multi-line text in the owner's message box, which one Enter sends to Auto

**Files**

- hooks/use-chat-session.ts (useChatScope)
- hooks/use-conversation-list.ts
- hooks/use-recent-chats.ts
- new hooks/use-refresh-history-on-turn-end.ts
- components/play/chat/play-chat-view.tsx
- lib/play/draft.ts
- new lib/text/plain-line.ts
- lib/looks/validate.ts
- new hooks/__tests__/use-conversation-list.test.tsx and use-recent-chats tests
- lib/play/__tests__/draft.test.ts
- components/play/chat/__tests__/play-chat.test.tsx

**Change**

1. M45: useChatScope() returns chatSessionKey(workspace.id, user.id) once the user is loaded and the workspace settled, else null (the readiness rule useChatSessionHydration uses, without hydrating). Both history queries put the scope in their key and set `enabled: scope !== null`.
2. M44: export CHAT_HISTORY_QUERY_PREFIX (['chat-history']) instead of repeating the literal; useRefreshHistoryOnTurnEnd(status) invalidates it when the status leaves 'streaming' (called from usePlayChat, 37 → 38 lines); Recent's key becomes stable (`[...RECENT_CHATS_QUERY_KEY, scope, limit]`) with an invalidation when activeChatId changes; the conversation list sets `refetchOnMount: 'always'`.
3. M56: move UNSAFE_TEXT out of lib/looks/validate.ts into lib/text/plain-line.ts as `oneLine(text)` (unsafe characters and whitespace runs → ' ', trimmed); validate.ts imports it (replace, don't shim); lib/play/draft.ts cutDraft applies it before MAX_DRAFT_CHARS, so draftHref and draftFrom both get one plain line.

**Proving test (fails before, passes after)**

- Scope 'A' resolves [chatA]; rerender with scope 'B' and a pending second call → the hook reports loading (not chatA) and getChatHistory was called twice; the same for useRecentChats.
- Real QueryClient: drive PlayChatView's status streaming → idle → getChatHistory called again. Recent with activeChatId A → B → A inside 30 s of fake time still lists B; Conversations unmounted and remounted within 60 s after the mock switches to [B, A] shows B.
- draft.test.ts: `draftFrom(new URLSearchParams({draft: 'Do X\n\n\nAbout "Q3": '}))` is `'Do X About "Q3":'`; a '\u202Eabc' draft becomes 'abc'. play-chat.test.tsx: ?draft= with newlines → the textbox value contains no '\n'. All fail today.

**Code shape:** usePlayChat stays ≤ 50 lines (one added call).

### P260-FX-11 — Deliverables routing under Play follows K8

**Severity:** medium

**Closes**

- **M13** (medium) Every `/deliverables?tab=outputs` link, the backend's own Deliverable link included, opens Studio's Deliverables page under Play (against K8)
  - #34 · medium · `frontend/app/deliverables/layout.tsx:9` — /deliverables?tab=outputs keeps Studio's Deliverables page under Play: Explorer's Back/Esc, 'Open Deliverables' and Studio's own Outputs tab all leave Play's shelf
  - #103 · medium · `frontend/app/deliverables/layout.tsx:9` — Every '/deliverables?tab=outputs' link (Explorer's 'Back to Deliverables' and Esc, Studio's Outputs tab and 'Back to feed', GenerateDocumentDialog's 'Open Deliverables') lands Play users on Studio's Outputs page instead of Play's shelf
  - #129 · medium · `frontend/app/deliverables/layout.tsx:9` — The backend's own Deliverable link (?tab=outputs&deliverable=&lt;id&gt;) opens Studio's page and sheet under Play, not Play's Deliverable page (K8); no test uses that shape
- **M14** (low) PlayRouteSwitch reads useSearchParams outside a Suspense boundary in the Deliverables layout (latent build break)
  - #108 · low · `frontend/app/deliverables/layout.tsx:18` — app/deliverables/layout.tsx has the same unwrapped useSearchParams as the chat layout: PlayRouteSwitch sits outside any Suspense

**Files**

- components/play/play-route-switch.tsx
- app/deliverables/layout.tsx
- app/deliverables/__tests__/deliverables-layout.test.tsx
- components/play/__tests__/play-route-switch.test.tsx

**Change**

1. M13: PlayRouteSwitch takes a serializable `playServes?: Readonly<Record<string, readonly string[]>>`: values of a keepPageFor parameter that Play's screen still serves. `pageAsked = keepPageFor.some((name) => params.has(name) && !(playServes[name] ?? NONE).includes(params.get(name) ?? ''))`. The Deliverables layout passes `{ tab: [PLAY_SHELF_TAB] }` with `PLAY_SHELF_TAB = 'outputs'` named beside CORE_DELIVERABLES_PARAMS. `?tab=outputs&deliverable=<id>` (the backend's link) then shows Play's Deliverable page, which PlayShelf already renders from `deliverable`, with DeliverableDeepLink closed; `?tab=outputs&artifact_type=x` still keeps the core page through artifact_type.
2. M14: rename the switch's body PlayRouteChoice and render `<Suspense fallback={children}><PlayRouteChoice …/></Suspense>`, so both layouts are covered and a static prerender renders the route's own page.

**Proving test (fails before, passes after)**

- deliverables-layout.test.tsx: 'tab=outputs' → shelf; 'tab=outputs&deliverable=d-laptop' → Play's detail, deep link not mounted; 'tab=outputs&artifact_type=code' and 'tab=brand' → core page. Fail today.
- play-route-switch.test.tsx: a playServes value lets Play's screen show; a useSearchParams mock that suspends once renders the children as the fallback.

**Code shape:** Both files are small.

### P260-FX-12 — Deliverables data stays on screen and Download gets the file

**Severity:** medium · **Depends on:** P260-FX-11

**Closes**

- **M37** (medium) The open Deliverable is replaced by an error (or 'may have been deleted') when a background refetch fails
  - #105 · medium · `frontend/components/play/deliverables/play-deliverable.tsx:66` — The open Deliverable is replaced by "Play couldn't open this Deliverable" (or "…may have been deleted") when a background refetch fails
- **M57** (medium) A success:false list answer on a background refetch replaces the loaded shelf and drops every Load-more page
  - #104 · medium · `frontend/lib/play/shelf.ts:127` — A list answer of success:false on a background refetch replaces the loaded shelf with "Your shelf didn't load" and drops every page loaded with Load more
- **M38** (medium) The shelf never refreshes while open: a new Deliverable and Auto's celebrate wait for a refocus or a navigation
  - #107 · medium (finder: low) · `frontend/components/play/deliverables/play-shelf.tsx:85` — The shelf no longer updates while it is open: a Deliverable filed by a running Mission only appears (and Auto only celebrates) after a tab refocus or a navigation
  - #56 · low · `frontend/components/play/deliverables/play-shelf.tsx:85` — The shelf never refreshes while open, so a new Deliverable doesn't land and Auto doesn't celebrate until a refocus or remount
- **M49** (medium) Download saves the API's JSON envelope, not the file, for every text Deliverable (Play's page and Studio's sheet)
  - #100 · medium (finder: high) · `frontend/lib/deliverables/download.ts:15` — Download saves the API's JSON envelope, not the file, for every text Deliverable (reports, blog posts, .md/.csv/.py files), in Play's page and Studio's sheet
- **M82** (low) A failed Download on Play's Deliverable page says 'Try again from Deliverables', the page the user is on
  - #109 · low · `frontend/components/play/deliverables/detail-panel.tsx:60` — A failed Download on Play's Deliverable page tells the user to "Try again from Deliverables", the page they are already on

**Files**

- components/play/deliverables/play-deliverable.tsx (detailState)
- lib/play/shelf.ts (shelfView, SHELF_REFRESH_MS)
- components/play/deliverables/play-shelf.tsx (useShelfList)
- hooks/use-deliverables-api.ts (useDeliverables)
- lib/deliverables/download.ts
- components/play/deliverables/detail-panel.tsx (DownloadButton)
- components/play/deliverables/deliverable-content.tsx (FileCard)
- components/workspace/gallery-view/deliverable-preview.tsx
- components/play/deliverables/__tests__/play-deliverable.test.tsx (:278-283)
- components/play/deliverables/__tests__/play-shelf.test.tsx
- new lib/deliverables/__tests__/download.test.ts
- lib/play/__tests__/detail.test.ts (:165)

**Change**

1. M37: detailState reads usable data first (`query.data?.success !== false && query.data?.deliverable` → ready), then the isError mapping (404 → missing, else failed), then loading. useLogFailure still logs the refetch error.
2. M57: useDeliverables' queryFn throws when a page answers `success: false` (message from page.error), so react-query keeps the last good pages on a failed refetch; remove shelfView's per-page success:false branch and failureDetail. This changes the core hook: confirm each consumer handles isError or missing data (gallery-view/index.tsx:47, task-deliverables-panel.tsx:94, mission-deliverables-panel.tsx:124, type-row, today-hero, system-strip).
3. M38: `useDeliverables(filters, options?: { refetchInterval?: number })` passed to useInfiniteQuery with refetchIntervalInBackground false; useShelfList passes a named SHELF_REFRESH_MS (lib/play/shelf.ts, e.g. 30_000).
4. M49: `deliverableDownloadUrl(d)` in lib/deliverables/download.ts maps `^/api/workspaces/([^/]+)/files/content\?path=` to `apiClient.getWorkspaceFileRawUrl(ws, path)` (path decoded from the query) and returns any other URL unchanged; used by DownloadButton, FileCard and DeliverablePreview's download action. The display URL is unchanged.
5. M82: `DOWNLOAD_FAILED = "Couldn't download the file. Try again in a moment."` passed to useApiFileDownload in DownloadButton and FileCard.

**Proving test (fails before, passes after)**

- play-deliverable.test.tsx: data `{success:true, deliverable}` with isError and a 500 error → the 'Laptop shortlist' heading stays, no alert, no NOT_ON_SHELF.
- play-shelf.test.tsx with the real hook over the mocked apiClient: a success page of 2 rows, then refetch() answered `{success:false}` → the 2 cards stay, no SHELF_FAILED; with fake timers, advancing SHELF_REFRESH_MS asks again and a new first item shows the Newest card and Auto's celebrate caption.
- download.test.ts: '/api/workspaces/ws-1/files/content?path=outputs%2Fa.md' → '/api/workspaces/ws-1/files/raw?path=outputs%2Fa.md'; '/api/documents/generated/x.pdf' unchanged. Change play-deliverable.test.tsx:282 and detail.test.ts:165, which lock the bug in, to the raw URL. The use-api-file-download mock captures DOWNLOAD_FAILED. All fail today.

**Code shape:** DeliverablePreview about 110 lines (under 150); useDeliverables about 20.

**Note:** M49 is on main for Studio's sheet (deliverable-preview.tsx used content_url there); the helper is new in this branch and Play's TRUNCATED note sends users to Download, so it is fixed here.

### P260-FX-13 — Deliverable content and labels: code blocks, raw HTML, provenance, captions, sources

**Severity:** medium · **Depends on:** P260-FX-01

**Closes**

- **M36** (medium) Fenced code on the Deliverable sheet and in Auto's replies renders in Studio's dark CodeBlock card; Play's own pre rules never apply
  - #169 · medium · `frontend/components/play/deliverables/deliverable-content.tsx:51` — Code on a Deliverable's sheet (and in Auto's replies) renders in Studio's dark CodeBlock card; Play's own pre rules never apply
- **M39** (medium) Raw HTML in a Deliverable or a reply prints as literal tags ('&lt;br&gt;', '&lt;script&gt;…'); the approved app dropped it
  - #174 · medium (finder: low) · `frontend/components/shared/markdown-view.tsx:62` — Raw HTML in a Deliverable prints as literal tags on the sheet ('&lt;br&gt;', '&lt;script&gt;…'); the approved Play app dropped it
- **M104** (low) 'Made by Auto' is shown for the owner's uploads and template documents
  - #171 · low · `frontend/lib/play/detail.ts:132` — 'Made by Auto' is shown for files Auto didn't make (the owner's Socials uploads and template documents), and their source reads 'Document'
- **M95** (low) Auto's celebrate caption shows the raw type id ('Your blog_post is ready')
  - #172 · low · `frontend/hooks/use-shelf-mascot.ts:49` — The shelf's celebration caption shows the raw type id: 'Your blog_post is ready'
  - #160 · low · `frontend/lib/mascot/media.ts:94` — The shelf's Auto announces a new blog post as 'Your blog_post is ready'
- **M81** (low) The Deliverable side panel drops the Play app's 'Sources · N' section
  - #57 · low · `frontend/components/play/deliverables/detail-panel.tsx:49` — The Deliverable side panel drops the Play app's 'Sources · N' section (extra.sources)

**Files**

- components/play/chat/play-markdown.tsx
- components/shared/markdown-view.tsx
- components/play/deliverables/deliverable-content.tsx
- components/play/deliverables/deliverable-content.module.css
- lib/play/detail.ts
- components/workspace/gallery-view/made-by.ts (export MADE_BY_SOURCE)
- components/play/deliverables/play-deliverable.tsx (:149)
- components/play/deliverables/detail-panel.tsx (:86) and detail-panel.module.css
- lib/mascot/media.ts (captionFor)
- tests: play-deliverable.test.tsx, lib/play/__tests__/detail.test.ts, lib/mascot/__tests__/media.test.ts

**Change**

1. M36: a plain `PrintedPre` (`<pre>{children}</pre>`) in PLAY_MARKDOWN_COMPONENTS ({ a, img, pre }), so Play's own pre rules apply on the sheet and in replies; `.prose pre code { color: var(--pl-color-sheet-text) }` (with FX-01's prefix).
2. M39: MarkdownView gets an optional `skipHtml?: boolean` forwarded to ReactMarkdown (default false, other surfaces unchanged); DeliverableContent and PlayMarkdown pass it.
3. M104: reuse made-by.ts's MADE_BY_SOURCE: madeBy/madeLine return the core's word for upload, document and social_post ('Uploaded', 'Made in Templates', 'Socials'); 'Auto' stays for chat, task and mission; SOURCE_WORDS gains those sources; pass deliverable.source_type.
4. M95: captionFor normalises the kind (`trim().toLowerCase().replace(/_+/g, ' ')`), one place for the chat and the shelf (typeLabel pluralises, so don't reuse it).
5. M81: `sourcesOf(deliverable)` in lib/play/detail.ts, validated off the wire (string or {url|title}, trimmed, URL → host without www, capped by a named limit); detail-panel renders 'Sources · N' only when non-empty.

**Proving test (fails before, passes after)**

- play-deliverable.test.tsx: a code Deliverable renders its text inside a &lt;pre&gt; in the article and there is no CodeBlock 'Copy' button; content 'a&lt;br&gt;b' and the existing &lt;script&gt; fixture leave no literal tag text; 'Sources · 3' with which.co.uk, rtings.com and the Asus sheet, and no section without extra.sources.
- detail.test.ts: madeBy(null,'upload') is not 'Auto'; madeLine(null,'document') is not 'Made by Auto'; madeBy('Researcher','upload') is 'Researcher'.
- media.test.ts: captionFor('celebrate','blog_post') is 'Your blog post is ready'; 'report' unchanged. All fail today.

**Code shape:** MarkdownView and DetailPanel are small; the markdown-view change is an additive optional prop.

### P260-FX-14 — Chat layout fidelity: first open on laptop heights, the Conversations header, receipts' Done tag

**Severity:** medium · **Depends on:** P260-FX-01

**Closes**

- **M31** (medium) First open: Auto and the hero overlap the page header and the example chips on laptop-height windows (`min-height: 0` lets the hero shrink)
  - #51 · medium · `frontend/components/play/chat/first-open.module.css:9` — First open: Auto and the hero overlap the page header and the example chips on laptop-height windows (the port added min-height:0, so the hero shrinks instead of the page scrolling)
  - #65 · medium · `frontend/components/play/chat/first-open.module.css:7` — The empty chat's hero is allowed to shrink (`min-height: 0`) and is centred, so on common laptop heights 'Hi, I'm Auto.' and its tagline overlap the example chips and the message box
- **M35** (medium) Conversations: the title and New chat span the whole column while the list is centred at 700 px
  - #55 · medium (finder: low) · `frontend/components/play/conversations/play-conversations.tsx:74` — Conversations: the title and New chat span the whole main column while the list is centred at 700px, so they no longer line up as in the Play app
- **M34** (medium) 'What Auto did' lost Play's accent 'Done' tag and the refused/skipped reason line
  - #52 · medium · `frontend/components/play/chat/receipts.tsx:36` — 'What Auto did' lost Play's accent 'Done' tag and the refused/skipped reason line, in the chat and on the style sheet

**Files**

- components/play/chat/first-open.module.css
- components/play/chat/composer.module.css
- components/play/conversations/play-conversations.tsx and conversations.module.css
- components/play/chat/receipts.tsx and turn.module.css
- components/looks/style/style-components.tsx (ReceiptsBox)
- tests: a static CSS test, play-conversations.test.tsx, play-chat-turn.test.tsx
- docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md

**Change**

1. M31: let the empty chat grow and scroll, as the approved app did: `.hero { flex: 1 0 auto; justify-content: safe center }` without `min-height: 0`, and `.chat[data-empty] { flex: 1 0 auto }`, so PageColumns' `.main` (already overflow-y: auto) scrolls on short windows while tall ones still centre. (The other proposal, scrolling inside the hero with a 42dvh mascot cap, changes the approved composition.)
2. M35: ConversationsScreen renders the reference's in-column header (h1 `.title` + New chat) as the first child of `.page` instead of PageHeader; `.title` follows FX-01's `:global(html.play)` rule.
3. M34: a pure `receiptBadge(r)` in receipts.tsx (write && done → 'Done', refused → 'Not done', skipped → 'Skipped', else null; named constants) rendered as `<span className={styles.badge}>`; restore `.badge` and its tone rules in turn.module.css; drop the whole-line error colour so only the mark and the tag carry the tone; the same tag in the style sheet's ReceiptsBox.

**Proving test (fails before, passes after)**

- Static CSS test: `.hero` declares no `min-height: 0` and `.chat[data-empty]` grows (fails today); RUN-checklist step: the first open at 1366×657 and at 125% zoom shows no overlap.
- play-conversations.test.tsx: the 'Conversations' heading, New chat and the search box share one parent region (fails today).
- play-chat-turn.test.tsx: a done write receipt renders 'Done' beside its line, a done read renders no tag, a refused one renders 'Not done' (fails today).

**Code shape:** ConversationsScreen about 35 lines; receipts.tsx 79.

### P260-FX-15 — Auto's player: blocked autoplay, the boundary race, the style sheet's 12 clips, clip caching

**Severity:** medium · **Depends on:** P260-FX-04

**Closes**

- **M26** (medium) When the browser blocks muted autoplay, Auto never reaches a clip boundary: stuck on hello/celebrate, picture frozen
  - #79 · medium · `frontend/components/mascot/auto-mascot.tsx:60` — Auto stays stuck on hello or celebrate, and his picture freezes, when the browser blocks muted autoplay
  - #95 · medium · `frontend/components/mascot/auto-mascot.tsx:105` — When the browser blocks muted autoplay, Auto never reaches a clip boundary: loops never swap, and hello/celebrate never end, so the reducer is stuck in the one-shot
- **M27** (medium) A celebrate queued behind a loop is usually skipped: the clip's final timeupdate and ended both apply clipBoundary before React re-renders
  - #94 · medium · `frontend/hooks/use-mascot-player.ts:69` — A celebrate queued behind a loop is usually skipped: the clip's final timeupdate and ended both apply clipBoundary before React re-renders, so the new one-shot is marked finished before it plays
- **M23** (medium) The style sheet autoplays all 12 mascot clips and posters at once (2.6-3.5 MB, 12 alpha decoders)
  - #78 · medium · `frontend/components/looks/style/style-mascots.tsx:29` — The style sheet autoplays all 12 mascot clips and their posters at once (2.6-3.5 MB, 12 alpha decoders)
- **M75** (low) Auto's clips are served with max-age=0, so every state swap waits a revalidation round trip
  - #81 · low · `frontend/components/mascot/auto-mascot.tsx:109` — Auto's clips are served with max-age=0, so every state swap waits a revalidation round trip

**Files**

- components/mascot/auto-mascot.tsx (MascotVideo, MascotStill, AutoMascotProps)
- hooks/use-mascot-player.ts
- components/looks/style/style-mascots.tsx
- next.config.js (headers())
- components/mascot/__tests__/auto-mascot.test.tsx
- components/looks/__tests__/style-screen.test.tsx
- new lib/__tests__/mascot-cache-headers.test.ts

**Change**

1. M26: MascotVideo gets `onBlocked` and a ref callback that calls `Promise.resolve(video.play())` in try/catch; a NotAllowedError rejection calls onBlocked (console.warn), an AbortError from a normal swap is ignored. AutoMascot passes `() => setVideoFailed(true)`, the existing stills path, so jumpTo swaps at once and one-shots end on the CLIP_MS timer.
2. M27: make the boundary idempotent per clip: MascotVideo's handlers carry the clip they belong to, and `setPlayer((current) => current.showing === clip ? clipBoundary(current).player : current)`, restarting only when player.showing === clip.
3. M23: `readonly playing?: boolean` (default true) on AutoMascotProps with `video = playing && !reducedMotion && !videoFailed ? format : null`; MascotStill gets `loading="lazy"`; StyleMascots gets a "Play Auto's states" toggle mirroring StyleMotion's PLAY_MOTION (aria-pressed, off by default, disabled under reduced motion) and passes `playing={on}`. T-008b's 'AutoMascot for all 12 states' still holds.
4. M75: next.config.js headers(): `{ source: '/auto/:file*', headers: [{ key: 'Cache-Control', value: `public, max-age=${MASCOT_MAX_AGE_S}, stale-while-revalidate=${MASCOT_SWR_S}` }] }` with named constants at the top of the file; not `immutable`, because the file names are unversioned.

**Proving test (fails before, passes after)**

- auto-mascot.test.tsx: `vi.spyOn(HTMLMediaElement.prototype, 'play').mockRejectedValue(new DOMException('blocked', 'NotAllowedError'))`, render `<AutoMascot state="hello" onClipEnded={fn}/>` under a Chrome UA → an &lt;img&gt; /auto/hello.png and no &lt;video&gt;; advancing CLIP_MS calls fn('hello'). Give the other video tests a resolved play() mock.
- auto-mascot.test.tsx: idle, rerender 'celebrate', then `playTo(oldVideo, CLIP_SECONDS); fireEvent.ended(oldVideo)` → src '/auto/celebrate.webm' and onClipEnded not called; the celebrate element's own ended calls it once.
- style-screen.test.tsx: with motion allowed, 12 stages with 12 &lt;img&gt; and no &lt;video&gt;; the toggle gives 12 &lt;video&gt; with .webm; under reduced motion the toggle is disabled.
- mascot-cache-headers.test.ts (loads next.config.js as csp-media-sources.test.ts does): a rule for /auto whose Cache-Control has max-age &gt; 0. All fail today.

**Code shape:** AutoMascot about 46 lines (component limit 150); useMascotPlayer small.

**Note:** M26 and M27 are inherited from the reference app (same code); M75's header config is on main, but only PRD-260's clips are re-requested this way.

### P260-FX-16 — Auto follows the model's reasoning, turn caps and the right tool kind

**Severity:** medium · **Depends on:** P260-FX-04

**Closes**

- **M54** (medium) Auto never returns to 'thinking' while the model reasons between tool rounds
  - #157 · medium · `frontend/lib/mascot/events.ts:160` — Auto never reacts to the reply's reasoning: between tool rounds he says 'Ready when you are' while the model is still thinking
- **M100** (low) Auto doesn't play oops when a turn hits a cap (limit_reached)
  - #11 · low · `frontend/lib/mascot/events.ts:142` — Auto doesn't play oops when a turn hits a cap (limit_reached), although Studio reads that event and the reducer supports it
- **M103** (low) platform_acknowledge_report shows Auto 'Reading' ('acknowledge' contains 'knowledge')
  - #97 · low · `frontend/lib/mascot/tools.ts:43` — platform_acknowledge_report shows Auto 'Reading' and the tool line 'Looking through your files': 'acknowledge' contains 'knowledge'

**Files**

- components/play/chat/message-text.ts (replyProgress)
- lib/mascot/events.ts (ReplyProgress, NO_REPLY, mascotEventsFromReply)
- hooks/use-auto-mascot.ts (useReplyEvents deps)
- lib/mascot/tools.ts (WORKING_TOOLS)
- lib/mascot/__tests__/events.test.ts
- hooks/__tests__/use-auto-mascot.test.tsx
- lib/mascot/__tests__/tools.test.ts

**Change**

1. M54: useChat writes reasoning only onto the reply (hooks.ts:409-417), so read it there: ReplyProgress gains `reasoningChars` (NO_REPLY 0); while streaming, with the same reply id, reasoning growth and no tool running emits `{ type: 'reasoning', delta: '' }` before the text check. The narration frame already resets the words each round, so the existing first-words event still ends this thinking.
2. M100: ReplyProgress gains `limitReached` (`Boolean(last.limitReached?.message)`); a transition emits one `{type:'limit_reached', …}` whatever the status (the frame and the stream's end can land in one render).
3. M103: add 'platform_acknowledge_report' to WORKING_TOOLS (exact names win over the patterns).
4. useReplyEvents adds the two primitives to its deps. useChat is not edited.

**Proving test (fails before, passes after)**

- events.test.ts through the reducer: send → tool-start → tool-end (idle) → reasoning grows (thinking) → the reply's words start (idle); reasoning growth while a tool runs changes nothing; the limit transition, a repeat giving nothing, and a new reply.
- use-auto-mascot.test.tsx: the same by rerendering `reply`; working → limit → finish ends on oops.
- tools.test.ts: `toolActivity('platform_acknowledge_report')` is 'working'. All fail today.

**Code shape:** Small pure functions; useReplyEvents stays ≤ 50 lines.

**Note:** The runner's one unverified finding (no `reasoning` event can reach the reducer) is the same defect as M54.

### P260-FX-17 — Studio tokens under a look keep their meaning and reach AA on Studio's own fills

**Severity:** medium · **Blocked:** F9 for M52 (M53 can land alone).

**Closes**

- **M52** (medium) Under Play, Studio's consequence colours follow the look's accent: --accent (Studio's error colour) turns mint/cream/gold, --destructive is the same orange as --primary
  - #37 · medium (finder: high) · `frontend/lib/looks/studio-tokens.ts:146` — Under Play, Studio's consequence/error colour (--accent) becomes the look's accent, so failures on Studio pages are drawn in mint, cream or gold
  - #82 · medium (finder: high) · `frontend/lib/looks/studio-tokens.ts:146` — The look overwrites Studio's --accent, which Studio's CSS uses as its error colour, so failures on Studio pages under Play show in the look's accent (mint under Aurora, cream under Obsidian)
  - #146 · medium · `frontend/lib/looks/studio-tokens.ts:139` — Under Ember, Workshop and Daylight, Studio's destructive buttons are the same orange as its primary buttons (1.13 to 1.17:1 apart)
- **M53** (medium) Under a look, Studio's --primary and --fg-3 are checked only on --background and --card, not on Studio's tinted, secondary and muted fills
  - #62 · medium · `frontend/lib/looks/studio-tokens.ts:132` — Studio's --primary becomes the look's accent, checked only on --background and --card, so Studio's tinted accent chips and text-primary on secondary or muted fall to 3.3 to 4.5:1 (Studio Light: about 13:1)
  - #148 · medium (finder: low) · `frontend/lib/looks/studio-tokens.ts:44` — --fg-3 (Studio's faded meta text) drops below AA on Studio's own --bg-2 and --muted fills: fadedText checks only bg and surface

**Files**

- lib/looks/studio-tokens.ts (studioGroundTokens, studioAccent, fadedText)
- lib/looks/contrast.ts (readableOnAll / readableOnTinted)
- lib/looks/__tests__/studio-tokens.test.ts
- lib/looks/__tests__/contrast.test.ts
- docs/frontend-architecture/play-theme.md (Studio-token table, lines 248-250)
- scripts/ralph/prd-260.json (D3 note, if F9 is approved)

**Change**

1. M52 (blocked on decision F9, an amendment to D3): write Studio's consequence tokens from Play's coral in studioGroundTokens: `--accent`, `--destructive` = `readableOn(ERROR_COLOUR, bg, surface, text)`, their foregrounds = the ground's bg when that pair reads ≥ 4.5:1, else the surface, `--accent-2` the tint at the existing ACCENT_TINT_ALPHA_*; `--primary` and `--ring` keep the look's accent. Studio itself sets `--accent` and `--destructive` to the same burnt orange (globals.css), so this keeps Studio's meaning. STUDIO_TOKEN_NAMES and the pre-paint table follow automatically.
2. M53: contrast.ts gains a list form, `readableOnAll(colour, grounds, text)` (and `readableOnTinted(colour, grounds, text, maxTint)` for tinted chips); the existing two-ground call delegates to it. studioAccent checks [bg, card, secondary, muted] flat and at PRIMARY_TINT_MAX = 0.2 (the strongest `bg-primary/20` in use); fadedText checks [bg, surface, secondary, muted] and falls back to `readableOn(text2, …, text)` when no fade reads.

**Proving test (fails before, passes after)**

- studio-tokens.test.ts: for all 24 ground × accent pairs `--accent` and `--destructive` are the ground's readable coral, never the look's accent fill; replace the 'no semantic token' assertion.
- contrast.test.ts, all 24 pairs: accent-foreground on accent, destructive-foreground on destructive, and accent/destructive on background and card ≥ 4.5; TEXT_PAIRS add ['--fg-3','--bg-2'], ['--fg-3','--muted'], ['--muted-foreground','--secondary'], and --primary over primary tints 0.1/0.15/0.2 on --background, --card, --secondary, --muted. These fail today for Paper × every accent and Obsidian × every accent.

**Code shape:** Each function stays under 20 lines; no new file.

### P260-FX-18 — Play's accent tints and page glow are inside the AA proof

**Severity:** medium · **Depends on:** P260-FX-17

**Closes**

- **M50** (medium) The AA proof measures flat grounds only: text over Play's accent tints and the page glow drops below 4.5:1, Daylight preset included
  - #8 · medium · `frontend/lib/looks/__tests__/contrast.test.ts:110` — "Every look stays readable (WCAG AA), proven by tests" is not met: the tests measure flat grounds only, and Play draws text over accent tints and the page glow, where it drops below 4.5:1, including on the Daylight preset
  - #60 · medium · `frontend/lib/looks/css.ts:151` — The page glow tints the ground under text, but the AA proof measures flat grounds only: Make it yours' editor labels drop to 2.54:1, and the chat caption detail to about 4.0:1

**Files**

- lib/looks/css.ts (accentTextVars, glowVars, a safe-glow cap)
- lib/looks/prepaint.ts (cap table)
- lib/looks/editor.ts (clampGlow) and the Glow slider
- app/play-theme.css (--pl-tint-accent-halo comment; the snag glow from a ground-specific variable instead of the literal 15% color-mix)
- lib/looks/__tests__/contrast.test.ts
- lib/looks/__tests__/prepaint.test.ts

**Change**

1. --pl-accent-text must read on [bg, surface, mix(bg, accent, ACCENT_HALO_ALPHA)] with ACCENT_HALO_ALPHA = 0.18 (named; the test asserts play-theme.css's literal matches it), using FX-17's readableOnAll.
2. A pure `safeGlowAlpha(ground, accent)`: the largest alpha at which text, text2, accent-text and error-text stay ≥ 4.5 over mix(bg, accent, alpha) at the bright opacity; glowVars emits `min(glow / GLOW_SCALE, safe)`; prepaint's table gains the cap per ground × accent; the slider's max and clampGlow use the same cap; a ground-specific `--pl-glow-snag` replaces the literal 15% mix. The presets' glows (24/6/20/14/10) already fit.

**Proving test (fails before, passes after)**

- contrast.test.ts, all 24 pairs: accent-text over mix(bg, accent, 0.18), 0.12 and 0.04 ≥ 4.5; text, text2, accent-text and error-text over the effective glow for glow 0..GLOW_MAX ≥ 4.5; the snag alpha per ground likewise. Fails today (Paper + Molten is 4.08:1 at 18%).
- prepaint.test.ts: a stored glow-40 look paints the clamped --pl-glow.

**Code shape:** Pure functions in lib/looks; no component over its limit is touched.

**Note:** Inherited from the reference app; the PRD's 'Every look stays readable (WCAG AA), proven by tests' makes it in scope.

### P260-FX-19 — Make it yours: selection and colour samples readable on Paper and in forced colours; long names

**Severity:** medium

**Closes**

- **M18** (medium) On Paper, Make it yours marks the chosen option, look card and swatches only in the raw accent (below 3:1); unselected swatches have no edge
  - #40 · medium · `frontend/components/looks/editor.module.css:57` — On Paper, the selected state in Make it yours is shown only by a raw-accent edge that falls below 3:1 (the Daylight preset itself is at 2.86:1)
  - #61 · medium · `frontend/components/looks/editor.module.css:57` — On Paper looks, selected states and swatches are drawn in the raw accent (1.07 to 2.94:1), so the chosen ground, type, shell, light and look card can't be seen
  - #119 · low · `frontend/components/looks/editor.module.css:71` — Cream, Mint and Gold accent swatches are invisible or faint on the Daylight (Paper) ground
- **M19** (medium) In forced colours (Windows High Contrast), Play's focus indicators, pressed states and colour samples disappear (`outline: none` + box-shadow, background-only samples)
  - #151 · medium · `frontend/components/looks/editor.module.css:71` — In Windows High Contrast (forced colours), Make it yours shows six blank accent circles, and no option or look card shows which one is chosen
  - #64 · medium · `frontend/components/play/deliverables/shelf.module.css:104` — In Windows High Contrast (forced colours), Play's focus indicators and pressed states disappear: nine port-added `outline: none` rules rely on box-shadow, which forced colours remove
- **M73** (low) A long look name squeezes 'Make it yours.' into a sliver and an unbroken name is clipped in its card
  - #155 · low · `frontend/components/looks/looks.module.css:57` — A long look name pushes 'Make it yours.' into a sliver at narrow desktop widths, and an unbroken name is clipped in its card

**Files**

- components/looks/editor.module.css (:57-62, :71-83, :173)
- components/looks/looks.module.css (:56-69, :253-272, :315-324)
- components/looks/style/style.module.css
- the ten `outline: none` rules in Play CSS (shelf.module.css :74/:105/:168, shelf-card.module.css :33, deliverable-detail.module.css, conversations.module.css, composer.module.css, editor/looks/style modules)
- app/play-theme.css
- docs/frontend-architecture/play-theme.md:229
- lib/__tests__/prd260-play-guards.test.ts
- docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md

**Change**

1. M18: every `[aria-pressed="true"]` / `[aria-current]` edge, ring and the slider's accent-color use `--pl-color-accent-text` (≥ 4.8:1 on bg and surface for every look; equal to the raw accent on dark grounds, so Ember is unchanged); `.swatch` gets `box-shadow: inset 0 0 0 1px var(--pl-color-text-2)`, the pressed swatch keeps its 2px text ring on top. Correct play-theme.md:229 (raw accents are not 3:1 on Paper).
2. M19: `forced-color-adjust: none` on colour samples (.swatch, .groundSwatch, .dot, .lightDot, .cardTop and its marks, .swatchChip, .step); a `@media (forced-colors: active)` block in editor.module.css and looks.module.css widening pressed borders; every `outline: none` becomes `outline: 2px solid transparent; outline-offset: 2px` (Tailwind's own outline-none) with the box-shadow kept; play-theme.css adds `@media (forced-colors: active) { html.play [aria-pressed="true"] { outline: 2px solid Highlight; outline-offset: 2px } }`.
3. M73: `.headerSide { flex: 0 1 auto; min-width: 0; max-width: 40% }`, `.status { overflow-wrap: anywhere; text-align: right }`, `.cardText { min-width: 0; overflow-wrap: anywhere }`.

**Proving test (fails before, passes after)**

- Guards (static; css:false): fail on a pressed/current rule in components/{looks,play}/**/*.module.css that reads --pl-color-accent or --pl-accent directly; fail on `outline: none|0` in Play, looks and mascot CSS; require forced-color-adjust on the swatch rules and the forced-colors blocks; require .headerSide not `flex: none` and .cardText to declare overflow-wrap. All fail today.
- RUN-checklist steps: Windows High Contrast on /looks shows the chosen accent, ground, type, shell, light and look card; a 40-character look name at 1024 px keeps the heading readable and the whole name in its card.

**Code shape:** CSS only. N19 polices variables, so the `Highlight` system colour is allowed; FX-28's widened colour guard must allow CSS system colours inside forced-colors blocks.

**Note:** M18 and M73 are byte-for-byte ports of the reference app.

### P260-FX-20 — Looks: tabs stay in sync, failed writes are told, the editor waits for the stored look, exports are honest

**Severity:** medium · **Blocked:** If F5 makes looks per user, the key scoping lands in this story (same files).

**Closes**

- **M25** (medium) A second tab's stale saved-looks list overwrites looks saved in another tab, and the tone cookie can disagree with the stored look
  - #38 · medium · `frontend/components/looks/use-look-store.ts:41` — A second tab overwrites saved looks with its stale list (data loss), and the tone cookie can disagree with the stored look
  - #117 · medium · `frontend/components/looks/use-look-store.ts:41` — A second tab's stale saved list overwrites looks saved in another tab (data loss)
- **M98** (low) Save, Duplicate and Import report success when localStorage refuses the write
  - #41 · low · `frontend/lib/looks/storage.ts:97` — Save, Duplicate and Import report success when localStorage refuses the write; the look is gone after a reload
  - #121 · low · `frontend/lib/looks/storage.ts:97` — The success notices claim 'Saved…', 'Added…' while blocked or full storage keeps nothing
- **M72** (low) /looks and /looks/style render Ember's selections (and 'reduced motion') until storage is read after hydration
  - #120 · low · `frontend/components/looks/look-provider.tsx:26` — /looks and /looks/style render Ember's selections (and 'reduced motion') until storage is read after hydration
- **M71** (low) Export always announces 'Downloaded &lt;name&gt;.look.json.' even when the browser blocked the download
  - #150 · low · `frontend/components/looks/download.ts:10` — Export always announces 'Downloaded &lt;name&gt;.look.json.' even when the browser blocked or cancelled the download
- **M99** (low) 'Copy tokens' exports next/font variable references instead of the face, so the :root block fails outside the app
  - #42 · low · `frontend/lib/looks/style.ts:119` — 'Copy tokens' gives a :root block whose display family points at next/font variables that exist only inside Automatos
  - #123 · low · `frontend/lib/looks/style.ts:119` — 'Copy tokens' copies next/font variable references instead of the face, so the :root block doesn't work outside the app

**Files**

- components/looks/use-look-store.ts (+ a small useStorageSync)
- lib/looks/store.ts (a `synced` action)
- lib/looks/storage.ts
- components/looks/look-provider.tsx
- components/looks/looks-screen.tsx
- components/looks/style/style-screen.tsx and style-motion.tsx
- components/looks/look-manager.tsx, look-import.tsx, look-notice.tsx
- components/looks/download.ts
- lib/looks/style.ts (rootBlock), lib/looks/catalogue.ts (a `generic` field on TypeStyle)
- tests: look-provider.test.tsx, look-manage.test.tsx, store.test.ts, storage.test.ts, looks-screen.test.tsx, style.test.ts

**Change**

1. M25: a `synced` reducer action sets look (active ?? DEFAULT_LOOK) and saved with lookChanged/savedChanged false, so it never writes back; useStorageSync(storage, dispatch) subscribes to window 'storage' for SAVED_LOOKS_KEY, ACTIVE_LOOK_KEY or null and dispatches readStoredLooks(storage). An identical setItem fires no event, so tabs can't ping-pong. The tone cookie follows through the existing useLookTone.
2. M98: writeSavedLooks/writeActiveLook return a boolean and readStoredLooks reports `persisted`; useLookStore exposes it through useLook(); LookManager and LookImport append a named LOOKS_NOT_KEPT line ('This browser isn't keeping looks, so it lasts until you close this tab.') to their success notices when it is false.
3. M72: expose `loaded` in LookContextValue; LooksScreen and StyleScreen render their frame with aria-busy='true' and no editor or sheet content until loaded; StyleMotion shows the reduced-motion note only once mounted.
4. M71: the Export notice says "Your browser is saving &lt;file&gt;. If it didn't appear, use Copy JSON." (named constant); download.ts's doc comment says it throws only when the Blob or URL can't be made.
5. M99: rootBlock writes `--pl-display-family: '<face>',<generic>` from the catalogue; lookToCssVars (in-app values) is unchanged.

**Proving test (fails before, passes after)**

- look-provider.test.tsx: set SAVED_LOOKS_KEY directly and dispatch `new StorageEvent('storage', { key: SAVED_LOOKS_KEY })` → useLook().saved holds it, and a following Save as new look leaves both looks under different ids; set ACTIVE_LOOK_KEY to Daylight → the tone cookie becomes 'light'.
- storage.test.ts: the write helpers return false when setItem throws; look-manage.test.tsx with a throwing storage: Duplicate's notice contains LOOKS_NOT_KEPT; Export's notice contains 'Copy JSON'.
- renderToString of /looks under Play with a stored Daylight look: aria-busy='true' and no pressed Ember card.
- style.test.ts: rootBlock(EMBER) contains "--pl-display-family: 'Instrument Serif',serif;" and no line of any preset's block contains 'var('. All fail today.

**Code shape:** useLookStore is about 35 of 50 lines: the sync effect goes in the useStorageSync callee. LookManager about 43 lines.

**Note:** M72 and M71 are inherited from the reference app.

### P260-FX-21 — Asleep and 404 edges: /health timeout, first render, countdown, offline, start command, 404 title

**Severity:** medium

**Closes**

- **M43** (medium) GET /health has no timeout: a frozen backend keeps the pill on 'Connected' and 'Automatos is asleep' never appears
  - #112 · medium (finder: low) · `frontend/hooks/use-backend-health.ts:62` — GET /health has no timeout: when the backend freezes, the pill stays 'Connected' and 'Automatos is asleep' never appears
- **M89** (low) The asleep gate starts awake on every mount, so a page mounts and fires its requests before the asleep screen returns
  - #113 · low · `frontend/hooks/use-backend-mascot.ts:27` — The asleep gate starts awake on every mount, so after browser Back while the backend is down the page mounts and fires its requests before the asleep screen returns
- **M94** (low) The asleep screen's countdown never shows that a check is running
  - #114 · low · `frontend/hooks/use-recheck-countdown.ts:40` — The asleep screen's countdown never shows that a check is running: it says 'Checking again in N seconds' while the request and its retries are in flight
- **M83** (low) The hosted asleep screen tells an offline user 'Nothing's broken on your side'
  - #74 · low · `frontend/components/play/edges/asleep-screen.tsx:24` — The hosted asleep screen tells an offline user 'Nothing's broken on your side'
- **M84** (low) The local asleep screen offers `docker compose up`, not the documented start command
  - #167 · low · `frontend/components/play/edges/start-command.tsx:15` — Under the local edition, the asleep screen tells the owner to run `docker compose up`, which is not the documented start command: it runs attached to the terminal, and on a `make up-images` install it rebuilds from source
- **M68** (low) The new client not-found drops the 404 document title for every style
  - #18 · low · `frontend/app/not-found.tsx:1` — The new client not-found drops Next's 404 page title for every style: tabs and screen readers now say 'Automatos AI Platform' on a 404
  - #36 · low · `frontend/app/not-found.tsx:16` — The new app/not-found.tsx drops the document title Next's default 404 set ('404: This page could not be found.') for every style

**Files**

- hooks/use-backend-health.ts
- hooks/use-backend-mascot.ts
- hooks/use-recheck-countdown.ts
- new hooks/use-online.ts
- components/play/edges/asleep-gate.tsx, asleep-screen.tsx, start-command.tsx
- app/not-found.tsx → a server component, plus new components/play/edges/not-found-view.tsx
- components/play/edges/__tests__/edges.test.tsx
- docs/frontend-architecture/play-theme.md and the checklist's asleep steps 41-43

**Change**

1. M43: a named BACKEND_HEALTH_TIMEOUT_MS passed as `signal: AbortSignal.timeout(BACKEND_HEALTH_TIMEOUT_MS)` (apiClient.request spreads options into fetch); a timed-out ask counts as a failure through the existing retries. Pick the value against the asleep screen's 10 s recheck (worst case is timeout × 4 + 3 × 2.5 s; e.g. 2 s gives about 15.5 s) and write the new worst case into the checklist.
2. M89: initialise useBackendMascot's reducer from the connection it mounts with (lazy init: 'down' → the backend-down state), so the gate shows the asleep screen on the first render.
3. M94: BackendHealth exposes `checking: query.isFetching`; AsleepGate passes it and AsleepScreen shows CHECKING while it is true; drop the '0 while one is being asked' line from the countdown's contract.
4. M83: useOnline (navigator.onLine plus the online/offline events); offline → OFFLINE_LINE in place of HOSTED_LINE and StartCommand.
5. M84: `START_COMMAND = 'make up'` (SETUP.md's day-to-day start) plus a named START_HINT line ('Installed with the published images? Run make up-images. No make? docker compose up -d'); Copy keeps writing START_COMMAND.
6. M68: app/not-found.tsx becomes a server component exporting `metadata = { title: NOT_FOUND_TITLE }` ('404: This page could not be found.') and rendering a client NotFoundView that holds today's LostScreen/PlainNotFound switch unchanged.

**Proving test (fails before, passes after)**

- edges.test.tsx: apiClient.request mocked to reject only when its signal aborts; advancing fake timers by the timeout × (retries + 1) plus the retry delays shows the asleep screen.
- Seed the health query into its error state, then render a new PlayFrame around a counting page: the page never mounts and the asleep title is there on the first render.
- With /health pending, Check again shows 'Checking…'; once it fails, 'Checking again in 10 seconds.'.
- navigator.onLine stubbed false with /health failing: OFFLINE_LINE, and neither HOSTED_LINE nor START_COMMAND.
- The local asleep screen shows 'make up' and the hint; Copy writes exactly START_COMMAND (update the existing expected value).
- Import `metadata` from app/not-found and assert its title. All fail today.

**Code shape:** All hooks stay small; the not-found split moves code unchanged into the client child.

### P260-FX-22 — Keyboard focus survives unmounts and frame swaps

**Severity:** medium · **Depends on:** P260-FX-21

**Closes**

- **M20** (medium) Keyboard focus falls to &lt;body&gt; when the control just used unmounts (Save, Save name, Cancel, Delete look, Keep it, Try again, the last Load more); Delete opens its confirmation without moving focus
  - #118 · medium · `frontend/components/looks/look-manager.tsx:141` — Keyboard focus falls to &lt;body&gt; after Save, Save name, Cancel, Delete look and Keep it
  - #68 · low · `frontend/components/play/chat/snag-box.tsx:37` — Keyboard focus is dropped to &lt;body&gt; when the button just used unmounts: Try again, the last Load more, Save name, Cancel, Delete look, Keep it
  - #153 · low · `frontend/components/looks/look-manage-forms.tsx:74` — Pressing Delete opens the confirmation without moving focus to it or announcing it, unlike Rename
- **M21** (medium) Whole-frame swaps (the asleep gate, 'Switch to Play') drop focus to &lt;body&gt; and announce nothing
  - #66 · medium · `frontend/components/play/edges/asleep-gate.tsx:34` — When the backend goes down or comes back, the asleep gate swaps the whole frame, dropping keyboard focus to &lt;body&gt; with nothing announced
  - #152 · low · `frontend/components/looks/play-only-route.tsx:79` — 'Switch to Play' on /looks and /looks/style drops keyboard focus to &lt;body&gt; as the whole frame is replaced, with nothing announced

**Files**

- components/looks/look-manager.tsx (a useReturnFocus hook; ToolButton `tool` prop as data-tool)
- components/looks/look-manage-forms.tsx (DeleteConfirm)
- components/play/chat/play-chat-view.tsx and message-box.tsx (a textarea ref)
- components/play/deliverables/shelf-body.tsx (Load more)
- components/play/edges/asleep-gate.tsx, asleep-screen.tsx
- components/play/shell/play-frame.tsx (&lt;main tabIndex={-1}&gt;)
- components/looks/play-only-route.tsx, looks-screen.tsx, style/style-screen.tsx
- tests: look-manage.test.tsx, play-chat-turn.test.tsx, play-shelf.test.tsx, edges.test.tsx, looks-screen.test.tsx, style-screen.test.tsx

**Change**

1. M20: LookManager keeps a toolbar ref and a `focusTo` state; an effect focuses `[data-tool=…]` after the re-render. Save → 'rename'; a successful rename and Cancel → 'rename'; Keep it → 'delete'; Delete look → 'duplicate' (always present). DeleteConfirm puts autoFocus on 'Keep it' (the safe default, as RenameForm autofocuses its field). Try again focuses the message box; when the last Load more disappears, focus moves to the first new card's link.
2. M21: AsleepGate renders one persistent visually hidden `<p role="status">` outside the swap, set to ASLEEP_TITLE when the gate closes and a named 'Automatos is back' when it opens; AsleepScreen's h1 gets tabIndex={-1} and focus on mount; the frame's &lt;main&gt; gets tabIndex={-1} and focus when it remounts after a nap (a `wasAsleep` ref in the gate). PlayOnlyRoute remembers it showed the note and, once Play renders, focuses the screen's h1 (marker PLAY_ONLY_HEADING exported from play-only-route.tsx, tabIndex={-1}).

**Proving test (fails before, passes after)**

- look-manage.test.tsx: user.keyboard through Save, Save name, Cancel, Keep it and Delete look; after each the expected toolbar button has focus and document.activeElement is not body; Delete → 'Keep it' has focus.
- play-chat-turn.test.tsx: Try again → the message box has focus; play-shelf.test.tsx: the last Load more → a card link has focus.
- edges.test.tsx: connected → down → connected: activeElement is the asleep h1, then the frame's main, and the status text follows each step.
- looks-screen/style-screen tests: renderLooks('studio'), click 'Switch to Play' → the 'Make it yours.' / 'Style sheet' h1 has focus. All fail today.

**Code shape:** LookManager about 43 lines (150); PlayChatView 66; useReturnFocus a small hook.

**Note:** M20's LookManager gaps are inherited from the reference app.

### P260-FX-23 — Load Play's code only when Play renders

**Severity:** medium · **Depends on:** P260-FX-11, P260-FX-21

**Closes**

- **M17** (medium) Play's frame, chat, shelf and 404 are statically imported at the core's seams, so their JS and CSS ship to every Classic and Studio user
  - #77 · medium · `frontend/components/layout/main-layout.tsx:13` — Play's frame, chat, shelf and 404 code and CSS ship to every Classic and Studio user
  - #84 · medium · `frontend/components/layout/main-layout.tsx:13` — Play's screens are statically imported at the core's seams, so every Classic and Studio page load downloads Play's frame, chat, shelf and CSS, and a theme installed from the Marketplace has nowhere to plug in
  - #16 · low · `frontend/components/layout/main-layout.tsx:13` — Classic and Studio users download Play's frame, chat, shelf and look engine on every page through static imports

**Files**

- components/layout/main-layout.tsx (the module-scope import only)
- new 'use client' modules: components/play/shell/play-frame-lazy.tsx, components/play/chat/play-chat-lazy.tsx, components/play/deliverables/play-shelf-lazy.tsx
- app/chat/layout.tsx, app/deliverables/layout.tsx
- the not-found view from FX-21
- lib/__tests__/prd260-play-guards.test.ts
- play-frame and play-route-switch tests (findBy* queries)

**Change**

1. Each lazy module is `'use client'` and exports `dynamic(() => import('./play-frame').then((m) => m.PlayFrame))` (and the same for PlayChat, PlayShelf, LostScreen), the pattern already in assignments-hub.tsx and activity-page.tsx. MainLayout and the two server layouts import the lazy modules instead, so the RSC payload references only the small slot and Play's chunks load only when Play renders. SSR stays on, so a Play first paint still has the frame and Next preloads the chunk only when it was rendered.
2. LookProvider stays a static import: it must react to a runtime switch to Play.

**Proving test (fails before, passes after)**

- Guard in prd260-play-guards.test.ts using its TypeScript-AST specifier walker: components/layout/**, app/**/layout.tsx and the not-found modules carry no static import of components/play/** (dynamic import() allowed). Fails today.

**Code shape:** MainLayout's body is untouched (K7: it sits near the 150-line limit); only the import line changes.

**Note:** The theme-registry half of M17 (a theme installed from the Marketplace) is decision F11.

### P260-FX-24 — Announcements and accessible names

**Severity:** low · **Depends on:** P260-FX-22

**Closes**

- **M22** (low) Status changes are not announced: a repeated notice, the Conversations filter result, a Deliverable link to a missing Deliverable
  - #154 · low · `frontend/components/looks/look-notice.tsx:18` — A repeat of the same outcome is never announced: importing the same look twice, copying twice, or the same failure twice is silent to screen readers
  - #69 · low · `frontend/components/play/conversations/play-conversations.tsx:121` — Filtering Conversations announces nothing: the result ('No conversation titles match …', or the new count) is not a status message
  - #175 · low · `frontend/components/play/deliverables/play-deliverable.tsx:114` — When a Deliverable link names a missing Deliverable, the 'isn't on your shelf' message replaces the status line without being announced
- **M74** (low) The style sheet's 12 mascot specimens are named with live status captions ('Something went wrong')
  - #67 · low · `frontend/components/looks/style/style-mascots.tsx:29` — The style sheet's 12 specimen Autos are exposed as images named with live-status captions ('Something went wrong', 'Automatos isn't running')
  - #122 · low · `frontend/components/looks/style/style-mascots.tsx:29` — Each style-sheet mascot specimen is announced with its live status caption ('Automatos isn't running', 'Something went wrong')

**Files**

- components/looks/look-notice.tsx
- look-manager.tsx, look-import.tsx, style-screen.tsx (a seq counter)
- components/play/conversations/play-conversations.tsx
- components/play/deliverables/play-deliverable.tsx
- components/mascot/auto-mascot.tsx (a `decorative` prop)
- components/looks/style/style-mascots.tsx
- tests: look-manage.test.tsx, play-conversations.test.tsx, play-deliverable.test.tsx, style-screen.test.tsx

**Change**

1. Each notice carries a `seq` from a useRef counter; LookNotice/CopyResult keep the polite &lt;p&gt; mounted and render `<span key={seq}>`; the role=alert div is keyed by seq, so a repeat is re-announced.
2. Conversations: one persistent `<p role="status">` under the search box, visually hidden for a match, with the count or the no-match line (named constants).
3. The Deliverable page's loading and missing texts share one persistent role=status paragraph.
4. AutoMascot `decorative` (default false): aria-hidden, no role or label; StyleMascots passes it (the slot's visible name already says the state).

**Proving test (fails before, passes after)**

- Paste the same JSON twice: the second 'Added … to your Looks.' span is a new node; two identical bad pastes re-insert the alert.
- Typing a query updates getByRole('status') with the count; a non-matching query shows the no-match text.
- A 404 answer yields getByRole('status') with NOT_ON_SHELF.
- On /looks/style, no img named 'Something went wrong' and the 12 slot names are present. All fail today.

**Code shape:** All components well under 150.

### P260-FX-25 — One width source for Play

**Severity:** low · **Depends on:** P260-FX-11, P260-FX-21, P260-FX-23

**Closes**

- **M69** (low) useIsTabletOrBelow starts false, so Play mounts for one render on phones and tablets, at five places, with side effects (deep link consumed, cold start raced, requests fired, 12 posters fetched)
  - #1 · low (finder: medium) · `frontend/components/play/play-route-switch.tsx:37` — Play's tree flashes for one frame on phones/tablets under Play before the width switch flips (useIsTabletOrBelow starts false)
  - #90 · low · `frontend/components/layout/main-layout.tsx:148` — The 'Play at ≥1024 px' check is re-implemented at five places, each with its own useIsTabletOrBelow that starts false, so Play's components mount for one render on phones
  - #124 · low · `frontend/components/looks/play-only-route.tsx:40` — PlayOnlyRoute mounts Play's full screen on phones and tablets before the width check flips (12 mascot PNGs and videos on /looks/style)

**Files**

- hooks/use-studio-theme.ts (usePlayLayout)
- components/layout/main-layout.tsx
- components/play/play-route-switch.tsx
- the not-found view
- components/looks/play-only-route.tsx
- components/play/conversations/play-conversations.tsx
- tests

**Change**

1. `usePlayLayout(): 'pending' | 'play' | 'fallback'` built on useSyncExternalStore over matchMedia('(max-width: 1023px)') with 'pending' as the server snapshot; the five places use it and render nothing Play-specific while 'pending' (MainLayout can keep its current fallback). useIsTabletOrBelow and its other callers are unchanged.

**Proving test (fails before, passes after)**

- Phone-width jsdom under Play: the chat layout never calls openChat or router.replace and never mounts PlayFrame (spy on useRecentChats); /looks/style's first render has no editor aside and no mascot slots. Fails today.

**Code shape:** MainLayout about 107 lines (150).

**Note:** useIsTabletOrBelow starting false is on main (Studio's phone flash is the same); the new side effects come from PRD-260's five call sites.

### P260-FX-26 — Small wiring: the Discuss hint, the widget's page, ⌘K's Play pages

**Severity:** low

**Closes**

- **M67** (low) Under Play, Discuss tells the owner to pick another agent 'in the message box', which Play's box can't do
  - #30 · low · `frontend/app/chat/layout.tsx:26` — The discussion bar over Play's chat tells the owner to pick another agent 'in the message box', which Play's box has no way to do
  - #14 · low · `frontend/components/chatbot/discussion-bar.tsx:64` — Under Play, Discuss tells the owner to "pick them in the message box", but Play's message box has no agent picker, so the ticket's agent can't be addressed from Play's chat
- **M70** (low) The floating Auto widget on Conversations and Make it yours says 'Ask me anything about Dashboard' and sends page 'dashboard'
  - #116 · low · `frontend/components/layout/main-layout.tsx:112` — Under Play, the floating Auto widget on Conversations, Make it yours and the style sheet says 'Ask me anything about Dashboard' and sends page 'dashboard'
- **M90** (low) ⌘K search under Play cannot find Play's own pages
  - #115 · low · `frontend/hooks/use-global-search.ts:25` — ⌘K search under Play cannot find Play's own pages (Conversations, Make it yours, the style sheet)

**Files**

- components/chatbot/discussion-bar.tsx and lib/discussion.ts (a pure discussionHint)
- components/layout/main-layout.tsx (getCurrentPage)
- components/chatbot/chat-widget.tsx (module-level PAGE_LABELS only)
- hooks/use-global-search.ts and components/shared/global-search.tsx
- tests

**Change**

1. M67: when Play's chat is on screen (useIsPlay() && !narrow), the hint reads "You are talking to Auto in Play's chat. To ask {agent}, switch to Studio from the theme menu."; Studio's text unchanged. An agent picker in Play's chat is decision F3.
2. M70: getCurrentPage maps /conversations → 'chat' and /looks → 'looks'; PAGE_LABELS gains looks: 'Make it yours'.
3. M90: PLAY_PAGES (Conversations, Make it yours, Style sheet) included in filterPages only when isPlay (GlobalSearch reads useIsPlay()).

**Proving test (fails before, passes after)**

- discussion-bar test under Play: no 'pick them in the message box', the Play hint present; Studio unchanged.
- main-layout test with the pathname /looks: the open widget says 'Make it yours', not 'Dashboard'.
- use-global-search: isPlay with 'conv' returns /conversations; under Studio it doesn't. All fail today.

**Code shape:** DiscussionBar about 44 lines; MainLayout about 107; AutoWidget is over the limit, so only the module-level PAGE_LABELS changes in chat-widget.tsx.

### P260-FX-27 — Hygiene: one /health poll owner, shared helpers and route constants, dead code

**Severity:** low · **Depends on:** P260-FX-21

**Closes**

- **M88** (low) GET /health is polled once per mounted useBackendHealth (twice per 30 s on /chat); the cookie writer and a Blob download helper are duplicated
  - #80 · low · `frontend/hooks/use-backend-health.ts:63` — GET /health is polled once per mounted useBackendHealth, so Play's chat sends it twice every 30 s
  - #89 · low · `frontend/lib/looks/tone.ts:35` — Small core helpers duplicated: a second copy of the cookie writer, another Blob download function, and a second /health poll
- **M93** (low) Routes and query-parameter names are re-declared as private constants across Play files
  - #88 · low · `frontend/hooks/use-play-conversation.ts:29` — Routes and query-parameter names are re-declared as separate constants across Play files instead of reusing the ones that already exist
- **M105** (low) lib/play/shelf.ts filterForType is dead code with no caller or test
  - #176 · low · `frontend/lib/play/shelf.ts:88` — lib/play/shelf.ts adds filterForType, which nothing calls or tests: the ?artifact_type= chip selection it was written for was never wired

**Files**

- hooks/use-backend-health.ts ({ poll } option; export the query key and fetcher)
- components/play/edges/asleep-gate.tsx (the poller)
- SessionModeTab (useSessionModeHealth reuses the exported query)
- new lib/cookies.ts (writeCookie) used by contexts/ui-style-context.tsx and lib/looks/tone.ts
- new lib/download-text.ts (downloadText; downloadMarkdownFile wraps it, keeping the delayed revoke)
- route constants: CHAT_HREF, LOOKS_HREF, LOOK_STYLE_HREF, DELIVERABLE_PARAM, DELIVERABLES_HREF (use-play-conversation.ts, app/conversations/page.tsx, theme-toggle.tsx, play-nav.ts, style-screen.tsx, deliverables.ts, play-deliverable.tsx)
- lib/play/shelf.ts (delete filterForType and its import)
- new hooks/__tests__/use-backend-health.test.tsx

**Change**

1. M88: `useBackendHealth({ poll }: { readonly poll?: boolean } = {})` with `refetchInterval: poll ? BACKEND_HEALTH_POLL_MS : false`; AsleepGate (always mounted in PlayFrame) polls, every other caller reads the shared query; SessionModeTab reuses the same query key so one cache serves both. One cookie writer and one Blob download helper.
2. M93: import the existing constants and delete the private copies (CHAT_ROUTE, OPEN_PARAM, the /looks literals). No behaviour change.
3. M105: delete filterForType and the isDeliverableType import (default; opening the shelf on a ?artifact_type= chip is decision F14).

**Proving test (fails before, passes after)**

- use-backend-health.test.tsx with fake timers: a polling consumer mounted at t=0 and a reader at t=5 s give one request per tick over 2 × BACKEND_HEALTH_POLL_MS; a reader alone never polls; Play's hook and SessionModeTab share one request. Fails today. The moves are covered by the existing tests.

**Code shape:** No function grows past its limit.

### P260-FX-28 — Guard and regression tests

**Severity:** low · **Depends on:** P260-FX-19

**Closes**

- **M96** (low) The prd260 guard misses oklch()/color-mix()/named colours and next/font/local paths and CSS @import/url() into the reference
  - #6 · low · `frontend/lib/__tests__/prd260-play-guards.test.ts:1` — The prd260 guard test's literal-colour check misses oklch(), color-mix() and named colours
  - #133 · low · `frontend/lib/__tests__/prd260-play-guards.test.ts:59` — The 'ported, never imported' guard misses next/font/local paths and CSS @import/url() references to the Play reference
- **M91** (low) The idle nap re-arming on input is untested
  - #134 · low · `frontend/hooks/use-idle-timeout.ts:40` — Untested: the idle nap re-arming on input (the Play reference's test was dropped), so Auto could nap while the owner is active

**Files**

- lib/__tests__/prd260-play-guards.test.ts
- new hooks/__tests__/use-idle-timeout.test.ts

**Change**

1. M96: the literal-colour guard flags oklch/oklab/lab/lch/hwb/color() and color-mix() whose arguments hold a literal colour, plus named colours; var(--pl-*), transparent and currentColor stay allowed, and CSS system colours are allowed inside `@media (forced-colors: active)` (FX-19). The 'ported, never imported' guard reads next/font/local `path`/`src` values and CSS `@import`/`url()` references and runs FORBIDDEN_PARTS on them; self-tests: a localFont call with a forbidden path and a CSS @import/url() count as 2 hits, a comment naming the folder as 0.
2. M91: port the reference's idle-timeout test: onIdle fires at IDLE_MS, not at IDLE_MS − 1; a keydown at IDLE_MS − 1000 pushes it a full IDLE_MS; onActive fires once on the first input after the nap.

**Proving test (fails before, passes after)**

- The new self-test fixtures fail against today's regexes and pass after; the idle-timeout cases document current behaviour (a regression net, not a bug fix).

**Code shape:** Tests only.

## A. Founder decisions

Behaviour or scope calls that a fix story can't make. Each has a recommendation and the fix it would take.

### F1 — Must Play's reply render the core's withdraw offer (F091-C2) and task cards?

**Findings**

- **M06** (high) Play's reply never renders the core's withdraw offer (F091-C2) or task cards, so a pending grant cannot be withdrawn from Play's chat
  - #25 · high (finder: medium) · `frontend/components/play/chat/auto-reply.tsx:104` — Play's reply drops the core chat's withdraw offer (F091-C2) and ticket cards
  - #10 · medium · `frontend/components/play/chat/auto-reply.tsx:100` — Auto's replies in Play never show the core's withdraw offer (F091-C2) or task cards (PRD-238 S6 / PRD-239 S2), so a pending request can't be withdrawn from Play's chat

**Severity:** high · **Recommendation:** Yes, before merge: it is a safety fix and costs one small component.

**If yes:** A ReplyCards component in AutoReply renders the core's WithdrawOfferCard (it calls useDenyApproval itself) and TaskCard for message.taskCards and stored `task_card` parts (a pure taskCardsOf in message-text.ts), framed by a --pl-* module. Test in play-chat-turn.test.tsx: withdrawOffer `{requests:[{grant_id:7}]}` renders Withdraw, and clicking it calls the mocked useDenyApproval mutateAsync(7); live taskCards and a stored task_card part each render a card. AutoReply is 120 of 150 lines, so the cards go in their own component.

### F2 — Must Play's chat show the SaaS onboarding and trial surfaces (opener, connect cards, power-up card, setup checklist, trial pill)?

**Findings**

- **M32** (medium) Play's chat drops the SaaS onboarding and trial surfaces chat.tsx shows (onboarding opener, connect cards, power-up card, setup checklist, trial pill)
  - #9 · medium · `frontend/components/play/chat/play-chat-view.tsx:211` — Play's chat drops the hosted-edition onboarding and trial surfaces chat.tsx shows: the onboarding opener, inline connect cards, the power-up card, the setup checklist and the trial balance pill
  - #72 · medium · `frontend/components/play/chat/play-chat-view.tsx:182` — Play's chat drops the SaaS trial and onboarding surfaces that Studio's chat shows: the trial balance pill, the power-up card, the inline app-connect cards, the setup checklist and the onboarding opener

**Severity:** medium · **Recommendation:** Yes for the trial pill, power-up and connect cards; land after P260-FX-07 (PowerUpCard calls refreshWorkspace). The alternative is to keep a workspace in onboarding on the core chat.

**If yes:** components/play/chat/onboarding-slot.tsx reuses the core's TrialBalancePill, PowerUpCard (stage 'powerup' and trial not exhausted, via isTrialBlocked), OnboardingConnectCards and SetupChecklistCard (stage 'completed') in PlayChatView's footer; at 'not_started' the core's OnboardingOpener replaces the hello line and the example chips are hidden. Tests per stage under NEXT_PUBLIC_AUTH_EDITION=saas: trial-balance-pill, power-up-card, onboarding-connect-cards, the checklist, onboarding-opener with no chips.

### F3 — Must Play's chat show the core's tool results (knowledge sources, database results with SQL and clarification questions, code snippets, canvas widgets), and an agent picker for Discuss (M67's fuller fix)?

**Findings**

- **M78** (low) Play's replies drop the tool results the core chat shows (knowledge sources, database results, code snippets, canvas widgets)
  - #163 · low (finder: medium) · `frontend/components/play/chat/auto-reply.tsx:104` — Play's replies drop the tool results the core chat shows: knowledge sources, database results (SQL, rows, clarification questions), code snippets and canvas widgets

**Severity:** low · **Recommendation:** Not in this merge: neither is in §4 or the approved design. P260-FX-26 makes the Discuss hint honest now.

**If yes:** Tool results: a components/play/chat/tool-results.tsx after the answer reading message.documents, database_results and codeSnippets. Agent picker: reuse AgentSelector and pass selectedAgentId to useChatWithReceipts.

### F4 — Stop: keep Auto's partial answer with 'You stopped this answer.' (the approved Play app) or drop it (Studio's useChat)?

**Findings**

- **M33** (medium) Stop erases Auto's partial answer and its tool lines; the approved Play app kept them with 'You stopped this answer.'
  - #23 · medium · `frontend/components/play/chat/play-chat-view.tsx:218` — Stop erases Auto's partial answer and its tool lines (the approved Play app keeps them; checklist step 18 expects 'stopped' lines)
  - #138 · medium · `frontend/components/play/chat/play-chat-view.tsx:218` — Stop discards Auto's partial answer, where the approved Play app kept it with 'You stopped this answer.'

**Severity:** medium · **Recommendation:** Keep it, Play-side, without editing useChat (over the length limit). Decides checklist step 18 (B, M61).

**If yes:** A hooks/use-stopped-reply.ts wrapper snapshots the reply being written on Stop, calls chat.stop(), and renders the snapshot in MessageList's `after` slot as a non-streaming AutoReply with STOPPED_NOTICE until the next send or Try again; running calls read 'Stopped: …' through toolStatusOf. Test with the real useChatWithReceipts over a stubbed stream: words plus a tool-start, Stop → the words stay, 'You stopped this answer.' and 'Stopped: searching the web'.

### F5 — Looks, the active look and the seen-Deliverables record: per browser, or per signed-in user?

**Findings**

- **M51** (medium) Looks, the active look and the seen-Deliverables record are kept per browser, not per user or workspace
  - #43 · medium (finder: low) · `frontend/lib/looks/storage.ts:12` — Looks are kept per browser, not per user: a second account in the same browser gets the first account's looks and active look
  - #125 · medium (finder: low) · `frontend/lib/looks/storage.ts:12` — Saved looks are shared by every account that uses the same browser, though PRD §2 says looks are saved per user
  - #13 · low · `frontend/lib/looks/storage.ts:11` — Goal "Looks are saved per user" is only per browser: every account on a browser shares the saved looks and the active look
  - #5 · low · `frontend/lib/play/seen.ts:34` — The 'seen Deliverables' record is per browser, not per user or workspace

**Severity:** medium · **Recommendation:** The verifiers split: #13/#43/#125 read §2's 'Looks are saved per user, starting in the browser' as per account now; rejected #73 read §6/D10 ('saving looks per user on the server' is a non-goal) as making per-browser storage intended. Recommended: per user in SaaS (key by user id; seen record by workspace), local edition unchanged; do it inside P260-FX-20.

**If yes:** lookStorageKeys(userId) → 'automatos-play.looks:&lt;id&gt;' / 'automatos-play.look:&lt;id&gt;' ('local' in the local edition); LookProvider re-reads on a user change; a pointer key 'automatos-play.look.current' for the pre-paint script, which runs before auth; seen.ts keyed by workspace. Tests: storage isolation per id, a user change swaps the saved list and the active look, prepaint reads the pointer key.

### F6 — Should Play's Deliverable page preview generate_document's PDF/DOCX/XLSX (and audio) through the core's FilePreview, as Studio's sheet does, or keep the AC's file card?

**Findings**

- **M55** (medium) Every generate_document PDF/DOCX/XLSX (and audio) opens in Play as "Play can't preview this document here."; Studio's sheet previews them
  - #99 · medium (finder: high) · `frontend/lib/play/detail.ts:87` — Every document Auto makes with generate_document (PDF/DOCX/XLSX) opens in Play as "Play can't preview this document here." Studio's sheet previews the same files, and RUN checklist step 15 cannot pass

**Severity:** medium · **Recommendation:** Preview: it is the Deliverable Auto makes most often. Decides checklist step 15 (B, M62).

**If yes:** lib/play/detail.ts adds ContentView `{kind:'preview'; previewType; url}` when inferPreviewType gives pdf/docx/xlsx/audio, and BINARY_TYPES gains m4a, flac, aac, doc, xls, ppt, odt, ods, odp, key, rar, 7z, bz2; deliverable-content.tsx renders `<FilePreview url previewType filename className={styles.filePreview}/>` (min-height 70vh). Tests: a 'generated' pdf yields kind 'preview'; an .m4a with text content is never 'text'; the page renders the frame, not the file card.

### F7 — Allow a backend change in this wave (K11, gate N3) to retract the last turn before Regenerate / Try again?

**Findings**

- **M08** (medium) Regenerate / Try again re-asks on top of the old turn: the server keeps the first question and its reply (or its error note)
  - #15 · high (finder: medium) · `frontend/lib/chat/use-chat-with-receipts.ts:41` — Regenerate / Try again re-asks on top of the old turn: the server keeps the first question and its reply (or its error note), so the model reads them and a reopened chat shows the question twice

**Severity:** medium · **Recommendation:** Yes, as a separate small backend PR; Play's Try again works meanwhile, with a duplicated question in the stored history.

**If yes:** ChatService.delete_last_turn(chat_id, workspace_id) deletes the last user message and everything after it in one transaction; POST /api/chat/{chat_id}/retract-last-turn (require_workspace_permission('agents:execute'), 404 outside the workspace); regenerate the route manifest; lib/chat/api.ts retractLastTurn through apiClient; useAskAgain awaits it when the chat has an id and on failure toasts and sends nothing. pytest for the service and route; vitest that reload awaits the retract and a rejected retract leaves the messages unchanged.

### F8 — Should receipts link a Deliverable (backend receipts.py `_link` reading deliverable_id; M76's backend half), so a reopened conversation shows the Deliverable card?

**Severity:** low · **Recommendation:** Later, with F7's backend PR. P260-FX-09 corrects the docs and accepts the real open_url meanwhile.

**If yes:** `_link` maps a done answer's string deliverable_id to '/deliverables?deliverable={id}', with a pytest case for a generate_document result with deliverable_id 'd-7'.

### F9 — Amend D3: under Play, Studio's consequence tokens (--accent, --destructive and their foregrounds and tint) follow Play's coral, not the look's accent (M52, P260-FX-17)?

**Severity:** medium · **Recommendation:** Yes. Studio itself sets --accent and --destructive to one burnt orange, and the founder's printed rule is 'Errors stay coral in every look'. Unblocks P260-FX-17 part 1.

**If yes:** See P260-FX-17.

### F10 — Add a 'Pause Auto' control so the looping clips, ring pulse and cursor blink can be stopped without the OS setting (WCAG 2.2.2, Level A)?

**Findings**

- **M24** (medium) Auto's loops, the ring pulse and the cursor blink run with no in-page way to pause them (WCAG 2.2.2, Level A)
  - #63 · medium · `frontend/components/mascot/auto-mascot.tsx:60` — Auto's clips loop indefinitely (12 at once on the style sheet), and the ring pulse and the cursor blink run on, with no way to pause them except the OS reduced-motion setting (WCAG 2.2.2, Level A)

**Severity:** medium · **Recommendation:** Yes, a per-browser preference.

**If yes:** LookProvider keeps a paused flag in localStorage (try/catch, as storage.ts); useLook().reducedMotion becomes `osReduced || paused` and drives data-pl-motion='reduce'; AutoMascot reads useLook().reducedMotion; a 'Pause Auto' / 'Play Auto' toggle (aria-pressed) under the caption in MascotColumn and in the style sheet's Motion section. Test: the toggle swaps &lt;video&gt; for &lt;img&gt;, sets data-pl-motion on &lt;html&gt; and survives a remount.

### F11 — Step-2 (Design Kit) architecture: move theme-agnostic chat parsing out of components/play, drive the Style axis from a per-style table, and add a theme registry for screens (M17's follow-up), now or at the start of step 2?

**Findings**

- **M28** (medium) Theme-agnostic chat logic (approval/mission/Deliverable parsing, deep link and cold start) lives under components/play and duplicates the core's copies, which already differ
  - #86 · medium · `frontend/components/play/chat/approvals.ts:114` — Chat logic that doesn't depend on the theme lives under components/play and duplicates the core chat's parsing (the copies already differ), so a second theme must import 'play' code or copy it again
- **M106** (low) The Style axis grows by literal 'play' checks and per-theme booleans, not a per-style table
  - #83 · low (finder: medium) · `frontend/lib/ui-style.ts:61` — The Style axis is extended by literal 'play' checks and per-theme booleans, not a per-style table, so adding a fourth style compiles but renders the Classic layout and is missing from the menu

**Severity:** medium · **Recommendation:** At the start of step 2. P260-FX-23 does the dynamic loading (the user-visible cost) now. Note that M28 also found Studio's mission card dropping the EU-AI-Act oversight fields Play's parser keeps (a Studio gap on main).

**If yes:** M28: pure moves to lib/chat/approvals.ts, lib/chat/mission-suggestion.ts, lib/chat/turn-deliverables.ts and renamed hooks. M106: rendersStudio from HTML_CLASSES_BY_STYLE, STYLE_LABELS and LOOK_EDITOR_HREF_BY_STYLE tables, the menu mapped from UI_STYLES, Tone locked by `forcedTheme !== undefined`.

### F12 — Copy: tool-line durations '2.1 s' (approved design) or the core's '2 s'; 'Saved just now' stamped on every saved look forever; the editor label "Auto's light · eyes + ring" when the eyes are baked into the clips?

**Findings**

- **M79** (low) Tool-line durations round to whole seconds ('2 s'); the approved design reads '2.1 s'
  - #54 · low · `frontend/components/play/chat/labels.ts:126` — Tool-line durations round to whole seconds ('2 s'), where the approved design reads '2.1 s'
- **M97** (low) Every look made with 'Save as new look' says 'Saved just now' forever
  - #156 · low · `frontend/lib/looks/editor.ts:74` — Every look made with 'Save as new look' says 'Saved just now' forever, in the Looks row and in its exported file

**Severity:** low · **Recommendation:** '2.1 s' through a Play-only formatter (SECONDS_DECIMALS = 1); SAVED_SUB = 'Your look'; label "Auto's light · ring".

**If yes:** Durations: toolSeconds(ms) in labels.ts, test '· 2.1 s' for 2140 ms. SAVED_SUB: lib/looks/editor.ts only, test asNewLook(...).sub has no time words. Label: look-options.tsx:138.

### F13 — Pre-existing defects on main (section C: M15, M40, M42, M48, M86, M92): fix in their own PRs off main, or ride in this branch?

**Severity:** medium · **Recommendation:** Own PRs off main; M40 (SVG scripts in the app's origin) and M92 (unencoded ?chatId=) first. M86 can ride in P260-FX-05 because it edits the same two lines.

**If yes:** See section C.

### F14 — (M105) ?artifact_type= links: open Play's shelf on that type chip, or keep sending them to Studio's page (today)?

**Severity:** low · **Recommendation:** Keep today's routing; P260-FX-27 deletes the dead filterForType.

**If yes:** Drop 'artifact_type' from CORE_DELIVERABLES_PARAMS and seed ShelfScreen's chip from filterForType(searchParams.get('artifact_type')), with a play-shelf test.

## B. Docs-only fixes

Text edits in `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md` and `docs/frontend-architecture/play-theme.md`; no test needed. Doc edits that come with a code change are listed in their story.

- **M58** (low) play-theme.md's T-008a/b and T-009 sections were never checked against the merged code; the /looks/style note is misquoted (also checklist step 46)
  - #92 · low · `docs/frontend-architecture/play-theme.md:409` — play-theme.md still says three sections were 'written from the acceptance criteria … check against the code once merged', although that code is in this branch, and one of those sections misquotes the code
  - #126 · low · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:180` — The RUN checklist expects the Make it yours note on /looks/style, which shows a different note
  - #141 · low · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:180` — /looks/style's note is 'The style sheet is part of the Play theme.', not the text step 46 and play-theme.md give; the T-008a/b/T-009 sections were never checked against the merged code
  - **Edit:** docs/frontend-architecture/play-theme.md: rewrite the T-008a, T-008b and T-009 sections from the merged code and drop the 'written from the acceptance criteria … check once merged' caveats (lines 66, 409, 417); quote each page's own note ('Make it yours is part of the Play theme.' on /looks, 'The style sheet is part of the Play theme.' on /looks/style); describe PlayOnlyRoute's three cases (Play ≥ 1024 px; Classic/Studio with Switch to Play; Play below 1024 px with the wider-screen note) and AsleepGate's whole-window swap; fill the 'How it mounts' cells (PlayOnlyRoute; not-found.tsx; AsleepGate in PlayFrame); add use-backend-mascot and use-recheck-countdown to the hooks line; replace 'the code on feat/prd-260-play-theme' with 'the code'. Checklist step 46: /looks shows 'Make it yours is part of the Play theme.', /looks/style shows 'The style sheet is part of the Play theme.', each with Switch to Play and no redirect loop.
- **M59** (low) The RUN checklist and ownerTest cover only the local edition
  - #76 · low · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:3` — The RUN checklist and ownerTest cover only the local edition: no step exercises Play in SaaS before merge
  - **Edit:** Checklist: add section 'O. The hosted edition', run on a NEXT_PUBLIC_AUTH_EDITION=saas build against a Clerk test instance with a trial workspace: switch to Play and sign out/in from Play's header; reload /deliverables and click New chat (a new chat opens; tabs survive in Studio and a second browser); exhaust the trial and save a key from the banner (the conversation stays); stop the backend (the hosted asleep line); walk the trial pill and onboarding stages. Map it in the ownerTest map.
- **M60** (medium) Checklist step 43 asks for Try again after the asleep screen, which removes it
  - #136 · medium · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:168` — Step 43 asks for Try again after the asleep screen, but the asleep swap removes the snag box, so Try again isn't there
  - **Edit:** Checklist step 43: stop the backend while words stream → the reply goes and 'I hit a snag' / "Auto couldn't answer that just now." with Try again shows under your message, with a toast, and Auto plays oops; press Try again while the backend is still stopped → the same snag (proves it re-sends); within the next poll plus about 7.5 s the asleep screen takes the window; start the backend → the same conversation, no snag box (a failed turn isn't kept), send again by hand. play-theme.md:420: the page and its unsaved state, a failed turn's Try again included, don't survive the swap. Re-check after P260-FX-07 and P260-FX-21 change the remount and the timings.
- **M61** (medium) Checklist step 18 expects Stop to keep the words and a 'stopped' tool line
  - #137 · medium · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:78` — Step 18 expects Stop to keep the streamed words and a 'stopped' tool line, but useChat removes the whole reply on Stop
  - **Edit:** Checklist step 18 (depends on F4): if Stop keeps the partial answer, expect the words, 'You stopped this answer.' and 'Stopped: …' tool lines; if not, expect the reply being written to go (words and tool lines, as in Studio), Auto to leave thinking, the next message to stream normally, and the backend log to show the first turn cancelled. Step 20 and play-theme.md:400 must say the same.
- **M62** (low) Checklist step 15 expects the step-14 PDF rendered on the sheet
  - #139 · low (finder: medium) · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:69` — Step 15 expects the step-14 Deliverable rendered as headings and lists on the sheet, but generate_document makes a PDF/DOCX, which Play shows only as a download card
  - **Edit:** Checklist step 15 (depends on F6): if Play previews documents, expect the PDF in the page; if not, expect the file card ("Play can't preview this document here." with 'Download the file'). Either way add a printed-sheet check on a markdown Deliverable (a Report from the shelf's Report chip, e.g. step 19's Mission report): headings and lists rendered, no raw HTML.
- **M63** (low) Checklist steps 41/42 and play-theme.md expect the pill beside the asleep screen
  - #140 · low · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:163` — Steps 41/42 and play-theme.md say the pill reads 'Not running' beside the asleep screen, but the screen replaces the whole frame, sidebar and pill included
  - **Edit:** Checklist step 41: 'only then does the whole window become "Automatos is asleep" (no sidebar): "Not connected" at the top, Auto sleeping, the start command (local edition), Check again, the countdown and "Automatos didn't answer at localhost."'; step 42 keeps 'the pill says "Connected · localhost"'. play-theme.md:420: AsleepGate shows it in place of Play's frame and its page (the whole window, Ember mock 07).
- **M64** (low) Checklist step 31 says Auto's eyes keep the light's colour under a Cobalt shell
  - #142 · low · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:125` — Step 31 says Auto's eyes keep the light's colour under a Cobalt shell, but the eyes are baked into the clips and take the shell filter
  - **Edit:** Checklist step 31: "Auto's body and eyes take the blue tint in the preview and, on /chat, in the mascot column, in the clip and in the still; his ring keeps the light's colour." play-theme.md:226 and the play-theme.css:34 comment: 'the light: Auto's ring and the connected dot (the eyes are part of the clips)'. The editor label is decision F12.
- **M65** (low) Checklist step 23 (Safari) doesn't say to switch Safari to Play first
  - #144 · low · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:94` — Step 23 (Safari) doesn't say to switch Safari to Play first, so its step 8 opens Studio's chat with no Auto
  - **Edit:** Checklist step 23: 'In Safari, choose Play from the theme menu first (Safari has its own cookies and storage), then repeat steps 8 and 12.'
- **M66** (low) Checklist step 30 expects Auto's ring to change between three presets that all use Mint
  - #149 · low · `docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:122` — RUN checklist step 30 expects Auto's ring to change between Ember, Obsidian and Daylight, but all three presets use the Mint light
  - **Edit:** Checklist step 30: name what changes (the sidebar, the page, the preview's accent and type, the tone) and add 'Auto's ring stays Mint: all three presets use the Mint light (Aurora uses Ice); the light is changed in step 32', or pick Ember, Aurora and Daylight so the ring visibly changes.

## C. Pre-existing on main, exposed by PRD-260

Same behaviour on main (checked with `git show d7affad605:<path>`, or the verifiers' diff); PRD-260 didn't introduce them, it adds another way to reach them. How to handle them is decision F13.

- **M15** (medium) Studio-Light colour literals in globals.css misbehave under the Dark tone (Playbooks primary button label, plan DAG steps, kanban drag-over)
  - #53 · medium · `frontend/app/globals.css:2865` — Studio pages inside Play's frame still paint Studio-Light literals under the forced Dark tone: the Playbooks primary button's label vanishes on hover, and the plan DAG and kanban drag-over turn near-white
  - **On main:** globals.css's five Studio-Light literals (`.l-btn--p:hover`, `.cc-kb-body.drag-over`, `.sh-chat-thread-close:hover`, dag-step queued/err) are byte-identical on main and already wrong in Studio Dark, the default tone. Play makes Dark the default for three of four grounds.
  - **Handling:** Own PR on main: swap each literal for a token in place (globals.css doesn't grow): `color-mix(in srgb, var(--fg) 88%, transparent)`, `var(--accent-2)`, `hsl(var(--cc-hover))`, `hsl(var(--info) / 0.08)` with `/ 0.3` borders, `hsl(var(--accent) / 0.08)`. Guard in dark-variant-tokens.test.ts: those selectors hold no literal hsl()/hex. Changes Studio Dark visuals (fixes them).
- **M40** (medium) An SVG image Deliverable, shown as a same-origin blob: URL, runs its scripts as the app when opened in a new tab
  - #170 · medium · `frontend/components/widgets/FileWidget/FilePreview.tsx:109` — An SVG image Deliverable, shown on Play's shelf and Deliverable page as a same-origin blob: URL, runs its scripts as the app when opened in a new tab (pre-existing in Studio's FilePreview, reused here)
  - **On main:** useAuthenticatedBlobUrl (FilePreview.tsx, unchanged by PRD-260) hands an SVG to &lt;img&gt; as a same-origin blob: URL; 'Open image in new tab' runs its scripts as the app (CSP allows 'unsafe-inline'). Studio's document-thumbnail.tsx:25 and deliverable-card.tsx:86 have it on main; Play adds the shelf card and the Deliverable page.
  - **Handling:** Own PR on main, first: for image/svg+xml (or a .svg path) with no forcedType, hand &lt;img&gt; a data: URL (FileReader.readAsDataURL; top-level navigation to data: is refused, img-src allows data:); keep revoked object URLs for every other type. Tests: SVG → 'data:image/svg+xml' src; PNG → 'blob:' revoked on unmount. useAuthenticatedBlobUrl is about 43 of 50 lines: put the branch in a small callee.
- **M42** (medium) The style cookie is written only on a switch, so Safari's 7-day cap on document.cookie silently drops Play (and Classic)
  - #35 · medium · `frontend/contexts/ui-style-context.tsx:40` — The style cookie is written only when the user switches style, so Safari's 7-day cap on document.cookie silently drops Play (and Classic); the localStorage mirror is never read
  - **On main:** The automatos-style cookie is written only on a switch (ui-style-context.tsx, unchanged mechanism since PRD-244), so Safari's 7-day cap on document.cookie drops Play and Classic alike; the localStorage mirror is never read.
  - **Handling:** Own PR on main: split writeUiStyleCookie out of applyUiStyleToDocument; on mount, restore from the mirror when the cookie is gone, else rewrite it to restart the window. Tests: cookie written on mount; no cookie + mirror 'play' → Play with both classes; the defaults migration still wins in a fresh browser.
- **M48** (medium) Stop during an awaited (detached) reply leaves Stop, a thinking Auto and a blocked box for 5 minutes
  - #162 · medium · `frontend/lib/chat/hooks.ts:96` — Stop during an awaited (detached) reply leaves Stop, thinking Auto and a blocked box for 5 minutes
  - **On main:** Stop during an awaited (detached) reply only POSTs cancel; the server sends chat_changed only for completed turns, so awaitingReply stays for the 5-minute guard. Same useChat in Studio's chat and the widget.
  - **Handling:** Own PR on main: useChat is far over the length limit, so move the awaited-reply state out by pure deletion into a hooks/use-awaited-reply.ts composed by useChatWithReceipts (placeholder, guard, merge, isLoading/status overlay), whose stop wrapper releases the wait after cancelChatTurn; switch chat-widget.tsx to useChatWithReceipts. Test: initialAwaitingReply true, stop() → cancelChatTurn called, isLoading false, status idle, no 'awaiting-reply' message.
- **M86** (low) The 'instruction' memory badge and the 'Cancelled' workflow badge paint their text in their own background colour (1:1)
  - #46 · low · `frontend/components/widgets/MemoryWidget/MemoryItem.tsx:57` — The 'instruction' memory badge and the 'Cancelled' workflow badge paint their text in their own background colour; the PR edited both lines and left the fix to a PRD-259 story that is not in this merge
  - **On main:** 'instruction' memory badge and 'Cancelled' workflow badge are `bg-warning text-warning` (1:1) on main; PRD-260 edited both lines (removing the redundant dark: copy) and left them, citing a PRD-259 story that isn't in this merge.
  - **Handling:** Recommended to ride in P260-FX-05 (same lines): `bg-warning text-warning-foreground` at MemoryItem.tsx:57 and WorkflowStatus.tsx:86, a guard against `bg-X text-X` on one semantic token, and render assertions for both badges.
- **M92** (low) ?chatId= from the URL goes unencoded into API paths and is stored unvalidated as the chat id
  - #2 · low · `frontend/hooks/use-play-conversation.ts:97` — ?chatId= from the URL goes unencoded into API paths and Play stores the URL value as the chat id
  - **On main:** ?chatId= goes unencoded into getChat/getChatMessages/cancelChatTurn paths and is stored unvalidated as the active chat id; byte-identical in app/chat/page.tsx and lib/chat/api.ts on main, repeated by Play's useDeepLink.
  - **Handling:** Own PR on main: validate the id against the chat-id pattern before openChat in both chats, encodeURIComponent in lib/chat/api.ts, and use the response's chat.id. Tests: '/chat?chatId=x/../..' opens nothing and makes no request outside /api/chat/&lt;encoded&gt;.

Partly pre-existing roots fixed inside this branch (PRD-260 made them reachable or worse, and the fix sits in its files):

| Merged | Where | What is on main |
|---|---|---|
| M07 | P260-FX-06 | Root re-render clobbering &lt;html class&gt; exists for Classic↔Studio on main. |
| M08 | F7 | No server-side retraction on main; Studio's old reload mostly sent nothing. |
| M09 | P260-FX-07 | The workspace-refresh hydration blip in use-chat-session.ts is on main; Play's consumer turns it into a stale remount. |
| M30 | P260-FX-09 | The empty-content resend existed in the old reload on main; moved into the new retry.ts. |
| M41 | P260-FX-05 | The dark: class strings are on main but generated no CSS until PRD-260's darkMode cherry-pick. |
| M49 | P260-FX-12 | Studio's sheet downloads the JSON envelope on main; the shared helper is new in this branch. |
| M57 | P260-FX-12 | useDeliverables' success:false handling is on main; Play's shelfView is the first consumer to read it. |
| M69 | P260-FX-25 | useIsTabletOrBelow starting false is on main (Studio's phone flash). |
| M75 | P260-FX-15 | next.config.js sets no cache header for public/ on main; only PRD-260's clips are re-requested. |
| M76 | P260-FX-09 + F8 | receipts.py never links a Deliverable on main (known in the reference app). |
| M36 | P260-FX-13 | MarkdownView's pre → CodeBlock default is on main; Play's composition exposes it. |
| M39 | P260-FX-13 | MarkdownView never set skipHtml on main. |
| M82 | P260-FX-12 | useApiFileDownload's default message is on main, written for other pages. |

## Rejected findings (14)

| # | Finding | Why rejected |
|---|---|---|
| #0 | Try again can drop the owner's message while a detached reply is awaited (useAskAgain guards only isLoading) (`frontend/lib/chat/use-chat-with-receipts.ts:35`) | useAskAgain reads the hook's returned isLoading, which already includes awaitingReply (hooks.ts:590); awaitingRef and the state always change together. |
| #3 | A second ?draft= link replaces text the user already typed in an empty new chat (`frontend/components/play/chat/play-chat-view.tsx:105`) | Not reachable: the only ?draft= link is on the Deliverable page, a different route from the chat. |
| #4 | PlayRouteSwitch reads useSearchParams without the Suspense boundary its own comment promises; it builds only because the root layout forces dynamic rendering (`frontend/app/chat/layout.tsx:1`) | The root layout awaits cookies(), so every route renders dynamically and nothing breaks today (the same latent pattern was confirmed as low in the Deliverables layout, M14, and P260-FX-11 covers both). |
| #73 | Looks are stored per browser, not per user as PRD §2 says, and sign-out leaves them for the next account (`frontend/lib/looks/storage.ts:12`) | Read §6/D10 as making per-browser looks intended; it contradicts three confirmed findings, so it is decision F5. |
| #85 | LookProvider clears &lt;html&gt;'s inline Studio tokens and resets the forced tone on every effect run while Play is off, not only when leaving Play, so it will wipe a second theme's look (`frontend/components/looks/look-provider.tsx:72`) | Only a hypothetical second theme's look would be wiped; that is step 2's design work, not a defect today. |
| #87 | Play's nav repeats StudioSidebar's role and plan filters instead of sharing one helper, so a new gate added to one sidebar won't reach the other (`frontend/components/play/shell/play-nav.ts:78`) | Play's nav imports the same filterNavForEdition and isNavItemVisible as StudioSidebar; a new gate reaches both. |
| #91 | The look engine and editor have generic names (lib/looks, components/looks, /looks, useLook) but are bound to Play, and the Studio-token mapping is keyed by Play's catalogue keys (`frontend/lib/looks/prepaint.ts:85`) | Generic names on Play-bound code is a step-2 (Design Kit) concern the PRD scopes out. |
| #98 | With memories injected, Auto shows 'Reading' through the whole answer: T-003b's 'words end thinking' never fires, because memory-injected moves him to reading and text only clears thinking (`frontend/lib/mascot/reducer.ts:143`) | Reading staying on screen through the answer after memory-injected is the approved app's own ranking (reading &gt; thinking), ported 1:1. |
| #106 | A failed Load more is remembered across a workspace switch: the next workspace's shelf says "More Deliverables didn't load." (`frontend/components/play/deliverables/play-shelf.tsx:87`) | No in-app workspace switcher exists, so a failed Load more can't carry across workspaces. |
| #131 | The K3 'exactly' and 'leaves nothing behind' look tests only read the names the code already lists, so an extra inline property can't fail them (`frontend/lib/looks/__tests__/helpers.ts:38`) | --pl-glow is part of LOOK_VAR_NAMES, so the K3 tests already cover every property the code sets. |
| #145 | Step 26's 'more than 24 Deliverables' ignores that the shelf leaves heartbeat outputs out (`docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md:107`) | No screen shows a heartbeat-inclusive Deliverables count to compare with the shelf's. |
| #166 | Send and Stop are two different buttons swapped in place, so keyboard focus drops to &lt;body&gt; when a turn starts and again when it ends (`frontend/components/play/chat/message-box.tsx:71`) | Send and Stop are the same &lt;button&gt; element at the same position; React updates it in place, so focus stays. |
| #168 | Under Play, a Deliverable's page has no Delete, Add to Knowledge or Open in Explorer, and no tags or summary, and K8 keeps Studio's sheet with those actions closed (`frontend/components/play/deliverables/detail-panel.tsx:39`) | Play's narrower Deliverable panel (no Delete, Knowledge, Explorer, tags) matches the AC and the reference; a product question, not a defect. |
| #173 | Auto's shelf caption reads the chip's count, not the shelf's: 'Your first Deliverable!' on a chip with one item, 'Your shelf is waiting' on an empty chip (`frontend/hooks/use-shelf-mascot.ts:47`) | Auto's shelf caption and the header count both read the active chip's total, as in the reference: consistent by design. |

The salvage file's one unverified item (no `reasoning` event can reach Auto's reducer) is the same defect as M54, fixed in P260-FX-16.

## Appendix: merged-finding index

| Merged | Severity | Finding | Sources | Goes to |
|---|---|---|---|---|
| M01 | high | Studio's element rules (`.studio h1/h2`, `.studio code/kbd`, `.studio .md-view a` and `h1-h3`) outrank Play's single-class CSS-module rules, because `<html>` carries `studio` under Play: every Play heading loses the look's type, inline code on the Deliverable sheet is invisible on dark grounds, sheet links read about 2.1:1 | #17, #32, #44, #48, #59, #102, #49, #50, #101, #58 | P260-FX-01 |
| M02 | high | Play's chat never mounts the chat_changed SSE lane (useBoardEventStream): an awaited (detached) reply never merges, the composer is blocked for 5 minutes, background posts never appear live | #127 | P260-FX-02 |
| M03 | high | The chat id adopted mid-turn is never cleared (every turn re-adopts): after New chat, reopening that conversation shows an empty first-open bound to it, and the next message lands in a hidden history | #20, #128 | P260-FX-03 |
| M04 | high | Play's New chat writes to the chat-session store before it is hydrated on any page but /chat: the previous conversation opens instead, and in SaaS an empty tab list is PUT to the server | #70, #110 | P260-FX-03 |
| M05 | high | A one-shot can be queued behind itself (deliverable-added mid-turn plus the done receipt at finish): Auto freezes on 'celebrate' for the rest of the conversation, or celebrates twice for one Deliverable | #93, #158 | P260-FX-04 |
| M06 | high | Play's reply never renders the core's withdraw offer (F091-C2) or task cards, so a pending grant cannot be withdrawn from Play's chat | #25, #10 | F1 |
| M07 | medium | A root re-render (router.refresh) after a client-side style switch rewrites `<html class>` from the server string and drops the tone class | #33 | P260-FX-06 |
| M08 | medium | Regenerate / Try again re-asks on top of the old turn: the server keeps the first question and its reply (or its error note) | #15 | F7 |
| M09 | medium | Play's conversation view is held above the remount boundary: after the asleep screen, or a workspace refresh (saving a key from the trial banner), the chat comes back from a stale snapshot | #22, #135, #71 | P260-FX-07 |
| M10 | medium | Play's cold start never cancels: a late history answer replaces a new chat or a discussion opened meanwhile | #161 | P260-FX-07 |
| M11 | low | /chat?chatId= with an empty value leaves Play's chat on 'Opening this conversation…' for good | #164 | P260-FX-07 |
| M12 | medium | New chat does nothing while the current new chat is still unnamed (its first send failed) | #24 | P260-FX-03 |
| M13 | medium | Every `/deliverables?tab=outputs` link, the backend's own Deliverable link included, opens Studio's Deliverables page under Play (against K8) | #34, #103, #129 | P260-FX-11 |
| M14 | low | PlayRouteSwitch reads useSearchParams outside a Suspense boundary in the Deliverables layout (latent build break) | #108 | P260-FX-11 |
| M15 | medium | Studio-Light colour literals in globals.css misbehave under the Dark tone (Playbooks primary button label, plan DAG steps, kanban drag-over) | #53 | C (pre-existing) |
| M16 | medium | Keyboard focus on the Deliverable sheet is drawn in the ground's accent text colour (1.06 to 2.62:1 on the sheet) | #39 | P260-FX-01 |
| M17 | medium | Play's frame, chat, shelf and 404 are statically imported at the core's seams, so their JS and CSS ship to every Classic and Studio user | #77, #84, #16 | P260-FX-23 |
| M18 | medium | On Paper, Make it yours marks the chosen option, look card and swatches only in the raw accent (below 3:1); unselected swatches have no edge | #40, #61, #119 | P260-FX-19 |
| M19 | medium | In forced colours (Windows High Contrast), Play's focus indicators, pressed states and colour samples disappear (`outline: none` + box-shadow, background-only samples) | #151, #64 | P260-FX-19 |
| M20 | medium | Keyboard focus falls to &lt;body&gt; when the control just used unmounts (Save, Save name, Cancel, Delete look, Keep it, Try again, the last Load more); Delete opens its confirmation without moving focus | #118, #68, #153 | P260-FX-22 |
| M21 | medium | Whole-frame swaps (the asleep gate, 'Switch to Play') drop focus to &lt;body&gt; and announce nothing | #66, #152 | P260-FX-22 |
| M22 | low | Status changes are not announced: a repeated notice, the Conversations filter result, a Deliverable link to a missing Deliverable | #154, #69, #175 | P260-FX-24 |
| M23 | medium | The style sheet autoplays all 12 mascot clips and posters at once (2.6-3.5 MB, 12 alpha decoders) | #78 | P260-FX-15 |
| M24 | medium | Auto's loops, the ring pulse and the cursor blink run with no in-page way to pause them (WCAG 2.2.2, Level A) | #63 | F10 |
| M25 | medium | A second tab's stale saved-looks list overwrites looks saved in another tab, and the tone cookie can disagree with the stored look | #38, #117 | P260-FX-20 |
| M26 | medium | When the browser blocks muted autoplay, Auto never reaches a clip boundary: stuck on hello/celebrate, picture frozen | #79, #95 | P260-FX-15 |
| M27 | medium | A celebrate queued behind a loop is usually skipped: the clip's final timeupdate and ended both apply clipBoundary before React re-renders | #94 | P260-FX-15 |
| M28 | medium | Theme-agnostic chat logic (approval/mission/Deliverable parsing, deep link and cold start) lives under components/play and duplicates the core's copies, which already differ | #86 | F11 |
| M29 | medium | Stored turn errors and background posts render as ordinary answers in Play (Studio badges them; the approved app shows a failed reply) | #27 | P260-FX-09 |
| M30 | medium | Try again / regenerate re-sends an empty message when the last user message was loaded from the server (rows carry parts, not content) | #29 | P260-FX-09 |
| M31 | medium | First open: Auto and the hero overlap the page header and the example chips on laptop-height windows (`min-height: 0` lets the hero shrink) | #51, #65 | P260-FX-14 |
| M32 | medium | Play's chat drops the SaaS onboarding and trial surfaces chat.tsx shows (onboarding opener, connect cards, power-up card, setup checklist, trial pill) | #9, #72 | F2 |
| M33 | medium | Stop erases Auto's partial answer and its tool lines; the approved Play app kept them with 'You stopped this answer.' | #23, #138 | F4 |
| M34 | medium | 'What Auto did' lost Play's accent 'Done' tag and the refused/skipped reason line | #52 | P260-FX-14 |
| M35 | medium | Conversations: the title and New chat span the whole column while the list is centred at 700 px | #55 | P260-FX-14 |
| M36 | medium | Fenced code on the Deliverable sheet and in Auto's replies renders in Studio's dark CodeBlock card; Play's own pre rules never apply | #169 | P260-FX-13 |
| M37 | medium | The open Deliverable is replaced by an error (or 'may have been deleted') when a background refetch fails | #105 | P260-FX-12 |
| M38 | medium | The shelf never refreshes while open: a new Deliverable and Auto's celebrate wait for a refocus or a navigation | #107, #56 | P260-FX-12 |
| M39 | medium | Raw HTML in a Deliverable or a reply prints as literal tags ('&lt;br&gt;', '&lt;script&gt;…'); the approved app dropped it | #174 | P260-FX-13 |
| M40 | medium | An SVG image Deliverable, shown as a same-origin blob: URL, runs its scripts as the app when opened in a new tab | #170 | C (pre-existing) |
| M41 | medium | dark: classes made live by the darkMode cherry-pick regress dark tones in Classic, Studio and Play: text below AA, a lost focus indicator and hover, overridden borders | #45, #177, #47, #178, #179 | P260-FX-05 |
| M42 | medium | The style cookie is written only on a switch, so Safari's 7-day cap on document.cookie silently drops Play (and Classic) | #35 | C (pre-existing) |
| M43 | medium | GET /health has no timeout: a frozen backend keeps the pill on 'Connected' and 'Automatos is asleep' never appears | #112 | P260-FX-21 |
| M44 | medium | Conversations and Recent stay stale after a turn (no invalidation; a 60 s staleTime; Recent keyed by the active chat) | #12, #111 | P260-FX-10 |
| M45 | medium | Play caches chat history under keys without the user or workspace, so the next account on the browser sees the last one's conversations | #21 | P260-FX-10 |
| M46 | medium | Approval cards live only inside the keyed chat screen: leaving the chat or switching conversation loses a pending card (Studio keeps it) | #26 | P260-FX-08 |
| M47 | medium | The shelf's Auto stays on 'Something went wrong' with the coral glow after Try again loads the shelf | #96 | P260-FX-04 |
| M48 | medium | Stop during an awaited (detached) reply leaves Stop, a thinking Auto and a blocked box for 5 minutes | #162 | C (pre-existing) |
| M49 | medium | Download saves the API's JSON envelope, not the file, for every text Deliverable (Play's page and Studio's sheet) | #100 | P260-FX-12 |
| M50 | medium | The AA proof measures flat grounds only: text over Play's accent tints and the page glow drops below 4.5:1, Daylight preset included | #8, #60 | P260-FX-18 |
| M51 | medium | Looks, the active look and the seen-Deliverables record are kept per browser, not per user or workspace | #43, #125, #13, #5 | F5 |
| M52 | medium | Under Play, Studio's consequence colours follow the look's accent: --accent (Studio's error colour) turns mint/cream/gold, --destructive is the same orange as --primary | #37, #82, #146 | P260-FX-17 |
| M53 | medium | Under a look, Studio's --primary and --fg-3 are checked only on --background and --card, not on Studio's tinted, secondary and muted fills | #62, #148 | P260-FX-17 |
| M54 | medium | Auto never returns to 'thinking' while the model reasons between tool rounds | #157 | P260-FX-16 |
| M55 | medium | Every generate_document PDF/DOCX/XLSX (and audio) opens in Play as "Play can't preview this document here."; Studio's sheet previews them | #99 | F6 |
| M56 | medium | ?draft= lets any link put hidden multi-line (or bidi) text in the owner's message box | #19 | P260-FX-10 |
| M57 | medium | A success:false list answer on a background refetch replaces the loaded shelf and drops every Load-more page | #104 | P260-FX-12 |
| M58 | low | play-theme.md's T-008a/b and T-009 sections were never checked against the merged code; the /looks/style note is misquoted (also checklist step 46) | #92, #126, #141 | B (docs) |
| M59 | low | The RUN checklist and ownerTest cover only the local edition | #76 | B (docs) |
| M60 | medium | Checklist step 43 asks for Try again after the asleep screen, which removes it | #136 | B (docs) |
| M61 | medium | Checklist step 18 expects Stop to keep the words and a 'stopped' tool line | #137 | B (docs) |
| M62 | low | Checklist step 15 expects the step-14 PDF rendered on the sheet | #139 | B (docs) |
| M63 | low | Checklist steps 41/42 and play-theme.md expect the pill beside the asleep screen | #140 | B (docs) |
| M64 | low | Checklist step 31 says Auto's eyes keep the light's colour under a Cobalt shell | #142 | B (docs) |
| M65 | low | Checklist step 23 (Safari) doesn't say to switch Safari to Play first | #144 | B (docs) |
| M66 | low | Checklist step 30 expects Auto's ring to change between three presets that all use Mint | #149 | B (docs) |
| M67 | low | Under Play, Discuss tells the owner to pick another agent 'in the message box', which Play's box can't do | #30, #14 | P260-FX-26 |
| M68 | low | The new client not-found drops the 404 document title for every style | #18, #36 | P260-FX-21 |
| M69 | low | useIsTabletOrBelow starts false, so Play mounts for one render on phones and tablets, at five places, with side effects (deep link consumed, cold start raced, requests fired, 12 posters fetched) | #1, #90, #124 | P260-FX-25 |
| M70 | low | The floating Auto widget on Conversations and Make it yours says 'Ask me anything about Dashboard' and sends page 'dashboard' | #116 | P260-FX-26 |
| M71 | low | Export always announces 'Downloaded &lt;name&gt;.look.json.' even when the browser blocked the download | #150 | P260-FX-20 |
| M72 | low | /looks and /looks/style render Ember's selections (and 'reduced motion') until storage is read after hydration | #120 | P260-FX-20 |
| M73 | low | A long look name squeezes 'Make it yours.' into a sliver and an unbroken name is clipped in its card | #155 | P260-FX-19 |
| M74 | low | The style sheet's 12 mascot specimens are named with live status captions ('Something went wrong') | #67, #122 | P260-FX-24 |
| M75 | low | Auto's clips are served with max-age=0, so every state swap waits a revalidation round trip | #81 | P260-FX-15 |
| M76 | low | The 'Deliverable card from receipts' path is dead: receipts never link a Deliverable, the port rejects the real absolute open_url, and the test uses a link shape the backend never sends | #130, #28 | P260-FX-09 |
| M77 | low | An answered loose approval stays under 'Approvals still open' | #31 | P260-FX-08 |
| M78 | low | Play's replies drop the tool results the core chat shows (knowledge sources, database results, code snippets, canvas widgets) | #163 | F3 |
| M79 | low | Tool-line durations round to whole seconds ('2 s'); the approved design reads '2.1 s' | #54 | F12 |
| M80 | low | 'Approvals still open' cards sit at the scroller's edge, outside the conversation column | #165 | P260-FX-08 |
| M81 | low | The Deliverable side panel drops the Play app's 'Sources · N' section | #57 | P260-FX-13 |
| M82 | low | A failed Download on Play's Deliverable page says 'Try again from Deliverables', the page the user is on | #109 | P260-FX-12 |
| M83 | low | The hosted asleep screen tells an offline user 'Nothing's broken on your side' | #74 | P260-FX-21 |
| M84 | low | The local asleep screen offers `docker compose up`, not the documented start command | #167 | P260-FX-21 |
| M85 | low | Under Play the theme menu's Tone is locked below 1024 px (where /looks can't open) and its items are unreachable by keyboard | #75, #147 | P260-FX-06 |
| M86 | low | The 'instruction' memory badge and the 'Cancelled' workflow badge paint their text in their own background colour (1:1) | #46 | C (pre-existing) |
| M87 | low | With localStorage unreadable, choosing Play flips straight back to Studio | #143 | P260-FX-06 |
| M88 | low | GET /health is polled once per mounted useBackendHealth (twice per 30 s on /chat); the cookie writer and a Blob download helper are duplicated | #80, #89 | P260-FX-27 |
| M89 | low | The asleep gate starts awake on every mount, so a page mounts and fires its requests before the asleep screen returns | #113 | P260-FX-21 |
| M90 | low | ⌘K search under Play cannot find Play's own pages | #115 | P260-FX-26 |
| M91 | low | The idle nap re-arming on input is untested | #134 | P260-FX-28 |
| M92 | low | ?chatId= from the URL goes unencoded into API paths and is stored unvalidated as the chat id | #2 | C (pre-existing) |
| M93 | low | Routes and query-parameter names are re-declared as private constants across Play files | #88 | P260-FX-27 |
| M94 | low | The asleep screen's countdown never shows that a check is running | #114 | P260-FX-21 |
| M95 | low | Auto's celebrate caption shows the raw type id ('Your blog_post is ready') | #172, #160 | P260-FX-13 |
| M96 | low | The prd260 guard misses oklch()/color-mix()/named colours and next/font/local paths and CSS @import/url() into the reference | #6, #133 | P260-FX-28 |
| M97 | low | Every look made with 'Save as new look' says 'Saved just now' forever | #156 | F12 |
| M98 | low | Save, Duplicate and Import report success when localStorage refuses the write | #41, #121 | P260-FX-20 |
| M99 | low | 'Copy tokens' exports next/font variable references instead of the face, so the :root block fails outside the app | #42, #123 | P260-FX-20 |
| M100 | low | Auto doesn't play oops when a turn hits a cap (limit_reached) | #11 | P260-FX-16 |
| M101 | low | Pressing Stop after a 'done' receipt makes Auto celebrate | #7 | P260-FX-04 |
| M102 | low | Hello on the first Play chat open replaces the thinking of an awaited reply | #159 | P260-FX-04 |
| M103 | low | platform_acknowledge_report shows Auto 'Reading' ('acknowledge' contains 'knowledge') | #97 | P260-FX-16 |
| M104 | low | 'Made by Auto' is shown for the owner's uploads and template documents | #171 | P260-FX-13 |
| M105 | low | lib/play/shelf.ts filterForType is dead code with no caller or test | #176 | P260-FX-27 |
| M106 | low | The Style axis grows by literal 'play' checks and per-theme booleans, not a per-style table | #83 | F11 |
| M107 | low | The dark: guard catches only neutrals and hex, so palette classes the audit rules 'fix' pass | #132 | P260-FX-05 |

