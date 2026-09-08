# PRD-237: Chat Continuity + Conversation Tabs — one conversation session for the page and the widget

> **Status:** BUILT 2026-09-07 on `feat/chat-continuity-tabs` (worktree `.claude/worktrees/feat+chat-continuity`, cut from `origin/main d1e056350` after the 09-07 merge wave). Awaiting CI → owner test in both editions → merge. `file:line` refs are grounded @ that tip — confirm by grep. **Owner ask (2026-09-07):** (1) the chat stays on the last conversation across page changes and reloads until *New Chat* — page and floating widget alike, so a conversation with Auto continues while moving around the app; (2) several conversations open at once as **tabs**, opened with *New Chat*, closed individually. **Both editions.**
>
> **Owner decisions (2026-09-07):** **D1** the editions persist differently — local keeps the session in the browser (one operator, one machine, open source); the hosted edition also stores it server-side so a subscriber's tabs follow them across devices. **D2** build "a turn finishes even if the browser reloads" now. **D3** the live voice call may still end on navigation while voice is in testing — the *thread* continues, the *call* does not.

---

## Framing (CLAUDE.md §3)

**Extension + consolidation.** PRD-220 built conversation persistence for the *widget* (`widget-session.ts`); the full chat page never got it and kept a second, page-local pointer that every navigation lost. This PRD promotes the widget's session into **the** conversation session — one store read by the chat page, the studio shell and the widget — adds an ordered open-tabs list to it, and makes the backend finish a reply whether or not the browser is still listening. Two pointer implementations become one; two dead page variants are deleted. Backend surface: one nullable JSONB column on `users`, three routes, one service module. **Size:** M. **Risk:** Low–Medium (the turn-runner touches every chat reply; it is covered by asyncio tests and keeps the wire protocol unchanged).

## Root cause (what was actually wrong)

- `frontend/app/chat/page.tsx` held the active conversation in `useState('')` — lost on every navigation or reload.
- `<Chat>` learned the backend-assigned id (`useChat`'s `chat-id` event) but never reported it upward, so the page's id stayed `''` for the whole life of a new conversation; the history panel's `isActive` prop had no caller and the studio shell never marked a fresh chat active.
- The widget's PRD-220 session was a *separate* pointer with a 24 h expiry, and the widget is hidden on `/chat` — page and widget never shared a conversation.
- The reply was produced inside the HTTP response generator: a reload or a closed tab cancelled the LLM stream and `save_message` never ran (save only on natural completion, no `finally`).
- `/chat/[id]` was a server component fetching with the browser client → always 404; four Activity links still pointed at it. `chat-page-content.tsx` was unused.

## What shipped

### S1 · One conversation session — model + store
- `frontend/lib/chat/chat-session.ts` (replaces `widget-session.ts`, deleted with its test): `{activeChatId, draftOpen, openChatIds, unreadChatIds, lastReadAt, titles, lastActiveAt}`; pure immutable updaters (`withChatOpened`, `withDraft`, `withChatIdAssigned`, `withTabClosed`, `withDraftClosed`, `withThreadRead`, `withChatTouched`, `withTitles`), `sessionTabs`, `mergeServerSession`/`toServerSession`; **no expiry**; `MAX_OPEN_CHATS = 8` (evicts the oldest tab that is neither the one being opened nor the one just left). Key `automatos:chat:session:<ws>:<user>` (`anon` in local — `auth-hooks.ts` returns no user there).
- `frontend/stores/chat-session-store.ts` (zustand): `hydrate(key)`, `openChat`, `newDraft`, `closeTab`, `closeDraft`, `chatIdAssigned`, `markRead`, `touchChat`, `setTitles`, `reload`. Every change → browser copy; in `saas` also a debounced (400 ms) `PUT /api/chat/session`; on hydrate the server copy wins when strictly newer (a fresh device inherits the tabs). Titles and unread flags are device-local. Edition seam = `isSaaS` only (D1).
- `frontend/hooks/use-chat-session.ts` — binds the store to workspace + user, follows `storage` events (other browser tabs), flags open background tabs on `automatos:chat-changed`.

### S2 · The chat page resumes the active conversation
- `frontend/app/chat/page.tsx` rewritten around the store: stored pointer → that chat; stored draft → a new chat (survives reload); nothing stored on this device → the most recent conversation (`getChatHistory(1)`), else a draft; `?chatId=` deep link wins once and is dropped from the URL; a stale pointer (deleted chat, other workspace) drops its tab. `<Chat>` announces the assigned id (typed and voice turns) and the generated title into the store; the history panel highlights the active chat and a delete also closes the tab; the studio shell's `selectedChatId` comes from the store.

### S3 · Conversation tabs
- `frontend/components/chatbot/chat-tabs.tsx` — the strip above the conversation (desktop + mobile, horizontally scrollable, `×` per tab, `+` = New Chat; the last tab cannot be closed). Studio shell: an **Open** section (tabs, close control, `new` pill for unread) above **Recent**. Switching remounts `<Chat>` with fetched messages; a tab whose reply finishes in the background gets an unread dot via the PRD-205 lane.

### S4 · The widget shares the session
- `chat-widget.tsx` `AutoChatTab` reads the store: the dropdown lists **the open tabs**, *New thread* = `newDraft`, close = `closeTab`, the same active conversation as the page, *Open in full chat* → `/chat` (the page lands on the same thread). Row metadata (Auto badge, time, preview) still comes from recent history. Widget stays hidden on `/chat`.

### S5 · Route + dead-code hygiene
- Deleted `frontend/app/chat/[id]/page.tsx` and `frontend/components/chatbot/chat-page-content.tsx`; the four Activity links now open `/chat?chatId=<id>`; relic tests updated (`model-selector-removed`, `prd184-us006-placebo-relics`).

### S6 · Hosted tabs follow the user (D1)
- Migration `prd237_users_chat_sessions`: `users.chat_sessions JSONB NULL`, keyed by workspace id — per user, per workspace, no membership dependency (personal-workspace owners have no `workspace_members` row). Head pin bumped in `test_prd209_alembic_single_head.py`.
- `GET /api/chat/session` (empty doc when none) · `PUT /api/chat/session` (validated: UUID ids, canonicalised + de-duplicated, ≤ 8 open, `activeChatId ∈ openChatIds`, numeric read-stamps capped to the newest 50; the JSONB is rebuilt, never mutated; `updatedAt` in UTC with offset so `Date.parse` is exact). Declared above `/{chat_id}` (the PRD-220 `/search` failure mode). Route manifest 785 → 788.

### S7 · A turn finishes even if the browser goes away (D2)
- `orchestrator/services/chat_turns.py`: the reply is a **producer task** with its own `SessionLocal` session; the HTTP response only consumes a queue. A disconnect records `client_gone` and the turn carries on to `save_message`; the caller's `on_complete` then fires `chat_changed` (PRD-205 lane) so a reloaded page merges the reply live — only when the client was gone, so a connected client never sees a duplicate. `TurnRegistry`: process-local tasks + Redis markers (`chat:turn:inflight:*`, `chat:turn:cancel:*`, TTL-bounded) because **production runs `uvicorn --workers 4`** (`orchestrator/Dockerfile` production stage); without Redis it degrades to process-local, the same scope the existing session queue already has.
- `api/chat.py`: `stream_chat` hands the turn to `run_detached_turn` (the request-scoped `db` is never touched by the producer); `POST /api/chat/{chat_id}/cancel` is the explicit Stop (process-local cancel + marker the producer polls every 0.5 s); `GET /api/chat/{chat_id}` carries `turnInFlight`.
- Frontend: `useChat` aborts its fetch on unmount on purpose (navigation/tab switch → detached completion), `stop()` also calls the cancel route, and `initialAwaitingReply` (from `turnInFlight`) shows the typing placeholder until the reply merges in or a 5-minute guard gives up.

### S8 · Voice across navigation — **not built (D3)**
The live call still ends on navigation; the thread it wrote to is the one that resumes.

## Tests (CI is the gate — nothing ran locally)
- `frontend/lib/chat/chat-session.test.ts` — storage, tab rules (draft lifecycle, close focus order, cap/eviction), read/unread, titles, server reconciliation.
- `frontend/stores/__tests__/chat-session-store.test.ts` — hosted hydrate/merge/debounced PUT/no-op saves/cross-tab reload; local edition never calls the API.
- `frontend/app/chat/__tests__/chat-page-resume.test.tsx` — pointer, draft, cold device, deep link (+URL cleanup), awaiting reply, stale pointer.
- `frontend/components/chatbot/__tests__/chat-tabs.test.tsx` — select/close/new/unread/last-tab.
- `orchestrator/tests/test_prd237_chat_turns.py` — disconnect does not cancel; full consumption; explicit cancel; producer exception → `e:` frame; registry without Redis.
- `orchestrator/tests/test_prd237_chat_session_routes.py` — route order, validation, GET/PUT handlers (db mocked).

## Assumptions taken (say if wrong)
No expiry on the resume pointer; a cold device lands on the most recent conversation; an empty draft gives way when another chat is opened (no orphan "New chat" tabs); tab cap 8; `/chat/[id]` deleted rather than redirected; the `kind='auto'` thread opens as a tab like any other; Stop = abort + explicit cancel (the partial reply is not saved, as before).

## Sequencing / notes for the merge
Single PR. Adds one migration (head pin bumped) and three manifest routes. Rebased on the 09-07 merge wave (#694 #695 #696 #697 #699 #700) — no overlap remained after the squashes.
