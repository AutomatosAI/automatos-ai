# PRD-221 — Context-Aware Auto Layer (page-aware chat → Command Centre intelligence)

> Origin: concept drafted by Auto in-product 2026-07-16; grounded against main and re-scoped same day
> (recon + 4 locked decisions). Coordinates with **PRD-205 Auto Speaks (#560)**, which owns
> background→chat delivery (ChatMessenger + per-user Auto thread) — #560 must merge before this
> branch cuts (launcher-enforced); S11 here is a regression guard, not a build.
> Two phases, one Ralph run: Phase 1 = the context mechanism, Phase 2 = Command Centre intelligence.
> Phase 1 alone ships value if the night ends early.

## 1. What this is

Auto should know what page the user is looking at, what that page is for, and which of his
platform actions apply there — and the Command Centre should stop being a stats-and-cards page
and start being an interpreted one: a plain-English "Auto's Read" digest, progress lines on feed
cards, scheduled-run output that lands in a chat instead of a log line, and one-click follow-up
tasks from failed items.

**What this does for the user:** ask "why is this one blocked?" about the row on screen and get a
real answer; open Command Centre and read three sentences instead of scanning six tabs; wake up to
the overnight run's output in a chat, not silence.

## 2. Current reality (grounded 2026-07-16, main @ `5ebd3febe` — grep to confirm, lines drift)

- **Page context today is one string.** The floating widget maps route → coarse bucket
  (`frontend/components/layout/main-layout.tsx` `getCurrentPage()` ~59-77), sends
  `{context:{page:"Agent Management"}}` (`frontend/lib/chat/hooks.ts` ~128). Backend reads ONLY
  `context["page"]`, truncates to 80 chars, appends `[Context: the user is currently on the X page]`
  to the last user message **after** the clean DB save (`orchestrator/api/chat.py`
  `_inject_page_context` ~209-236, call ~377-381). The main `/chat` page sends **nothing** and the
  widget is hidden there (`main-layout.tsx` ~80).
- **Dead type:** `frontend/types/chat.ts` ~247-261 `ChatRequest.context`
  (`currentPage/selectedItems/userRole/recentActions`) is unused and field-mismatched with the wire.
  It includes `userRole` — authz from the client is never acceptable.
- **No page→tool map exists.** 173 `platform_*` actions ride one `platform_execute` dispatcher
  (`orchestrator/modules/tools/tool_router.py` ~493-514); per-request narrowing is semantic top-K
  over the **user's words only** (~125-260; query set at `consumers/chatbot/service.py` ~2330).
  Role gates: `is_admin`, `is_super_admin` fail-closed (~333-362, ~592-597). The only manifest in
  the repo is the route contract (`orchestrator/reports/route-manifest.json`, 767 `{method,path}`).
- **A structured context→preamble pattern already exists** — but only in the storefront plane:
  `orchestrator/integrations/shopify/widget_proactive.py` `_build_page_context_preamble` ~382-409
  with the `context_fields.py` allow-list. Generalise the pattern; **do not touch that plane**
  (in-app chat ≠ `api/widgets/` embed SDK).
- **Command Centre** (renamed from Activity; there is no `/activity` index route) already has
  humanized cards (`frontend/components/activity/activity-feed-item.tsx` STATUS_MAP), pulse
  StatsBars (`activity-page.tsx` ~133-198), and tabs. Feed = on-the-fly aggregation of
  chats/routines/playbook-executions/board_tasks (`orchestrator/services/activity_service.py`
  `get_feed` ~72-120). **No LLM digest endpoint exists.** Studio shell has Watchlist/Governance
  tabs; classic does not.
- **PRD-204 watches are the supervision substrate** (`core/models/watches.py`: interval columns,
  append-only `watch_events` with `requires_attention`; `services/watch_ticker.py` writes only on
  meaningful change). Fine-grained work progress lives in `orchestration_events`
  (~50 **underscore** `EventType` values in `core/models/orchestration_enums.py` ~67-129 —
  dotted names like `mission.status.updated` do not exist here).
- **Scheduled output was dropped — now fixed upstream.** `scheduled_task_service.py`
  `_trigger_agent_chat` historically logged 200 chars and dropped the run output.
  **Superseded 2026-07-17 by PRD-205 Auto Speaks (#560):** `services/chat_messenger.py` +
  per-user Auto thread (`chats.kind='auto'`, `messages.source`) now own background→chat
  delivery. This PRD only regression-guards that wiring (S11) and adds no schema there.
- **Reusable plumbing:** `messages.retrieval_context` + `messages.context_trace` off-`parts` JSONB
  side-channels (`core/models/core.py` ~1144-1176; writers in `modules/context/service.py`,
  `consumers/chatbot/service.py`); `chats.last_context`; `core/cache/service.py` `CacheService`
  (Redis, workspace-isolated keys) + `core/redis/client.py`; `BoardTask.source_type/source_id/tags`
  (`core/models/core.py` ~1524-1548) — task-from-event needs **no schema change**.

## 3. Decisions — LOCKED (Gerard, 2026-07-16)

1. **Page awareness = checked-in page manifest + CI contract.** One hand-authored registry
   (`orchestrator/contracts/page-manifest.json`), validated in tests: every `actions[]` entry
   exists in the action registry; every route exists in the frontend app router; a generated,
   committed TS mirror keeps the frontend in sync (route-manifest discipline).
2. **References, not payloads.** Per-message context = `{page, route, tab, selected{type,id},
   filters, visible_ids}` — IDs only; Auto fetches detail through his existing tools. Never
   serialize page data into messages.
3. **Auto's Read = cached + event-invalidated.** Regenerate only when the workspace state-hash
   changes or the cached digest is stale; deterministic fallback if the LLM errors. Never a
   per-pageview LLM call.
4. **One PRD, two phases, one overnight branch** (`ralph/prd-221-context-aware-auto`). Phase 1
   completes before Phase 2 begins.

Naming: the new surfacing is **progress events** — never "heartbeats" ("heartbeat" is taken twice:
agent/routine heartbeats + the watch tick). Canonical terms per CLAUDE.md §10 (Playbook, Mission,
Command Center, Auto).

## 4. Stories (test-first; sized S/M; grep every anchor before editing)

### Phase 1 — Page Context & Capability Layer

**S1 · Page manifest + loader — M · `orchestrator/contracts/page-manifest.json` + `core/page_manifest.py`**
Hand-authored manifest, one entry per page: `key` (kebab, canonical), `route`, `title`, `purpose`
(1-2 plain sentences), `entities[]`, `tabs[]`, `actions[]` (real `platform_*` names), `quick_prompts[]`
(strings; optional `admin_only` flag). Seed all 12 primary pages: chat, command-center, missions,
agents, documents, playbooks, deliverables, analytics, settings, team, tools, marketplace.
Loader: cached typed access by key + route→key resolution.
**Success:** pure tests prove every `actions[]` name exists in the action registry, keys unique,
routes well-formed, purposes non-empty; unknown key → None (no raise).

**S2 · Structured context intake + server-side preamble — M · `services/page_context.py` + `api/chat.py`**
Allow-listed sanitizer (Shopify `context_fields` discipline): accepted fields exactly
`page,route,tab,selected{type,id},filters(≤8 str:str),visible_ids(≤16 str)`; caps on every string;
unknown keys dropped; **client-supplied role/permission fields are never read**. Renderer builds a
compact preamble from manifest entry + sanitized state and replaces `_inject_page_context` — same
placement property (injected after the clean DB save, never stored in `parts`). A label-only or
unknown-page context renders the minimal one-line form through the **same renderer** — one code
path, the old helper is deleted (no shim). Preamble ends with an instruction to fetch details via
platform tools rather than assume page contents.
**Success:** unknown fields dropped; oversized input truncated; preamble carries purpose +
selected ref; role text never echoed; legacy label path renders through the same renderer.

**S3 · Persist what Auto saw — S · `modules/context/service.py` + chatbot service**
Add a `page_context` section (the **sanitized** dict) to the existing per-turn `context_trace`
so "what did Auto know when he said that" includes page state.
**Success:** trace contains sanitized context; the raw client dict is never stored.

**S4 · Page-prior tool exposure — M · `tool_router.py` + `consumers/chatbot/service.py`**
Thread the sanitized page key through the chat call path; resolve manifest `actions[]` and pass
`page_actions` into tool selection: those actions are guaranteed present in the exposed
`platform_execute` enum (union with semantic top-K, total capped ≤40). Role gates still win —
an admin/super-admin-gated action in a manifest is **still excluded** for an unauthorized principal.
**Success:** page actions present when context supplied; gated actions stay excluded; behaviour
identical when no page context.

**S5 · Frontend structured capture, both surfaces — M · `frontend/lib/page-context.ts` + widget + main chat**
Replace the dead `ChatRequest.context` type with the real shape. New `usePageContext()` builds
`{page:<manifest key>, route, tab?, selected?, filters?, visible_ids?}` from the pathname plus a
lightweight setter pages can call (Command Centre sets active tab; detail pages set `selected`).
Widget sends the structured object (manifest **key**, not display label). Main `/chat` sends its
own context too (page `chat`, selected pinned agent / mode).
**Success:** vitest proves both send paths carry the structured object; unknown route → key
`unknown` (backend renders minimal line).

**S6 · Manifest TS mirror + CI contract — S · `orchestrator/scripts/gen_page_manifest_ts.py` + `frontend/lib/generated/page-manifest.ts`**
Deterministic stdlib-only generator (dump_routes discipline); mirror is committed. Vitest contract
test reads the JSON via fs (test-time cross-package read, exactly like route-contract): every
manifest route exists under `frontend/app/`, and the committed mirror byte-matches a fresh render
of the JSON.
**Success:** contract test red on drift (route removed / mirror stale), green on main.

**S7 · Quick-prompt chips in the widget — S · `chat-widget.tsx`**
Widget empty state renders the current page's `quick_prompts` from the mirror as tappable chips
(today it is plain text, ~425-435); tapping sends the prompt. `admin_only` prompts hidden for
non-admin roles (role-context); the tool layer remains the enforcer.
**Success:** vitest — chips render per page, tap sends, admin-only hidden for members.

### Phase 2 — Command Centre Intelligence

**S8 · Digest snapshot builder (pure) — M · `services/workspace_digest.py`**
`build_digest_snapshot(db, workspace_id, period)` reuses `activity_service` feed/stats + live
watches: counts, needs-attention items (name + one-line reason), active work, recent completions,
and a stable `state_hash` (sha256 over a canonically-ordered projection).
**Success:** pure tests (mocked sources): hash stable under reordering, changes when a
needs-attention item appears; snapshot carries names not raw event dumps.

**S9 · Auto's Read endpoint + cache — M · `api/activity.py` + `CacheService`**
`GET /api/activity/digest?period=`: snapshot → if cached digest for (workspace, state_hash) is
fresh (TTL ~15 min) return it; else ONE bounded LLM call (existing LLM manager pattern; ≤150
words; plain English; must name blocked/failed items; no log dumps) → cache
(`core/cache/service.py`, workspace-isolated key). LLM failure → deterministic template from the
snapshot — never a 500. Response `{text, generated_at, state_hash, needs_attention_count}`.
**Route-manifest updated** (add entry + bump count — the committed contract, PRD-185 S12 lesson).
**Success:** same-hash second call does not re-invoke the LLM (mock-proven); fallback on LLM error;
digest names the blocked item; route present in route-manifest.json.

**S10 · PRD-221 migration (digest_feedback only) + feedback endpoint — S · one alembic revision**
One revision, stacked on the current single head (which includes `prd205_auto_speaks` from #560):
`digest_feedback` table (id, workspace_id, user_id, state_hash, rating ∈ {-1,1}, created_at) —
**no other schema change** (background→chat needs no column; #560's Auto thread owns delivery).
`POST /api/activity/digest/feedback` writes a row (422 on bad rating). Route-manifest updated.
Check `alembic heads` is single before authoring; check no open PR carries a rival merge-heads.
**Success:** migration lints (single head after); POST persists; invalid rating 422.

**S11 · Guard: background→chat stays wired — S · test-only (PRD-205 Auto Speaks is canonical)**
#560 already delivers scheduled/background output via `services/chat_messenger.py` to the
per-user Auto thread. This story adds (or cites, if #560 shipped one) a pure regression test:
`_trigger_agent_chat` delivers via the ChatMessenger seam and a bare log line is never the
terminus for run output. **No production changes; no per-task chats; `chat_messenger.py`
untouched.**
**Success:** guard test exists (new or cited); scheduled_task_service references the seam;
no diff on `chat_messenger.py`.

**S12 · Progress line on feed items — M · `services/activity_service.py` (+ small label helper)**
Mission/board items gain `last_progress {summary, at, requires_attention}` from the latest
`orchestration_events` row for the run (bounded: one grouped query, no N+1), mapped to plain
English via a table keyed by **real** `EventType` values (`run_replanning` → "Re-planning after a
failed step", `task_verification_failed` → "Output failed verification — retrying", …).
Watch-supervised targets flag `requires_attention` from the live watch.
**Success:** seeded events → correct label + flag; query count bounded; unknown event type →
graceful generic label (no KeyError).

**S13 · Auto's Read panel + thumbs — M · `frontend/components/activity/autos-read.tsx`**
One component used by BOTH shells (classic ActivityPage summary area + studio summary-tab):
renders digest text, needs-attention count, generated-at, 👍/👎 posting
`{state_hash, rating}` to S10's endpoint; react-query hook `use-digest-api.ts` in the existing
60s-poll family. Feed card `last_progress` (S12) renders as the card's update line.
**Success:** vitest — renders digest, thumbs fire POST, progress line renders with
needs-attention emphasis; both shells import the same component.

**S14 · Follow-up task from a feed card — S · `activity-feed-item.tsx` + existing board API**
Failed / needs-attention cards get "Create follow-up task" → existing board-task creation API with
`source_type='activity'`, `source_id='<type>:<id>'`, title/description carrying the plain-English
context (existing `BoardTask` columns — **no schema change**). Board card links back via the
existing source fields.
**Success:** vitest — action shown only for failed/needs-attention, click calls the API with
source fields populated.

## 5. Sequencing

**Precondition (launcher-enforced): PR #560 (PRD-205 Auto Speaks) merged to main before the cut**
— it owns background→chat and carries the migration S10 chains after; it also touches
`api/chat.py` / `consumers/chatbot/service.py`, so cutting before it merges would conflict with
S2–S4 at PR time.

S1 → S2 → S3 → S4 (backend spine) → S5 → S6 → S7 (frontend) — **Phase 1 gate** — then
S8 → S9 → S10 → S11 → S12 (backend) → S13 → S14 (frontend). No story depends on a later one.
If the night ends inside Phase 2, Phase 1 is independently shippable.

## 6. Verification (CI is the gate; the overnight loop runs the local proxies)

- Backend: `cd orchestrator && python3 -m pytest -q` green per story (pure tests, `@integration`
  skips without DB; `_sys_guard` block in new test files that import `modules.*/consumers.*/core.*`).
- Frontend: `cd frontend && npm run -s test` (vitest) green per story; touched files tsc-clean.
- `bash scripts/ralph/acceptance-prd221.sh` exit 0 = PRD-level done.
- Per-story push; CI (`test.yml`) is the protected-regression gate.

## 7. Explicitly out of scope (each is a future decision, not silently dropped)

- The `api/widgets/` embed-SDK / storefront plane (Shopify) — pattern donor only, untouched.
- Classic-shell Watchlist/Governance parity (PRD-204 surface question).
- Notification-channel expansion, mute/grouping preference controls, per-page "technical mode"
  (the execution page already serves raw detail).
- Autonomy, approval-policy, or permission-model changes (PRD-163/143 planes).
- Auto-generated manifest entries for admin/marketplace sub-routes beyond the 12 seeded pages.

## 8. Auto's-draft open questions — answered here so the run has none

1. Digest on load vs on request → **cached + event-invalidated (S9)**. 2. Progress-event
configurability → **not configurable v1; derived from existing streams**. 3. Grouping threshold →
**feed already aggregates; one `last_progress` line per item**. 4. Technical mode → **out (§7)**.
5. Muting → **out (§7)**. 6. Frequency by type → **inherited from existing producers (watch tick /
orchestration events)**. 7. Proactive notify → **existing watch/notification paths only**.
8. Quick actions config vs hardcoded → **manifest-driven (S1/S7)**.
