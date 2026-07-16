# PRD-205 — Memory, Continuity & Personal Context ("Auto Remembers")

> **Status:** DRAFT for review — spec only, no build yet. Authored from Gerard + Auto's product draft (2026-07-17), grounded file:line against `main @ ecfe3a11a`.
> **North star (Gerard's words, kept verbatim):** *Auto should carry conversations, follow users page-to-page, build/teach for them, and use memory to make the platform feel personal, continuous, and human.* The auntie test: Auto knows what you're building, has chats with you, reminds you about shit, and remembers that Gerard likes black.
> **Discipline:** the S10 memory baseline (PRD-198 gate, `docs/runbooks/S10-MEMORY-BASELINE-FREEZE.md`) is the before-number this layer is measured against. Continuity becomes a tracked number, not a vibe (T3).

---

## 1. What this is

The product layer on top of the repaired memory substrate: **active project memory, thread summaries + "where we left off", decision & open-loop memory, page-aware recall, a user-facing memory panel with edit/pin/forget, proactive recall, and memory-to-action.** The waves fixed storage (PRD-187), quality plumbing (PRD-159), DR + observability (PRD-197 S3/S4) and measurement (PRD-198 S1); none of them built the part a user *feels*. Auto today is a well-plumbed goldfish: it stores curated facts but cannot say "last time we were tightening the Academy tutor flow — pick up there?"

**Framing (CLAUDE.md §3): Extension-heavy with two justified net-new seams.** Most stories extend live surfaces: the typed distill taxonomy, the memory tools, the Memory Explorer UI, the widget's route hint, the context-assembly section registry, the notifications inbox. Net-new is confined to (a) the project-memory entity and (b) the proactive-recall engine — both justified in §2 because nothing existing fits.

**Build size:** L across 10 stories (each S–M individually; phased §5). · **Risk:** Medium — the hot path (context assembly) is touched additively; every behavioural change sits behind the existing floor/exclusion guards, and the eval slice (S10 here, not to be confused with PRD-198's S10 run) measures continuity before/after.

---

## 2. Current reality (grounded — what EXISTS vs what is NET-NEW)

**Substrate (exists, extend):**
- **Typed distill taxonomy** — `MEMORY_FACT_TYPES = {tool_outcome, task_learning, playbook_pattern, user_fact, business_fact, preference, procedure}` (`consumers/chatbot/smart_memory.py:30-38`); one cheap LLM call per turn distills `{fact, type, importance}` (`:567,614-654`). **No `decision` / `open_loop` / `thread_summary` / project linkage** — net-new types on an existing seam.
- **Memory tools already fire on "remember/forget":** `platform_store_memory` (with `source_type`/`confidence`/`evidence_uri` — `modules/tools/discovery/actions_workspace.py:60,79-95`), `platform_search_memory` (`actions_search.py:46`), `platform_browse_memories` (`actions_workspace.py:112`), `platform_delete_memory` (destructive-gated, `:144`), `platform_get_memory_stats` (`:38`).
- **Metadata contract is split-brain:** the tool path writes provenance; the distill path writes only `category`+`importance` (`smart_memory.py:484-488`); the L3 payload has no first-class type/confidence columns — everything rides the metadata dict (`modules/memory/durable_store.py:231-239`). Unify, don't add a store.
- **Lifecycle exists:** relevance floor at L3 + merged-set (`durable_store.py:50-60,273`; `smart_memory.py:222-234`), junk-type injection exclusions (`modules/memory/injection_filter.py:19-23`), hourly decay, daily L2→L3 promotion (type-aware), monthly graphify archival, contradiction-based consolidation (`unified_memory_service.py:1629-1708`), daily Qdrant snapshots (PRD-197 S3).
- **Retrieval ranking is thin:** semantic score + floor + exclusions only; `importance` gates promotion but does **not** rank recall; **no recency weighting, no memory pins** (`context/sections/memory.py:266`; the only "pin" is documents-to-chat, `core/models/core.py:1125`).

**Continuity (partial):**
- Chats are persistent + multi-threaded with deep links (PRD-220: `frontend/components/chatbot/chat-widget.tsx:99-230`, `/chat?chatId=` at `:225`; `app/chat/page.tsx:35,75-100`); `chats.updated_at` is orderable (`core/models/core.py:1109`); every message stores its **context trace** incl. injected memory ids (`:1166`, PRD-201). **But `chats` has no `summary` column**, and the L1 session "summary" is a 500-char string concat whose own comment says "Phase 2 adds LLM summarisation" — never done (`unified_memory_service.py:1604-1606`). "Where did we leave off" has nothing to stand on.
- **Page awareness exists as an ephemeral hint only:** the in-app widget sends the current route (`chat-widget.tsx:605-607` via `main-layout.tsx:59-71`), and `_inject_page_context` appends `"[Context: the user is currently on the X page]"` to the last user message, prompt-side only (`api/chat.py:212-236`). It is **not** a context section (`modules/context/sections/__init__.py:31-49` — no page/route/selected-object section), never stamped into memories, never used for recall.

**Surfaces (partial / mis-wired):**
- **A Memory Explorer UI already exists** (`frontend/components/knowledge/memory-tab.tsx`, mounted in Documents at `document-management.tsx:1202`, with sidebar/viewer/health-banner) — but it calls `/api/v1/memory`, which is **router-wide super-admin** (`api/memory_stats.py:30`), so real users can't see it. The workspace-scoped user router **exists and is unused by any UI** (`api/widget_memory.py:27` — list/search/store/delete under `knowledge:*` permissions). The "memory panel" is largely a **rewire**, not a build.
- **PRD-166's workspace-field UI gap is still open** — field memory has a read-only context section and inspection views, no user surface.

**Net-new (nothing fits — justification per CLAUDE.md §4 rule 1):**
- **No project entity.** `OrchestrationRun` is an execution run with a goal + plan (`core/models/orchestration.py:39,67-68`); `BoardTask` is a work item (`core/models/core.py:1524`). A *living topic* — purpose, status, decisions, open questions, last-summary, next action, links — is an aggregate neither models; stuffing it into mission goals or graph blobs (rebuild-on-write JSON, `workspace_graphs`) would abuse both. One new table is justified (S4; box Q11 offers the alternative for veto).
- **Proactive recall does not exist anywhere**, and the natural engine is literally discarding its fuel today: `scheduled_task_service._trigger_agent_chat` runs a background task, extracts the text, and `logger.info(...[:200])`s it into oblivion (`services/scheduled_task_service.py:387-419`). This PRD absorbs the "background→chat injection" candidate that was pencilled as PRD-205.

---

## 3. Findings → fix → story

| # | Finding (grounded) | Fix | Story |
|---|---|---|---|
| 1 | Metadata split-brain: tool path carries provenance, distill path doesn't; no `decision`/`open_loop`/`thread_summary` types (`smart_memory.py:30-38,484-488`) | One write contract (type, confidence, source_type, page, thread/project link) across both paths; extend the taxonomy | **S1** |
| 2 | No thread summaries; L1 "summary" is a 500-char concat with an unfulfilled "Phase 2" promise (`unified_memory_service.py:1604`) | `chats.summary` + LLM checkpoint/end-of-thread distill; decisions + open loops extracted as typed memories | **S2** |
| 3 | "Where did we leave off" unanswerable | Resume payload (active threads + projects + decisions + open loops + next steps) as an API + chat action + button | **S3** |
| 4 | No project entity (§2) | `project_memories` table + platform tools + auto-linking of thread summaries | **S4** |
| 5 | Memory Explorer super-admin-locked; user router UI-orphaned (`memory_stats.py:30`; `widget_memory.py`) | Rewire the panel onto the workspace router; add edit / **pin** / forget / archive; project cards UI | **S5** |
| 6 | Page context is an ephemeral string; never stamped, never recalled (`api/chat.py:212-236`) | Structured `page_context` assembly section + selected-object param; stamp page provenance into memories; page-aware recall boost | **S6** |
| 7 | Recall ignores importance/recency; no pins (`sections/memory.py:266`) | Composite recall scoring (semantic × recency × importance × pin) **above** the existing floor/exclusions | **S7** |
| 8 | Proactive recall absent; background output discarded (`scheduled_task_service.py:416-419`) | Open-loop lifecycle + background→chat/notification injection + strict anti-spam rules | **S8** |
| 9 | Memory can't become work | "Thread/project → PRD / tasks" actions reusing deliverables + board tools | **S9** |
| 10 | Continuity is unmeasured | Continuity slice in the memory gold set, riding the PRD-198/S10 eval discipline | **S10** |

---

## 4. Stories (test-first; CI is the only gate — no local runs)

### S1 · One memory-write contract + the missing types — S
Extend `MEMORY_FACT_TYPES` with **`decision`**, **`open_loop`**, **`thread_summary`** (`smart_memory.py:30-38`); the distill prompt learns to emit them with the same `{fact, type, importance}` shape. Unify metadata across BOTH write paths (distill + `platform_store_memory`): every stored memory carries `type, importance, confidence, source_type, page (nullable), chat_id (nullable), project_id (nullable)` in the L3 metadata dict — no schema change to the durable store, one contract module both paths import. Preferences keep influencing tone via the existing identity/preference injection (the "Gerard likes black" path already works — it stays).
**Test:** `test_distill_emits_new_types` (mocked LLM → typed facts validated); `test_write_contract_unified` (both paths produce the same metadata keys). Pure/mocked.

### S2 · Thread memory: checkpoints + end-of-thread summaries — M
`chats.summary` JSONB (migration): `{topic, decisions[], open_questions[], last_summary, next_step, updated_at}`. A **checkpoint distill** runs (a) at conversation idle/end (extend the existing L1→L2 consolidation seam — the code's own unfulfilled "Phase 2 adds LLM summarisation", `unified_memory_service.py:1604`) and (b) on demand mid-chat (the §14 checkpoint: "want me to save where we are?"). Extracted `decision`s and `open_loop`s are stored as typed L3 memories linked by `chat_id`; the summary itself is `thread_summary`. Long-chat checkpoint offers ride the existing chat action-card pattern (PRD-163 in-chat cards) — no new UI primitive.
**Test:** `test_checkpoint_writes_chat_summary_and_typed_memories` (mocked LLM); `test_checkpoint_idempotent_updates` (re-checkpoint updates, not duplicates). Pure/mocked.

### S3 · "Where did we leave off?" — S
One resume payload: `GET /api/memory/resume` (workspace-scoped router, `widget_memory.py`) → last-N active threads (`chats.updated_at` + `summary`), active projects (S4), recent `decision`s, open `open_loop`s, suggested next steps (from summaries). Chat-side: a `platform_resume_context` tool so "where did we leave off?" in ANY chat answers from it; frontend button in the chat header + widget (deep-links threads via the existing `/chat?chatId=`). Route-manifest updated.
**Test:** `test_resume_payload_shape`; `test_resume_orders_by_recency`. Pure/mocked.

### S4 · Active project memory — M
New table `project_memories` (the §2-justified entity): `id, workspace_id, name, status (active|paused|blocked|completed|archived), purpose, importance, last_summary, next_action, decisions JSONB, open_questions JSONB, links JSONB ({chats[], documents[], prds[], tasks[], pages[]}), owner_user_id, pinned bool, last_touched_at, created_at`. Platform tools: `platform_upsert_project_memory`, `platform_get_project_memory`, `platform_list_project_memories` (3-file pattern). S2's checkpoint links thread summaries to a project when the topic matches (name/alias match first — LLM matching only if the eval later proves it earns its cost). "Save this as a project" = an in-chat action card. Missions/tasks stay what they are; a project card may LINK to them, never replace them.
**Test:** `test_project_upsert_roundtrip`; `test_thread_summary_links_to_project`; `test_project_status_lifecycle`. Pure/mocked; migration chains on the current single head.

### S5 · The user memory panel + project cards UI — M
Rewire the existing Memory Explorer (`memory-tab.tsx` et al.) onto the **workspace-scoped** `/api/memory` router (extend that router with update/pin/archive; permissions stay `knowledge:*`); super-admin `/api/v1/memory` remains the ops view. Panel sections per the draft: About you / Preferences / **Projects** / Decisions / Open loops / Recent thread summaries / Archived. Actions: edit, delete, pin, archive, mark-complete (open loops), correct. **Project cards** render from S4 with "Ask Auto", "Resume thread" (deep-link), "Create task", "Generate PRD" (S9). Honest-UI rules apply (no fake counts, PRD-196 lesson).
**Test:** panel-rewire route test (frontend hits `/api/memory/*`, grep-guard against the su router); vitest render of a project card incl. empty/honest states.

### S6 · Page context becomes real — S/M
Promote the route hint to a **`page_context` assembly section** (registry `sections/__init__.py`) carrying `{page, selected_object?}` from the existing widget payload (`chat-widget.tsx:605-607` — extend with selected object where pages have one); keep the prompt hint for compat. Memories created mid-chat get `page` stamped (S1 contract). Recall: a modest same-page boost in S7's scoring, and page-aware suggestions ("we discussed this while on Activity") become possible because provenance exists.
**Test:** `test_page_section_renders_when_context_present`; `test_memory_write_stamps_page`. Pure/mocked.

### S7 · Recall that ranks like an auntie remembers — S
Composite recall score on the merged candidate set (`smart_memory.py:222-234` seam): `semantic × recency-decay × importance × pin-boost × same-page/same-project boost`, applied **after** the existing floor and type exclusions (floors/exclusions are load-bearing — untouched). Memory `pinned` lives in L3 metadata (S1 contract) + panel toggle (S5). Weights are config knobs, defaulted conservatively; the S10 eval slice is the referee, not taste.
**Test:** `test_recall_ranking_composite` (fixture memories → deterministic order); `test_floor_and_exclusions_still_apply`. Pure.

### S8 · Proactive recall + open loops (helpful, never creepy) — M
Open-loop lifecycle: created by S2 checkpoints or explicitly ("that's an open loop"); completed/expired via panel or chat; surfaced ONLY on the draft's trigger list (§13): project-page visit, project-name mention, "what next?", return-after-gap. Two channels: (a) **in-chat contextual recall** — a short "this connects to X from last night" preamble assembled from S4/S2 data when triggers fire; (b) **notifications inbox** for return-after-gap digests — reusing the notifications plane, never a new one. **Fix the discard**: `scheduled_task_service._trigger_agent_chat` output (`:416-419`) lands as a chat message in its owning thread + optional notification (the absorbed PRD-205 candidate). Anti-spam: per-day cap, dismiss-suppresses, pinned-projects-only for unprompted surfacing, config kill-switch default ON for in-chat / OFF for notifications until the eval says otherwise.
**Test:** `test_background_output_lands_in_thread` (the discard is dead — grep + behavioural); `test_proactive_triggers_and_caps`; `test_dismiss_suppresses`. Pure/mocked.

### S9 · Memory to action — S/M
"Turn this into a PRD / tasks" from a thread or project card: tools that assemble S2/S4 context and call the EXISTING deliverables generation + board-task creation paths (no new generators). The draft's exact scenario — "we talked about Academy last night, turn that into a PRD" — becomes: resume-context fetch → deliverable generation with that context. 
**Test:** `test_memory_to_prd_assembles_thread_and_project_context` (generation service mocked, asserts the assembled input); `test_memory_to_tasks_creates_board_tasks`.

### S10 · Continuity becomes a number — S
Extend the memory gold set (the PRD-198/S10 kit) with a **continuity slice**: resume queries ("where did we leave off on X"), decision recall ("what did we decide about Y"), open-loop recall — scored recall@5 like the rest. Runs in the existing `memory-recall-eval` lane; the frozen S10 baseline is the before-number; this layer must move the continuity slice, or the number says so honestly.
**Test:** slice present in gold-set schema; harness scores it per category. Pure.

---

## 5. Sequencing / phases (maps to the draft's MVP phases)

- **Phase 1 = S1+S2+S3+S7** (foundation: types, thread memory, resume, ranking) — the fastest path to "Auto continues".
- **Phase 2 = S4+S5** (project memory + the panel/cards — memory gets a UI shape).
- **Phase 3 = S6** (page-context plane) — small, unblocks page-aware suggestions.
- **Phase 4 = S8+S9** (proactive + memory-to-action) — only after S10's slice exists so helpfulness is measured, not vibed.
- **S10 lands with Phase 1** (instrument first — the house discipline).
- Academy learner memory (draft §21) is **deliberately not in this PRD** — different pod/repo; this layer is built reusable (workspace+user scoping, typed contract) and the Academy integration is its own spec there (box Q13).

## 6. Verification (CI is the only gate — no local runs)
Pure/mocked tests throughout; migrations (chats.summary, project_memories) chain single-headed and self-apply on boot; new routes → route-manifest; source-grep guards for the retired discard path and the panel's router rewire; the continuity eval slice reports in the non-required lane (number = deliverable). Frontend: vitest for panel/cards/resume button; hooks mocked per the existing strip-test pattern.

## 7. Safety, privacy, scope (the draft's §16 — binding)
Workspace + user scoping on every read/write (the existing `knowledge:*` permission model); no cross-workspace recall (PRD-157's lesson stands); secrets/credentials/payment data never stored as memory (extend the distill prompt's exclusions + a validator in the S1 contract); provenance visible in the panel; proactive recall dismissible + capped + kill-switched; emotional/relationship context stored only as plain `user_fact`s with clear utility ("excited about Academy because it solves trust") — no sentiment profiling, user-deletable like everything else.

---

## 8. Open questions — Gerard's call (decide, don't let me defer — CLAUDE.md §12)

The draft's ten, plus three of mine. Recommendations inline; build proceeds on the recommendation where unanswered, flagged per the house flip-and-recut pattern — EXCEPT Q3/Q7 (privacy-shaped), which block their stories until answered.

1. **Proactive cadence** — daily mentions vs context-triggered only? **Rec: context-triggered only** (the §13 trigger list); a daily digest is a later opt-in.
2. **Project cards visible day one?** **Rec: yes** — cards are what make memory legible; ship with Phase 2.
3. **⏸ Approval granularity** — approve all memories vs sensitive/major only? **Rec: the draft's own §15.3** (silent low-risk, lightweight confirm for project memories, explicit ask for sensitive) — confirm this is the consent model you want.
4. **Admin visibility of shared project memories?** **Rec: yes, workspace-scoped** (they're workspace objects, like tasks).
5. **Personal vs workspace tabs in the panel?** **Rec: one panel, a scope badge per memory** — two tabs when real users ask.
6. **Learner memory resettable per course?** Deferred with Q13 to the Academy spec.
7. **⏸ "Private to me" vs "shared with workspace" toggle** — **Rec: yes for `user_fact`/`preference` (default private-to-me), project/decision/open-loop default workspace-shared.** This is the one schema-shaping consent question — S1 needs it answered.
8. **"Save this as a project?" prompt on long strategic chats?** **Rec: yes** — it's S2's checkpoint card with a project button.
9. **Pinned projects on the dashboard/homepage?** **Rec: not yet** — Command Center is already dense; revisit after cards prove used.
10. **Memory-powered notifications now or chat-recall first?** **Rec: chat-recall first, notifications behind the S8 kill-switch (default OFF)** until the eval + your own usage say they're wanted.
11. **Project entity shape** — new `project_memories` table (rec) vs riding mission goals / graph blobs? Table is a new-table-rule exception; §2 argues nothing fits. Veto here if you disagree.
12. **The panel rewire** — extend `/api/memory` (workspace router, rec) vs de-admin `/api/v1/memory`? Rec keeps the ops/user split clean.
13. **Academy learner memory** — separate PRD in the academy pod consuming this layer's contract? **Rec: yes** (one flag line, not a plan — cross-pod).

---

*Traceability: grounded against `main @ ecfe3a11a` (scout sweep 2026-07-17 — every EXISTS/NET-NEW claim carries file:line above). Extends PRD-159 (typed distill, consolidation, floor), PRD-187 (durable store), PRD-197 S3/S4 (DR + substrate telemetry), PRD-198 S1 + S10 kit (the eval this layer answers to), PRD-201 (context assembly + per-message context trace), PRD-220 (persistent multi-thread chats), PRD-163 (in-chat action cards), PRD-166 (field memory — UI gap noted, not owned here). Absorbs the former "PRD-205 candidate" (background→chat injection) into S8. The T1 verdict (HOLD graph memory substrate) stands — this is a product layer over the existing stores, not a substrate migration. PILOT lens; no moat framing; the auntie stays.*
