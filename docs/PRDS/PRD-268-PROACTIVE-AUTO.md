# PRD-268: Proactive Auto: notice, suggest, ask, never flood

**Status:** **Approved to build**, 28 Oct 2026 (Gerard: "get it all started"). Every §12 recommendation is the default until Gerard changes it. Still gated: §12-Q17 (Always allow), Q22 (calibration consent), Q4 (reply chips on the phone) and Q27 (the push-tap bug before a fourth push type). Tracking issues: #1358 (W1), #1359 (W2), #1360 (W3), #1361 (W4). v1 (27 Oct) is kept beside this file as `PRD-268-PROACTIVE-AUTO.v1-2026-10-27.md`. · **Owner:** Gerard · **Written:** 27 Oct 2026; v2 28 Oct 2026, grounded on `automatos-ai` `main @ 28a775307`, `automatos-go` `main @ 289a8e9`, `automatos-design-kit` `origin/main @ c584f0d`, and `CopilotKit/OpenDuet` `main @ 60fd4c7` (MIT, created 21 Oct 2026). Every v1 citation was rechecked against these commits: all still hold.
**Why now:** Gerard read OpenDuet on 27 Oct: "This is the new thing — 'proactive' — even though Automatos was building for 18 months, so we need to keep ahead." On 26 Oct VECTOR's failing heartbeat filed one board card per tick (#0846, #0848, #0850, #0853 and more), each "ROUTINE · HIGH · BLOCKED · APPROVAL", and 12 cards needed Gerard's eyes (`<workspace>/.claude/BUGS-FOR-FIXER-2026-10-26.md` §2).
**Scope:** automatos-ai (core and web: Studio and Classic), automatos-design-kit (the Play and HUD themes) and automatos-go (the app). Stories for the kit and the app are written here so the whole design can be reviewed in one place. On approval they move into each repo's PRD, as PRD-266 and the app's PRD-001 were split (§12-Q14).
**Depends on:** PRD-266 W2 (push types, device rows, digest; merged). The server voice route `POST /api/voice/speech` is live (core #1348, 27 Oct; discovery lists `tts`), and automatos-go build 13 plays it; both are needed for the spoken brief only.

**What v2 changes.**
- It merges the Claude Design handoff (`<workspace>/DUMPING AREA/design_handoff_proactive_auto/`, "the handoff"), which designed the same features for the Play and HUD web themes. Its display contracts become §6.11, corrected where they disagreed with the code.
- A new §6.12 places every surface in Studio, Classic, Play, HUD and every installed theme, and in Go's three layouts. The MOBILE session agreed the Go placements on 28 Oct. The PLAY session reviewed the kit side the same day; its measurements and file anchors are in §6.12.2, §6.12.4, US-C21, US-T01 and Wave K.
- The Go placements of §6.5.2 move there: one card above the composer, not a row under the needs strip.
- The handoff's glyphs are corrected to the app's existing vocabulary, and its fixed Ember colours to look-relative roles.
- OpenDuet's provenance rule is added (§5.3, §6.2.3).
- New stories US-C17–C22, US-W04–W06, US-G05–G09, US-T01 and US-K01–K02; decisions §12-Q16–Q28.

Labels: **[H]** from the handoff; **[M]** agreed with the MOBILE session (28 Oct); **[T]** read in code on the commits above; **[O]** read in OpenDuet's code; **[I]** inferred or estimated; **[Dn]** a decision from the 20 Oct mobile Q&A (`<workspace>/reports/mobile-review-2026-10-20/decisions.md`); **§12-Qn** an open decision in §12.

**Needs Gerard's approval before code (AGENTS.md "stop and ask first"):**

| Item | Where | Why it is on this list |
|---|---|---|
| Agent heartbeats stop filing board tickets and become read-only check-ins | §6.2, §12-Q1 | Changes what every existing heartbeat does |
| A new tool-execution lane that refuses every non-read tool | §6.2.3 | A guardrail in the executor every agent passes through |
| A fourth push type, `suggestion`, with lock-screen actions | §6.5.3, §12-Q3 | Revises [D18] and [D22] |
| Reply chips in the phone's thread | §6.6, §12-Q4 | Revises [D29] ("No suggestion chips") |
| New tables (`proactive_suggestions`, `proactive_runs`) and columns on two existing ones | §6.2.4, §6.3, §6.4 | Schema |

---

## 1. Introduction

Automatos already does a lot without being asked. Agents run heartbeats on a schedule, a watcher supervises Missions and playbooks, scheduled tasks fire, Missions narrate into the chat they came from, Auto's Read summarises the workspace, a daily digest is posted to each phone user, and pushes reach the phone for questions, reviews and Auto's own messages [T, §4].

The weak part is how that work reaches the person. Today it reaches them as **board tickets, reports and notifications**. A heartbeat routed to Auto files a new ticket on every tick, whether or not anything changed (`orchestrator/services/heartbeat_service.py:1466-1536`). Each such ticket then costs a second model run when Auto picks it up, and on SaaS it waits for an approval before it runs (`heartbeat_service.py:1533-1536`). Nothing records whether a person found any of it useful.

OpenDuet shows a cleaner shape. A background check-in may only read; most check-ins end silent; when one finds something it records a **suggestion** that the person starts or hides; a started suggestion becomes a task with its own conversation; follow-ups are capped; the morning brief is waiting the first time the person opens the app that day [O, §5].

**Goal:** one voice. Auto notices, suggests and asks. It does not act on its own initiative, it does not file work nobody asked for, and it does not repeat itself. Every proactive message is either something new that needs the person, or silence.

## 2. Goals

- Background checks read; they never write. Enforced in the executor, not by the prompt.
- What a check finds reaches the person as a suggestion with **Start** and **Hide**, on the web, in the Auto thread and on the phone; a started suggestion becomes a board ticket, a Mission or a playbook run that the person can follow on the Board.
- Heartbeats file **zero** board tickets on their own. A failing agent produces one suggestion per failing streak, not one card per tick.
- Auto can say "I'll check again in an hour" and keep to it, within caps that are counted, locked and audited.
- The brief is waiting the first time the person opens Automatos each local day, on any device, once. It can be heard in Auto's server voice.
- When Auto asks the person to choose, the likely answers appear as taps.
- Each of these has an eval with a number to hit before it goes live, and a live metric afterwards.
- A proactive day costs little: most check-ins make no model call at all (§6.9).
- One status vocabulary on every surface: a glyph, a word and the next step, never colour alone (§6.11.1).
- The brief says what it couldn't check (§6.11.3), and an approval shows the exact action and the rule that asked (§6.11.4).
- Every proactive surface works in Studio, Classic, Play, HUD and every installed theme, and in Go's three layouts, from one contract (§6.12).

## 3. Non-goals

- Auto acting on its own initiative. Every action that comes out of a check-in passes through a person's Start. "Act" heartbeats remain only where an owner explicitly keeps them (§12-Q1).
- Reading the person's mail or calendar. OpenDuet's check-ins read Gmail and Calendar [O]; ours read the workspace (board, runs, reports, documents, playbooks) and whatever read-only tools an agent already has. Personal-inbox triage is a separate product question.
- A new task list. The Board is the task list; started suggestions are ordinary tickets.
- Team routing of suggestions to the right member (v1 has one addressee per suggestion, §12-Q2).
- Real-time speech-to-speech. The spoken brief uses the text-to-speech route only.
- Replacing the watcher (PRD-204/205). Watches supervise launched work; check-ins look for new work. They share delivery, not purpose.

## 4. What exists today

### 4.1 Capabilities, where they live, and the gap

| Capability | Where in code [T] | Gap this PRD closes |
|---|---|---|
| Agent heartbeats | `services/heartbeat_service.py:1020-1189` (`_agent_tick`); config in `agents.configuration.heartbeat`; scheduled per agent (`:501-527`), active hours (`:542-609`), "already ran this period" dedupe (`:615-668`) | Runs with the agent's **full toolset**, Composio included (`modules/context/modes.py:167-175`, `tool_loading="full"`, 128,000-token context). "Report findings only" is one prompt line (`heartbeat_service.py:1101-1110`); nothing enforces it |
| Heartbeat → Auto | `report_to=auto` creates a `BoardTask` assigned to Auto on every tick (`heartbeat_service.py:1408-1418`, `:1466-1536`); failures get priority `high` (`:1517`) | No "anything new?" check, no dedupe, no streak handling: this is the 26 Oct flood |
| Heartbeat → notification | `NotificationDispatcher.dispatch(event_type="heartbeat_complete")` (`heartbeat_service.py:1432-1450`) | Every tick rings, changed or not |
| Heartbeat report | A report is filed for every tick, even one that found nothing (`heartbeat_service.py:1709-1716`, `services/heartbeat_report.py`) | Report volume with no reader |
| Orchestrator (Auto) heartbeat | Opt-in per workspace (`heartbeat_service.py:388-395`); `proactive_level` silent/notify/act_notify/autonomous (`:812-821`); default `notify` (`core/seeds/seed_auto_agent.py:415`) | `silent` and `notify` send the same instruction (`:816-817`); output is a 500-character summary, not something a person can act on |
| Auto cadence loops | Daily Brief, Weekly Org Review, HARNESS review, Team Review as prompt blocks, all default off (`core/services/auto_cadence.py:30-41`, `:48-55`) | A second "daily brief" beside the digest; Team Review's "never edit" is prompt-only (`:104-120`) |
| Triage judgement | PRD-248 decision engine asks "The owner should act on this today" and a severity for each heartbeat (`heartbeat_service.py:1399-1406`; `core/llm/decisions/judgements.py:453-462`) | Off by default and **shadow-only by construction**: a `live` setting is read as `shadow` (`core/llm/decisions/engine.py:129-136`). Nothing acts on its verdict |
| Watcher (PRD-204/205) | `services/watch_ticker.py:1-33`; `core/models/watches.py` (one live watch per target, `DEFAULT_ACTION_BUDGET = 2`, `:41-44`); verdicts post to the origin chat or the Auto thread (`watches.py:83-87`) | Supervises launched work only; it does not look for new work |
| ChatMessenger and the Auto thread | `services/chat_messenger.py`: one Auto thread per (workspace, person) (`:52-98`); background posts with provenance (`:122-229`); `auto_message` announced for scheduled tasks and the digest only (`services/notification_events/auto_message.py:44-49`) | The seam to reuse. A post carries text only (`chat_messenger.py:208-214`), no card |
| Scheduled tasks | `platform_schedule_task` (`modules/tools/discovery/actions_scheduling.py:17-30`); caps 10 per agent, 25 recurring and 50 operator rows per workspace (`services/scheduled_task_service.py:34-38`) | No minimum interval, no horizon, no per-person daily cap: a one-shot only has to be in the future and a cron only valid (`services/scheduled_task_steps.py:49-62`). Usable as the base for follow-ups with caps added |
| Missions and plan approval | `platform_create_mission` (`actions_mission_create.py:93`); `mission_plan_ready` notification (`services/coordinator_service.py:3101-3106`); narration into chat (`:644-655`) | The way a large suggestion becomes multi-agent work. Already pushes as `needs_answer` |
| Mission suggestion card | Emitted mid-turn as a stream data part when the router judges a message complex (`api/chat.py:305-309`); web card `frontend/components/chatbot/mission-suggestion-card.tsx:19-67`; Go `automatos-go:src/chat/cards/MissionSuggestionCard.tsx:96-101` | Exists only in the live stream: "The saved reply carries no card" (`automatos-go:src/chat/cards/model.ts:4-8`). "No thanks" is local state on both clients (web `:26-31`, Go `:101`), so a dismissal is never recorded |
| Auto's Read | `services/digest_service.py:113-142` (cached per workspace state hash, `:4-7`); `GET /api/activity/digest` and thumbs up/down `POST /api/activity/digest/feedback` (`api/activity.py:71-121`); web `frontend/components/activity/autos-read.tsx:39-137` | Per workspace, not per person; the phone never calls the feedback route [T, Go report] |
| Daily digest | `services/mobile/digest.py:1-27`: posted at each phone's chosen time, claimed once per (workspace, person, local date) in `mobile_digest_posts` (`:180-190`) | Only people with an active **phone that has a timezone** get one (`:73-90`). Web-only people never do. Not "waiting on open" |
| Needs-you | `services/needs_you.py:1-50`, `:204-223`; `GET /api/activity/needs-you` (`api/activity.py:329-346`) | Per workspace, filtered by role. The right home for "what needs you"; suggestions are a separate, lighter list ("what Auto thinks you might want") |
| Questions with options | `platform_ask_human` parks work and offers options as buttons (`modules/tools/discovery/actions_asks.py:13-80`); answers also by Telegram reply (`:6-7`) | Options exist for parked work only, not for ordinary chat questions |
| Push | Three types: `needs_answer`, `review`, `auto_message` (`services/push/payload.py:41-45`); signed v2 payload (`:188-227`); channels (`:82-86`); questions carry up to three options (`:66-68`) | No suggestion type |
| Push preferences | Per-person rows, else the workspace's; no row means on (`services/push/preferences.py:1-26`, `:42-47`) | Needs a fourth switch |
| Quiet hours | Workspace-level only (`core/services/auto_reporting.py:12-17`, `:99-124`); pushes are held in quiet hours unless urgent (`core/services/notification_dispatcher.py:35-38`) | No per-person quiet hours |
| Tool permission levels | Every platform action is `read`, `write` or `destructive` (`modules/tools/discovery/action_registry.py:64`); the registry can filter by level (`:166-191`); the policy gate classifies risk from it (`modules/policy/policy_document.py:238-272`); the executor already branches on write/destructive (`modules/tools/discovery/platform_executor.py:932`) | The building block for an enforced read-only lane |
| Spoken style | `SPOKEN_OUTPUT_CONTRACT`: one to three short sentences, no markdown (`modules/voice/spoken_style.py:24-44`), applied on spoken turns (`consumers/chatbot/service.py:920-924`) | No eval measures it |
| Tool prompt chips (web) | PRD-40/41 `ToolSuggestionBar` (`frontend/components/suggestions/types.ts:7-37`) | Canned prompts per connected app, not answers to Auto's question |
| Cost controls | One ask per heartbeat tick (`heartbeat_service.py:796`); background lanes never paid from the trial (`services/credits/asks.py:35-38`; `heartbeat_service.py:674-700`); daily spend ceiling refuses new autonomous work (`services/daily_spend_guard.py:1-17`); heartbeats use the `orchestrator_llm` setting (`core/llm/manager.py:47`); platform default model `claude-haiku-5-5` (`core/llm/defaults.py:19`) | The spend guard is called by board pickup and Mission dispatch only (`api/board_tasks.py:1942`, `services/coordinator_service.py:3032`), not by heartbeats |

### 4.2 Automatos Go today

| Surface | Where [T] | Note |
|---|---|---|
| Home per style | `src/styles/homes/StyleHome.tsx:43-48` picks Studio (`src/chat/screens/HomeScreen.tsx`), Play (`src/styles/homes/PlayHome.tsx`) or HUD (`src/styles/homes/HudHome.tsx`) | |
| Needs-you | Studio draws a card strip (`src/chat/home/NeedsStrip.tsx`, `src/chat/home/needs-strip.ts`); Play a slim line (`src/styles/play/companion/PlayHeader.tsx`); HUD "COMMAND · N" (`src/styles/hud/core/HudTopBar.tsx`); one source, `GET /api/activity/needs-you` (`src/api/needs-you.ts`) | A Suggested row sits beside each of these |
| Chat cards | `suggestion`, `plan`, `tool`, `withdraw` (`src/chat/cards/model.ts:23-28`); "Nothing here is persisted" (`:8`) | |
| Push | Lock screen: a question offers Reply… (unlocked phone) and Open; everything else Open only; "No Approve and no option buttons… Every action opens the app" (`src/push/platform/categories.ts:1-6`) | [D18] |
| Face ID | `src/step-up/step-up.ts` is the only `expo-local-authentication` user; gates approve and answer, never deny | Start will join it |
| Composer | "No suggestion chips, no Plan toggle (D30)" (`src/chat/composer/Composer.tsx:4`) | [D29] |
| Digest | A push setting and a time (07:00/08:00/09:00) (`src/push/settings/digest.ts`); no brief card, no playback | |
| Voice | Server voice is live: `src/voice/server/*` calls `POST /api/voice/speech` and plays it through expo-audio (build 13); on-device speech (`src/voice/speech/speaker.ts`) is the fallback | Listen on the brief can use it now (v2) |

### 4.3 What is per workspace and what is per person

| Thing | Scope today [T] |
|---|---|
| Heartbeats, their active hours and timezone | Per agent, so per workspace (`heartbeat_service.py:542-609`) |
| Auto thread | Per (workspace, person) (`chat_messenger.py:52-73`) |
| Auto's Read text | Per workspace (`digest_service.py:121-126`); digest delivery per person per local date (`mobile/digest.py:14-16`) |
| Needs-you | Per workspace, filtered by role (`needs_you.py:46-49`) |
| Push switches | Per person, falling back to the workspace (`push/preferences.py:1-26`) |
| Question redaction | Per workspace (`push/payload.py:73-75`) |
| Quiet hours | Per workspace (`auto_reporting.py:46-51`) |
| Credits and spend ceiling | Per workspace |

Suggestions, follow-ups and the brief are **per person** in this design; check-ins and their caps are **per workspace**.

## 5. OpenDuet compared

OpenDuet is a six-day-old open-source assistant on CopilotKit's OpenBot. It is single-person: everything is scoped to `ownerUserId` [O].

### 5.1 What OpenDuet does that we do not

| Mechanic | How OpenDuet does it [O] |
|---|---|
| Read-only background runs, enforced | "FAIL CLOSED. Anything not positively classified as a read is refused" (`openbot/server/src/proactive/restriction.ts:15`); allowed tools are those with `effect === "read" && destructive !== true` (`:104`); the write-up turn may call only `offer_task` (`:70-73`); refusal text tells the model to "Suggest it as a next step instead" (`:125`) |
| Only what is new | The window starts at the last **successful** check-in, else a day back (`proactive/engine.ts:104`, `checkInWindow`); the prompt says "Most check-ins should record nothing… silence is the right outcome" (`engine.ts:304`) |
| Suggestions with Start and Hide | `status: "open" \| "dismissed" \| "started"` (`proactive/types.ts:4`); up to 5 per run (`:14`); open and hidden titles are never suggested again, hidden ones for 14 days (`engine.ts:105-106`, `:305-312`); dedupe throws rather than drops (`engine.ts` `addSuggestion`); `POST /suggestions/:id/start` and `/dismiss` (`proactive/routes.ts`); one Start wins and opens a task with its own conversation (`engine.ts` `start`) |
| Tasks with states | Working, Needs input, Ready for review, Resolved (`app/src/lib/tasks/queries.ts:14-18`, `:108-113`); origin asked / suggestion / check_in (`engine.ts:117`) |
| Check-in interval bounds | 15 minutes to 24 hours (`proactive/types.ts:11-12`); default 4 hours via the API (`:27`), 1 hour from the one-button UI (`app/src/lib/proactive.ts:167`) |
| Bounded wake-ups | Min 1 minute, max 7 days, at most 5 pending and 20 per day per person and bot (`openbot/server/src/agents/wake-up.ts:32-38`); enforced under a Postgres advisory lock (`:184`); audited on schedule and cancel (`:118-147`); a failed follow-up is not retried (`:403-406`) |
| Brief waiting on open | One claim per local date across tabs and devices (`openbot/server/src/channels/routes.ts`, `morningBriefOn`); the date comes from the browser, checked within a day of UTC (`openbot/shared/morning-brief.ts:96-99`, `openbot/server/src/channels/morning-brief.ts:20-26`) |
| First day "gets to know you" | Reads recent mail and the week's calendar, answers in three parts with "Sounds right" / "Something's off" (`openbot/shared/first-day.ts:125-144`) |
| Reply chips | A `suggest_replies` tool; never on the brief turn (`openbot/shared/morning-brief.ts:30-31`) |
| Evals with numbers | Triage: 62 invented emails, committed run 100% precision and recall on `claude-sonnet-5-5` (`evals/triage/README.md:7`, `:36`). Reply chips: 10 choice questions, pass at 2–3 chips of at most 24 characters on 7 of 10 turns (`evals/reply-chips/README.md:9`). Voice: fail if the median answer exceeds 45 words or any exceeds 80 (`evals/voice/README.md:16`); its own manual audit found 2 of 12 answers leaked a source caveat the checker missed |
| Kill switch | Per person, per bot pause checked at every unattended turn start, fail closed (`openbot/server/src/agents/lifecycle.ts:10-14`) |

### 5.2 What we do that OpenDuet does not

| Automatos strength | Where [T] | OpenDuet [O] |
|---|---|---|
| Multi-agent Missions with a plan the owner approves | `coordinator_service.py`, `mission_plan_ready` | Coordination tools are zeroed in proactive runs; a started suggestion is one bot's task |
| A shared Board with numbers, review gates, SLAs and one needs-you count | `needs_you.py` | Per-person task list |
| Playbooks with schedules, missed-run and bench detection | `watch_ticker.py:16-21` | None found |
| A watcher that supervises launched work to a verdict, with a corrective budget | `core/models/watches.py:41-44` | None found |
| Workspace knowledge (RAG) and agent reports as evidence | `modules/rag`, reports | Mail and calendar only |
| Native phone app with signed push, lock-screen answers, Face ID before approve | PRD-266; `automatos-go:src/push`, `src/step-up` | A `push` transport flag (`lifecycle.ts:30-40`) |
| Quiet hours | `auto_reporting.py:99-124` | None found |
| Credits per ask, trial never pays for background work, a daily spend ceiling | `services/credits`, `daily_spend_guard.py` | No cost guard found in the proactive code |
| Telegram and Slack answers to questions | `actions_asks.py:6-7` | Slack, Teams, SMS notification transports |

The gap is delivery and discipline, not machinery. §6 closes the gap with what we already have, and §6.8 lists where we go further.

### 5.3 Re-read on 28 Oct (`60fd4c7`): what to take and what to leave

**Take (behaviour only):**
- provenance tied to reads the run really made (§6.2.3);
- a once-a-day claim done in one row update that only moves forward in date, so phones in different time zones can't take turns;
- "couldn't sort it" goes to Needs you, with a summary row;
- "couldn't check" rows that the model is told never to contradict.

**Leave:**
- research switched on by default when a person first talks to a bot;
- follow-ups that run with the bot's full tools and approval gate (ours run in the read-only lane, §6.3);
- Always allow scoped to the whole app rather than the recipient, with an AI reviewer still deciding;
- a started suggestion or brief row sent as the person's own message (§6.2.3);
- telemetry on by default and a hosted key needed to boot.

**Where the README claims more than the code:**
- It says a card number or health detail never goes out; the content check detects credentials only.
- It says the brief reads mail, calendar and Slack; it reads Gmail and Calendar.
- The shipped default triage model (`gpt-5.6-terra`) wasn't run on the current 62-email set.

**Their phone app has none of it:** no brief, suggestions, tasks or chips. The Expo client shows raw approval JSON. Proactive on a native phone, with signed push and Face ID, is where Automatos leads (§6.12.5).

---

## 6. Design

### 6.1 One pipeline

```
signal ──► check-in (read-only) ──► 0..3 suggestions ──► delivered (web · Auto thread · phone · push)
                                         │                       │
                                         │               Start ──┴── Hide (reason)
                                         ▼                 │
                                   proactive_runs          ▼
                                     (audit)        board ticket · Mission · playbook run
                                                           │
                                                     watch (PRD-204) ──► outcome posted back
```

Every producer that today files a ticket or rings the bell on its own initiative (agent heartbeats, Auto's heartbeat, cadence loops) feeds this pipeline instead. The watcher, scheduled tasks the person created, Missions and questions keep their current routes: they are answers to something the person started.

### 6.2 (a) Read-only check-ins that suggest

#### 6.2.1 Two kinds of check-in

1. **Auto's workspace check-in** replaces the orchestrator heartbeat in `silent`/`notify` mode and the cadence "Daily Brief" (`auto_cadence.py:48-55`). It starts with a **deterministic signal diff** over the window since its last successful run: tickets that became failed, blocked or stuck; heartbeat results with `status = error` or `objective_met = false`; watch events (`missed_run`, `benched`, `expired`, `budget_exhausted`, `watch_enums.py:71-80`); reports with `requires_approval` not acknowledged; approval grants lapsing in the next 24 hours; skipped scheduled runs; `credits_low`. **No signal, no model call**: the run is recorded as `silent` at zero cost. Signals present: one Haiku call with the read-only tool set writes up 0–3 suggestions.
2. **Agent check-ins** are converted agent heartbeats. The agent's own prompt runs as today, but in the new `CHECK_IN` context mode with read-only tools only and a lean context (12,000 tokens instead of `HEARTBEAT_AGENT`'s 128,000, `modes.py:167-175`). Its output is 0–3 suggestions, never a ticket. An agent whose domain is outside the workspace (VECTOR reads SEO data through read-only connector calls) cannot be pre-checked deterministically, so its interval and model are its cost levers (§6.9).

Both use OpenDuet's window rule: "new since the last **successful** run, else the last 24 hours"; a failed run's reads are not trusted.

#### 6.2.2 Failing streaks

A tick that errors (provider 401, tool failure, timeout) makes **no model call and no ticket**. The runner keys the failure by `(agent, error class)`:

- First failure of a streak: one suggestion, e.g. "VECTOR's check-ins fail: the AI provider refuses its key (HTTP 401) since 09:00. Start: Auto moves VECTOR to the working model." Proposed action: a ticket for Auto, or a settings change through the existing approval card.
- Later failures of the same streak update `occurrences` and `last_seen_at` on that suggestion. No new card, no push.
- After 3 consecutive failures the agent's check-in interval doubles each run, up to 24 hours, and the card says so. The first success resets it and resolves the suggestion as `expired` with "VECTOR's check-ins work again".

Under this rule, 26 Oct would have produced one card, not twelve.

#### 6.2.3 The read-only lane (enforced)

- A new `ContextMode.CHECK_IN` (in `modules/context/modes.py`) loads platform actions with `permission_level == "read"` only (`action_registry.py:166-191`), plus one write tool, `platform_suggest_next_step`. No Composio tools, no plugin tools, no `platform_execute` meta-tool, no sub-agent or Mission tools.
- The executor refuses in depth: every call made under `caller_context.lane = "check_in"` is checked in `PlatformActionExecutor.execute` (`platform_executor.py:869`, beside the existing write/destructive branch at `:932`). Anything not positively `read`, including an unknown action, is refused with "Not available in a check-in, which can only read. Suggest it as a next step instead." This mirrors OpenDuet's fail-closed rule (`restriction.ts:15`, `:125`).
- A CI test walks the registry and asserts every action marked `read` touches no write path that the test's database spy can see (inserts, updates, deletes, outbound HTTP). 114 actions declared `read` on 27 Oct [T, count over `modules/tools/discovery`]. At least one more has been added since: `platform_get_social_plan_slots` (`actions_socials.py:254`) takes the default without stating a level, which is exactly the leak described below. The field defaults to `read` when a new action omits it (`action_registry.py:64`). A mislabelled or unlabelled one is a leak, so US-C02 removes the default (an action must state its level, checked at registration) and the lane refuses an action with none.
- **Provenance (from OpenDuet, v2).** A suggestion's evidence must name reads that the same run actually made and that succeeded, checked against the run's tool-call log; otherwise the tool refuses it. OpenDuet does this against its audit table [O, `engine.ts:443-460`].
- Everything read is untrusted data. Suggestion text is built from structured fields; evidence is quoted, never followed. Starting a suggestion never posts its text as the person's own message. OpenDuet does post it that way, and its approval reviewer then reads it as "the person asked" [O, `policy.ts:158-168`].

#### 6.2.4 Data model

`proactive_suggestions` (new):

| Column | Meaning |
|---|---|
| `id` (UUID), `workspace_id`, `user_id` | Addressee: one person per suggestion in v1 (§12-Q2) |
| `source_type`, `source_agent_id`, `run_id` | `check_in`, `heartbeat`, `follow_up`, `streak`; the agent; the run that made it |
| `title` (≤ 120), `detail` (≤ 600) | What Auto says |
| `evidence` (JSONB) | Up to 5 refs: ticket numbers, report ids, document ids, run ids; each opens in the client |
| `proposed_action` (JSONB) | `{kind: board_task \| mission \| playbook_run \| chat, ...payload}`: what Start does, shown on the card |
| `subject_key` | Dedupe key, e.g. `agent:578:error:401`, `ticket:0846`, normalised title |
| `severity` | `today` or `can_wait`, from triage (§6.7) |
| `status` | `open`, `started`, `hidden`, `expired` |
| `occurrences`, `last_seen_at` | Streak counting |
| `started_ref` | `{board_task_id \| mission_id \| execution_id}` |
| `decided_via`, `hide_reason`, `decided_at` | `web`, `go`, `push`, `chat`, `telegram`; `not_now`, `not_useful`, `wrong_agent` |
| `est_credits` | Estimated cost of Start (§6.8) |
| `created_at`, `delivered_at`, `expires_at` | Open suggestions expire after 7 days |

`proactive_runs` (new, the audit): `id`, `workspace_id`, `agent_id`, `kind`, `window_since`, `window_until`, `outcome` (`silent`, `suggested`, `error`, `skipped_quiet`, `skipped_cap`, `skipped_credits`, `skipped_paused`), `signals`, `suggestions`, `model`, `tokens`, `cost_usd`, `created_at`. Kept 90 days.

Heartbeat config gains `mode: "check_in" | "act"` inside `agents.configuration.heartbeat` (no new table). `act` keeps today's behaviour for agents an owner explicitly allows (§12-Q1).

**Dedupe:** a new suggestion is refused (the tool returns an error the model reads) when an `open` suggestion has the same `subject_key` or normalised title, or a `hidden` one has it within 14 days. The prompt also lists open and recently hidden titles, as OpenDuet's does (`engine.ts:305-312`).

#### 6.2.5 Start and Hide

- `POST /api/proactive/suggestions/{id}/start` moves `open → started` in one `UPDATE … WHERE status = 'open' RETURNING`: one Start wins. The click is the person's approval (the human-direct rule, `modules/policy/gate.py:216-224`).
  - `board_task`: a ticket with `source_type = 'suggestion'`, `source_id = <suggestion id>`, `created_by_type = 'user'`, assigned to the proposed agent, `review_mode = 'human'` by default (§12-Q5), its brief built from the suggestion's fields and evidence.
  - `mission`: `platform_create_mission`'s path; the Mission stops at plan approval as today.
  - `playbook_run`: the named playbook runs with the inputs shown on the card.
  - `chat`: opens the Auto thread with the suggestion as context ("Discuss"), like "Ask Auto about #0419" [D31].
  - Each started item gets a watch (PRD-204), so its outcome posts back into the thread that carried the suggestion.
- `POST /api/proactive/suggestions/{id}/hide` with an optional reason. Hide is never gated.
- **Started work on the Board.** A "From suggestions" filter (`source_type = 'suggestion'`) on the web and Go Boards shows the states OpenDuet calls Working (assigned, in progress), Needs input (blocked on a question or approval), Ready for review (review), Resolved (done) and Failed. No new list.

### 6.3 (b) Bounded follow-ups

Auto can say "I'll check again at 15:10" and does.

- **Tools:** `platform_schedule_follow_up(reason, in_minutes | at)`, `platform_list_follow_ups`, `platform_cancel_follow_up`, available in chat to Auto. A follow-up on a launched Mission, playbook run or ticket is refused with "use a watch" (the watcher already supervises those).
- **Storage:** reuse `agent_scheduled_tasks` (one-shot rows, `scheduled_task_service.py`) with `deliver_as = 'follow_up'`, `for_user_id`, and the origin chat captured server-side (never a model argument, as `chat_messenger.py:104-119` requires).
- **Caps** (counted under `pg_advisory_xact_lock` on (workspace, person, agent), so five parallel calls cannot pass a cap of five):

| Cap | Value | OpenDuet [O] |
|---|---|---|
| Earliest | 10 minutes from now | 1 minute |
| Latest | 7 days from now | 7 days |
| Pending per person and agent | 5 | 5 |
| Scheduled per person and agent per day (fired or not) | 20 | 20 |
| Scheduled per workspace per day | 100 | none |
| Chain depth | A follow-up's own turn cannot schedule another follow-up (§12-Q9) | No retry of a failed one |

- **When it fires** the turn runs in the `CHECK_IN` lane with the reason and the origin chat's last 5 messages. Three outcomes: nothing changed, so it records `silent` and posts nothing; something to say, so it posts one message in the origin chat (origin `follow_up`, announced as `auto_message`, `auto_message.py:47`); work to do, so it records a suggestion. A follow-up never acts. A failed follow-up turn is recorded and not retried.
- **Audit:** schedule, cancel, fire and outcome each write a workspace audit row (`core/workspaces/audit.py`) and a `proactive_runs` row.
- **In the thread** the client shows "Follow-up at 15:10 · Cancel" under the message that scheduled it, read from the tool result.

### 6.4 (c) The brief, waiting on open

Today the digest is pushed at a chosen time to people with a phone (`mobile/digest.py:73-90`). This adds "waiting on open" for everyone and keeps one brief per person per day.

- **Claim:** reuse `mobile_digest_posts`, already unique on (workspace, person, local date) (`mobile/digest.py:180-190`), adding `message_id` and `via` (`schedule` or `open`). The scheduled digest and the on-open brief share the claim: whichever comes first posts, the other returns the same message.
- **Route:** `POST /api/proactive/brief/open {local_date, timezone}`. The client calls it once per local date when the web app loads or the phone comes to the foreground. The server checks `local_date` is within one day of UTC (as OpenDuet does). Claim won: build, post into the Auto thread (origin `daily_digest`), return `{chat_id, message_id, posted: true}`. Claim taken: return the existing message. The brief on open runs only where the workspace's effective proactive mode (W1's `effective_mode`: the platform mode where the workspace switch is on, else off) is shadow or live, as §10 rolls it out (elsewhere the route answers 404 `brief_off` before any claim or build), and the scheduled digest is unchanged.
- **Content:** Auto's Read (cached per workspace state, `digest_service.py:113-142`, unchanged), then a deterministic per-person block with no second model call: "3 things need you" with the needs-you link, up to 3 open suggestions as Start/Hide cards, and what yesterday's started suggestions became (Ready for review, Resolved, Failed).
- **Feedback:** the thumbs up/down that Auto's Read has on the web (`api/activity.py:91-121`) is offered on the brief card everywhere.
- **Spoken:** a "Listen" button sends the brief's text, passed through `speechify` (`modules/voice/spoken_style.py`), to `POST /api/voice/speech`. On a refusal code (voice off), the client falls back to device speech (`automatos-go:src/voice/speech/speaker.ts`; the browser's speech synthesis on the web). Never auto-play (§12-Q12). Go plays server audio on build 13, in the voice for the person's language (#25/#26).
- **Structure:** the brief is posted as a structured `brief@1` part beside its text (§6.11.2), so every surface can draw it as a card.
- **A failed build releases the claim**, so the next open tries again. OpenDuet never releases it, and the day goes without a brief [O, `channels/routes.ts`].
- The cadence "Daily Brief" block (`auto_cadence.py:48-55`) is retired (§12-Q11).

### 6.5 (d) Delivery

#### 6.5.1 Web

- **Command Center:** a "Suggested" card list beside Auto's Read in the summary tab (`frontend/components/command-center/summary-tab.tsx`), each with title, detail, evidence links, "Start does: …", estimated credits, **Start** and **Hide** (with an optional reason).
- **Auto thread:** each suggestion delivered to a person is also posted in their Auto thread through `deliver_background_message` with `source.suggestion_id` (the persisted provenance, `chat_messenger.py:195-203`). The client draws the card from that id with live state, so it survives reload (unlike today's Mission suggestion). The newest open suggestion is pinned above the composer in the Auto thread. Origin `suggestion` is not in `ANNOUNCED_ORIGINS` (`auto_message.py:47`), so the post does not also ring as "Auto messaged you".
- **Bell:** a `suggestion_ready` in-app row, linked to the card (the bell-link fixture is updated, as PRD-266 US-012 did).
- **Per theme** (Studio, Classic, Play, HUD, other installed themes): §6.12.4.

#### 6.5.2 Automatos Go

v1 put a "SUGGESTED · N" row under the needs strip. v2 replaces it with the placements agreed with MOBILE (§6.12.5) [M]:
- one open suggestion as a card directly above the composer in Studio and Play, collapsing to one line while the keyboard is up;
- on HUD, a "NEXT ▸ …" line in the top bar's status row, never above TYPE · HOLD TO TALK;
- a count in the header in all three layouts, opening a sheet of every open suggestion.

The rest of v1 holds:
- **Start** asks Face ID through `src/step-up`; **Hide** never does.
- **Board:** the "From suggestions" filter.
- **One query:** `GET /api/proactive/suggestions?status=open`, refetched on a new `suggestion_changed` live event and on the 60-second poll that needs-you already uses. Suggestions come from this endpoint, never from thread cards, which aren't persisted.
- **No credits on the phone** (D8): the cost line in §6.8 shows on the web only.

#### 6.5.3 Push

- A fourth push type, `suggestion` (`push/payload.py:41-45`), for the event `suggestion_ready`; channel `suggestions`; default priority (`:305`); the AI badge in the title as every push.
- **Text:** generic, "Auto has a suggestion", unless the workspace switch `push.suggestion_titles` is on, in which case the title (§12-Q7).
- **When:** only `severity = today` suggestions; one push per run however many it made; at most 2 per person per day; never in quiet hours (they wait for the brief); a person's own switch in `/api/notification-preferences` (`push/preferences.py:42-47`).
- **Lock screen:** category `automatos.suggestion` with **Start…** (unlock required, opens the app on the card, Face ID, then starts) and **Open**. Hide from the lock screen without opening the app needs a background action handler that the app does not have today ("Every action opens the app", `categories.ts:6`); it is a later story once proven on a build (§12-Q3).

#### 6.5.4 Telegram and Slack

When a workspace routes Auto's reporting to Telegram or Slack (`auto_reporting.py:10`), a suggestion arrives as one line with "Reply start or hide". The reply path that answers questions today (`actions_asks.py:6-7`) is reused. Wave W4.

### 6.6 (e) Reply chips

When Auto asks the person to choose, the likely answers appear as taps.

- **Mechanism:** a terminal tool, `platform_offer_replies(replies: string[2..3])`, available to Auto in chat. "Terminal" means the runtime renders the chips and ends the turn **without another model call**, so chips add no round trip and no cost beyond a few output tokens.
- **Limits:** 2–3 replies, each 1–3 words and at most 24 characters (OpenDuet's bar). Only when the reply asks the person to choose. Never on the brief. On spoken turns the chips are shown but never spoken.
- **Storage:** the chips are saved on the assistant message (a `reply_chips` entry in its source), so they survive reload; the client hides them once any newer message exists.
- **Tap:** sends the chip's text as the person's message. Free text stays available.
- **Clients:** under the last assistant message on the web chat and in all three Go threads. On the phone this revises [D29], which ruled out suggestion chips in the composer; these are answers to Auto's own question, not canned prompts (§12-Q4).
- Questions parked with `platform_ask_human` keep their option buttons as today.

### 6.7 (f) Evals

Runners live in the repo (`orchestrator/evals/proactive/`); gold sets stay local and are never committed (public repo). Runs use Haiku 5.5, the platform default, and report numbers only. Each must pass before its feature goes live; each has a live metric after.

| Eval | Set | Pass bar | Live metric (target at 4 weeks) |
|---|---|---|---|
| **Triage** ("act today" vs "can wait", with a one-line reason ≤ 120 characters) | 80 labelled heartbeat results and reports: c1's history (all fake data) plus hand-written cases incl. 401 streaks, missed playbooks, budget pauses | Recall for "act today" ≥ 95%, precision ≥ 90%, every verdict has a reason. OpenDuet: 62 emails, 100/100 on Sonnet [O] | Pushes per person per day ≤ 1 on average |
| **Check-in silence** | 40 workspace snapshots where nothing new needs the person; 40 with one planted need | ≥ 95% of the first end with no suggestion; ≥ 85% of the second name the planted need; 0 duplicates of an open or hidden title | Share of check-ins that end silent ≥ 70% |
| **Suggestion precision** | Live only | n/a | Started ÷ (started + hidden) ≥ 40% per workspace; started **and** passed review ≥ 30% of decided; per source agent, below 20% after 20 decisions slows that agent (§6.8) |
| **Reply chips** | 20 choice questions, 10 open questions | Chips on ≥ 16 of 20 choice turns, all within the limits; 0 turns where the chips replace the answer; chips on ≤ 1 of 10 open questions | Chip tap rate ≥ 30% of turns that showed chips |
| **Spoken brevity** | 30 spoken-mode asks incl. 5 briefs | Median ≤ 40 words, none over 80, no markdown after `speechify`, no URL, path or id read aloud; every answer also audited by hand (OpenDuet's checker missed 2 of 12) | n/a |
| **One brief** | Deterministic test: 3 devices, 2 tabs, DST change, a phone in another timezone | Exactly one brief per person per local date | Duplicate briefs in logs = 0 |

The triage eval also decides whether PRD-248's `report_triage` judgement may run live for the check-in lane only (§12-Q15); today it is shadow-only by construction.

### 6.8 (g) Beyond OpenDuet

Each of these uses something Automatos has and OpenDuet does not. None is a new subsystem.

1. **A suggestion can start a team.** `proposed_action.kind = mission` carries a goal and the planner's draft steps. Start creates the Mission at plan approval, through the existing `mission_plan_ready` push. OpenDuet's proactive runs cannot coordinate at all (`restriction.ts:160-166`) [O].
2. **A suggestion can run what already exists.** When a check-in finds a missed or benched playbook (`watch_ticker.py:16-21`), the suggestion's Start runs that playbook with the inputs shown, instead of an ad-hoc task.
3. **Evidence you can open.** Up to 5 refs per suggestion (ticket numbers, report ids, documents from workspace knowledge), each a link in every client. OpenDuet carries one source app and ref [O].
4. **Supervised outcome.** Every started suggestion gets a watch, so its verdict posts back into the thread that carried the suggestion, and precision is measured on outcome ("started and passed review"), not only on clicks.
5. **Learns from Hide, deterministically.** A hidden subject is suppressed for 14 days. A source agent whose suggestions are hidden in ≥ 80% of its last 20 decisions has its check-in interval doubled, and the owner is told once in the brief. Hide reasons `wrong_agent` and `not_useful` are written to that agent's field memory (`modules/context/sections/field_memory.py`), which its next check-ins read.
6. **Cost on the card.** "Start (about 40 credits)": the median credits of the proposed agent's last 10 runs of that kind (PRD-265 per-ask metering). Unknown when there is no history. Web only: the phone shows no cost or credit language (Go D8, D-NO-BILLING).
7. **One card per failing streak** (§6.2.2), at zero model cost: platform health reaches the owner without filling the Board.
8. **The phone and the voice.** Signed pushes with lock-screen Start, Face ID before work starts, three home styles, and the brief read in Auto's server voice.
9. **Channels.** Start or hide from Telegram or Slack (§6.5.4).

### 6.9 (h) Cost and safety

**Model and spend.** Check-ins, triage and follow-up turns use the workspace's `orchestrator_llm` setting (`core/llm/manager.py:47`), default `claude-haiku-5-5` (`core/llm/defaults.py:19`); the 11 Oct benchmark put Haiku 5.5 level with Opus on Auto's set at about 3% of its cost [I, memory]. Per check-in: at most 4 tool iterations, 800 output tokens, 12,000 tokens of context, and a cost ceiling of $0.05 after which the run stops and records `error`. Every run is one ask (`heartbeat_service.py:796`), never paid from the trial (`asks.py:35-38`), and refused by the daily spend ceiling, which today does not cover heartbeats (`daily_spend_guard.py`). Start of a suggestion goes through the same ceiling as board pickup (`api/board_tasks.py:1942`).

**Estimate [I].** A workspace with 6 agents checking in every 4 hours makes 36 runs a day. If 30% find a signal and make one Haiku call at about $0.01, the day costs about $0.11, plus Auto's check-in, which makes a call only when its signal diff is non-empty. Today the same agents run a full-context heartbeat on every tick and file tickets that cost a second run each.

**Caps** (workspace settings under `settings.proactive`, owner/admin editable; values proposed, §12-Q8):

| Cap | Default |
|---|---|
| Check-in interval per agent | 30 minutes to 24 hours; migrated heartbeats keep their interval within those bounds |
| Check-in runs that reach the model, per workspace per day | 48 |
| Suggestions per run | 3 |
| New suggestions per person per day | 10 |
| Open suggestions per person | 20 (oldest expire first) |
| Open suggestions per source agent | 3 |
| Suggestion pushes per person per day | 2 |
| Follow-ups | §6.3 |

**Quiet hours.** Add per-person quiet hours (start, end, timezone from the person's phone, else the workspace's) to the person's notification settings. In quiet hours, check-ins still run and record, but nothing pushes; suggestions wait for the brief. The workspace window (`auto_reporting.py:99-124`) still applies on top.

**Kill switches, checked at every run start, failing closed:** a platform setting `proactive.enabled` (super admin); a workspace switch `settings.proactive.enabled` (owner/admin); a per-person "Pause suggestions" on web and Go. A paused person gets no suggestions and no pushes; the brief still comes unless they switch the digest off.

**No write tools in check-ins.** §6.2.3. Follow-up turns run in the same lane. Composio and plugin tools are not loaded at all in it.

**Audit.** `proactive_runs` for every run, workspace audit rows for every Start, Hide, follow-up schedule, cancel and fire, and the existing `llm_usage` rows for cost.

**Trust.** Suggestion text never auto-runs; Start shows what it does first; evidence is data. A ticket brief built from a suggestion quotes the evidence and states the person's Start as its instruction.

### 6.10 Where the code goes

| Repo | Path | What |
|---|---|---|
| core | `orchestrator/modules/proactive/` (pure) | Window, signal diff, dedupe keys, caps, streak rules |
| core | `orchestrator/services/proactive/` | Runner, delivery, brief claim, follow-ups, Start/Hide |
| core | `orchestrator/api/proactive.py` | Suggestions, brief, settings, runs (route manifest regenerated) |
| core | `orchestrator/core/models/proactive.py` + one Alembic revision | Two tables, columns on `mobile_digest_posts` and `agent_scheduled_tasks` |
| core | `modules/context/modes.py`, `platform_executor.py:932` | `CHECK_IN` mode and the lane refusal |
| core | `services/heartbeat_service.py:1083-1170`, `:1408-1418` | Mode switch, streaks, no ticket |
| core | `services/push/payload.py`, `push/preferences.py`, `notification_dispatcher.py` | `suggestion` push type, switch, event |
| web | `frontend/components/proactive/`, `hooks/use-suggestions.ts` | Cards, list, brief card, reply chips |
| Go | `src/proactive/`, `src/chat/home/`, `PlayHeader.tsx`, `HudTopBar.tsx`, `src/push/platform/{channels,categories}.ts`, `src/push/settings/prefs.ts`, `src/push/links/targets.ts`, `src/step-up/policy.ts` | Rows, sheet, push type, Face ID on Start, chips, brief card |
| core (v2) | `orchestrator/services/brief/`, `services/readiness/`, `services/chat_messenger.py` (`parts`), `modules/tools/discovery/actions_brief.py` (`get_brief`), `services/board_approval.py`, a rule-sentence strings table beside `modules/policy/` | Brief@1, readiness, message parts, rule sentences |
| web (v2) | `frontend/components/proactive/` (defaults), `lib/theme-sdk/host-cards/`, `lib/theme-sdk/surface/*.json`, `components/chatbot/message.tsx`, `components/chatbot/chat.tsx:1363`, `components/settings/SettingsPanel.tsx` (Orchestrator) | Default components, SDK 1.3.0, placements |
| kit (v2) | `themes/play/src/components/play/chat/`, `themes/hud/src/chat/` (BRIEFING, NEXT, SYSTEMS, MISSION rows), `packages/theme-cli/src/core/mobile-manifest.ts`, `packages/theme-cli/src/check/mobile-contrast.ts` | Play and HUD surfaces, skin status tokens |
| Go (v2) | `src/brief/`, `src/readiness/`, `src/chat/schemas.ts`, `src/chat/cards/` (on `MissionFrame`), `src/styles/dress/status-glyphs.ts`, `src/ui/variants.ts`, `app/(app)/settings/index.tsx`, `src/styles/themes/paint.ts` | Brief per layout, status vocabulary, approval sheet, readiness, skin tokens |

### 6.11 Display contracts (from the handoff, corrected)

The handoff's `CONTRACTS.md` defines what every surface reads. They are adopted here with the corrections below. Core implements each as a Pydantic model; the SDK mirrors it in `frontend/packages/theme-sdk/areas/*.d.ts` (through `orchestrator/services/themes/data/sdk-surface.json`); Go mirrors it as a zod schema in the feature's `schema.ts`. Every shape carries a version (`@1`); unknown fields are ignored everywhere; every id is a string.

**Copy is the server's, in the person's language.** Greeting, `why`, `next_step`, the rule sentence, readiness `detail` and brief section labels are written by the server in the person's language (English and Portuguese (Brazil) today, from the person's "I speak" setting, Go #25/#26). Clients never compose these strings; they only draw them. Each lives in one `strings` table per feature with both languages, and a test fails when a key is missing in either.

#### 6.11.1 Status: one vocabulary, computed once

The server computes the state; clients only draw it. The handoff put a pure `displayState()` in `frontend/lib/proactive/` for Go to vendor **and** had the server add `display_state` to rows. That is two sources for one fact. The server already has to build `next_step`, so it sends both, and clients map the state to a glyph, a word and a colour role.

Rows that gain `display_state`, `next_step` (≤ 90 characters) and `retry` (`rerun_mission` | `replan_mission` | `retry_ticket` | null): needs-you rows (`GET /api/activity/needs-you`), the Missions list (`useMissions`), board task reads used by the activity feed and the Go Board, and started suggestions ("From suggestions").

The glyphs are the ones the product already uses. Go's `src/styles/dress/status-glyphs.ts` (from HUD's README) defines them, and the board's words come from the web's `STATUS_CONFIG` through `src/ui/variants.ts`. Every glyph means one thing everywhere:

| Glyph | Means | Statuses today | Added by this PRD |
|---|---|---|---|
| ● | in motion | working, in progress, live | `working` |
| ○ | waiting, or not known yet | inbox, assigned, queued | readiness `not_checked` |
| ◆ | a person must act | review, blocked, needs you | `needs_you`; readiness `needs_setup` |
| ✓ | finished, or fine | done, approved | `ready` ("Ready to look at"), `done`; readiness `ready` |
| ✕ | failed | failed, error | `failed` ("Couldn't finish") |
| — | stopped | cancelled, closed, paused, idle, offline, unknown | none |
| ▸ | a running step | running | the next-step line |

| State | From | Glyph and word (Studio, Classic, Play) | HUD |
|---|---|---|---|
| `needs_you` | any `needs_you()` row of kind review, question, approval or stuck | ◆ Needs you | ◆ NEEDS YOU |
| `working` | board `in_progress`/`assigned` (not stuck); run `running`/`planning`; board `review` that a Mission checks, not the owner | ● Working | ● WORKING |
| `ready` | a finished run or ticket whose Deliverable is unopened (needs the seen flag, §12-Q18; until then, finished since the last brief) | ✓ Ready to look at | ✓ READY |
| `done` | `done`, `closed`, a completed run | ✓ Done | ✓ DONE |
| `failed` | kind `failed`; board `failed`; a run in a terminal failed state | ✕ Couldn't finish | ✕ FAILED |
| none | unknown, or not yet read | nothing | nothing |

Rules:
- **Unknown draws nothing.** Never a placeholder "Working".
- **The handoff's glyphs are corrected to this table.** It drew Done as —, which means stopped in the app, and readiness Ready as ●, which means in motion. It also put ○ and — on the CAN WAIT and TODAY section labels. Section labels carry no glyph.
- **Colour is a role, never the only signal.** The roles per surface are in §6.12.2.
- **Needs-you is never recounted.** Every badge, brief section and push reads `needs_you()` (PRD-252 R5).
- Brand Kits may relabel a word ("Awaiting approval"); glyphs and colour roles stay.
- `next_step` sources, in order: the needs-you row's reason (each `STUCK_*` constant, `services/needs_you.py:68-74`, mapped to a sentence in the strings table); the approval's action ("Allow me to send the message, or refuse"); the running step's title and progress ("42 of 60 made"); the failure's `error_message`, first sentence only.

#### 6.11.2 `Brief@1`: the structured brief

§6.4 decides when the brief is made and that it is made once. This is what it contains. It merges §6.4's per-person block with the handoff's sections.

```ts
type Brief = {
  v: 1
  date: string                       // the person's local date
  greeting: string                   // "Morning, Sam." in the person's language
  prose: string                      // Auto's Read (digest_service), unchanged: the fallback for any surface that can't draw the card
  sections: {
    needs_you: BriefRow[]            // from needs_you(); counts.needs_you equals the badge
    suggestions: BriefRow[]          // up to 3 open suggestions (§6.2), each with Start / Hide
    today: BriefRow[]                // activity schedule for the local day (calendar: §12-Q20)
    since: BriefRow[]                // what yesterday's started suggestions became, and work finished since the last brief
    unchecked: BriefRow[]            // Readiness items the brief depends on that are not `ready`; never hidden
  }
  counts: { needs_you: number; suggestions: number; today: number; since: number }  // exact, even when rows fold
  state_hash: string
  generated_at: string
}
type BriefRow = {
  id: string; title: string; why: string; at?: string; number?: string     // "#0419"
  display_state?: DisplayState
  opens?: { type: 'ticket' | 'mission' | 'question' | 'chat' | 'deliverable' | 'suggestion' | 'settings'; id: string }
  action?: { verb: 'start' | 'answer' | 'open' | 'connect' | 'hide'; label: string; done_label?: string }
}
```

- The handoff's **Can wait** section is renamed **Since your last brief** until §12-Q18 decides what "can wait" means. A section with a definition we can't compute today would be filled with guesses.
- Each section shows 5 rows and folds the rest behind "Show N more"; a section only one row over shows whole (fold at 5, never at 6). A row with nowhere to go has no `opens` and draws no chevron.
- An empty brief says so in words: "Nothing new needs you, and nothing is scheduled today."
- **Delivery as a message part.** The brief is posted into the Auto thread (§6.4) with `parts = [{type: 'text', text: prose}, {type: 'brief@1', brief}]`. Today `post_background_message` writes a single text part (`services/chat_messenger.py:211`) and Go's `MessagePartSchema` drops part types it doesn't know (`automatos-go:src/chat/schemas.ts`), so both change: the messenger takes `parts`, the web thread and Go's schema add `brief@1`. Old clients keep reading the prose; on Go, build 12 and older drop the part silently and still show the text. A client that draws the card shows the prose inside it and does not draw the text part a second time. The suggestion card in the thread (§6.5) uses the same seam (`suggestion@1` carrying `suggestion_id`), so it is built once.
- **Tool `get_brief`** (the 3-file pattern, `modules/tools/README.md`): Auto answers "what needs me?", "brief me" and "what's on today" in any chat, including voice and Go's quick-ask, from the same object as the route. Spoken, Auto says the titles only; the screen shows the rest.

#### 6.11.3 `Readiness@1`: say what couldn't be checked

```ts
type Readiness = { v: 1; checked_at: string; items: ReadinessItem[] }
type ReadinessItem = {
  id: 'core' | 'model_route' | 'web_search' | 'calendar' | 'push' | 'microphone' | 'speech' | 'server_voice' | string
  label: string; detail: string
  state: 'ready' | 'not_checked' | 'needs_setup'
  fix?: { label: string; href: string }          // an app path, never an outside URL
  needed_by: Array<'brief' | 'voice' | 'push' | 'check_in'>
}
```

- Glyphs, from §6.11.1: **✓ Ready · ○ Not checked · ◆ Needs setup** (a person must act). "Not checked" means the probe didn't answer in time; it never stands in for a failure the server knows about.
- One probe per server item in `orchestrator/services/readiness/`, each with a timeout from config. Reuse `api/system_health_checks.py`'s checks where one exists. `GET /api/readiness`, `POST /api/readiness/check` (rate-limited).
- `microphone`, `speech` and `push` on the phone, and `microphone`/`speech` on the web, are added by the client (`useVoice().available` on the web; the native permission, `src/voice/` and the push registrar on Go). The server never claims them.
- `detail` is edition-aware: the local edition may show `localhost:8000 · 12 ms`; SaaS never shows hostnames or latency.
- It sits **beside** the PRD-222 setup checklist (`services/onboarding_state.py:627`), which tracks onboarding progress, not dependency health. `useHealth` (backend reachable) stays as is.

#### 6.11.4 `ApprovalView@1`: the exact action and the rule that asked

Anything that sends, pays, signs, deletes or publishes pauses on the host's protected approval card (`ToolApproval`, `MissionApproval`; `frontend/lib/theme-sdk/host-cards/approvals.tsx`). This extends what the card shows; themes still place it and never redraw it, so every theme gets it at once.

```ts
type ApprovalView = {
  grant_id: string
  asked_by: { kind: 'auto' | 'agent'; name: string }
  action: { verb: 'send' | 'pay' | 'sign' | 'delete' | 'publish' | 'run' | string; summary: string }
  target?: { label: string; detail?: string }            // "Aisha Rahman · WhatsApp"
  preview?: { kind: 'text' | 'markdown' | 'diff'; body: string }   // the exact payload, redacted by the existing redaction
  rule: { id: string; sentence: string }                 // "Ask before anything that sends to other people"
  answers: Array<'refuse' | 'allow_once' | 'always_allow'>
  expires_at?: string
}
```

- **The rule sentence is new work.** No plain-language rule exists today: the gate's reason is machine text (`"posture={p} routes {risk} to ask"`, `modules/policy/gate.py:251`), and board approvals still store the raw policy key (`services/board_approval.py:128`, shown by `frontend/components/activity/board/blocked-reason.ts`). Build one strings table over posture × risk class (`modules/policy/policy_document.py`) and the Mission `approval_policy` values, and stop writing the raw key.
- **Always allow is gated on §12-Q17** (it writes policy, and no per-tool or per-target allow exists today; only workspace-wide `route_overrides` by risk class). Build the view, Allow once and Refuse now.
- **Words.** Until Always allow ships, the buttons keep today's words: web Approve / Deny, Go "Approve & run" / "Deny" (`src/chat/cards/ToolApprovalCard`). When it ships, every surface changes together to Allow once / Always allow / Refuse. "Allow once" only reads right beside "Always allow".
- **Buttons.** Web: Refuse · Always allow · Allow once, Allow once primary; keys `1` `2` `3` work while the card has focus (the HUD relies on them, and a theme can't add keys to a protected surface). **Phone:** the card in the thread stays as the record (D30), and tapping it, or the push, opens a bottom sheet where the person answers. HUD draws no thread cards (its transcript shows only the newest turn as text), so on HUD the sheet is the only surface. In the sheet, Allow once is full width and primary and Refuse is secondary below it. Always allow is a text link that opens a confirm sheet naming the rule it will create. Three equal buttons with the permanent one in the middle is a mis-tap risk at 390 pt, and the handoff's own Play mock already wraps "Allow once" onto two lines at about 340 px.
- **Voice** (§6.12.6): Refuse by voice always; Allow once by voice only on the web and only for verbs other than send, pay, sign, delete and publish; Always allow never by voice. On the phone every approve goes through Face ID (`automatos-go:src/step-up/policy.ts`), spoken or tapped.

#### 6.11.5 `Calibration@1`: day one

On a person's first chat in a workspace, Auto says back what it thinks it knows, from the titles and kinds of their recent Deliverables, Conversations and Knowledge (never file contents), and offers three things it can do from the installed skills. The person can accept all or reject any line; acceptance writes memories and a rejection writes a correction, through the existing memory service.

```ts
type Calibration = {
  v: 1; id: string
  lines: Array<{ id: string; text: string; sources: Array<{ type: 'deliverable' | 'conversation' | 'document' | 'skill'; id: string; label: string }> }>
  offers: string[]               // 3 things Auto can do, from installed skills
  empty: boolean                 // nothing to read yet: say so and offer 3 starters
}
```

- **No confidence percentages.** The handoff's HUD mock shows "85%" and "70%"; those are numbers the model makes up, shown as if measured. Show the source count ("from 6 Deliverables") instead.
- OpenDuet's version is a free-text answer with two chips [O, `shared/first-day.ts`]; the structured, line-by-line version is ours and net new.
- Offered once per person per workspace. Consent: §12-Q22. Routes: `POST /api/auto/calibration`, `POST …/{id}/lines/{line_id}/reject`, `POST …/{id}/accept`.

### 6.12 Surfaces: every theme, phone first

The handoff designed two web themes (Play Ember, HUD ARC) at desktop width. The product has more surfaces than that:
- **Web, built in:** Studio (the default, light and dark) and Classic.
- **Web, installed:** Play, HUD, and any third-party theme. A third-party theme draws the pages it owns; Studio draws the rest inside its frame.
- **Automatos Go:** three native layouts, Studio, Play and HUD. Classic and every installed theme fall to the Studio layout, painted by the theme's phone skin (24 colour tokens, fonts and mascot; no code, D58).

This section makes every proactive surface work on all of them from one contract. The phone layouts come first.

#### 6.12.1 Three layers

1. **Data (core):** the contracts in §6.11, including the server-written copy in the person's language.
2. **One pattern spec per component** (§6.12.3). It describes the parts, every state, the screen-reader label and motion, with colour named only by role (§6.12.2). It is written once and holds for every theme.
3. **Renderers:**
   - **Web:** core ships a default component for every pattern, styled by tokens so it reads right in Studio and Classic. A theme that owns the page places the default or draws its own from the SDK hooks. A theme that doesn't own the page gets Studio's.
   - **Go:** one data module per feature and one component per layout. The Studio component also carries Classic and every skin.

No new theme slots are needed. Play and HUD already own `/chat`, so they draw from the hooks; everything else falls to the defaults.

#### 6.12.2 Colour roles

Every theme already names its colours with Studio's token names: the web paints every Studio page under a look by them (PRD-260 D3), and Go's three token files fill the same `ColourName` set (`automatos-go:src/styles/tokens/types.ts`). So a status needs **one** role-to-token mapping, and each theme's token file decides the colour. This extends Go's existing mapping (`src/ui/variants.ts` `PIP_TOKENS`, which follows the web's `STATUS_CONFIG`). Theme palette names (gold, mint, molten) never appear in a component.

| Role | Token | Studio light / dark (default look) | Classic | Play (what Ember resolves to) | HUD (any Look) |
|---|---|---|---|---|---|
| ◆ needs_you | `warning` on dark grounds; on light grounds glyph **and** word in `warn-ink` (`warning` is 1.58:1 on Studio light and Play Daylight, so it fails as text and as a 3:1 mark) | warn-ink / sand | warn-ink / amber | warn-ink or warning by ground tone | `--hud-accent-text`, plus the one live border |
| ● working | `running`: a **new real token** (light hsl(40 20% 36%), 5.20:1; dark hsl(40 20% 65%)); today it is only a literal at `globals.css:782` and Tailwind has no such colour | warm grey | `info` until Classic gains `running` | `running` by ground tone | `--hud-accent-text` |
| ✓ ready | `success`, never on a tinted fill (4.38:1 on `secondary`) | olive | green | `success` by ground tone | `--hud-accent-text` |
| ✓ done | `success` | olive | green | `success` by ground tone | text-2 |
| ✕ failed | `destructive` | burnt orange | red | `destructive` | `--hud-error-text` (never themeable) |
| ◆ needs_setup | `info` | navy / steel | blue | `info` by ground tone | `--hud-accent-text`: ◆ is the "a person must act" glyph, painted in the accent role in `status-glyphs.ts` (the handoff's gold and an earlier text-2 are dropped) |
| ○ not_checked | `muted-foreground` | faded ink | grey | `muted-foreground` | text-2 |

- **No status ever goes through `--accent` on the web.** Under every Play look `--accent` *is* the error colour (`packages/look-kit/src/studio-tokens.ts:192`). Status uses the status tokens only.
- **Play paints from the look.** `lookToStudioTokens` emits `--destructive`, `--muted-foreground` and `--accent`, but not `success`, `warning`, `info` or `warn-ink` (`studio-tokens.ts:26-28`), so Studio's light or dark values apply by tone. It gains those, plus `running`, **per ground tone**. That fixes Daylight and the phone-skin parity at once (US-T01). Contrast is checked on **every accent × ground combination** the look catalogue allows (Paper included), not only the five presets. On Ember, Obsidian, Aurora and Workshop every role already passes (6.1–11.2:1, PLAY's measurement on 28 Oct).
- **HUD keeps its own rule.** Every status uses `--hud-accent-text` except failed (`--hud-error-text`) and quiet states (text-2), as `panels.module.css:46-52` does, and Go's `hud.ts` maps the status tokens to the accent. Accent against error is only 1.18:1 on FORGE and 1.03:1 on NOVA, so the glyph and the WORD always carry the meaning. HUD gains a **one-live-border rule**: today the live border is per panel (`panel.tsx:31`), so NEEDS YOU and MISSION can both glow. Only the item that needs the person glows.
- **Contrast.** Every role's text pair is measured on `background` and `card` in both tones: core's contrast tests on the web, Go's token tests, the kit's `check`.
- **Phone skins** (US-T01). A skin today sets 24 tokens, and none is a status colour. Under a dark skin the Studio layout keeps Studio's olive, sand and navy, which nothing measures; Go's status words read `--olive`, `--warn-ink` and `--navy`, at 2.43, 3.00 and 1.71:1 on a dark skin. The fix runs through the web first, because every skin token must be one the web paints (`mobile_tokens.py:11-14`), and is specified in US-T01. It matters only for third-party skins: Go reads neither Play's nor HUD's mobile block.

#### 6.12.3 The pattern spec

Each component is specified once. Renderers may change density, type and placement; they may not drop a part or a state.

| Component | Parts | States | Screen reader |
|---|---|---|---|
| **Brief card** | date, greeting, sections (§6.11.2) with title + why rows, the row action as text ("Start →"), "Show N more", Listen, thumbs | loading (the text part shows; no card skeleton); full; empty sentence; folded; a row pressed → "✓ {done_label}" in place; offline (cached copy, actions disabled with a line) | the row's action names the row: "Start: Lisbon flights" |
| **Suggestion** (card + pinned line) | "Auto suggests" label, the text in Auto's voice, "Start does: …", estimated credits when known, Start, Hide (reason menu) | open; starting (Face ID on the phone); started → "Started. Auto is on it." for 4 s; hidden; refused (spend ceiling, offline: one line, stays put); expired; keyboard up → one line | Start and Hide are actions on the card and on the pinned line |
| **Status row** | title, glyph + word, `▸ next_step`, progress rail while working, Try again when `retry` | needs_you (sorted first), working, ready, done, failed; unknown draws nothing | "Needs you: Ask Aisha to split the Airbnb. Allow me to send the message, or refuse." |
| **Approval view** (host-owned) | asked by, action, target, exact preview, rule sentence, expiry, Refuse / Allow once / Always allow | waiting (expiry counts down); answering (Face ID on the phone); answered (who, how); expired | the rule is read before the buttons |
| **Readiness list** | label, detail, glyph + word, the fix, Check again | checking; each item ready, not checked or needs setup; rate-limited "Checked a moment ago" | "Calendar: needs setup. Connect calendar." |
| **Calibration** | up to 3 lines, each with its source count and Reject; 3 offers; Sounds right / Something's off | full; empty (says so, offers 3 starters); a line rejected → struck through with "I'll forget that"; accepted | each line's Reject names the line |
| **Reply chips** | 2–3 answers ≤ 24 characters | shown under Auto's last message only; tapped → sent as the person's message; gone once a newer message exists | "Reply: Window" |

Motion everywhere: one entrance per component (fade plus 4 px rise, 180 ms); the in-place swap on Start; nothing loops; under reduced motion, a cross-fade only. The mascot reacts to the turn, never to old rows: `hello` for the first brief of the day, `listening` while an approval waits, `idle` otherwise.

#### 6.12.4 Web placements

| Surface | Studio and Classic (core) | Play (kit, owns `/chat`) | HUD (kit, owns `/chat`) | Other installed themes |
|---|---|---|---|---|
| Brief | the `brief@1` part draws a card in the Auto thread (`frontend/components/chatbot/message.tsx:112` maps parts and drops unknown types today, so it gains `brief@1`); `AutosRead` on the Command Center summary gains the structured rows under the prose (both shells already mount it: Studio `summary-tab.tsx`, Classic `activity-page.tsx`) | on an empty chat, in `FirstOpenHero`'s place (`play-chat-view.tsx:173-179`, where the hello already plays); in a conversation, the `brief@1` message in the `MessageList`. A card fixed at the head of a conversation would scroll away. | **three columns** (≥ 1180 px of screen, beside the 232 px rail): the right column becomes BRIEFING (numbered rows, built in time with speech), folding back into TODAY and NEEDS YOU after 60 s or any press. There is no SPEND panel; TODAY stands in for it (`today-panel.tsx:7`). **Two columns:** the same, in the column TODAY occupies. **One column and the phone layout** (< 1024 px viewport): the transcript slot, as on Go | Studio's card in the theme's frame |
| Suggestion | one card in the stack directly above `MultimodalInput` (`frontend/components/chatbot/chat.tsx:1363`, where `ToolSuggestionBar` and the Mission-mode banner already sit), plus the Command Center's Suggested list beside Auto's Read | between `PlayTrialBanner` and `MessageBox` (`play-chat-view.tsx:186-187`). One suggestion surface: while a turn's own Mission suggestion (`PlayMissionSuggestion` in `TurnEnd`, `turn-end.tsx:68`) is unanswered, the pinned card hides. The same rule applies to Studio's `MissionSuggestionCard`. | desktop: a `▸ NEXT` line in `chat-footer.tsx:42`, between `TrialBanner` and `VoiceDeck` (waveform + VoiceButton), above the composer; phone layout: in `phone/status-line.tsx`, never above the controls (Go's 375 × 667 finding). NEXT UP keeps the schedule | Studio's card |
| Status rows | needs-you widget, Missions list, board cards: glyph + word + next step | the Missions page is Studio's inside Play's frame (Play owns only `/chat` and `/deliverables`) | a **new MISSIONS panel**: the existing MISSION panel shows one mission's steps (`mission-panel.tsx:24-53`), not a list with a state word and a next step | Studio's |
| Approval | host card, unchanged placement; gains target, preview, rule and keys | inline in Auto's reply, as today (`auto-reply.tsx:115`, `approval-cards.tsx:56`; loose cards in `TurnEnd`, `:229`); the mascot's `listening` while a card waits already exists (`caption.ts:38`). **No web sheet:** an overlay over a protected surface is what theme review flags. The sheet is Go-only. | inline where HUD places host cards today; no overlay | host card |
| Readiness | Settings → Orchestrator (`frontend/components/settings/SettingsPanel.tsx`), a "What Auto depends on" section; `/settings` is host-owned, so it looks the same under every theme. Plus the brief's "Couldn't check" rows | the Settings section, plus a "What I depend on" card on Chat only when an item the brief needs isn't ready | a SYSTEMS panel from `useReadiness`; there is no THIS MACHINE panel to replace | Settings |
| Calibration | first chat: three Auto messages + Sounds right / Something's off | Ember bubbles, mascot `hello` | a CALIBRATION panel | Studio's |
| Reply chips | under Auto's last message | 40 px pills | `SAY "WINDOW"` chips, also tappable | Studio's |

Studio rules from `frontend/DESIGN_SYSTEM.md` §22 apply to every new web surface:
- flat paper, 1 px border, radius 6–10 px, no glass or glow;
- the primary button is ink, not orange, and there is one per view;
- Newsreader serif only for the brief's greeting;
- labels use `studio-eyebrow` and statuses `studio-pill`.

Classic renders the same components through its tokens (rounded, glass) with no fork. New components use `rounded-xl`/`rounded-2xl`, which Studio's radius override already maps; not `rounded-3xl`, which it doesn't.

#### 6.12.5 Automatos Go placements

Agreed with the MOBILE session on 28 Oct, read on main `289a8e9`.

| Piece | Studio | Play | HUD |
|---|---|---|---|
| Brief | a card at the top of the Auto thread (`src/chat/components/MessageList.tsx`, `src/chat/cards/ThreadCards.tsx`), styled as Auto's box (D72). The needs strip shrinks to its count while the card is on screen. | a card at the top of `PlayThread` (Play has no needs strip; `PlayHeader` keeps the count) | HUD has no thread. When the brief is the newest message it fills the transcript slot, and "OPEN BRIEF" opens a `Sheet` with numbered rows |
| Suggestion | a card directly above the composer (`src/chat/composer/Composer.tsx` in `ChatFrame.tsx`); with the keyboard up, one line; a count in the header opens a sheet of all of them | above `PlayComposer`; the count in `PlayHeader` | a "NEXT ▸ …" line in `HudTopBar`'s status row, never above TYPE · HOLD TO TALK (no room on a 375 × 667 phone after #20 and #26); "IDEAS · N" opens a `Sheet` |
| Approval | the card in the thread stays as the record (D30); the card or the push opens a bottom sheet to answer | same, in `PlayThread` | the sheet only |
| Status | `statusGlyph()` (`src/styles/dress/status-glyphs.ts`) + `PIP_TOKENS`, on Command Centre board rows, tickets and Questions; new statuses are added there | same, dressed | `glyphLine()` in capitals |
| Readiness | Settings, a "What Auto depends on" row beside Notifications, Voice and Privacy & legal. Push, microphone, speech and server voice states come from the app (`src/push/registration/state.ts`, the voice policy, discovery `tts`) | same | same |
| Calibration | first chat: three Auto boxes + Sounds right / Something's off | Play bubbles, mascot `hello` | a CALIBRATION sheet |
| Reply chips | under the last Auto message (revises D29; answers to Auto's own question only, §12-Q4) | same | in the transcript slot, tappable |
| Push | category `automatos.suggestion` with Start… (authentication required, opens the app) and Open, registered at runtime (JS, ships over the air); sent only to apps that report a new discovery capability `suggestion_push` | | |

Phone rules:
- **Build on the existing primitives.** New cards use `MissionFrame`'s head, body and foot (`src/chat/cards/MissionFrame.tsx`). The Settings section is one `Card` of `ListRow`s. Both then dress themselves per layout: Studio 8–10 pt corners, Play 14 pt on the ground's surface, HUD square bracketed panels. The approval sheet follows `src/cc/social/SocialSheet.tsx`.
- **No cost or credit language on the phone** (D8, D-NO-BILLING). The "about 40 credits" line on a suggestion (§6.8) shows on the web only.
- **Suggestions come from the server endpoint**, never from thread cards; cards aren't persisted and would vanish on reload.
- **VoiceOver:** Start and Hide are `accessibilityActions` on the card and the pinned line. The Face ID step is announced.
- **Dynamic Type:** the pinned line wraps to two lines, then truncates. Cards grow; nothing is clipped.
- **Offline:** Start refuses with one line; nothing is queued.
- **Quiet hours:** the suggestion push is never urgent.
- **iPhone only** (`app.json`): no iPad layouts.
- **Listen** uses `POST /api/voice/speech` in the person's language voice (#25/#26).
- **Native vs over the air:** everything here ships over the air on build 13. Home-screen widgets and Live Activities need a native extension and wait until after WebSummit. A background Hide from the lock screen works with installed modules but must be proven on a device first.
- **Before the fourth push type:** fix the open bug where tapping a push doesn't open its ticket on Gerard's phone.

#### 6.12.6 Voice

- Auto speaks titles only; the screen shows the why. Suggestions are never spoken.
- Lists Auto reads are numbered, so "start two" is a sentence a person can say. A spoken "start two" or "open one" is the person's own request in their own turn. Auto resolves the number against the `get_brief` result it just read and calls the Start or open route.
- Approvals by voice follow §6.11.4: Refuse always; Allow once on the web only and never for send, pay, sign, delete or publish; Always allow never. On the phone, every approval goes through Face ID.
- No surface speaks first. The brief is spoken only after a press or "brief me" (§12-Q12). A theme never captures or plays audio itself; the host voice session does (`areas/voice.d.ts`).

---

## 7. User stories

Every story: tests go in with the story; CI is the gate and nothing runs on a maintainer's machine; route manifest regenerated when routes change; new settings in `config-surface.json`; functions and components within the AGENTS.md limits; commits signed off. Waves are one branch, one push, one PR and one CI run per repo.

### Wave W1: Check-ins that suggest (core)

#### US-C01: Suggestion and run tables
**Description:** As the platform, I need one record for every suggestion and every proactive run, so delivery, decisions and cost are counted in one place.

**Acceptance Criteria:**
- [ ] Alembic revision on the current head adds `proactive_suggestions` and `proactive_runs` as in §6.2.4, with a partial unique index on (`workspace_id`, `user_id`, `subject_key`) where `status = 'open'`
- [ ] Both head pins move (`test_prd209_alembic_single_head.py`, `test_prd236_w1_routes.py`)
- [ ] Models in `core/models/proactive.py`; workspace delete cascades

#### US-C02: The read-only lane
**Description:** As an owner, I need background checks that cannot change anything, so a check-in can never send, edit or delete.

**Acceptance Criteria:**
- [ ] `ContextMode.CHECK_IN` loads only `read` platform actions plus `platform_suggest_next_step`; no Composio, plugin, `platform_execute`, sub-agent or Mission tools (test lists the loaded names)
- [ ] `PlatformActionExecutor.execute` refuses any non-`read` or unknown action when `caller_context.lane == "check_in"`, with the refusal text in §6.2.3 (tests for write, destructive, unknown, Composio)
- [ ] A registry sweep test asserts each `read` action performs no insert, update, delete or outbound HTTP against a spy session and transport; failures name the action
- [ ] `ActionDefinition.permission_level` has no default; registering an action without one fails at import (test)

#### US-C03: Auto's workspace check-in
**Description:** As an owner, I want Auto to look at what changed and tell me only what is new, so I am not told the same thing twice.

**Acceptance Criteria:**
- [ ] The signal diff of §6.2.1 runs over [last successful run, now], else the last 24 hours
- [ ] No signal: a `silent` run row, zero `llm_usage` rows (test)
- [ ] Signals: one model call in the `CHECK_IN` lane, at most 3 suggestions, dedupe per §6.2.4 (open, and hidden within 14 days)
- [ ] Respects the platform, workspace and person switches, the daily spend ceiling and `background_allowed`; each skip writes a run row with its reason

#### US-C04: Agent heartbeats become check-ins
**Description:** As an owner, I want my agents' scheduled checks to suggest, not to file tickets.

**Acceptance Criteria:**
- [ ] `heartbeat.mode` (`check_in` | `act`) is read by `_agent_tick`; `check_in` runs in the `CHECK_IN` lane and records suggestions
- [ ] `report_to=auto` no longer creates a `BoardTask` in either mode (test on `_route_heartbeat_to_auto`'s replacement)
- [ ] `heartbeat_complete` bell rows are written only when a run made a suggestion or errored for the first time in a streak
- [ ] The per-tick report is filed only when the run made a suggestion or changed state (closes the "failed tick files a report" item noted 16 Sep)
- [ ] Migration per §12-Q1; the agent page shows the mode

#### US-C05: Failing streaks
**Description:** As an owner, I want one card for an agent that keeps failing, not one per tick.

**Acceptance Criteria:**
- [ ] An erroring tick makes no model call; the first failure of a (agent, error class) streak creates one suggestion, later ones update `occurrences` and `last_seen_at` (test: 12 failing ticks → 1 suggestion, 1 push at most)
- [ ] After 3 consecutive failures the interval doubles per run up to 24 hours; the first success resets it and expires the suggestion with a "works again" line
- [ ] Replays 26 Oct's VECTOR case from fixtures: 0 board tickets

#### US-C06: Start and Hide
**Description:** As a person, I want to start or hide a suggestion and see what Start did.

**Acceptance Criteria:**
- [ ] `GET /api/proactive/suggestions`, `POST …/{id}/start`, `POST …/{id}/hide` as in §6.2.5; owners, admins and editors may start, the addressee may hide; viewers neither (the PRD-266 [D16] rule)
- [ ] Concurrent Starts: exactly one creates work (test with two sessions)
- [ ] Each `proposed_action.kind` creates its item with `source_type = 'suggestion'`; a board ticket defaults to `review_mode = 'human'`; each started item gets a watch
- [ ] Start passes the daily spend ceiling check; a refusal leaves the suggestion `open` and says why

#### US-C07: Kill switches, caps, quiet hours, audit
**Description:** As an owner, I need to bound what proactive work costs and when it reaches me.

**Acceptance Criteria:**
- [ ] The three switches of §6.9 are checked at every run start and fail closed (test: a settings read error skips the run)
- [ ] Every cap in §6.9 is enforced and each refusal is a run row (`skipped_cap`)
- [ ] Per-person quiet hours are stored, read with the workspace window, and hold pushes (tests across midnight and DST)
- [ ] Start, Hide and every switch change write workspace audit rows

### Wave W2: Delivery and the brief (core, web, Go)

#### US-C08: Suggestions in the Auto thread
**Acceptance Criteria:**
- [ ] A delivered suggestion is posted through `deliver_background_message` with `source.suggestion_id` and origin `suggestion`; it raises no `auto_message`
- [ ] A `suggestion_changed` event rides the existing live lane on create, start, hide and expire

#### US-C09: The `suggestion` push type
**Acceptance Criteria:**
- [ ] `PUSH_TYPE_EVENTS` gains `suggestion: ("suggestion_ready",)`, channel `suggestions`, generic line "Auto has a suggestion"; `suggestion_ready` is in `VALID_EVENT_TYPES` with an in-app default and a bell link
- [ ] Title carried only when `settings.push.suggestion_titles` is true (signed field set unchanged in order; the phone's signature vectors fixture is extended)
- [ ] Only `severity = today`, one per run, at most 2 per person per day, never in quiet hours (tests)
- [ ] The preferences GET lists the fourth switch; no row means on

#### US-C10: The brief on open
**Acceptance Criteria:**
- [ ] `POST /api/proactive/brief/open` claims (workspace, person, local date) in `mobile_digest_posts` shared with the scheduled digest; a second call from any device returns the same message (test: 3 devices, 2 tabs)
- [ ] A `local_date` more than one day from UTC is refused with 422
- [ ] The per-person block of §6.4 is deterministic (no second `llm_usage` row)
- [ ] Web-only people get a brief (test with no phone row)

#### US-W01: Suggested on the web
**Acceptance Criteria:**
- [ ] Command Center summary shows the open suggestions beside Auto's Read with Start, Hide (reason menu), evidence links, "Start does", estimated credits
- [ ] The Auto thread draws cards from `source.suggestion_id` with live state; the newest open one sits above the composer; state survives reload
- [ ] The bell links `suggestion_ready` to the card; the Board gains "From suggestions"

#### US-W02: Brief on open (web)
**Acceptance Criteria:**
- [ ] The app calls the brief route once per local date on load; the brief card shows thumbs up/down (posting to the existing feedback route) and Listen (server voice, falling back to the browser's speech on a refusal code)

#### US-G01: Suggested on the phone
**Acceptance Criteria:**
- [ ] (v2, §6.12.5) Studio and Play: the newest open suggestion as a card directly above the composer, one line while the keyboard is up. HUD: a "NEXT ▸ …" line in `HudTopBar`'s status row, never above TYPE · HOLD TO TALK. All three: a count in the header opens a sheet of every open suggestion. All hidden when there is none.
- [ ] VoiceOver: Start and Hide are `accessibilityActions` on the card and the line; Dynamic Type wraps the line to two lines, then truncates; offline Start refuses with one line; no credits shown (D8)
- [ ] Start asks Face ID through `src/step-up` (on by default, as approve); Hide never does
- [ ] The Auto-thread card is drawn from `source.suggestion_id`; the Board gains "From suggestions"
- [ ] One query, refetched on `suggestion_changed` and the 60-second poll

#### US-G02: The `suggestion` push on the phone
**Acceptance Criteria:**
- [ ] Channel `suggestions` and category `automatos.suggestion` with Start… (unlock required, opens the card, Face ID, then starts) and Open; tap opens the card
- [ ] The Notifications screen shows the fourth switch
- [ ] Signature vectors match core's extended fixture byte for byte

#### US-G03: Brief on open (phone)
**Acceptance Criteria:**
- [ ] On the first foreground of each local date the app calls the brief route and opens the Auto thread at the brief
- [ ] Listen plays server audio when build 13 is installed, else device speech; never auto-plays

### Wave W1 and W2 additions: display contracts (core, web, Go)

These stories carry §6.11 and §6.12. W1 holds US-C17 and US-C18; the rest ride W2.

#### US-C17: Status, next step and retry on every row
**Description:** As a person, I want every piece of work to say where it stands and what happens next, in the same words everywhere.

**Acceptance Criteria:**
- [ ] Needs-you rows, the Missions list and board task reads carry `display_state`, `next_step` (≤ 90 characters) and `retry` as in §6.11.1; a table test covers every board, run and grant status, and unknown gives `null`
- [ ] One strings table maps each `STUCK_*` constant (`services/needs_you.py:68-74`) to a sentence in English and Portuguese (Brazil); a missing key in either language fails a test
- [ ] Copy follows the person's language setting; the default is English

#### US-C18: Readiness
**Description:** As a person, I want to see what Auto depends on and what it couldn't check, so a quiet day never hides a broken connection.

**Acceptance Criteria:**
- [ ] `services/readiness/` has one probe per server item, each with a timeout setting; a timeout reads `not_checked` (test for each probe's three states)
- [ ] `GET /api/readiness` and `POST /api/readiness/check` (rate-limited, test); routes in the manifest; works in both editions
- [ ] `fix.href` is an app path only (test); SaaS `detail` never contains a hostname or latency (test)
- [ ] Probes reuse `api/system_health_checks.py` where a check exists

#### US-C19: Message parts and `Brief@1`
**Description:** As a person, I want the brief as a card I can act on, and as plain text on any client that can't draw it.

**Acceptance Criteria:**
- [ ] `post_background_message` accepts `parts`; the brief message carries `[text(prose), brief@1]` and its text equals today's prose (test)
- [ ] `Brief@1` per §6.11.2: `counts.needs_you == needs_you_counts()["total"]` (test); a viewer's brief has no question or approval rows (D16); fold at 5, never at 6 (tests for 6 and 7 rows); an unconnected source gives an `unchecked` row
- [ ] Tool `get_brief` (3-file pattern) returns the same object as the route (test)
- [ ] A failed build releases the day's claim, so the next open tries again; OpenDuet never releases it (test)
- [ ] The web thread (`frontend/components/chatbot/message.tsx`) draws `brief@1` and still drops unknown part types

#### US-C20: Approval view and rule sentences
**Description:** As a person, I want to see exactly what Auto will do, to whom, and which of my rules asked, before I allow it.

**Acceptance Criteria:**
- [ ] `ApprovalView@1` is built from the grant: `target` and `preview` from the tool call's arguments through the existing redaction; `rule.sentence` from one strings table over posture × risk class and the Mission approval policy (tests for every pair)
- [ ] `services/board_approval.py` no longer writes the raw policy key into `reason`; `blocked-reason.ts` shows the sentence (test)
- [ ] `ToolApproval` and `MissionApproval` render target, preview and rule; keys `1`, `2`, `3` work only while the card has focus; buttons keep today's words until §12-Q17 ships Always allow
- [ ] The cards stay protected surfaces; no theme CSS reaches them (the existing protected-surface test is extended)

#### US-C21: SDK 1.3.0 and the default components
**Description:** As a theme author, I want hooks and default components for every proactive surface, so my theme gets them without redrawing anything.

**Acceptance Criteria:**
- [ ] A new SDK area **`proactive`**. Not `brief`: that name is already a kit CLI command, `automatos-theme brief`. It holds `useBrief`, `useReadiness`, `useCheckReadiness`, `useSuggestions`, `useStartSuggestion`, `useHideSuggestion`, `useCalibration`, `useAnswerCalibration`; `pages/assignments` rows gain `display_state`, `next_step`, `retry`; `theme` re-exports `DISPLAY_STATE_LABELS` (glyph and word per state)
- [ ] Default components `BriefCard`, `SuggestionCard`, `StatusLabel`, `ReadinessList`, `CalibrationCard`, `ReplyChips` under `frontend/components/proactive/`, exported through `lib/theme-sdk/host-cards/` and the surface JSON (the recipe in `docs/theme-host/sdk-surface-map.md`); `surface.test.ts` pins the names
- [ ] Each default component renders in Studio light, Studio dark and Classic; browser-lane screenshots of each; axe reports no critical issue
- [ ] A new **host-owned voice control** in the `voice` area, `BriefVoice` (protected like `voice-control`), for Listen and the `▸ SPEAKING` sync. The 1.2.0 area (`VoiceButton`, `useVoice`, `readVoiceLevels`) has nothing that starts a session or reads the brief, and themes never speak.
- [ ] Every default component marks Auto's words with `data-ai-generated`: brief rows, suggestion text and chips
- [ ] `--running` becomes a real web token in `globals.css` and Tailwind (light hsl(40 20% 36%), dark hsl(40 20% 65%)), and `lookToStudioTokens` emits `success`, `warning`, `warn-ink`, `info` and `running` per ground tone (`packages/look-kit/src/studio-tokens.ts`, and core's copy)
- [ ] SDK and host contract versions bump to 1.3.0, and the d.ts files are regenerated for the kit to vendor (step 1 of Wave K's order)

#### US-W04: Status vocabulary on the web
**Acceptance Criteria:**
- [ ] The needs-you widget, the Missions list and board cards show glyph + word + `next_step` from the row and Try again when `retry`, in Studio (`studio-pill`) and Classic; needs-you rows sort first
- [ ] A grep test finds no status shown by colour alone in `components/activity` and `components/missions`
- [ ] The brief's needs-you count equals the badge on the same screen (browser lane)

#### US-W05: "What Auto depends on" (web)
**Acceptance Criteria:**
- [ ] Settings → Orchestrator gains the section from `useReadiness` plus the client's `microphone` and `speech` rows from `useVoice().available`; Check again calls the check route
- [ ] With the calendar not connected, the brief shows a "Couldn't check" calendar row only if §12-Q20 brings the calendar into v1

#### US-G05: Status vocabulary on the phone
**Acceptance Criteria:**
- [ ] The new states are added to `src/styles/dress/status-glyphs.ts` and `src/ui/variants.ts` `PIP_TOKENS` per §6.12.2; their tests read the full lists
- [ ] Command Centre board rows, tickets and Questions show glyph + word + `next_step`, and Try again when `retry`, in all three layouts; never colour alone (D-A11Y)

#### US-G06: The brief on the phone
**Acceptance Criteria:**
- [ ] `src/brief/` holds the zod `Brief@1`, routes (with `pendingUpstream` until the core pin includes them) and the hook (memory only, refetched on `board_changed`)
- [ ] `MessagePartSchema` gains `brief@1`. Studio and Play draw the card at the top of the thread, styled as Auto's box (D72). HUD fills its transcript slot and "OPEN BRIEF" opens a sheet with numbered rows.
- [ ] Studio's needs strip shrinks to its count while the card is on screen; the brief's count equals the strip and the app badge (test)
- [ ] Start and Answer route by `opens.type`; question rows open Command Centre → Questions
- [ ] Shown only when discovery lists `brief`; an older server shows today's app unchanged

#### US-G07: "What Auto depends on" on the phone
**Acceptance Criteria:**
- [ ] Settings gains one `Card` of `ListRow`s beside Notifications, Voice and Privacy & legal; `push`, `microphone`, `speech` and `server_voice` rows come from the app (`src/push/registration/state.ts`, the voice policy, discovery `tts`)
- [ ] Shown only when discovery lists `readiness`

#### US-G08: The approval sheet
**Acceptance Criteria:**
- [ ] The card in the thread stays as the record (D30); the card or the push opens a sheet on the `SocialSheet` pattern showing `ApprovalView`: asked by, action, target, preview and rule
- [ ] Allow once is full width and primary, Refuse secondary. Always allow appears only when `answers` includes it, as a link opening a confirm sheet that names the rule.
- [ ] Face ID before allow, tapped or spoken; never before Refuse (`src/step-up/policy.ts`)
- [ ] HUD uses the sheet only (it draws no thread cards)

#### US-T01: Status colours in phone skins
**Acceptance Criteria (gated on §12-Q25; reworked with PLAY, 28 Oct):**
- [ ] Web first. Every skin token is one the web paints (`mobile_tokens.py:11-14`), so `--running` is a real web token before it is a skin token (US-C21), and `lookToStudioTokens` emits the status tokens per ground tone.
- [ ] The mobile token list adds `--success`, `--warning`, `--warn-ink`, `--info` and `--running`. Go's status words also read `--olive` and `--navy`: either they join the list, or Go's words move to the status tokens (US-G05 decides, and records which). Core's parity tests change in the same PR: `mobile-tokens.test.ts:59-63` (the list equals what `lookToStudioTokens` writes), `:75` (the count) and `look-kit.test.ts:55`.
- [ ] Fallbacks are chosen by the **ground tone** the skin paints, not by the scheme's name. Today a "light" scheme on a dark ground gets Studio Light's navy (1.71:1) and olive (3.59:1); the fix covers Go's `paint.ts` and the kit's `phone-paint.ts:9-11`. A fallback in use is measured, never reported as "not measured" (`check/mobile-contrast.ts:99-103`, `mobile_contrast.py:105`).
- [ ] New contrast pairs: each status token on `--background` and `--card` (3:1 for a mark, 4.5:1 where it is text), plus `--accent` on `--background`, which PLAY flagged on 24 Oct. They go in core's `mobile_contrast.py`, the kit's `check/mobile-contrast.ts` and Go's `paint.ts`.
- [ ] Older hosts refuse unknown token names (`mobile-manifest.ts:94`), so each new token carries a minimum host version, and `THEME_HOST_VERSION` bumps (`contract.py:33`)
- [ ] The kit's harness: the skin designer, contrast panel, export and `automatos-theme skin` follow once re-vendored. The Studio phone preview gains the new states (`src/phone/parts.ts` `PIP_COLOURS` and `TEXT_TONES`; `app/phone/studio/copy.ts` fixtures), mirroring US-G05.

### Wave K: the kit themes (automatos-design-kit, after SDK 1.3.0 is vendored)

Order (agreed with PLAY, 28 Oct). Each step waits for the one before it:
1. **Core** (US-C21): `lib/theme-sdk/proactive.ts`, `host-cards/*`, `surface/*.json`, `packages/theme-sdk`, the shared-modules `sdkVersion`, `services/themes/contract.py:34`, `data/sdk-surface.json`, `docs/theme-host/sdk-surface-map.md`.
2. **Kit vendor PR:** `packages/theme-sdk`, `theme-cli/src/core` and its sha256 rows, `CORE_SHA` (`check/report.ts:29`), and the harness fakes **in the same PR**: `src/sdk/proactive.ts`, `index.ts`, `internal/page-views.ts:38`, `host/sample-data.ts:46`. They can't land earlier: `surface.test.ts:57` requires the fakes to equal the vendored d.ts, `check` errors on a name the SDK doesn't export, and themes type-check against it.
3. **Play and HUD versions:**
   - `engines.sdk` `^1.3.0` and `proactive` added to `uses`;
   - the `package.json` peer, pinned exactly to 1.2.0 today;
   - HUD's `manifest.test.ts:47,210`, which hard-codes 1.2.0;
   - CHANGELOGs.

   Each version takes its own review and sign round.
4. The lockfile dispatch.
5. Core browser-e2e `KIT_REF` (`:221`) and `PLAY_PACKAGE` (`:224`), only after step 1 is on main, because `^1.3.0` refuses a 1.2.0 host. SaaS runs core with SDK 1.3.0 before either theme installs.

Vendor SDK 1.3.0 **before** the Kit 1.0 freeze (B-07b), or the freeze's surface fixture takes a minor bump.

One feature per PR. Never redraw `ToolApproval` or `MissionApproval`; place them. Never call speech APIs. Brief rows, suggestion text and chips are Auto's words: each carries `data-ai-generated` (pattern: Play `auto-reply.tsx:104`; the fence question is §12-Q29). Acceptance is **harness showcase shots** of the brief, suggestion and approval in both themes at 1440 and at 390, not hand mocks.

#### US-K01: Play
**Acceptance Criteria:**
- [ ] Placements per §6.12.4:
  - the brief in `FirstOpenHero`'s place on an empty chat, and as the `brief@1` message in a conversation;
  - the suggestion card between `PlayTrialBanner` and `MessageBox`, hidden while a turn's `PlayMissionSuggestion` is unanswered;
  - reply chips;
  - approvals inline as today (no sheet), with the mascot `listening` while one waits.
- [ ] Both layouts: at 1440 (sidebar, main column, mascot column) and below 1024 px (Play's phone layout; one breakpoint, `lib/play/phone-layout.ts:15`)
- [ ] Every colour comes from the active look through Studio's token names, and no status goes through `--accent`. Contrast passes on **every accent × ground combination** the look catalogue allows, Paper included (`automatos-theme check`).
- [ ] Harness showcase shots of the brief, suggestion and approval at 1440 and 390; Auto's words carry `data-ai-generated`
- [ ] Calibration bubbles with the mascot `hello` (with US-C22)

#### US-K02: HUD
**Acceptance Criteria:**
- [ ] BRIEFING:
  - at three columns (≥ 1180 px of screen) it takes the right column, numbered rows, `▸ SPEAKING` while the host voice session reads it (`BriefVoice`), and folds back into TODAY and NEEDS YOU after 60 s or any press;
  - at two columns (940–1179) it takes TODAY's column;
  - at one column (under 940) and on the phone layout it takes the transcript slot.
- [ ] `▸ NEXT` with START and HIDE: desktop in `chat-footer.tsx:42` between `TrialBanner` and `VoiceDeck`; phone in `phone/status-line.tsx`. SAY chips can be tapped. Never spoken.
- [ ] A new MISSIONS panel (the MISSION panel stays one mission's steps): rows with glyph + WORD + `▸ next_step` and a 3 px rail while working, in `--hud-accent-text` / `--hud-error-text`
- [ ] A SYSTEMS panel from `useReadiness`
- [ ] The one-live-border rule: only the item that needs the person glows (today `panel.tsx:31` lets NEEDS YOU and MISSION both glow)
- [ ] Harness showcase shots at 1440 and 390 on all five Looks (ARC, FORGE, GHOST, NOVA, SIGNAL) pass `automatos-theme check`; nothing moves under reduced motion; Auto's words carry `data-ai-generated`

### Wave W3 addition: calibration (core, web, Go)

#### US-C22: Calibration
**Acceptance Criteria (gated on §12-Q22):**
- [ ] `POST /api/auto/calibration` reads titles and kinds only (a test asserts no document body is read); returns `Calibration@1` with source counts and no confidence numbers
- [ ] Reject writes a memory correction and accept writes the lines, through the existing memory service; offered once per person per workspace (test); `empty: true` offers three starters

#### US-W06 / US-G09: Calibration on the web and the phone
**Acceptance Criteria:**
- [ ] On the first chat per person per workspace, when the `calibration` capability is present: three Auto messages, each line rejectable, then Sounds right / Something's off
- [ ] Go: accepting isn't gated by Face ID, because it writes memory, not an action; Studio boxes, Play bubbles with the mascot `hello`, HUD a sheet

### Wave W3: Follow-ups, reply chips, evals (core, web, Go)

#### US-C11: Bounded follow-ups
**Acceptance Criteria:**
- [ ] The three tools of §6.3; a follow-up on a Mission, playbook run or ticket is refused with "use a watch"
- [ ] Caps of §6.3 under an advisory lock (test: 6 parallel calls with a cap of 5 → 5 rows)
- [ ] A firing runs in the `CHECK_IN` lane and ends silent, posts one message, or records a suggestion; a follow-up turn cannot schedule another; a failed one is not retried
- [ ] Schedule, cancel, fire and outcome are audited

#### US-C12: Reply chips
**Acceptance Criteria:**
- [ ] `platform_offer_replies` is terminal: the turn ends with no further model call (test counts `llm_usage` rows)
- [ ] 2–3 replies, each ≤ 24 characters and 1–3 words, else the tool returns an error the model reads
- [ ] Chips are stored on the message and returned with history; not offered on the brief or in check-ins

#### US-W03 / US-G04: Reply chips in the thread
**Acceptance Criteria:**
- [ ] Chips render under the last assistant message only; a tap sends the text; chips hide once a newer message exists
- [ ] Go: all three styles; spoken turns show chips and never speak them

#### US-C13: Eval runners
**Acceptance Criteria:**
- [ ] Runners for the six evals of §6.7 in `orchestrator/evals/proactive/`, reading gold sets from a path in an environment variable; no gold data committed (a CI check refuses `*.jsonl` under that folder)
- [ ] Each prints its pass bar and exits non-zero on a miss; results are summarised as numbers in the PR that turns a feature on

### Wave W4: Beyond OpenDuet (core, web, Go)

#### US-C14: Mission and playbook proposals
**Acceptance Criteria:**
- [ ] A check-in may propose `mission` (goal and draft steps) or `playbook_run` (an existing playbook and its inputs); Start uses the existing create paths
- [ ] Missed or benched playbooks produce a `playbook_run` suggestion instead of a bell row

#### US-C15: Learning from Hide, cost on the card
**Acceptance Criteria:**
- [ ] The 80%-of-20 rule of §6.8 doubles that agent's interval and adds one line to the next brief
- [ ] `wrong_agent` and `not_useful` reasons are written to the agent's field memory
- [ ] `est_credits` is the median of the agent's last 10 comparable runs, or absent

#### US-C16: Telegram and Slack
**Acceptance Criteria:**
- [ ] A suggestion routed to Telegram or Slack arrives as one line with "Reply start or hide"; the reply starts or hides it through the same route and records `decided_via`

## 8. Functional requirements

- FR-1: A check-in must be able to call only `read` platform actions and `platform_suggest_next_step`; any other call must be refused in the executor.
- FR-2: A check-in must read only what changed since its last successful run, else the last 24 hours.
- FR-3: Auto's workspace check-in must make no model call when its signal diff is empty.
- FR-4: Heartbeats must not create board tickets.
- FR-5: A failing streak must produce at most one open suggestion per agent and error class.
- FR-6: A suggestion must not repeat an open one, or one hidden in the last 14 days.
- FR-7: Only one Start may create work for a suggestion; Start must pass the daily spend ceiling.
- FR-8: Started work must carry `source_type = 'suggestion'` and get a watch.
- FR-9: Follow-ups must respect the caps in §6.3 under a lock, and must never act.
- FR-10: Exactly one brief per person, workspace and local date must be posted, whichever device or schedule comes first.
- FR-11: The `suggestion` push must carry generic text unless the workspace allows titles, must never be sent in quiet hours, and must respect the per-person daily cap.
- FR-12: Reply chips must end the turn without another model call.
- FR-13: The platform, workspace and person switches must be checked at every run start and fail closed.
- FR-14: Every run, Start, Hide and follow-up event must be auditable.

## 9. Technical considerations and risks

- **Mislabelled read actions.** The lane is only as good as `permission_level`. The sweep test in US-C02 is the control; a failing action is relabelled, never excepted.
- **Prompt injection.** A check-in reads agent output and documents. Suggestions are structured, never auto-run, and the person sees what Start does. A started ticket's brief quotes evidence as data.
- **A second list people ignore.** If precision stays under 40%, suggestions become noise like the heartbeat tickets they replace. The caps, Hide learning and the triage bar are the controls; the live metric is reviewed at 2 and 4 weeks.
- **iOS background actions.** Hide without opening the app needs a background response handler; the app opens on every action today (`categories.ts:6`). Not in v1.
- **Two briefs.** The cadence Daily Brief and the digest overlap; retiring the cadence block (§12-Q11) removes one.
- **Existing acting heartbeats.** Agents with `auto_act = true` act today. Converting them silently would change a workspace's behaviour; §12-Q1 decides.
- **Schema churn.** Alembic heads move fast; re-read at rebase.
- **Board review load.** Human review on started tickets raises needs-you by design; if that becomes the new flood, §12-Q5 can switch the default.
- **Server voice availability.** The spoken brief uses the Kokoro route (live); the fallback is device speech, so nothing blocks on it.
- **Old phone builds.** Build 12 and older drop `brief@1` silently and show the prose, which is acceptable. The `suggestion` push goes only to apps that report the `suggestion_push` capability.
- **Theme drift.** Play and HUD draw their own versions from the hooks. The pattern spec (§6.12.3) and the kit's harness shots on every preset and Look are the control. A theme that drops a state fails review.

## 10. Rollout and sequencing (proposed; dates are Gerard's call)

| Step | What | Where |
|---|---|---|
| 0. Shadow (one week) | W1 merged with delivery off: check-ins record suggestions next to today's heartbeats on Gerard's workspace and c1; PRD-248 `report_triage` turned to shadow; compare tickets filed vs suggestions recorded; run the triage and silence evals | SaaS |
| 1. Live for Gerard | W2 on: web, Go (OTA for rows and push type; build 13 for server audio), `report_to=auto` tickets stop | SaaS, Gerard's workspace and c1 |
| 2. SaaS default | Workspace switch on for paid workspaces; never on trial (`background_allowed`) | SaaS |
| 3. Local edition | Same release; off by default until the operator switches it on (their own keys) (§12-Q10) | Local |
| K. Kit themes | Wave K (US-K01 Play, US-K02 HUD) once SDK 1.3.0 is vendored into the kit; releases through the usual tag and signed listing | Kit, then SaaS and local installs |
| W3, W4 | Follow-ups, chips, evals, calibration; then the differentiators | Both |

W1 and W2 are the minimum that closes the OpenDuet gap and stops the flood. Parallel lanes per wave: W1 L1 US-C01 → US-C03 → US-C05 · L2 US-C02 → US-C04 · L3 US-C06, US-C07. W2 L1 US-C08 → US-C09 → US-G02 · L2 US-C10 → US-W02, US-G03 · L3 US-W01, US-G01. v2 adds: W1 L4 US-C17 → US-C18 · W2 L4 US-C19 → US-C20 → US-C21 → US-W04, US-W05 · W2 L5 (Go) US-G05 → US-G06 → US-G07 → US-G08 · US-T01 when §12-Q25 is answered · Wave K after US-C21 merges.

## 11. Success metrics

- Board tickets created by heartbeats per day: **0** after step 1 (today: one per `report_to=auto` tick).
- Check-ins that end silent: **≥ 70%**; check-ins that reach the model: within the 48-per-day cap with headroom.
- Suggestion precision (started ÷ decided): **≥ 40%**; started and passed review: **≥ 30%** of decided.
- Pushes of type `suggestion`: **≤ 1 per person per day** on average; opt-out of the switch **< 10%** of people with the app.
- Brief: duplicates **0**; people who see the brief on a day they open Automatos **≥ 90%**.
- Proactive spend per workspace per day (check-ins, triage, follow-ups, chips): **≤ $0.25** on Haiku [I].
- Evals: every bar in §6.7 met before its feature goes live.

## 12. Open decisions for Gerard (each with a recommendation)

1. **Heartbeat conversion scope.** Convert every agent heartbeat to a read-only check-in, or only those with `report_to=auto`? *Recommendation:* all heartbeats default to `check_in`; agents whose config has `auto_act = true` today stay `act`, flagged on their agent page, and `act` becomes an owner-only switch.
2. **Who receives a suggestion.** The workspace owner, the person who set up the agent, or every editor? *Recommendation:* one addressee per suggestion in v1, the workspace owner (or the person whose chat scheduled the follow-up); team routing after WebSummit.
3. **Lock-screen Start.** Offer Start… on the lock screen (opens the app, Face ID, then starts), or Open only? Either way it extends [D18] and adds a fourth type to [D22]. *Recommendation:* Start… and Open in v1; Hide in the background only after a build proves it.
4. **Reply chips on the phone.** Revise [D29] for chips that answer Auto's own question? *Recommendation:* yes; no canned prompts, ever.
5. **What Start makes by default.** A board ticket with human review (OpenDuet's "Ready for review"), or auto review? *Recommendation:* human review for suggestion-started tickets.
6. **Brief.** Keep the scheduled digest push [D24] and add the on-open brief with one shared claim, so web-only people get one too? *Recommendation:* yes.
7. **Suggestion push text.** Title or generic? *Recommendation:* generic by default, a workspace switch for titles.
8. **Cap values.** Accept the defaults in §6.3 and §6.9? *Recommendation:* yes for the shadow week; revisit with its numbers.
9. **Follow-up chains.** May a follow-up's turn schedule another? *Recommendation:* no in v1.
10. **Local edition default.** On or off? *Recommendation:* off until the operator switches it on; on for paid SaaS workspaces, never on trial.
11. **Retire overlapping loops.** Fold the cadence Daily Brief and the orchestrator heartbeat's `silent`/`notify` modes into the check-in? *Recommendation:* yes; `act_notify` and `autonomous` stay as they are.
12. **Spoken brief.** Auto-play on HUD, or a Listen button everywhere? *Recommendation:* Listen button only.
13. **Cost on the card.** Show estimated credits on Start? *Recommendation:* yes, when there is history.
14. **PRD split.** Keep the app stories here, or move them to an automatos-go PRD-002 on approval? *Recommendation:* move on approval, as PRD-266 and PRD-001 were split.
15. **Live triage.** Allow PRD-248's `report_triage` to run live for the check-in lane once the triage eval passes (it is shadow-only by construction today)? *Recommendation:* yes, for this lane only.

**Added in v2, from the handoff and the cross-theme review:**

16. **One PRD.** This document merges the handoff, which proposed itself as PRD-268 (its Q7), into the 27 Oct draft. Its stories split per repo on approval, as Q14 says. *Recommendation:* confirm, owner Gerard.
17. **Always allow** (handoff Q1). It writes policy, and no per-tool or per-target allow exists today, only workspace-wide `route_overrides`. *Recommendation:*
    - scope it to the same tool and the same target ("send to Aisha on WhatsApp"), never the whole tool;
    - only owners and admins create such rules;
    - list them in Settings → Approvals with an undo;
    - never by voice;
    - on the phone, a confirm sheet that names the rule.

    OpenDuet scopes it to the whole app, and its AI reviewer still decides; we don't copy that.
18. **"Can wait"** (handoff Q2). (a) Deliverables the person hasn't opened, which needs a per-person seen flag (none exists); (b) finished since the last brief; (c) low-priority needs-you rows, which would split one number. *Recommendation:* (a), with (b) shipped first as the section "Since your last brief".
19. **Spaces** (handoff Q3). No precedent in core: `use-projects.ts` is mock data, and OpenDuet's version is called Channels. It is an organising concept, not a proactive one. *Recommendation:* take it out of PRD-268 into its own PRD that decides build, map or drop.
20. **Calendar in TODAY** (handoff Q4). §3 makes reading the person's calendar a non-goal, and no server calendar read exists. *Recommendation:* the activity schedule only in v1. A calendar comes later through an agent's read-only check-in, not a new client.
21. **Speaking first** (handoff Q5) is Q12. *Recommendation:* press-only everywhere.
22. **Calibration consent** (handoff Q6). It sends the titles of Deliverables, Conversations and documents to the model. *Recommendation:* one consent line the first time ("Auto reads the titles of your recent work to say what it knows"), and the Go privacy facts reviewed in the same PR.
23. **Navigation.** The handoff's Play tabs (Chat · Missions · Shelf) and HUD phone tabs (CORE · MISSIONS · VAULT · MORE) would redesign navigation. Go has no tab bar (D73 header and menu); web Play is Chats · Shelf; PRD-001 keeps HUD's desktop pages out of v1. *Recommendation:* keep today's navigation and place every feature in the existing frames. MOBILE agrees.
24. **Approvals by voice.** *Recommendation:*
    - Refuse by voice, always;
    - Allow once by voice on the web only, and never for send, pay, sign, delete or publish;
    - Always allow never by voice;
    - on the phone, Face ID for every approval.

    The handoff allowed "allow once" by voice for anything.
25. **Status colours in phone skins.** *Recommendation:* add `--success`, `--warning`, `--running` and `--info` as optional skin tokens with contrast pairs, plus the `--accent`-on-`--background` pair PLAY flagged on 24 Oct (US-T01). The alternative, deriving status from `--primary` and `--destructive`, loses the difference between needs-you and done.
26. **Calibration confidence.** *Recommendation:* show source counts ("from 6 Deliverables"), never the model's percentages.
27. **The push-tap bug before a fourth push type.** Tapping a push doesn't open its ticket on Gerard's phone. *Recommendation:* fix it before US-C09 and US-G02.
28. **Approval words.** *Recommendation:* keep Approve / Deny on the web and "Approve & run" / "Deny" on Go until Always allow ships, then change to Allow once / Always allow / Refuse everywhere at once.
29. **The AI-label fence and proactive controls.** `AiLabelFence` wraps only the composer. Brief rows, suggestion text and chips carry `data-ai-generated`. Should Start and the chips sit inside the fence, or carry their own small AI caption? *Recommendation:* each carries `data-ai-generated`, with no new caption. The brief card and the suggestion card each show the existing tiny "Auto is an AI" caption once (AI-label minimal rule), and Start and the chips sit outside the fence.

## Related

- PRD-204/205 (watcher, ChatMessenger), PRD-221 S9/S10 (Auto's Read and feedback), PRD-225 (questions), PRD-248 (decision engine), PRD-252 R5 (needs-you), PRD-265 (credits), PRD-266 (push, digest), automatos-go PRD-001.
- OpenDuet: `openbot/server/src/proactive/`, `openbot/server/src/agents/wake-up.ts`, `openbot/server/src/channels/`, `evals/{triage,reply-chips,voice}/`.
- `<workspace>/.claude/BUGS-FOR-FIXER-2026-10-26.md` §2 (the VECTOR flood).
- The Claude Design handoff: `<workspace>/DUMPING AREA/design_handoff_proactive_auto/` (CONTRACTS, CORE, WEB, GO, OPEN-QUESTIONS, and HTML references for Play and HUD). Its token tables for Play are superseded by §6.12.2, and its glyphs by §6.11.1.
- Cross-theme mocks of every surface: `<workspace>/reports/proactive-2026-10-27/design/`.
