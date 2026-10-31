# The brief, status and readiness

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/api/proactive_brief.py](../../../orchestrator/api/proactive_brief.py)
- [orchestrator/services/brief/on_open.py](../../../orchestrator/services/brief/on_open.py)
- [orchestrator/services/brief/claim.py](../../../orchestrator/services/brief/claim.py)
- [orchestrator/services/brief/build.py](../../../orchestrator/services/brief/build.py)
- [orchestrator/services/brief/model.py](../../../orchestrator/services/brief/model.py)
- [orchestrator/services/brief/rows.py](../../../orchestrator/services/brief/rows.py)
- [orchestrator/services/mobile/digest.py](../../../orchestrator/services/mobile/digest.py)
- [orchestrator/services/digest_feedback.py](../../../orchestrator/services/digest_feedback.py)
- [orchestrator/modules/tools/discovery/handlers_brief.py](../../../orchestrator/modules/tools/discovery/handlers_brief.py)
- [orchestrator/services/proactive/display_state.py](../../../orchestrator/services/proactive/display_state.py)
- [orchestrator/services/readiness/service.py](../../../orchestrator/services/readiness/service.py)
- [orchestrator/services/readiness/probes.py](../../../orchestrator/services/readiness/probes.py)
- [orchestrator/services/readiness/items.py](../../../orchestrator/services/readiness/items.py)
- [orchestrator/api/readiness.py](../../../orchestrator/api/readiness.py)
- [frontend/components/proactive/brief-on-open.tsx](../../../frontend/components/proactive/brief-on-open.tsx)
- [frontend/hooks/use-brief.ts](../../../frontend/hooks/use-brief.ts)

</details>

## The brief

One brief per person, workspace and local date, posted into the person's Auto thread. Two paths make it, and they
share one claim:

| Path | Trigger | Gate | Announced as `auto_message` |
|---|---|---|---|
| The scheduled digest (PRD-266) | `run_digests` on the leader's scheduler, at the time of the person's most recently seen phone with a timezone | The person's `daily_digest` push switch; not the proactive mode | Yes |
| The brief on open | `POST /api/proactive/brief/open {local_date, timezone, locale?}`, sent once per local date by `BriefOnOpen` (`frontend/components/providers.tsx`) and by the phone on foreground | The effective mode is shadow or live; off is 404 `brief_off` before any claim | No: opening the app is the person's own action |

`post_now` ("Send me today's brief now", `POST /api/v1/mobile/digest/now`) posts the same brief with no claim, as it
did before.

### The claim is a lease

The claim is the `(workspace, person, local date)` primary key of `mobile_digest_posts`, unchanged by PRD-268
(`services/brief/claim.py`, `services/mobile/digest.py`):

1. **Claim.** `INSERT … ON CONFLICT DO NOTHING`, committed. The winner builds and posts.
2. **Taken.** A loser looks for the day's message in the person's own Auto thread, by its source fields (`origin
   'daily_digest'`, `local_date`). Found: `{chat_id, message_id, posted: false}`. Not yet: `{posted: false, pending:
   true}`.
3. **Stale.** A claim whose `posted_at` is older than `BRIEF_CLAIM_STALE_SECONDS` (default 600, at least 60) with no
   day's message was left by a claimant that died. The next open or tick takes it over with one compare-and-set
   (`digest.take_over`) and builds as the winner.
4. **Still held.** A winner checks `still_held` just before posting. One whose day was taken over posts nothing, so
   a slow build costs a second build, never a second brief.
5. **Release.** A build or post that fails gives back the claim this caller holds (never a taker's), so the next open
   or tick tries again. The route answers 503 `open_unavailable`.

Claims older than `DIGEST_POSTS_KEEP_DAYS` are deleted by the digest job. The 600-second default outlasts the
OpenAI-compatible clients' 180 s × 3 tries; see the comment on `BRIEF_CLAIM_STALE_SECONDS` in
`orchestrator/config.py` for the Anthropic client's case.

`local_date` more than one day from the server's UTC date is 422 `open_out_of_window`. `timezone` must be an IANA
name. The language is `locale` (a BCP 47 tag), else `Accept-Language`, else English; strings exist in English and
Portuguese (Brazil).

### The message

`[text(prose), brief@1]` through `deliver_background_message`, source `{origin: 'daily_digest', local_date, via:
'schedule' | 'open'}`. Clients that can't draw `brief@1` show the prose. The cadence "Daily Brief" prompt block is
retired (`core/services/auto_cadence.py`): a stored `daily_brief` cadence adds nothing.

### `Brief@1`

Built by `build_brief` (`services/brief/build.py`, model in `services/brief/model.py`). It writes nothing.

| Field | What |
|---|---|
| `v`, `date`, `generated_at` | 1, the person's local date, now |
| `greeting` | By the local hour in the person's zone, with their first name when known |
| `prose` | Auto's Read (`services/digest_service.generate_digest`, cached per workspace state): the brief's only model call, and none on a cache hit |
| `sections` | `needs_you`, `suggestions`, `today`, `since`, `unchecked`, each a list of `BriefRow` |
| `counts` | Exact counts for `needs_you`, `suggestions`, `today`, `since`, even when rows fold |
| `labels`, `copy` | Section labels, "Show {n} more", the empty sentence: the server's words |
| `spoken` | The greeting and the numbered row titles through `speechify`, never a row's `why`; what Listen and `platform_get_brief` speak |
| `state_hash` | Auto's Read's hash: what the thumbs rate |

`BriefRow`: `id`, `title`, `why`, optional `at`, `number`, `display_state`, `opens {type, id}`, `action {verb, label,
done_label}`.

| Section | Reader | Rule |
|---|---|---|
| `needs_you` | `services/brief/needs.py` | `needs_you()`'s rows with display state and next step. Its count is `needs_you()`'s total, the badge's. A person who can't answer (a viewer) gets no question or approval rows, and no Mission awaiting plan approval; the count stays the badge's. |
| `suggestions` | `services/brief/suggested.py` | Up to 3 open suggestions addressed to the person, `today` first then newest; past `expires_at` left out without writing. |
| `today` | `services/brief/today.py` | `ActivityService.get_schedule` for one day, kept to the local date. No calendar. |
| `since` | `services/brief/since.py` | Since the person's previous claim: started suggestions and what their ticket became, tickets Done or Closed, Missions completed. |
| `unchecked` | `services/brief/unchecked.py` | Readiness items needed by the brief that aren't `ready`. Never hidden. |

Each section folds at 5 and never at 6 (`services/brief/rows.py`). Row text is workspace data, quoted.

### `platform_get_brief`

A `read` action (`actions_brief.py`, `handlers_brief.py`) that answers "brief me", "what needs me?" and "what's on
today" from the same `build_brief`. The person, their super-admin flag and language come from the server's caller
context (`carries_the_brief_caller` strips a supplied `_brief_caller`). A call made for nobody (a ticket, a Mission,
a check-in, a widget visitor) is refused. It claims nothing and posts nothing. A turn that called it never ends on
reply chips.

### The thumbs

The brief's thumbs post `POST /api/activity/digest/feedback {state_hash, rating: 1 | -1}` and read `GET
/api/activity/digest/feedback?state_hash=`, both `agents:execute`. `services/digest_feedback.py` keeps one row per
(workspace, person, `state_hash`): the write takes a transaction-scoped advisory lock on that key and updates or
inserts (there is no unique index). A caller with no person writes a row with NULL `user_id` and always reads null.

## Status: `DisplayState`

`services/proactive/display_state.py` computes one vocabulary on the server; clients only map it to a glyph, a word
and a colour role (`frontend/lib/proactive/display-state.ts`, re-exported to themes as `DISPLAY_STATE_LABELS`).

| State | From |
|---|---|
| `needs_you` | A `needs_you()` row of kind review, question, approval or stuck |
| `working` | A ticket Assigned or In progress; a Mission planning, running, replanning or checking |
| `done` | A ticket Done or Closed; a completed Mission; a granted approval |
| `failed` | Kind failed; a failed ticket or Mission |
| `ready` | Not produced by `display_state.py` (it needs the seen flag, §12-Q18); the brief's `since` rows use it for finished work |
| `null` | Inbox, Blocked, cancelled, paused, unknown: the client draws nothing |

Rows that carry `display_state`, `next_step` (≤ 90 characters, from `services/proactive/strings.py` in the person's
language) and `retry` (`retry_ticket`, `replan_mission`, `rerun_mission` or null): needs-you rows (`GET
/api/activity/needs-you`), the Missions list (`services/mission_view.py`), and the board's ticket reads
(`services/board_task_view.py`).

## Readiness: `Readiness@1`

`{v: 1, checked_at, items: [{id, label, detail, state, fix?, needed_by}]}`, served by `GET /api/readiness` and
`POST /api/readiness/check` (`api/readiness.py`, any member, both editions).

| Item (`services/readiness/probes.py`) | Needed by | Probe |
|---|---|---|
| `core` | brief, voice, push, check-in | The server and its database answer |
| `model_route` | brief, check-in | Auto's model is chosen and has a key |
| `web_search` | voice | A search engine is configured (OpenRouter, Composio Search or SearXNG), or web search is off |
| `push` | push | Notifications can reach a phone; on the local edition, the Expo consent |

The client adds `microphone` and `speech` (the web, from browser support) and the phone adds its own; the server
never claims them.

- `services/readiness/service.py` runs every probe at once, each in its own daemon thread on its own session, and
  waits `READINESS_PROBE_TIMEOUT_SECONDS` (default 3). A probe still running reads `not_checked`; one that raises reads
  `needs_setup` with "Couldn't be read", the error in the log.
- The last check is held per workspace (per worker) for the `readiness_check` window; `GET` answers it while held.
  `POST /api/readiness/check` is limited to 3 per workspace per minute (`core/security/rate_limiter.py`), counted in
  Redis; over it is 429 `checked_recently` with `Retry-After`.
- `fix.href` is always an app path (`/settings?tab=…`). On the local edition `detail` adds the host and latency; the
  hosted edition never does.
