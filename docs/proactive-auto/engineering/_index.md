# Proactive Auto: engineering

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [docs/PRDS/PRD-268-PROACTIVE-AUTO.md](../../PRDS/PRD-268-PROACTIVE-AUTO.md)
- [scripts/ralph/prd-268w1.json](../../../scripts/ralph/prd-268w1.json)
- [scripts/ralph/prd-268w2.json](../../../scripts/ralph/prd-268w2.json)
- [scripts/ralph/prd-268w3.json](../../../scripts/ralph/prd-268w3.json)
- [orchestrator/modules/proactive/__init__.py](../../../orchestrator/modules/proactive/__init__.py)
- [orchestrator/services/proactive/__init__.py](../../../orchestrator/services/proactive/__init__.py)
- [orchestrator/router_manifest.py](../../../orchestrator/router_manifest.py)
- [frontend/lib/theme-sdk/proactive.ts](../../../frontend/lib/theme-sdk/proactive.ts)

</details>

Proactive Auto (PRD-268) turns heartbeats into read-only **check-ins** that record **suggestions**, delivers those
suggestions and a once-a-day **brief** to each person, lets Auto keep **follow-ups**, offers **reply chips**, and
shows every approval as an **ApprovalView**. This topic is for people who work on the platform or run a server. What
members see is in [Proactive Auto](../_index.md).

It shipped in three waves on main: W1 check-ins (#1367, #1368), W2 delivery (#1375, then #1379 and SDK 1.4.0 in
#1383) and W3 follow-ups (#1385). The PRD (`docs/PRDS/PRD-268-PROACTIVE-AUTO.md`) is the plan; these pages describe
what the code on main does. Where they differ, see [Where the code differs from the PRD](#where-the-code-differs-from-the-prd).

## Pages in this topic

| Page | What it covers |
|---|---|
| [Check-ins](check-ins.md) | The runner, its start gates, the tool loop, the read-only lane and how it fails closed, signals, the suggestion tool (provenance, dedupe, caps), failing streaks, the heartbeat bridge, delivery and the suggestion push |
| [The brief, status and readiness](brief-status-readiness.md) | The brief claim as a lease, `Brief@1` and its sections, `platform_get_brief`, the thumbs, `DisplayState`, `Readiness@1` |
| [Follow-ups and reply chips](follow-ups-and-chips.md) | The follow-up row, its caps under a workspace lock, the firing in the lane, privacy, the terminal reply-chip turn and the turn's parts |
| [Approvals and calibration](approvals-and-calibration.md) | `ApprovalView@1`, rule sentences, the view route, and calibration (shipped off) |
| [Data and routes](data-and-routes.md) | Tables, the three migrations, every route with its auth and refusal codes, message parts, live events, discovery |
| [Configuration](configuration.md) | Every PRD-268 setting in `config-surface.json`, the platform mode, the workspace switch and caps, the person's pause, the hard-coded run limits |
| [Operations](operations.md) | Turning on shadow for one workspace, going live, the kill switches, what to watch in the logs, the evals |

## The pipeline

```
heartbeat fire ──► heartbeat bridge ──► check-in runner ──► gates ──► signals / agent prompt
                    (off: tick only;                          │            │
                     shadow: tick + check-in;          refused: a run     one bounded tool loop
                     live: check-in only)              row, no model      in the check_in lane
                                                                               │
                                                     platform_suggest_next_step (the one write)
                                                                               │
                                     proactive_runs row ◄── run ends ──► proactive_suggestions rows
                                                                               │
                                                     live only: deliver_run ──► Auto thread post (suggestion@1)
                                                                               └► suggestion_ready bell ─► push (off)
person ──► Start ──► board ticket under watch, or a chat     person ──► Hide (reason)
```

The brief, follow-ups, reply chips and approval views hang off the same parts: the brief reads open suggestions,
a follow-up fires one turn of the same loop in the same lane, and every structured card rides a message part.

## Code map

| Path | What lives there |
|---|---|
| `orchestrator/modules/proactive/` | Pure rules, no database: the check-in window, dedupe keys and normalised titles, provenance, caps, follow-up caps and times, failing-streak arithmetic, heartbeat modes, the locale chooser |
| `orchestrator/services/proactive/` | The runner, gates, loop, signals, run rows, switches, settings and limits, quiet hours, decisions (Start and Hide), the start ticket, expiry, delivery, the heartbeat bridge and bell, failing streaks, display state, follow-ups and their firing, live events, and the strings tables |
| `orchestrator/services/brief/` | The brief claim, build, its five section readers, `Brief@1`, strings, the on-open flow |
| `orchestrator/services/readiness/` | Readiness probes, the held check, `Readiness@1`, strings |
| `orchestrator/services/calibration/` | Calibration's claim, sources (titles only), prompt, build, answers, memory writes, strings, its switch |
| `orchestrator/services/approval_views.py`, `orchestrator/modules/tools/formatting/approval_view.py`, `orchestrator/modules/policy/rule_sentences.py` | `ApprovalView@1` from a card or a grant row, and the rule sentences |
| `orchestrator/modules/tools/discovery/check_in_lane.py`, `orchestrator/core/read_only_call.py` | The read-only lane's two decorators, chat-only actions, and the no-side-records context |
| `orchestrator/modules/tools/discovery/actions_proactive.py`, `handlers_proactive.py`, `suggestion_fields.py` | `platform_suggest_next_step` |
| `orchestrator/modules/tools/discovery/actions_brief.py`, `handlers_brief.py` | `platform_get_brief` |
| `orchestrator/modules/tools/discovery/actions_follow_ups.py`, `handlers_follow_ups.py`, `follow_up_context.py` | `platform_schedule_follow_up`, `platform_list_follow_ups`, `platform_cancel_follow_up` |
| `orchestrator/modules/tools/discovery/actions_replies.py`, `handlers_replies.py`, `orchestrator/modules/tools/execution/reply_chips_turn.py`, `orchestrator/consumers/chatbot/turn_parts.py` | `platform_offer_replies`, the terminal turn, the saved turn's parts |
| `orchestrator/services/chat_parts.py`, `orchestrator/services/chat_messenger.py` | Structured message parts on background posts |
| `orchestrator/services/push/suggestion_rules.py`, `payload.py`, `fanout.py`, `preferences.py` | The fourth push type, `suggestion` |
| `orchestrator/api/proactive.py`, `proactive_brief.py`, `proactive_follow_ups.py`, `readiness.py`, `approval_views.py`, `auto_calibration.py` | The routes, registered through `RouterSpec` in `orchestrator/router_manifest.py` |
| `orchestrator/core/models/proactive.py` | `ProactiveSuggestion`, `ProactiveRun`, `ProactivePersonSettings`, `ProactiveCalibration` |
| `orchestrator/evals/proactive/` | The six eval runners and their pure scorers |
| `frontend/components/proactive/`, `frontend/lib/proactive/` | The web defaults: brief card, suggestion card and pinned line, Suggested list, reply chips, follow-up line, readiness list, status label, calibration card, and the wire types |
| `frontend/hooks/use-brief.ts`, `use-suggestions.ts`, `use-follow-ups.ts`, `use-readiness.ts`, `use-calibration.ts` | The data hooks, all through `apiClient` |
| `frontend/lib/theme-sdk/proactive.ts`, `frontend/lib/theme-sdk/host-cards/` | The Theme SDK's `proactive` area and the host cards |

Tests: `orchestrator/tests/test_prd268_*.py`, `test_prd268w2_*.py` and `test_prd268w3_*.py`, the vitest files under
each `__tests__` folder beside the components, and the browser specs `e2e/browser/specs/proactive-*.spec.ts`.

## For theme authors: the Theme SDK

The SDK's proactive surface is already documented, name by name, in the theme host's docs. Read those rather than a
copy here:

- [SDK surface map §5](../../theme-host/sdk-surface-map.md#5-sdk-names-no-import-above-maps-to): every proactive
  hook (`useBrief`, `useReadiness`, `useCheckReadiness`, `useSuggestions`, `useStartSuggestion`,
  `useHideSuggestion`, `useCalibration`, `useAnswerCalibration`, `useRateBrief`), the cards (`BriefCard` and its
  `thumbs` prop, `SuggestionCard`, `StatusLabel`, `ReadinessList`, `CalibrationCard`, `ReplyChips`), `BriefVoice` and
  its `onSpeakingRow`, and `DISPLAY_STATE_LABELS`.
- [Theme host §7](../../theme-host/_index.md#7-the-host): the `proactive` area (SDK 1.3.0) and what 1.4.0 added (`knownParts`,
  the brief row Listen is reading, `useRateBrief`).
- `frontend/packages/theme-sdk/CHANGELOG.md`: what each SDK version added.

Three facts the theme-host pages don't spell out:

- The SDK is at **1.4.0** and the host contract at **1.1.0** (`orchestrator/services/themes/contract.py`). PRD-268
  didn't bump the host contract.
- The approval cards' data carries an optional `approval_view` (`ApprovalViewData`, cards area, 1.3.0). A theme places
  `ToolApproval` and `MissionApproval` and never redraws them.
- `HostChatMessage.knownParts` reads `text`, `brief@1`, `suggestion@1` and `reply_chips@1`. The host also posts
  `follow_up@1` and `calibration@1` parts, which only the host's own thread draws: a theme that owns `/chat` doesn't
  see them.

## Where the code differs from the PRD

| PRD | What the code does |
|---|---|
| A suggestion's addressee is decided per suggestion (§12-Q2) | Every check-in suggestion is addressed to the workspace owner (W1 K11). A follow-up's suggestion is addressed to the person whose chat scheduled it. |
| Start can create a board ticket, a Mission, a playbook run or a chat | Only `board_task` and `chat`. `mission` and `playbook_run` are accepted by the schema and refused with 409 `not_available` (K12). Check-ins propose only the two. |
| The brief on open is for everyone | It runs only where the effective mode is shadow or live; elsewhere `POST /api/proactive/brief/open` is 404 `brief_off` before any claim. The phone's scheduled digest is not gated. |
| The brief claim adds `message_id` and `via` columns to `mobile_digest_posts` | The table is unchanged. `via` and `local_date` ride the message's `source`; the claim is a lease on `posted_at` (`BRIEF_CLAIM_STALE_SECONDS`). |
| The brief's thumbs reuse the Read's feedback | They do, and `services/digest_feedback.py` now keeps one rating per person per `state_hash`, with a new `GET /api/activity/digest/feedback`. |
| "Start (about 40 credits)" on the card | `est_credits` exists on the row and the web can draw it, but nothing writes it, so it never shows. |
| The suggestion push ships | It is built and ships off (`SUGGESTION_PUSH_ENABLED=false`). The bell row is still written, and the per-day cap is counted from bell rows. |
| `report_triage` decides severity and may run live | Severity (`today` or `can_wait`) is set by the check-in's model in the tool call. The PRD-248 triage stays shadow-only. |
| Always allow; buttons Allow once / Always allow / Refuse | Not built. `answers` is only `refuse` and `allow_once`. The buttons keep "Approve & run" / "Deny" (tool) and "Approve & run" / "Reject" (Mission); key 2 does nothing. |
| Follow-up caps under a lock per (workspace, person, agent) | One `pg_advisory_xact_lock` per workspace (W3 K9), so the workspace's 100 a day can be bounded. |
| Chips stored as `reply_chips` in the message's `source` | They are a message part, `{type: 'reply_chips@1', replies}` (W3 K12). |
| Hide learning, field-memory writes from Hide reasons, Telegram and Slack, Mission and Playbook proposals | Not built (the PRD's W4). |
| The agent page shows the heartbeat mode | The mode is in the heartbeat config API (`mode` on `GET`/`PUT /api/heartbeat/agents/{agent_id}/config`); no web control sets it. |
| Workspace and person switches in Settings | API only: `PUT /api/proactive/settings` and `PUT /api/proactive/me`. No web UI. |
| Brief rows carry Start, Answer, Open | `Brief@1` rows carry their `action` and `opens`, but the web host passes no row handlers, so the web card draws no row actions and "Show N more" is text. A theme's `cards.BriefCard` can supply them. |
| Calibration follows the proactive mode | Calibration is gated only by `CALIBRATION_ENABLED`, not by the mode. |
| The approval view shows its expiry counting down (§6.12.3) | The view carries `expires_at`; the web card doesn't draw it. |

## Pending

- **#1386** (draft): the daily brief's push will carry its `message_id`, so the phone opens the thread at the brief.
  Not on main.
