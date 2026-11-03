# Operations

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/services/proactive/switches.py](../../../orchestrator/services/proactive/switches.py)
- [orchestrator/api/system_settings.py](../../../orchestrator/api/system_settings.py)
- [orchestrator/api/proactive.py](../../../orchestrator/api/proactive.py)
- [orchestrator/api/heartbeat.py](../../../orchestrator/api/heartbeat.py)
- [orchestrator/services/proactive/heartbeat_bridge.py](../../../orchestrator/services/proactive/heartbeat_bridge.py)
- [orchestrator/services/proactive/runner.py](../../../orchestrator/services/proactive/runner.py)
- [orchestrator/modules/tools/discovery/check_in_lane.py](../../../orchestrator/modules/tools/discovery/check_in_lane.py)
- [orchestrator/evals/proactive/__main__.py](../../../orchestrator/evals/proactive/__main__.py)
- [orchestrator/evals/proactive/bars.py](../../../orchestrator/evals/proactive/bars.py)
- [orchestrator/evals/proactive/gold.py](../../../orchestrator/evals/proactive/gold.py)

</details>

Every request below is authenticated as usual and names the workspace with the `X-Workspace-ID` header. Nothing in
Proactive Auto needs a restart except the environment settings (`PROACTIVE_MODE_DEFAULT`, `SUGGESTION_PUSH_ENABLED`,
`CALIBRATION_ENABLED` and the caps' defaults); the system setting, the workspace switch, the caps and the pause are read
at every run start.

## Turn on shadow for one workspace

Shadow runs a check-in beside each heartbeat that converts, records runs and suggestions, and delivers nothing.
Today's heartbeats keep running unchanged.

1. **The platform mode.** As a system admin, create the system setting:

   ```http
   POST /api/system-settings/
   {"category": "proactive", "key": "mode", "value": "shadow"}
   ```

   If the row exists, `PUT /api/system-settings/{setting_id}` with `{"value": "shadow"}`. On a local install you can
   instead set `PROACTIVE_MODE_DEFAULT=shadow` in `envs/api.local` and restart; a stored row wins over it. Workspaces
   are off until their own switch is on, so this alone changes nothing.

2. **The workspace switch.** As an owner or admin of the workspace:

   ```http
   PUT /api/proactive/settings
   {"enabled": true}
   ```

   The answer is `{enabled, caps, platform_caps}`. Optionally tighten caps in the same call, e.g.
   `{"enabled": true, "caps": {"max_model_runs_per_day": 12}}`.

3. **The heartbeats.** Check-ins ride the heartbeat schedule, so something must fire:
   - Auto's workspace check-in needs the orchestrator heartbeat on (Settings → Orchestrator → Enable Heartbeat) at
     Proactive Behavior Level **Silent** or **Notify Only**.
   - An agent's heartbeat converts by default unless its config has `auto_act` set; set `"mode": "check_in"` explicitly
     with `PUT /api/heartbeat/agents/{agent_id}/config` to convert one that acts. Claude Code agents don't convert.
   - Check-ins run only inside each heartbeat's active hours.

4. **Compare.** Over the shadow period, read what the check-ins did against the tickets the heartbeats filed:
   - `GET /api/proactive/runs?since=<ISO time>`: every run, its outcome, signals, suggestion count, model, tokens and
     cost.
   - `GET /api/proactive/suggestions?status=open` (and `hidden`, `started`, `expired`): as an owner or admin, everyone's.
   - The workspace owner also sees open suggestions in the Command Center's Suggested list, the brief and the pinned
     line in their Auto thread; the brief on open now runs for every member.

## Go live

Set the platform mode to `live` (the same system setting). Then, for every workspace whose switch is on:

- converted heartbeats run the check-in **instead of** the tick: no tickets from them, no `report_to` routing, no
  per-tick reports, and `heartbeat_complete` only on a suggestion or a streak's first failure;
- each new suggestion is posted once in its addressee's Auto thread, with one `suggestion_ready` bell per run when it
  is `today`;
- Auto can schedule follow-ups, and pending ones fire.

Suggestions recorded during shadow are not delivered retroactively; they stay in the list and the brief. The platform
mode is global: you can't keep one workspace in shadow while others are live, so switch that workspace off instead.

The suggestion push stays off until `SUGGESTION_PUSH_ENABLED=true` and the phone app registers the `suggestion`
category. Calibration stays off until `CALIBRATION_ENABLED=true`.

## Kill switches

From widest to narrowest. Each takes effect at the next run start, without a restart unless noted.

| Switch | Effect |
|---|---|
| `proactive.mode` = `off` (or delete the row with `PROACTIVE_MODE_DEFAULT` off) | Every workspace off: heartbeats run their ticks again, no brief on open, no delivery, follow-ups refuse to schedule and pending ones end `skipped_off` |
| `PUT /api/proactive/settings {"enabled": false}` | The same, for one workspace |
| A cap at 0, e.g. `max_model_runs_per_day: 0` | The workspace's check-ins are refused at their start (`skipped_cap`) |
| `heartbeat.mode` = `act` on one agent, or the orchestrator's level at Act & Notify | That heartbeat runs its tick only |
| `PUT /api/proactive/me {"paused": true}` | One person: no suggestions to them, no follow-ups |
| `SUGGESTION_PUSH_ENABLED=false` (restart) | No suggestion push; bells continue |
| `CALIBRATION_ENABLED=false` (restart) | Calibration routes 404, nothing shows |
| `PROACTIVE_MAX_SUGGESTION_PUSHES_PER_DAY=0` (restart) | No `suggestion_ready` bell rows |

The spend ceiling (`SPEND_CEILING_USD`, `services/daily_spend_guard.py`) also refuses check-ins and Starts when
reached, and only credits beyond the trial pay for check-ins on the hosted edition.

## What to watch in the logs

| Log line | Means |
|---|---|
| `[proactive] <kind> run <id> in <workspace>: <outcome>, <n> suggestion(s), $<cost>` | A run finished |
| `[proactive] <what> refused: <outcome> (<reason>)` | A run start was refused (`skipped_cap (max_model_runs_per_day)` and so on); the run row has the same |
| `[proactive] run <id> stopped at $<x>, past its $0.05 ceiling` | The cost ceiling stopped a run |
| `[proactive] agent <id> in <workspace>: failure <n> of streak <key>` and `… works again: <n> streak card(s) expired` | A failing streak and its reset |
| `[proactive] run <id>: <n> suggestion(s) posted, <n> bell(s) rung` | Delivery, live only |
| `[proactive] the proactive.mode setting could not be read: check-ins are off` | The platform read failed; failing closed |
| `[proactive] the workspace switch of <id> could not be read: off`, `the pause of user … could not be read: paused`, `the caps of … could not be read: check-in refused` | Other fail-closed reads |
| `[check_in_lane] <action> refused in a check-in (workspace=<id>)`, `[check_in_lane] tool <name> refused in a check-in` | The lane refused a call. A steady stream for one agent means its heartbeat prompt needs connector or write tools: keep it `act`. |
| `[proactive] a heartbeat's check-in in <workspace> failed` | The bridge caught a check-in failure; the tick result stands |
| `Brief on open: person <id> in <workspace> failed (<class>)`, `… was taken over`, `… was not posted` | The brief claim's failure, takeover and lost-lease paths |
| `[follow-up] <id> scheduled …`, `… fired as run <id> …: <outcome>, $<cost>`, `… cancelled by user …`, `… failed` | Follow-ups |
| `[calibration] …` | Calibration builds and memory writes |
| `readiness: the <item> probe failed` | A readiness probe raised; its item reads "Couldn't be read" |

Logs never carry a suggestion's words, a follow-up's reason, or the brief's text; failures are logged by error class.

## The evals

Six runners in `orchestrator/evals/proactive/`, each with a pass bar from §6.7 (`evals/proactive/bars.py`). They are
run by hand, never by CI. CI tests only the pure scorers on inline synthetic cases, and fails on any tracked
`*.jsonl` under `orchestrator/evals/proactive/` (`tests/test_prd268w3_no_gold_sets.py`; `.gitignore` covers it).

```bash
cd orchestrator
PROACTIVE_EVAL_GOLD_DIR=/path/outside/the/repo PROACTIVE_EVAL_WORKSPACE_ID=<workspace id> \
  python -m evals.proactive <triage|silence|reply_chips|spoken|precision|one_brief> [--json]
python -m evals.proactive precision --workspace <workspace id>
```

| Eval | Reads | Pass bar |
|---|---|---|
| `triage` | `triage.jsonl` | Recall of "act today" ≥ 0.95, precision ≥ 0.90, every verdict with a reason ≤ 120 characters |
| `silence` | `silence.jsonl` | ≥ 0.95 of quiet snapshots silent, ≥ 0.85 of planted ones naming the need, 0 duplicates of an open or hidden title |
| `reply_chips` | `reply_chips.jsonl` | Chips on ≥ 16/20 choice turns, 0 outside the limits, 0 replacing the answer, ≤ 1/10 on open questions |
| `spoken` | `spoken.jsonl` | Median ≤ 40 words, none over 80, none heard as nothing, no markdown, URL, path or id; prints every answer for a hand audit |
| `precision` | The database, in a READ ONLY transaction | Started ÷ decided ≥ 0.40, started and passed review ≥ 0.30 of decided; lists source agents under 0.20 after 20 decisions |
| `one_brief` | The database | Exactly one brief per person per local date; it **writes a scratch workspace** in the configured database and removes it |

- The gold sets are one JSON object per line, read from `PROACTIVE_EVAL_GOLD_DIR`, which must exist and be outside
  the repository. The fields each case needs are read in that eval's module. Never commit them: this is a public
  repository.
- The model is the platform's `orchestrator_llm` on the workspace `PROACTIVE_EVAL_WORKSPACE_ID` names, and it spends
  that workspace's credits. Every model call goes through a purpose in `services/credits/classification.py`.
- Exit 0 is a pass, 1 a miss, 2 "cannot run" (no gold set, no workspace, a model call that failed).
- A run under a bar is a miss, reported as it is; the bars are never tuned to a run. Put the numbers, and only the
  numbers, in the PR that turns the feature on.
