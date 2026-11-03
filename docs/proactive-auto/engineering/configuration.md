# Configuration

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/config.py](../../../orchestrator/config.py)
- [orchestrator/reports/config-surface.json](../../../orchestrator/reports/config-surface.json)
- [orchestrator/services/proactive/switches.py](../../../orchestrator/services/proactive/switches.py)
- [orchestrator/services/proactive/limits.py](../../../orchestrator/services/proactive/limits.py)
- [orchestrator/services/proactive/settings.py](../../../orchestrator/services/proactive/settings.py)
- [orchestrator/services/proactive/quiet.py](../../../orchestrator/services/proactive/quiet.py)
- [orchestrator/services/proactive/loop.py](../../../orchestrator/services/proactive/loop.py)
- [orchestrator/modules/proactive/heartbeats.py](../../../orchestrator/modules/proactive/heartbeats.py)
- [orchestrator/core/security/rate_limiter.py](../../../orchestrator/core/security/rate_limiter.py)

</details>

Proactive Auto is controlled at four levels. Each is read at the start of every run and fails closed: a value that
can't be read, or isn't one the code knows, is off.

| Level | Where | Who changes it | Values |
|---|---|---|---|
| **Platform mode** | `system_settings` row `(category 'proactive', key 'mode')`; no row reads `PROACTIVE_MODE_DEFAULT` | A system admin (`system_role` `admin` or `super_admin`) through `/api/system-settings`; on the local edition the local operator is the super admin | `off` (default), `shadow`, `live` |
| **Workspace switch** | `workspaces.settings.proactive.enabled` | Owners and admins, `PUT /api/proactive/settings` | Off unless exactly `true` |
| **Workspace caps** | `workspaces.settings.proactive.caps` | Owners and admins, `PUT /api/proactive/settings` | Each cap over its config default; tighten only |
| **The person's pause** | `proactive_person_settings.paused` | The person, `PUT /api/proactive/me` | A pause that can't be read is a pause |

`effective_mode(db, workspace_id)` (`services/proactive/switches.py`) is the platform mode where the workspace switch
is on, else `off`. Every gate, the bridge, delivery, the brief on open and follow-ups read it.

Per heartbeat, `agents.configuration.heartbeat.mode` (`check_in` or `act`; unset is `act` when `auto_act` is set,
else `check_in`) and, for the orchestrator, `proactive_level` decide whether that heartbeat converts.

## Settings in `config.py`

Every name below is in `orchestrator/reports/config-surface.json`. Set them in the environment (on the local edition,
`envs/api.local` overrides `envs/api.defaults`); none is in `envs/api.defaults`.

### Mode and switches

| Setting | Default | What it does |
|---|---|---|
| `PROACTIVE_MODE_DEFAULT` | `off` | The platform mode when no `proactive.mode` row exists. A value other than `off`, `shadow` or `live` is off. |
| `SUGGESTION_PUSH_ENABLED` | `false` | The `suggestion` push type. Off: no suggestion push is sent and discovery doesn't list `suggestion_push`; the bell row is still written. |
| `CALIBRATION_ENABLED` | `false` | Calibration. Off: `GET /api/auto/calibration` answers `available: false`, every POST is 404, discovery doesn't list `calibration`. |

### Check-in caps (§6.9)

A workspace's `settings.proactive.caps` may set any of these seven (`EDITABLE_CAPS`); each must be at least as tight as
config's. A looser value is 422 `caps_looser` on write and held to config's on read. Caps that don't fit together
(negative, or a minimum interval past the maximum) are 422 `caps_wrong`, and stored wrong they refuse every check-in.

| Setting | Workspace cap | Default | Counts |
|---|---|---|---|
| `PROACTIVE_MAX_MODEL_RUNS_PER_DAY` | `max_model_runs_per_day` | 48 | Check-ins that reached the model in the last day, per workspace. A silent run with no model call doesn't count. |
| `PROACTIVE_MAX_SUGGESTIONS_PER_RUN` | `max_suggestions_per_run` | 3 | Suggestions one run may record |
| `PROACTIVE_MAX_NEW_PER_PERSON_PER_DAY` | `max_new_per_person_per_day` | 10 | New suggestions per person |
| `PROACTIVE_MAX_OPEN_PER_PERSON` | `max_open_per_person` | 20 | Past it, the person's oldest open suggestions expire; nothing is refused |
| `PROACTIVE_MAX_OPEN_PER_AGENT` | `max_open_per_agent` | 3 | Open suggestions from one source agent |
| `PROACTIVE_MIN_INTERVAL_MINUTES` | `min_interval_minutes` | 30 | A check-in sooner than this after the last that ran is refused (less a 5-minute scheduler grace). A workspace tightens it by raising it. |
| `PROACTIVE_MAX_INTERVAL_MINUTES` | `max_interval_minutes` | 1440 | The ceiling a failing streak's doubling interval reaches |

`0` allows nothing: a suggestion cap at 0 refuses the check-in at its start, since it could record nothing.

### Delivery

| Setting | Default | What it does |
|---|---|---|
| `PROACTIVE_MAX_SUGGESTION_PUSHES_PER_DAY` | 2 | `suggestion_ready` bell rows per person per workspace since their local midnight. 0 rings none. The push follows the bell. |
| `BRIEF_CLAIM_STALE_SECONDS` | 600 (minimum 60) | A brief claim older than this with no brief posted is taken over by the next open or tick |
| `READINESS_PROBE_TIMEOUT_SECONDS` | 3 (minimum 0.1) | How long each readiness probe has before it reads `not_checked` |

### Follow-ups (W3)

| Setting | Default | What it does |
|---|---|---|
| `PROACTIVE_FOLLOW_UP_MIN_MINUTES` | 10 | Earliest a follow-up may be |
| `PROACTIVE_FOLLOW_UP_MAX_DAYS` | 7 | Latest a follow-up may be |
| `PROACTIVE_FOLLOW_UP_MAX_PENDING` | 5 | Pending follow-ups per person (from Auto) |
| `PROACTIVE_FOLLOW_UP_MAX_PER_DAY` | 20 | Scheduled per person in the last 24 hours, fired or not |
| `PROACTIVE_FOLLOW_UP_MAX_PER_WORKSPACE_PER_DAY` | 100 | Scheduled per workspace in the last 24 hours |
| `PROACTIVE_FOLLOW_UP_CONTEXT_MESSAGES` | 5 | How many of the origin chat's last messages a firing quotes |
| `PROACTIVE_FOLLOW_UP_REASON_CHARS` | 500 | A reason's maximum length |

### Evals (W3)

| Setting | Default | What it does |
|---|---|---|
| `PROACTIVE_EVAL_GOLD_DIR` | empty | A directory **outside** the repository holding `<eval>.jsonl` gold sets. Empty, missing or inside the repository is refused. |
| `PROACTIVE_EVAL_WORKSPACE_ID` | empty | The workspace whose model settings and credentials the eval runners use |

## Hard-coded limits

These are constants, not settings:

| Limit | Value | Where |
|---|---|---|
| Model calls per check-in | 4 | `services/proactive/loop.py` `MAX_ITERATIONS` |
| Tool calls per model call | 8 | `MAX_CALLS_PER_TURN` |
| Cost per check-in | $0.05 | `MAX_COST_USD` |
| Output tokens per call | 800 | `core/llm/output_budget.py`, purpose `check_in` |
| Check-in context | 12,000 tokens | `modules/context/modes.py`, `ContextMode.CHECK_IN` |
| Window when there's no successful run | 24 hours | `modules/proactive/window.py` `LOOKBACK` |
| Open suggestion lifetime | 7 days | `services/proactive/expiry.py` |
| Dedupe window for hidden suggestions | 14 days | `modules/proactive/dedupe.py` `HIDDEN_WINDOW` |
| Evidence refs per suggestion | 5 | `modules/proactive/provenance.py` `MAX_EVIDENCE` |
| Brief rows per section before folding | 5 (never folds at 6) | `services/brief/rows.py` |
| Readiness checks | 3 per workspace per minute | `core/security/rate_limiter.py` `readiness_check`; overridable by `RATE_LIMIT_READINESS_CHECK_MAX` and `RATE_LIMIT_READINESS_CHECK_WINDOW_SECONDS` |

The check-in's model is the platform's `orchestrator_llm` system setting through `LLMManager(service_name='heartbeat')`
(`claude-haiku-5-5` by default, `core/llm/defaults.py`).

## Person settings

`PUT /api/proactive/me {paused?, quiet_start?, quiet_end?, timezone?}`:

- `paused`: no check-in suggestions addressed to them (the run is `skipped_paused`), no follow-ups scheduled or fired.
  The brief still comes.
- `quiet_start`, `quiet_end` (`HH:MM`, sent together, different, or both null to clear) and `timezone` (IANA; null
  falls to the workspace quiet hours' zone, else UTC). A window whose start is after its end runs over midnight. The
  workspace's `settings.auto_reporting.quiet_hours` window applies on top. Quiet hours hold the suggestion bell and
  push only; check-ins still run.

Other workspace settings: `settings.push.suggestion_titles: true` shows a suggestion's title in its push instead of
"Auto has a suggestion".
