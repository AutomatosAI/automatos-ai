# Simulation harness — `tests/sim/` (PRD-247)

Drives the **local** platform the way a customer would and scores the night
on four rows: usability, cost, quality, usefulness. Report-only. Standard
library only, so the scheduled job runs on the system `python3` (3.11+).

## Set up once

```sh
mkdir -p ~/.automatos-sim
cat > ~/.automatos-sim/env <<'EOF'
# all optional — these are the defaults
SIM_API_URL=http://localhost:8000
SIM_MODEL_PROVIDER=openrouter
SIM_MODEL_ID=openai/gpt-4.1-mini        # pinned on every seeded agent and on the global LLM tiers for the run
SIM_BUDGET_USD=5                        # the runner stops before the scenario that would cross this
SIM_TASK_TIMEOUT_S=900
# the standalone judge (quality/usefulness rows) — leave unset and the rows fall back to the packs' cheap checks
# OPENROUTER_API_KEY=...
# SIM_JUDGE_MODEL_ID=openai/gpt-4.1-mini
EOF
cp tests/sim/personas.example.toml ~/.automatos-sim/personas.toml   # optional; edit; never commit
```

The local stack must be up (`automatos_backend`, `automatos_postgres` containers):
provision/purge run the two `orchestrator/scripts/*_test_workspace.py` scripts
inside the backend container, and cost is read with `psql` inside postgres. The
runner never sees a database URL or password.

**Credential: none.** On the local edition the harness is the anonymous local
operator (the instance's super admin) and every call carries `X-Workspace-ID:
<sim workspace>` — that header is the boundary. The per-workspace `ak_srv_` key
the create script can mint is *not* accepted as `X-Api-Key` by
`get_request_context_hybrid` (that header is the single static
`ORCHESTRATOR_API_KEY`, empty locally), so the sim mints none (`--no-key`).
Set `SIM_API_KEY` only for a stack that requires the static key.

## Run

```sh
cd automatos-ai
python3 -m tests.sim.night packs                  # list + validate packs, show tonight's rotation
python3 -m tests.sim.night run --pack smoke       # ~10 min, under $1: proves the instrument
python3 -m tests.sim.night run --pack agents      # the Agents pack
python3 -m tests.sim.night run --pack auto        # tonight's pack from packs/rotation.toml
python3 -m tests.sim.night latest                 # path of the last scorecard
```

Useful flags: `--budget 2` · `--keep` (don't purge; `night purge --workspace-id …` later) ·
`--only <scenario-id>` · `--no-judge` · `--no-global-models` (leave Auto on its usual model) ·
`--model`/`--provider`.

Exit codes: `0` pack completed (a bad scorecard is still 0 — the report is the
product) · `3` budget stopped it · `2` the harness could not run.

## Schedule (macOS launchd, the CLI host's pattern)

```sh
python3 -m tests.sim.schedule install --hour 1 --minute 30 --pack auto
python3 -m tests.sim.schedule status | run-now | uninstall
```

Runs under `caffeinate` from the repo root; stdout in `~/.automatos-sim/logs/launchd.log`.

## What a night does

1. `create_test_workspace.py --slug sim-<pack>-<utc-stamp> --no-key` → a fresh isolated workspace per run (no credential);
   a second run on the same machine waits for nothing — it is refused while `~/.automatos-sim/run.lock` is held.
2. Snapshots the global `chatbot`/`*_llm` rows of `system_settings` to the run directory, points them at the cheap model
   (that table has no workspace column, so this touches every workspace on the stack — restored in `finally`, and
   `--no-global-models` skips it).
3. Seeds the pack's agents (`POST /api/agents/`) — each must echo the sim workspace id or the run stops
   before a ticket exists — and pins their model (`PUT /api/agents/{id}/model-config`).
4. Runs each scenario in order, checking the dollar ceiling first:
   * **task** — files a ticket assigned to an agent, polls to a terminal state (done/review/failed/blocked/cancelled or
     timeout), answers questions the agent raises, reads back deliverables and reports.
   * **chat** — talks to Auto (`POST /api/chat`), keeps every tool call and result, measures effects before/after
     (tasks and agents that now exist).
   * **crud** — create/update/get/list/delete an agent through the API, each step timed.
5. Scores, writes the record, restores the global tiers, purges the workspace.

## Outputs — `~/.automatos-sim/runs/<run_id>/`

| file | what |
|---|---|
| `scorecard.md` | the four rows, one line per scenario, the findings (all trace-backed) |
| `run.json` | the whole record — rewritten after every scenario, so a crash keeps what finished |
| `trace.jsonl` | every HTTP exchange with full request and response bodies |
| `run.log` | the runner's own log |
| `llm-settings-snapshot.json` | the global tiers before the run (restore by hand from this if the run died mid-way) |

`~/.automatos-sim/campaign.sqlite` keeps one row per run and per scenario for night-over-night comparison.

## Packs

TOML under `tests/sim/packs/`; validated before anything is created. See the docstring in
`packs.py` for the schema and `packs/agents.toml` for a full example. Scenario shapes may be
committed; personas and anything derived from real customers stay under `~/.automatos-sim/`.

## Safety

* Refuses any `SIM_API_URL` that is not loopback unless `SIM_ALLOW_REMOTE=1` — `tests/.env` points the API suite at
  Railway production and this runner must never inherit that.
* Refuses a `DOCKER_HOST` that is not the local socket (the scripts and the cost SQL run inside containers there).
* Refuses the operator's default workspace by id, and any workspace slug not starting with `sim-`.
* The purge script refuses a workspace not stamped `managed_by=create_test_workspace.py` with a `sim:` purpose.
* The harness is the anonymous local super admin: `X-Workspace-ID` is the *only* thing scoping it, so it is sent on
  every call, every created row is checked to echo the sim id, and the run stops on a mismatch.
* No secrets in the repo; the OpenRouter key for the judge lives in `~/.automatos-sim/env`.

## Unit tests (CI job `sim-units`)

```sh
python3 -m pytest tests/sim/tests --noconftest --import-mode=importlib -q
```
