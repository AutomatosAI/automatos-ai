# Customer night

Claude plays the operator overnight, in the operator's own local workspace, with the
real tools: talks to Auto, builds/edits/deletes agents, hands out a dozen real briefs
(research with the web, copy, an SOP, a one-page website by a Claude session agent, a
mission, uploaded documents and questions about them), answers the questions agents
raise, reviews everything that comes back, and writes a diary and a morning report.
It never touches the code; what it finds is what a customer would find.

## Before launching (each night)

1. Local stack up and on a branch that carries the PRD-245 session-tools bridge (sessions need platform tools).
2. The CLI host running and paired to this workspace at the version the backend expects (`make cli-host-status`; `make cli-host-install` from the same checkout after a branch switch).
3. The OpenRouter key present in the platform (Auto and API agents think with it).
4. `python3 -m tests.sim.customer preflight` — the runner refuses to start if this fails.

## Launch (human-only — you start it before bed)

```sh
./scripts/ralph/overnight-customer.sh
# env: RALPH_MODEL=claude-opus-5  RALPH_MAX_ITERS=10  RALPH_STOP_AT=06:30  CUSTOMER_PERSONA=harbourline
```

Runs under `caffeinate`; iterations of up to 50 minutes continue from the diary until
the stop time; usage limits are waited out the way the PRD Ralph runs do. The driver
model is pinned; the session agents use whatever their configuration says (your
subscription); Auto and API agents use the platform's OpenRouter key.

## In the morning

```
~/.automatos-sim/customer-night/<date>/MORNING-REPORT.md   what I built · what I got (graded) · tools used · friction · broken · cost · would I pay · fix first
~/.automatos-sim/customer-night/<date>/DIARY.md            every action, in order, with how it felt
~/.automatos-sim/customer-night/<date>/logs/iter*.log      the full transcripts
```

Everything the persona created carries the tag `sim-night-<date>`. Leave it to look
at, or clear it: `python3 -m tests.sim.customer purge --tag sim-night-<date> --yes`
(dry run without `--yes`).

## Rules the persona is under

Tag everything; never touch untagged rows; drafts only — no send/post/publish/pay/delete
on any connected app; no code, git, Docker or database; stop starting new work at the
stop time; stop at $15 of model spend. Personas live in `personas/`; add one per
company you want simulated and pick it with `CUSTOMER_PERSONA`.
