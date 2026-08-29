# PRD-222 Q1 — Tier definitions (v1 APPROVED)

**Status: APPROVED v1 — Gerard, 2026-08-28.** Names: **Basic / Pro / Business** (+ **Enterprise** as a "coming soon" label only — no code path). Display pricing (no Stripe yet, plan *assignment* only per Q5): **$19 / $49 / $99**, config-driven strings labeled early-access so repricing stays free. Numbers below locked as v1 explicitly for trial-and-error: everything is config-driven (W2·S1 AC), Gerard tests all three tiers and adjusts without redeploy. `workspaces.plan` default `"starter"` → W2b renames the default to `basic` + backfills existing rows (that is W2b's one migration).

## Grounding (what exists in code right now)

- `workspaces.plan` is `String(50)`, default **`"starter"`** — free-form, no enum, no migration needed for any naming below as long as one tier keeps the name `starter` (or we add a rename/backfill to the W2b kit).
- `plan_limits` (JSONB) keys already read by live code: `budget` (config fallback at `config.py:794`), `max_agents`, `max_workflows`.
- **No `seats` key exists.** W2·S4's invite checklist item ("only if plan has seats") and any team gating need it added — that lands in W2b regardless of the numbers chosen.
- The W1 trial is **orthogonal**: it meters platform-funded *spend*; tiers gate *capabilities*. They compose, they don't overlap.

## The proposal (3 tiers)

| | **Basic** *(becomes the column default)* | **Pro** | **Business** |
|---|---|---|---|
| Display price (no billing wired) | $19/mo | $49/mo | $99/mo |
| Who | solo operator proving value | small team running the business on it | org with multiple pods |
| Seats (`plan_limits.seats`, NEW) | 1 | 5 | 25 |
| Agents (`max_agents`) | 5 | 20 | 0 (= unlimited) |
| Mission concurrency | 1 | 3 | 10 |
| Marketplace depth | curated starter set, full catalog *visible* with plan labels (D5) | full catalog installable | full + private/custom entries |
| CodeGraph | — | ✓ | ✓ |
| NL2SQL / analytics | — | ✓ | ✓ |
| Team features (invites, roles, org chart) | — | ✓ | ✓ |
| Watchers / scheduled (PRD-204) | 1 | 5 | unlimited |
| Voice (Retell, PRD-207) | — | — | ✓ |
| Budget default (`plan_limits.budget`) | modest | mid | custom |

## The only three decisions that block W2b

1. **Names** — `starter / growth / scale` as written, or your words.
2. **The ✓/— placements** — especially: which side CodeGraph and NL2SQL land on, and where Voice sits.
3. **The numbers** — seats, agents, concurrency, watchers.

Everything else (the exposure config map, nav trimming, tool-surface trimming, plan labels in the marketplace, the recommendation logic in the proposal stage) derives mechanically from this table and is already specced in W2·S1/S2.

## Explicitly out of this decision

- Pricing/billing/checkout — PRD §12 Q5 says plan *assignment* only in Wave 2; commerce is a separate call.
- A "Free" tier below Starter — the trial + waitlist covers the free mile today; add later without schema impact if wanted.
- Enforcement hardening (quota rejections etc.) — W2b builds exposure + recommendation; hard quota enforcement beyond what exists is its own later story if you want it.

**To approve:** edit this table in place (or reply with changes in any form). The W2b kit gets cut from whatever this file says the day you call it done.
