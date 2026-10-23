# PRD-265: Billing, plans and credits for hosted Studio (Clerk auth + Stripe billing)

**Status:** Draft for Gerard, 20 Oct 2026. · **Owner:** Gerard · **Written:** 20 Oct 2026, grounded on `origin/main` that day.

**Decisions taken (Gerard, 20 Oct 2026):**
- **P1, stack:** Clerk for sign-in and entitlements, Stripe for billing. Stripe runs the subscriptions, credit top-ups, Checkout (3-D Secure), the customer portal and Stripe Tax for VAT. **Clerk Billing is not used:** it has no 3-D Secure, no VAT and no one-off purchases.
- **P2, pricing unit — REVISED 20 Oct 2026 (Gerard): metered, like OpenRouter.** Every platform-funded model call costs its real cost in cents × one markup (`CREDITS_MARKUP`, default 1.2×). The earlier fixed credits per *ask* by band (Rikki's proposal) is dropped: a fixed charge could not price a $0.50 report and a $10 app build alike.
- **P3, timing:** the beta opens Tue 3 Nov on the trial plus customers' own keys, with nothing charged. **Paid plans open Mon 9 Nov 2026**, before Web Summit (10–12 Nov).
- **P4, legal entity:** left open. It gates the Stripe account and the VAT registrations (§9).

**Source:**
- `<workspace>/reports/Automatos billing options.md`: billing research of 19 Oct, with vendor and code claims re-verified.
- `<workspace>/DUMPING AREA/airtime-model/BILLING-PLAN.md`, `CREDITS.md` and `DEEP-DIVE.md`.
- `<workspace>/.claude/PLAN-PRICING-PUBLIC-RECORD-EU-AI-ACT.md` (D0–D10, decision 6 "Own keys = Pro").
- `<workspace>/.claude/PUBLIC-FACTS-2026-10-19.md` §4 (public price copy).
- `<workspace>/.claude/CLERK-PRODUCTION-CUTOVER.md` (Clerk runbook).

**Scope:**
- plan tiers in code;
- a credit ledger and the gate in front of the platform's AI keys;
- the leak fixes that must land before any money is taken;
- Stripe checkout, webhooks, portal and tax;
- the in-app billing surfaces;
- the Clerk production cutover as a dependency;
- go-live tests.

**Depends on:**
- automatos-ai #1198 (Clerk production readiness) and the Clerk cutover (the dev instance is capped at 100 users).
- #1216 (`azp` check, Gerard's call).
- P4, the entity, which gates the Stripe account.

**Ralph config:** launch is human-only, with `RALPH_MODEL` pinned. One branch, one push, one PR and one CI run per wave. At most four builders across all Automatos work.

**Human-gated:**
- P4 entity;
- the Stripe account and VAT registrations;
- Terms v2.3 and counsel;
- the markup on metered calls (§4, D-1);
- the Clerk dashboard, DNS and secret steps;
- live card tests;
- the 9 Nov opening switch.

Labels below: **[T]** read in code (`origin/main`, 19–20 Oct, via the billing report's verified file list); **[R]** from the research report; **[I]** inferred.

---

## 1. Introduction

Hosted Studio has public prices but no way to charge:
- **Public prices** (facts sheet §4, live on studio.automatos.app): **Own keys $9**, **Pro $29**, **Business $99** (by invite in the beta), **Enterprise coming soon**, a one-time **500-credit trial**, and 1 credit = 1 US cent of AI usage at Automatos prices.
- **No payment code [T]:** there is no Stripe dependency and no billing route. `orchestrator/config.py` `_PLAN_TIERS_DEFAULTS` still holds $19/$49/$99 and is marked "DISPLAY PRICING ONLY". The trial is a $5 dollar allowance, not credits. Several paths run AI on Automatos's own keys with no meter (§6).

This PRD builds the money path:
- **Credits:** a ledger that every platform-funded AI ask debits.
- **Gate:** a check that pauses work at zero.
- **Plans:** sold through Stripe Checkout, with webhooks that keep the plan and credits in step.
- **App surfaces:** a billing tab and a balance in the header.

Customers on their own keys (the Own keys plan, or any plan after the trial) are never debited for those calls.

## 2. Goals

- **G1:** Paid plans open **Mon 9 Nov 2026**. A customer can buy Own keys, Pro or a top-up with an EU or UK card, including 3-D Secure, and the workspace changes plan and balance within 60 seconds of payment.
- **G2:** No platform-funded AI call runs without a positive credit balance or an explicit platform allowance. All six leaks in §6 are closed and covered by tests.
- **G3:** Every credit movement is in an append-only ledger, idempotent per source event, and reconcilable against Stripe.
- **G4:** Business is invite-only. Auto can never put a workspace on a paid plan without payment.
- **G5:** VAT is calculated and collected by Stripe Tax for EU and UK buyers, with reverse charge for valid VAT numbers. The seller named on receipts is the P4 entity.
- **G6:** Public copy switches from the facts sheet's "beta pricing, not charged until billing opens" block to the paid block in the same deploy that opens checkout.

## 3. Plans (target state)

| Plan | Price | Credits | Seats / agents / render | Platform AI | Notes |
|---|---|---|---|---|---|
| Trial | $0 | 500 once, per person | Pro limits | Yes, until credits run out | A ledger grant. The daily global trial cap stays. |
| Own keys | $9/mo | 0 | **Same as Pro** (decision 6) | **No.** Calls use the customer's keys only; there is no platform fallback for any provider. | Background calls: see D-3. |
| Pro | $29/mo | 2,000/mo | 5 seats, 20 agents, 60 render min | Yes | Monthly grant spent before top-ups. |
| Business | $99/mo | 8,000/mo (D-5: Rikki may rebalance) | 25 seats, unlimited agents, 240 render min, voice | Yes | **By invite.** Admin-issued checkout link only. |
| Enterprise | Custom | — | — | — | "Coming soon · Contact Automatos". Not sold through Stripe. |
| Top-up | $10 | 1,000 | — | — | One-off, on any paid plan, valid 12 months. |

**Code keys:**
- Keep the DB key `basic` for Own keys, so no data migration of `workspaces.plan` is needed. `workspaces.plan` defaults to `basic`, and `ENTRY_TIER` and fallbacks reference it **[T]**.
- Display name becomes "Own keys"; add a `platform_ai: false` flag.
- `pro` and `business` stay.
- New per-tier field `monthly_credits`.
- Prices are display fields plus Stripe Price IDs from the environment.

## 4. The credit model

### 4.1 Units

- **Credit:** 1 credit = 1 US cent at Automatos prices.
- **Metered calls (P2, revised 20 Oct):** a platform-funded model call costs `ceil(its cost in cents × CREDITS_MARKUP)` credits. Its cost is `llm_usage.total_cost` — the provider's own figure when it reports one (OpenRouter), else the model's registered price — one price source.
- **$0 that is real:** the customer's own key; a free route (NVIDIA, price multiplier 0); a CLI subscription session (the person's own plan pays); the local edition (never metered).
- **$0 that is not:** a paid route whose price is unknown books $0; the meter charges it at the conservative estimate and logs the model, so its price gets added.
- **Ask:** one user turn (every agent, tool and sub-call it causes) or one run of a mission task, Task, scheduled task, heartbeat or playbook. Asks group calls (receipts: "this cost N credits") and carry an in-flight budget.

### 4.2 Guards

- **Paused at zero:** platform-funded work needs spendable credits; at zero it pauses with a clear way out (add a key, top up, upgrade). Own-key calls keep working.
- **No overdraft by a long run:** an ask's budget is the balance at its first call; once its running cost reaches it, the next call stops.
- **Overhead absorbed (D-3):** routing, memory, retrieval, verification and similar calls are never charged; they are recorded (`charge_class = 'overhead'`, rolled up nightly) for margin review.
- **Alerts:** "running low" at `CREDITS_ALERT_LOW_SHARE` (20%) of the allowance (the month's credits on a paid plan, else the trial), and "run out" at zero — once per crossing, to the workspace's notification destinations.
- **Margin** is the markup plus the plan fee (Pro $29 includes $20 of credits). The ledger records each call's real cost beside its charge.

### 4.3 Buckets and order

- **Buckets:** `trial`, `monthly` and `topup`.
- **Spend order:** trial, then monthly, then top-up. Background work never spends the trial.
- **Expiry:** monthly credits expire at period end; top-ups expire after 12 months.
- **Paused state:** at zero, platform-funded work pauses, and the workspace and own-key calls keep working.

## 5. User stories

Waves:
- **W1 (ledger and gate):** merge by **Fri 30 Oct**.
- **W2 (Stripe):** working in Stripe test mode by **Fri 30 Oct**, live keys by **Wed 4 Nov**.
- **W3 (go-live):** **4–9 Nov**.
- **C (Clerk):** runs in parallel; dates are in the runbook.

Every story: CI is the gate, tests go in with the story, and nothing runs locally.

### W1: ledger and gate

#### US-001: Plan tiers in code match the public sheet
**Description:** As the platform, I need plan tiers that match what we sell, so gating and display agree with the website.

**Acceptance Criteria:**
- [ ] `_PLAN_TIERS_DEFAULTS`: `basic` → "Own keys" $9 with `platform_ai: false` and Pro's limits; `pro` $29 with `monthly_credits: 2000`; `business` $99 with `monthly_credits: 8000`, assignable by admin only; enterprise unchanged.
- [ ] `price_label` "beta" is removed when paid plans open (config flag `BILLING_OPEN`).
- [ ] Every test that pins old prices or limits is updated to the new contract, not deleted (list in the PR body).
- [ ] `AUTOMATOS_PLAN_TIERS_JSON` interim overrides are removed once this merges.

#### US-002: Credit ledger and balances
**Description:** As finance and support, I need an append-only record of every credit movement, so balances are auditable.

**Acceptance Criteria:**
- [ ] One Alembic revision adds `credit_ledger` and `credit_balances` **[R §6]**:
  - **Ledger:** append-only, `kind` grant/topup/debit/refund/adjust/expire, `bucket`, signed `credits`, `expires_at`, unique `(source_type, source_ref)`, `actor`, `reason`.
  - **Balances:** one row per workspace, updated in the same transaction as each ledger insert.
- [ ] Both Alembic head pins move (AGENTS.md).
- [ ] Inserting the same source event twice changes nothing (test).
- [ ] Billing records key on `workspace_id`, never on a Clerk user ID.

#### US-003: Metered credits (revised 20 Oct: tokens × model price, like OpenRouter)
**Description:** As Automatos, I want every platform-funded call charged at its real cost × one markup, so a cheap report and an expensive build are both priced fairly.

**Acceptance Criteria:**
- [ ] The meter debits each customer call once (keyed on its `llm_usage` row) at `ceil(cost in cents × CREDITS_MARKUP)`, from `llm_usage.total_cost` (one price source) **[R leak 5]**.
- [ ] Free routes (NVIDIA), subscription sessions, own keys and the local edition cost nothing; an unpriced paid route is charged at the conservative estimate.
- [ ] Each ledger row carries the band (`metered`), the credits charged and the real cost.
- [ ] An ask stops at its next call once its running cost reaches the balance read at its start.
- [ ] Running-low (20%) and run-out alerts, once per crossing.

#### US-004: The gate in front of platform keys
**Description:** As Automatos, I need every platform-funded call to require credits, so nothing runs on our keys unmetered.

**Acceptance Criteria:**
- [ ] `_resolve_trial_decision` / `resolve_trial_routing` become credit routing **[T]**:
  - Own keys pass.
  - The platform key requires a positive balance, otherwise `CreditsExhaustedError`, rendered the way `TrialExhaustedError` is (`turn_errors.py`).
- [ ] **The Own keys plan has no platform lane at all, including the OpenRouter fallback.** Test: an Own keys workspace without an Anthropic key gets a "add a key" error, not a platform call.
- [ ] Heartbeats, the playbook scheduler and scheduled tasks skip on "no credits", as they skip on "trial exhausted" today.
- [ ] The policy-plane budget excludes `is_byok` rows (`budget.py` `spend_to_date`) and stays the owner's own cap, never an entitlement.

#### US-005: Trial becomes a 500-credit grant
**Description:** As a new user, I get a one-time 500-credit trial shown in credits, matching the website.

**Acceptance Criteria:**
- [ ] A new account's first workspace receives a `trial` grant of 500 credits through the ledger. Once per person, as today; the daily global cap is kept.
- [ ] `_accrue_trial_spend` is retired.
- [ ] The trial pill and Auto's onboarding copy show credits (`trial-balance-pill.tsx`, `onboarding.py`).
- [ ] Existing trial workspaces are migrated: the remaining dollars × 100 become credits, floored.

#### US-006: Close the leaks before charging
**Description:** As Automatos, I need the known free paths closed before we charge anyone.

**Acceptance Criteria (each with a test):**
- [ ] **No trial means no platform AI.** No trial (daily cap reached, trial disabled, or a second workspace) gives the paused state, not `passthrough` on the platform key (`trial_ledger.py:296-301`, `key_resolver.py:36-48`) **[T]**.
- [ ] **No paid plan without payment.** Auto's onboarding (`handlers_onboarding.py:369-371`) produces a checkout link for paid plans and never calls `assign_plan` for one.
- [ ] **Business** has no "no ceiling" path; it gets the monthly grant.
- [ ] **System calls** (31 call sites **[R leak 4]**): each is classified as platform overhead (absorbed, recorded with `bucket=overhead` for margin reporting) or customer work (charged), per D-3. No call site is left unclassified (a test lists them).

#### US-007: Admin credits and Business invites
**Description:** As a platform admin, I can grant or refund credits with a reason, and issue a Business checkout link to one workspace.

**Acceptance Criteria:**
- [ ] `POST /api/v1/admin/workspaces/{id}/credits` writes an `adjust` ledger row with `actor` and `reason`; platform admin only.
- [ ] `POST /api/v1/admin/workspaces/{id}/business-invite` returns a single-workspace checkout link (hidden Stripe Price).
- [ ] Route manifest regenerated; access tests for non-admins (403).

#### US-008: Balance and paused state in the app
**Description:** As a workspace member, I can see my balance and why work paused.

**Acceptance Criteria:**
- [ ] `GET /api/v1/billing/summary` (members): plan, status, renewal date, balance by bucket.
- [ ] `GET /api/v1/billing/ledger` (owner/admin): paged ledger (a line per metered call), plus the last 30 days' credits by model.
- [ ] The header shows the balance. A paused banner appears at zero, with "Top up" or "Add your own key" actions.
- [ ] Verify in the browser on a CI preview or staging, not locally.

### W2: Stripe

#### US-009: Stripe adapter and Checkout
**Description:** As a workspace owner, I can buy a plan or a top-up with a card that needs 3-D Secure.

**Acceptance Criteria:**
- [ ] `services/billing/` defines a provider interface (`create_checkout`, `portal_link`, `parse_webhook`), with a Stripe implementation. A merchant of record can be added later without touching the ledger.
- [ ] Checkout Sessions:
  - `mode=subscription` for Own keys and Pro; Business only through the admin link.
  - `mode=payment` for top-ups.
  - `automatic_tax` on, `tax_id_collection` on, `client_reference_id = workspace_id`.
  - The EU withdrawal-consent text for consumers buying credits (wording from counsel, D-4).
- [ ] Stripe keys, Price IDs and the webhook secret come from the environment only; none in code.
- [ ] `POST /api/v1/billing/checkout` (owner/admin) and `POST /api/v1/billing/portal` (owner/admin).
- [ ] The CSP adds `js.stripe.com`, `checkout.stripe.com` and `billing.stripe.com` where needed.

#### US-010: Webhooks keep plan and credits in step
**Description:** As Automatos, I need Stripe events to change plans and credits exactly once.

**Acceptance Criteria:**
- [ ] `POST /api/v1/billing/webhooks/stripe`: signature-checked, outside Clerk auth. Every event is stored in `billing_events` (unique `event_id`) before processing, and is idempotent.
- [ ] **Event handling:**
  - `checkout.session.completed` and `invoice.paid` grant the monthly credits and set the plan through `assign_plan`.
  - `customer.subscription.updated` / `deleted` handle plan changes and cancellations at period end.
  - `charge.refunded` and `charge.dispute.created` claw credits back (to a negative balance if already spent; work pauses).
  - `invoice.payment_failed` triggers dunning, then a downgrade to the paused state.
- [ ] One Alembic revision adds `billing_accounts` (with a `provider_account` column for a later entity move) and `billing_events`.

#### US-011: Billing tab and pricing surfaces
**Description:** As an owner, I can see my plan, invoices and balance and manage payment.

**Acceptance Criteria:**
- [ ] A Settings → Billing tab shows the plan, renewal date, balance, a ledger, "Change plan", "Top up" and "Manage payment" (portal).
- [ ] The in-app upgrade page uses the facts sheet §4 block.
- [ ] The Studio site pricing section switches to the paid block on the opening deploy (G6).
- [ ] Verify in the browser on staging.

### W3: go-live

#### US-012: Live tests with real cards (Wed 4 – Fri 6 Nov)
**Acceptance Criteria:**
- [ ] Each of these passes end to end, with the ledger and Stripe reconciled:
  - an EU personal card with 3-D Secure;
  - a UK card;
  - an EU business with a VAT number (reverse charge);
  - a top-up;
  - a refund (clawback);
  - a failed renewal (dunning, then paused).
- [ ] The results are recorded in the PR.

#### US-013: Open paid plans (Mon 9 Nov)
**Acceptance Criteria:**
- [ ] `BILLING_OPEN=true` deployed. The public copy flips to the paid block on all surfaces in one deploy.
- [ ] The Business invite path works.
- [ ] A rollback switch (`BILLING_OPEN=false`) restores beta copy and closes checkout without touching balances.

### C: Clerk (dependency, parallel)

#### US-014: Clerk on production before beta traffic
**Acceptance Criteria:**
- [ ] #1198 merged. #1216 decided.
- [ ] The runbook (`.claude/CLERK-PRODUCTION-CUTOVER.md`) is executed by **Tue 27 Oct**:
  - production instance, DNS, OAuth apps;
  - user import and the relink script (dry run, then `--apply`);
  - keys switched on Railway (frontend, API, Studio);
  - sign-in verified.
- [ ] The dev instance's 100-user cap is no longer in the path.

## 6. Functional requirements

- **FR-1:** Every platform-funded AI ask must debit credits per §4. Calls on the customer's own keys are never debited.
- **FR-2:** At a zero balance, platform-funded work pauses. Own-key calls, reading data and the workspace UI keep working.
- **FR-3:** The Own keys plan must never use platform keys for any provider.
- **FR-4:** No code path may put a workspace on a paid plan except a verified Stripe event or a platform admin action.
- **FR-5:** Every credit movement must be a ledger row, unique per source event.
- **FR-6:** Stripe webhooks must be signature-verified and processed exactly once.
- **FR-7:** Refunds and disputes must take back the credits they paid for.
- **FR-8:** VAT is calculated and collected by Stripe Tax. Receipts name the P4 entity.
- **FR-9:** Prices, Price IDs, the markup and the opening flag live in config or the environment, never hard-coded.
- **FR-10:** Billing records key on `workspace_id`.
- **FR-11:** Public copy follows the facts sheet: beta block until `BILLING_OPEN`, then the paid block.

## 7. Non-goals

- Clerk Billing (its gaps are in §1 and §8).
- A merchant of record for the November launch. The adapter keeps that possible later.
- Annual plans, auto top-up, and per-seat Business proration.
- Multi-currency (USD only, D1 of the plan).
- Enterprise billing (contract and invoice, off Stripe Checkout).
- Marketplace creator payouts (they need Stripe Connect; a separate PRD).
- Changing what the public pages say before `BILLING_OPEN`.

## 8. Technical considerations

- **Why not Clerk Billing [R]:**
  - USD only, no VAT, not a merchant of record, no refunds.
  - **No 3-D Secure**, which fails most EU/UK cards under PSD2.
  - No one-off purchases, so top-ups would need Stripe anyway.
  - Two integrations, and the largest code work of any option.
  - Revisit only if Clerk ships all of these.
- **Stripe direct means Automatos is the seller [R]:**
  - VAT registrations are needed: home country plus Union OSS, and UK.
  - Stripe Tax calculates and collects, but Union OSS filing is ours or a partner's.
  - In Portugal, invoices must use certified software.
  - All of this is gated on P4.
- **Clerk cutover first:** the dev instance is capped at 100 users. Billing records must not depend on Clerk user IDs (the relink moves them).
- **Shared files with PRD-264:**
  - Alembic: #1198 and #1208 also add revisions; whoever merges second re-chains and moves both pins.
  - `orchestrator/reports/route-manifest.json`: regenerate it.
  - The changed-lines code-shape gate checks whole enclosing functions; split before editing large ones.
- **OpenRouter is parked** (19 Oct): embeddings and OpenRouter-routed agents are down in production. Platform-funded asks must route to providers that work (Anthropic `claude-*` bare IDs; `anthropic/…` IDs re-route to OpenRouter via `vendor_fit`).
- **Policy plane is off in production:** the budget gate is not an entitlement mechanism; the credit gate (US-004) is.

## 9. Open decisions (Gerard, with Rikki)

| # | Decision | Needed by | Default if not decided |
|---|---|---|---|
| D-1 | The markup on metered calls (`CREDITS_MARKUP`), and render/voice rates | Thu 29 Oct | 1.2× on every model (P2 revised 20 Oct: metered); render and voice unpriced until decided |
| D-2 | **Legal entity (P4).** Which company or sole trader sells, its country, VAT registrations, and who owns the Stripe account | **Tue 27 Oct** (Stripe live keys) | Without it, 9 Nov runs on trial and admin grants only; no checkout |
| D-3 | Background and system calls: absorb as overhead, or charge to the customer | **Decided 20 Oct (Gerard): absorb** | Absorbed and recorded as overhead (`bucket = overhead`); margins reviewed after two weeks |
| D-4 | Terms v2.3 (seller, renewal, cancellation, withdrawal, credit terms) and counsel | Mon 2 Nov | No checkout until reviewed |
| D-5 | Business credits: keep 8,000 or rebalance (Rikki: fewer credits, more non-token value) | Thu 29 Oct | Keep 8,000 |
| D-6 | Does voice ship in the beta, and at what per-minute rate | Thu 29 Oct | Voice stays Business-only and unpriced until decided |
| D-7 | Prices shown including or excluding VAT for EU consumers (about $5 difference per Pro sale **[R]**) | Thu 29 Oct | Excluding VAT for business buyers; including for consumers |
| D-8 | Beta users on their own keys: offered the $9 plan on 9 Nov, or kept free for a period | Fri 6 Nov | Offered, with 30 days free |

## 10. Timeline

| By | What | Who |
|---|---|---|
| Tue 20 – Fri 23 Oct | D-3 decided (20 Oct: absorb); W1 build starts (20 Oct) | Gerard; engineering |
| Tue 27 Oct | **D-2 entity**; Stripe account live mode; **Clerk cutover** (US-014) | Gerard |
| Thu 29 Oct | D-1, D-5, D-6, D-7 decided | Gerard + Rikki |
| Fri 30 Oct | W1 merged; W2 working in Stripe test mode | Engineering |
| Mon 2 Nov | Terms v2.3 live (D-4) | Gerard + counsel |
| Tue 3 Nov | Product Hunt; beta on trial + own keys; nothing charged | — |
| Wed 4 – Fri 6 Nov | Live card tests (US-012) | Gerard + engineering |
| **Mon 9 Nov** | **Paid plans open** (US-013) | Gerard |
| Tue 10 – Thu 12 Nov | Web Summit | — |

**Fallback:** if D-2 or Stripe isn't ready by 6 Nov, Web Summit runs on trial credits and admin grants. The ledger means nothing is lost, and checkout opens when ready.

## 11. Success metrics

- 100% of platform-funded `llm_usage` rows have a matching ledger debit, so the nightly reconciliation finds 0 misses after day 3.
- 0 platform-key calls from Own keys workspaces, from workspaces with a zero balance, or without a trial (production check by query).
- Each Stripe event's effect on the ledger is applied once (idempotency test plus a production duplicate count of 0).
- All live-card test cases pass before 9 Nov.
- Gross margin per model and per plan is visible from day 1 (credits charged against real cost on every ledger row).

## 12. Size [R, I]

| Wave | Contents | Engineer-days |
|---|---|---|
| W1 | US-001 to US-008 (metered per call since 20 Oct) | 8–12 |
| W2 | US-009 to US-011 | 4–6 |
| C | Clerk cutover (code merged; runbook steps) | 1–2 |
| W3 | Live tests and opening | 1–2 |
