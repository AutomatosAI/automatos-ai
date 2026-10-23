# Billing architecture (developer guide)

**For:** engineers changing billing, credits or plans. Read the [overview](README.md) first for what the product does.
**Code:** `orchestrator/services/credits/` (the ledger, from PRD-265 W1) and `orchestrator/services/billing/` (Stripe, W2). **Last checked:** 23 Oct 2026, on `main` after #1283.

---

## The pieces

```
                       ┌──────────────── Settings → Billing (frontend/components/settings/billing/) ──────────────┐
 browser ──apiClient──▶│ GET  /api/v1/billing/summary | plans | ledger       (api/billing.py)                      │
                       │ POST /api/v1/billing/checkout          → in-page Checkout Session client secret            │
                       │ card · invoices · cancel · plan change (api/billing_manage.py)                             │
                       └───────────────────────────────┬─────────────────────────────────────────────────────────────┘
 Stripe.js card fields (iframes in our form) ──────────┼──▶ Stripe ── webhook ──▶ POST /api/v1/billing/webhooks/stripe
                                                       │                           (api/billing_webhooks.py)
                                                       ▼                                       │
                                 services/billing/  (checkout, manage, provider, stripe_*)     ▼
                                                       │                     events.receive → apply_fact → grants
                                                       ▼                                       │
 every model call ─▶ LLMManager ─▶ credits/gate ─▶ llm_usage row ─▶ credits/meter (60 s) ─▶ credits/ledger.record
                                    (pause at 0)     (charge_class)                          (credit_ledger + credit_balances)
```

## Data

| Table | What | Written by |
|---|---|---|
| `credit_ledger` | Append-only, one row per movement per bucket. Columns: `kind` (grant, topup, debit, refund, adjust, expire), `bucket` (trial, monthly, topup, overhead), signed `credits`, `source_type` + `source_ref`, and for each metered call `charged_credits` / `actual_cost_usd`. Unique on `(source_type, source_ref, bucket)`. | `services/credits/ledger.record` only |
| `credit_balances` | One row per workspace: `trial`, `monthly`, `topup`, `period_end` (when the monthly credits expire), `alert_level`. It moves in the same transaction as every ledger insert. | `ledger.record`, `grants.grant_period` (`period_end`), `alerts.check_alerts` |
| `llm_usage.charge_class` / `ask_id` | `customer` until metered, then `charged`; `overhead` is absorbed. `ask_id` groups a turn's calls. | Stamped by a `before_insert` listener in `core/models/core.py`, from `usage_scope` |
| `billing_accounts` | One row per workspace that has a Stripe customer. Holds: the subscription id and status, the paid plan, `current_period_end`, `cancel_at_period_end`, the credited period (`credited_period_start`/`_end` + `period_credits`), a scheduled downgrade (`schedule_id`, `scheduled_plan`, `scheduled_at`), and `business_offer_until`. | `services/billing/` (`accounts`, `apply`, `checkout`, `manage`) |
| `billing_events` | Every verified webhook, unique on `(provider, event_id)`, stored before it is processed. Also holds `status`, `payment_ref` and `grant_source` (how a refund finds its grant) and `payload` (JSONB). | `services/billing/events.py` |

Migrations: `prd265_credit_ledger` (W1), `prd265_billing` (W2, already applied in production), and `prd265_billing_in_studio` (four nullable `billing_accounts` columns).

## How credits are charged (W1)

1. **Classification** (`credits/classification.py`): every `LLMManager` has a purpose.
   - Overhead purposes (routing, memory, retrieval, checks) are absorbed (decision D-3).
   - Anything else is customer work. An unknown purpose is treated as customer work, and `tests/test_prd265_call_site_classes.py` fails on an unclassified call site.
2. **The gate** (`credits/gate.py`, called by `LLMManager` before every model call):
   - Waved through: own-key calls (`is_byok`), the local edition, overhead, and allowance plans (enterprise).
   - Otherwise the workspace needs spendable credits, or the call raises `CreditsExhaustedError` (`error_code = "credits_exhausted"`, which the chat renders as the paused banner).
   - Inside an **ask** (`credits/asks.py`: a chat turn, or a run of a mission, Task, heartbeat or playbook), the balance read at the first call is the ask's budget. Once the ask's running cost reaches it, the next call stops.
3. **Recording:** `LLMManager._record_usage` writes the `llm_usage` row under `usage_scope(charge_class=…)`, and the listener stamps `charge_class` and `ask_id`.
4. **The meter** (`credits/meter.py`) runs every `CREDITS_METER_INTERVAL_SECONDS` (60) on the unified scheduler (`credits/jobs.py`). For each `customer` row:
   - It debits `ceil(total_cost × 100 × CREDITS_MARKUP)` credits (`routing.credits_for`), keyed on the row's id.
   - A paid route with an unknown price is charged at the conservative estimate, with a warning naming the model.
   - It marks the row `charged`.
5. **Who pays** (`credits/routing.py`):
   - Trial credits pay on any plan.
   - Monthly and top-up credits pay only on a plan with `platform_ai` (Pro, Business), never on Own keys.
   - Background work never spends the trial.
   - The spend order is trial → monthly → top-up.
6. **Alerts** (`credits/alerts.py`): "running low" at `CREDITS_ALERT_LOW_SHARE` (20%) of the allowance, and "run out" at zero. Each fires once per crossing, through `NotificationDispatcher`.
7. **The trial** (`credits/trial.py`): 500 credits once per person (keyed on the owner), behind the `TRIAL_ENABLED` kill switch and the `TRIAL_GLOBAL_DAILY_CREDITS` cap.
8. **Overhead** (`meter.record_overhead`) is rolled up nightly at 03:15 UTC, for margin review.

## How Stripe moves plans and credits (W2)

**The provider seam**
- `services/billing/provider.py` defines `BillingProvider`. It covers the customer, the in-page checkout, webhook parsing and translation, and the in-app management calls.
- Stripe implements it in `stripe_provider.py` (checkout, webhooks) plus `stripe_manage.py` (card, invoices, cancel, plan change). It talks to Stripe through `stripe_api.py`, a thin `httpx` client; there is no Stripe SDK, so the hash-locked `requirements.lock` is unchanged.
- Stripe's object shapes are read only in `stripe_events.py`, which turns events into provider-neutral facts (`facts.py`). A merchant of record added later writes its own translator and reuses `apply.py`, `grants.py` and the storage.

**The API version**
- `STRIPE_API_VERSION = 2026-09-30.endive`, sent as the `Stripe-Version` header on every request and set on the webhook endpoint when the setup script creates it.
- It must stay on the same release train as the frontend's `@stripe/stripe-js@10` / `@stripe/react-stripe-js@7`: each Stripe.js major is bound to one train.
- The readers also accept the older (basil) shapes.

**Buying (in-page; nothing redirects)**
1. `POST /checkout` → `checkout.start_plan_checkout` / `start_topup_checkout`. The rules:
   - Before `BILLING_OPEN`, only platform admins (`checkout_open_for`).
   - Plans: Own keys and Pro; Business only while `business_offer_until` is in the future.
   - No second subscription: a live one answers 409, and the customer changes plan instead.
   - Top-ups need a live Pro/Business plan.
   - When `BILLING_WITHDRAWAL_CONSENT_TEXT` is set, `accept_terms` is required and stamped as `metadata.terms_accepted_at`.
2. `ensure_customer` creates the Stripe customer once and links `billing_accounts`. That happens only here and in `offer_business`, so a card change can never create a customer.
3. `stripe_provider.checkout_params` builds the session:
   - `ui_mode=elements`, `return_url`, `automatic_tax`, `tax_id_collection`, `customer_update` (address and name `auto`), `billing_address_collection=required`;
   - `client_reference_id` and the metadata name the workspace;
   - subscriptions use `billing_mode=classic`; top-ups get `invoice_creation` + `payment_intent_data`.
4. The browser mounts `CheckoutElementsProvider` with the client secret (`frontend/components/settings/billing/checkout-panel.tsx`):
   - `BillingAddressElement`, `PaymentElement`, our VAT field (`updateTaxIdInfo`) and the totals;
   - `confirm({redirect: 'if_required'})`.

   Stripe.js loads lazily (`@stripe/stripe-js/pure`, `frontend/lib/stripe.ts`).

**Webhooks** (`events.receive`)
1. `StripeProvider.parse_webhook` verifies the `Stripe-Signature` HMAC over the raw body (`stripe_signature.py`, 5-minute tolerance) and checks the livemode against the key. The body is capped at 1 MB on the wire (`main.PATH_BODY_LIMITS`).
2. `store` inserts the event and commits (a redelivery inserts nothing). `process` locks the row and skips a settled event. It then runs `translate` → `apply.apply_fact` → `_settle` in one transaction. After the commit come the notices (`notices.py`) and the alert re-check.
3. On a failure the row is marked `failed`, the route answers 500, and Stripe redelivers.

Events handled (`stripe_events.WEBHOOK_EVENTS`): `checkout.session.completed` / `async_payment_succeeded`, `invoice.paid`, `invoice_payment.paid`, `invoice.payment_failed`, `customer.subscription.created` / `updated` / `deleted`, `charge.refunded`, `charge.dispute.created` / `closed`.

**Grants** (`grants.py`)
- **A paid subscription invoice** (`grant_period`) is the only thing that puts a workspace on a paid plan (`assign_plan`, FR-4).
  - In a new period it expires last month's leftover (`credits/expiry.expire_monthly`), then grants the plan's month.
  - In the same period, an upgrade grants `(new month − already granted) × share of the period left`. The share is `share_left`, from the invoice line's start, so a last-day upgrade can't buy a whole month.
- **A top-up** grants 1,000 credits with `expires_at` 12 months out.
- **A refund or dispute** takes back the paid-for share of the grant, using the running refund total, even into a negative bucket. A dispute won restores it.
  - A refund finds its grant through `billing_events.payment_ref`.
  - The basil+ API leaves the payment out of the `invoice.paid` payload. So a subscription payment is linked to its invoice by the `invoice_payment.paid` event.
- **A subscription that ends** (deleted, or `unpaid` / `canceled`) moves the workspace back to `basic`, expires the month and notifies the owner.

**Expiry** (`credits/expiry.py`, hourly job)
- A month expires `BILLING_MONTHLY_EXPIRY_GRACE_HOURS` (24) after its period end when no renewal came. The expiry key `(workspace, period end)` is shared with the renewal path, so it happens once.
- Top-up packs expire after their year. The amount still left in a pack is worked out spending the oldest pack first (`_left_per_pack`), and each expired pack gets a marker row.

**Managing, in the tab** (`manage.py` → `stripe_manage.py`, routes in `api/billing_manage.py`, all `require_workspace_admin`)
- **Card:** `card_on_file` (the subscription's default, else the customer's). To change it: `start_card_change` (a SetupIntent), then `finish_card_change`, which checks the setup belongs to this customer and sets it as the default on the customer and the subscription.
- **Invoices:** listed from Stripe. Each PDF is fetched server-side by `StripeApi.fetch_file`: `https://*.stripe.com` only, re-checked after redirects, no `Authorization` header, a 10 MB cap. Our API serves it to the browser.
- **Cancel / resume:** `cancel_at_period_end`. Cancelling releases a scheduled downgrade first.
- **Plan change:**
  - **Upgrade:** `preview_plan_change` (`invoices/create_preview`) returns `proration_date`. The client sends it back, so `upgrade_now` charges exactly what was shown (accepted while under an hour old).
  - `upgrade_now` uses `payment_behavior=pending_if_incomplete`. If the bank wants 3-D Secure, it returns `latest_invoice.confirmation_secret` and the tab calls `stripe.handleNextAction`.
  - **Downgrade:** a subscription schedule (`from_subscription`, then two phases, `end_behavior=release`). "Keep my plan" calls `/release`.
- **Business offer:** `checkout.offer_business` sets `business_offer_until` (`BUSINESS_OFFER_DAYS`) and notifies the owner.

## Routes

| Route | Who |
|---|---|
| `GET /api/v1/billing/summary` | Any member (the header pill and the paused banner) |
| `GET /api/v1/billing/plans` | Any member: plans, top-up, subscription, publishable key, consent text, Business offer |
| `GET /api/v1/billing/ledger` | Owner/admin |
| `POST /api/v1/billing/checkout` | Owner/admin; pre-opening, platform admins only |
| `GET/POST /api/v1/billing/payment-method`, `POST …/payment-method/setup` | Owner/admin |
| `GET /api/v1/billing/invoices`, `GET …/invoices/{id}/pdf` | Owner/admin |
| `POST /api/v1/billing/subscription/cancel` / `resume` | Owner/admin |
| `POST /api/v1/billing/plan-change/preview`, `…/plan-change`, `…/plan-change/cancel` | Owner/admin |
| `POST /api/v1/billing/webhooks/stripe` | Stripe (signature only, no Clerk auth) |
| `POST /api/admin/workspaces/{id}/credits` | Platform admin: grant or take back credits |
| `POST /api/admin/workspaces/{id}/business-invite` | Platform admin: offer Business |

All billing routes answer 404 in the local edition.

## Configuration

Everything is set in `orchestrator/config.py` and overridable by environment variable:

- **Stripe:** `STRIPE_SECRET_KEY`, `STRIPE_PUBLISHABLE_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_API_VERSION`, `STRIPE_PRICE_BASIC/PRO/BUSINESS/TOPUP`, `STRIPE_ACCOUNT_ID`, `STRIPE_WEBHOOK_TOLERANCE_SECONDS`, `STRIPE_TIMEOUT_SECONDS`.
- **Billing:** `BILLING_OPEN`, `BILLING_WITHDRAWAL_CONSENT_TEXT`, `BILLING_MONTHLY_EXPIRY_GRACE_HOURS`, `BUSINESS_OFFER_DAYS`.
- **Credits and the trial:** `CREDITS_MARKUP`, `CREDITS_METER_INTERVAL_SECONDS`, `CREDITS_ALERT_LOW_SHARE`, `TOPUP_CREDITS` / `TOPUP_PRICE_USD` / `TOPUP_VALID_DAYS`, `TRIAL_ENABLED`, `TRIAL_CREDITS`, `TRIAL_GLOBAL_DAILY_CREDITS`.
- **Plans:** `AUTOMATOS_PLAN_TIERS_JSON` overrides the tier numbers. Each name must also be listed in `orchestrator/reports/config-surface.json`.

## Tests

- **Real Postgres:** `test_prd265_credit_ledger_realdb.py`, `test_prd265_asks_realdb.py`, and `test_prd265_w2_billing_realdb.py`. The last drives whole Stripe events through to the ledger: renewals, upgrades, top-ups, refunds, disputes, ended plans, retries and the sweep.
- **Unit and route:** `test_prd265_*.py` and `test_prd265_w2_*.py`. They cover the client, signatures, the translator in both API shapes, checkout params, management calls, service rules, every route's auth and edition, the setup script against a fake Stripe, and the proration share.
- **Frontend:** `frontend/components/settings/billing/__tests__/`, `frontend/hooks/__tests__/use-billing-*.test.tsx`, `frontend/lib/__tests__/{stripe,stripe-csp,stripe-appearance,api-client-blob}.test.ts`. Stripe.js is mocked in `stripe-mocks.tsx`.

## Traps (each one cost a CI round)

- **Never edit a migration that has been merged.** `main` deploys to production within minutes. Add a new revision with `ADD COLUMN IF NOT EXISTS`, and move both head pins (`test_prd209_alembic_single_head.py`, `test_prd236_w1_routes.py`).
- **CodeQL `py/clear-text-logging-sensitive-data` reacts to names.** A variable called `billing`, `account` or `*secret*` inside a printed f-string fails the PR. Never print a secret: the setup script points to the Dashboard instead.
- **gitleaks** flags high-entropy fixture strings next to `key`/`secret`. Keep fake keys short (`pk_test_1`, `whsec_t1`).
- **The changed-lines code-shape gate** fails any edited line inside a function over 50 lines (or a component over 150). Big files (`api-client.ts`, the admin workspaces page) are reached through small new modules.
- **Signed-webhook tests** must sign when the test runs, not at module import: collection can be minutes earlier than the 5-minute tolerance allows.
- **Guard tests to keep in step:**
  - the route manifest (`python -m scripts.dump_routes`, or hand-edit and keep `route_count` equal to `len(routes)`);
  - the authz sweep (`test_p2w2_authz_boundary_sweep.py`; the webhook is classified "own auth");
  - `config-surface.json`;
  - F232's Settings-tab pin (`test_f232_auto_knows_what_automatos_is.py`).
- **Stripe API shapes:** basil removed `invoice.payment_intent` / `charge.invoice` and moved line prices to `pricing.price_details.price`. Endive renamed the `ui_mode` values (`elements`, not `custom`) and made `billing_cycle_anchor` an object.

## How to …

- **Change a price.** Stripe Prices can't be edited.
  1. Change `_PLAN_TIERS_DEFAULTS`.
  2. Archive the old Price in Stripe.
  3. Re-run `python -m scripts.stripe_setup`; the new Price takes the lookup key.
  4. Set the printed `STRIPE_PRICE_*`. Existing subscribers stay on the old Price until moved.
- **Change the markup.** Set `CREDITS_MARKUP` (decision D-1); it affects calls metered from then on.
- **Handle a new Stripe event.** Add it to `stripe_events.WEBHOOK_EVENTS` with a reader that returns a fact, handle the fact in `apply._HANDLERS`, re-run the setup script (it updates the endpoint's events) and add a real-Postgres test.
- **Add a merchant of record.** Implement `BillingProvider` and a translator to the same facts, then pick the provider in `api/billing.billing_call`. `billing_accounts.provider_account` already records which account a customer lives in.
