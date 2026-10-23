# Billing, plans and credits: how it works

**For:** the Automatos team: founders, support, and engineers new to billing. **Applies to:** hosted Studio only; the local edition never bills.
**Built by:** PRD-265 W1 (#1259, the credit ledger) and W2 (#1280, #1283, Stripe). **Last checked against the code:** 23 Oct 2026.

Related:
- [Developer architecture](architecture.md)
- [Stripe setup runbook](../runbooks/stripe-billing.md) (test mode, going live, rolling back)
- [PRD-265](../PRDS/PRD-265-BILLING-PLANS-CLERK-STRIPE.md) (decisions and open questions)

---

## In one page

- **Who does what:**
  - **Clerk** signs people in.
  - **Stripe** takes the money.
  - **Automatos** keeps a **credit ledger**: every AI call made on Automatos's own AI keys is charged in credits.
- **One credit = one US cent** of AI usage at Automatos prices. A call costs its real cost × a markup (1.2 by default), rounded up.
- **The customer never leaves Studio.**
  - Buying, changing card, invoices, cancelling and changing plan all happen on **Settings → Billing**.
  - Stripe's secure card fields sit inside our own form. A bank's 3-D Secure check is the only thing that may appear over the page.
- **Paid plans open on Mon 9 Nov 2026.**
  - Until then, `BILLING_OPEN=false`: customers see beta prices and can't buy, while platform admins can buy for testing.
  - Everyone runs on the trial and then their own AI keys.

## Plans

| Plan | Price | Monthly credits | AI on Automatos's keys | Notes |
|---|---|---|---|---|
| **Own keys** | $9/month | none | No, only while trial credits last | The same service as Pro (seats, agents, render minutes). The AI runs on the customer's own Anthropic, OpenAI or OpenRouter keys. |
| **Pro** | $29/month | 2,000 | Yes | |
| **Business** | $99/month | 8,000 | Yes | **By invite:** a platform admin offers it to one workspace. 25 seats, unlimited agents, voice. |
| **Enterprise** | Custom | n/a | Yes, on an allowance | "Coming soon". Not sold through Stripe, and its calls are recorded but never blocked. |
| **Top-up** | $10 | +1,000 | n/a | One-off, valid for 12 months, sold only on a live Pro or Business plan. |

- Every new workspace starts on **Own keys**, the entry plan; its database key is `basic`.
- The numbers live in `orchestrator/config.py` (`_PLAN_TIERS_DEFAULTS`). The Stripe Prices are created from the same numbers by the setup script.

## Credits

**What is charged:**
- Only calls on **Automatos's own AI keys** are charged.
- These cost nothing: calls on the customer's own keys, free models (NVIDIA), Claude Code / CLI subscription sessions, and the local edition.
- Background platform work (routing, memory, retrieval, checks) is **absorbed**. We record it as overhead for margin review, but never charge the customer for it.

**Where credits sit:** three buckets, spent in this order.
1. **Trial:** 500 credits, once per person, granted with their first workspace. Trial credits pay for AI on any plan, including Own keys. Scheduled background work never spends them.
2. **Monthly:** the plan's month of credits, granted when Stripe confirms the monthly payment. Whatever is left expires at the end of the paid month.
3. **Top-up:** bought packs, each valid for 12 months. They are spent oldest first, and only on Pro and Business.

**Running low and running out:**
- At **20% left** the owner gets a "running low" notification, and the header balance turns amber.
- At **zero**, AI work on Automatos's keys **pauses**. The owner gets a "run out" notification, and chat shows a banner offering "Top up" or "Add your own key".
- The workspace itself, and any calls on the customer's own keys, keep working.
- A long job can't overspend: an ask stops at its next AI call once it has used the balance it started with.

## A customer's journey

1. **Signs up.** The first workspace gets the 500-credit trial and the Own keys plan.
2. **Uses Auto and agents.** Each AI call is charged within about a minute (the meter runs every 60 s). The header shows the balance. The Billing tab's ledger shows each charge, plus 30 days of usage by model.
3. **Trial runs out.**
   - On Own keys: they add their own AI key in Settings, and everything keeps working at no charge.
   - Or they buy a plan.
4. **Buys Pro** (from 9 Nov):
   1. Settings → Billing → Pro → the card form opens in the page.
   2. They enter card, billing address and an optional VAT number. Stripe Tax adds VAT, or applies reverse charge for a valid EU business VAT number.
   3. They pay. Stripe tells our server, and within seconds the plan is Pro and 2,000 monthly credits are in.
5. **Every month.** Stripe charges the card. Last month's leftover expires, and a fresh 2,000 credits arrive.
6. **Tops up** when they need more: $10 → 1,000 credits, kept for a year.
7. **Changes plan** in the Billing tab:
   - **Upgrade:** the tab shows the prorated price first. The change is charged at once, and credits for the rest of the month are added.
   - **Downgrade:** nothing is charged. The change happens at the end of the month, and "Keep my plan" undoes it.
8. **Card trouble.**
   - If a renewal fails, the account shows "payment failed" and the owner is notified. Stripe retries on its schedule.
   - If every retry fails, Stripe cancels. The workspace goes back to Own keys, the month's credits expire, and AI on our keys pauses.
9. **Cancels** in the Billing tab. The plan runs to the end of the paid month, then goes back to Own keys. "Resume plan" undoes it.
10. **Refunds and chargebacks** are done in the Stripe Dashboard. Credits for the refunded share are taken back automatically, even if that takes the balance below zero; a dispute we win gives them back.

## Where it shows in the app

| Place | What it shows |
|---|---|
| **Settings → Billing** (`/settings?tab=billing`) | Plan, status, renewal date, balance by bucket, the plan picker, top-up, the card on file, invoices (PDF), cancel/resume, the ledger and usage by model. Members see the plan and balance; owners and admins can buy and manage. |
| **Header** | The credit balance pill: amber when low, "Paused" at zero. |
| **Chat** | The paused banner, with "Top up" or "Add your own key". |
| **Notifications** | Running low, run out, payment failed, plan ended, Business offered. |
| **Auto** | On hosted Studio, Auto knows the Billing tab exists and sends owners there. Auto never puts anyone on a paid plan: only a verified Stripe payment or a platform admin can. |

## What admins can do

- **Offer Business:** operator console → workspaces → **Offer Business**. The owner gets a notification and can buy Business on their Billing tab for 14 days.
- **Grant or take back credits:** `POST /api/admin/workspaces/{id}/credits` with an amount, a bucket and a reason. It is recorded in the ledger as "Adjustment by Automatos". *There is no button for this in the console yet; it's an API call.*
- **Refund, cancel or see a payment:** in the Stripe Dashboard. The app follows within seconds.
- **Test purchases before 9 Nov:** platform admins can buy even while billing is closed.

## Before and after 9 Nov

| | Now (`BILLING_OPEN=false`) | From 9 Nov (`BILLING_OPEN=true`) |
|---|---|---|
| Who can buy | Platform admins only | Every workspace owner or admin |
| Prices shown | Labelled "beta", with the beta wording | The paid wording |
| Credits | The trial, then own keys | Trial, monthly, top-ups |
| Rolling back | — | Set `BILLING_OPEN=false`: buying stops, while balances, plans and subscriptions stay. |

## Connecting Stripe (summary)

There is no setup page in the app. Stripe is connected with environment variables on the Railway API service (`automatos-ai-api`), by design: keys never live in app settings.

| Variable | Where it comes from |
|---|---|
| `STRIPE_SECRET_KEY` | Stripe → Developers → API keys (`sk_test_…` / `sk_live_…`) |
| `STRIPE_PUBLISHABLE_KEY` | Same page (`pk_test_…` / `pk_live_…`). The card form needs it. |
| `STRIPE_WEBHOOK_SECRET` | Stripe → Developers → Webhooks → our endpoint → Reveal |
| `STRIPE_PRICE_BASIC`, `_PRO`, `_BUSINESS`, `_TOPUP`, `STRIPE_ACCOUNT_ID` | Printed by `python -m scripts.stripe_setup` |
| `BILLING_OPEN` | `false` until 9 Nov |
| `BILLING_WITHDRAWAL_CONSENT_TEXT` | Counsel's EU withdrawal wording (D-4); empty means no consent box |

**Test and live** use the same variables with different values. Do test mode first; switch to live after the entity decision (D-2). The [runbook](../runbooks/stripe-billing.md) has every step and the test cards.

Without `STRIPE_SECRET_KEY`, billing is "not configured": buying answers "Billing isn't set up yet" and nothing breaks.

## Open decisions and known gaps

These are tracked in [PRD-265 §9](../PRDS/PRD-265-BILLING-PLANS-CLERK-STRIPE.md).
- **D-2, the legal entity:** gates live payments.
- **D-7, VAT:** whether prices include or exclude VAT. A live Stripe Price can't change later.
- **D-4, counsel's terms and consent text:** due before opening.
- **D-1, D-5, D-6:** the markup, Business credits, and voice pricing.
- **D-8, Own keys isn't charged yet:** nothing makes a workspace pay $9 to stay on Own keys. It's the free default during the beta. What happens to beta workspaces on 9 Nov is still open.
- **No console button** for credit grants (API only), and **no "Stripe connected" indicator** in the operator console.
