# Stripe billing: set up, test, go live (PRD-265 W2)

**For:** whoever holds the Stripe account. **Time:** about 30 minutes in test mode.

**How the parts fit:**
- **The customer never leaves Studio.**
  - The card form is part of Settings → Billing, drawn in our theme. The card-number fields inside it are Stripe's secure fields, so card numbers never touch our servers.
  - When a bank asks for 3-D Secure, its check appears over our page or in the customer's banking app.
  - Changing card, downloading invoices, cancelling and changing plan are all buttons in the same tab. There is no Stripe page and no Stripe portal.
- **Stripe takes the money:**
  - Stripe Tax works out VAT, and it takes a VAT number from business buyers.
  - Webhooks are how Stripe tells our API that money moved. The API then sets the plan and grants the credits.
- **`orchestrator/scripts/stripe_setup.py` makes the Stripe account match our config:**
  - four Prices: Own keys $9 a month, Pro $29 a month, Business $99 a month, and a top-up of 1,000 credits for $10;
  - the webhook endpoint, at the API version our code reads (`2026-09-30.endive`).
- **The script is safe to re-run.** It creates only what's missing. At the end it prints the variables to set on Railway, and it never prints a key or a secret.

Stripe's Dashboard moves its menus around. If a setting isn't where this page says, use the Dashboard's search box.

## Test mode

1. **Stripe account.** Sign in at dashboard.stripe.com, or create an account. Test mode needs no company details. Switch to test mode (the "Test mode" toggle, or a sandbox) and stay in it for every step below.
2. **Stripe Tax.** Go to Tax. Set the head-office address to the home country and turn Stripe Tax on. Add a registration for that country under Tax → Registrations, so that checkout shows VAT. Nothing is filed in test mode.
3. **Failed payments.** Go to Billing settings, then the retries section (Revenue recovery → Retries on newer dashboards).
   - Turn on **Smart Retries**.
   - Set "if all retries fail" to **cancel the subscription**.
   - When Stripe cancels the subscription, the app moves the workspace back to Own keys and pauses platform AI. Own-key calls keep working.
4. **Emails.** In the customer-email settings, turn on Stripe's emails for **failed payments** and for **receipts** (successful payments and refunds). The app also tells the owner in its notifications. Stripe sends few or no customer emails in test mode, so check each customer's page in the Dashboard instead.
5. **Apple Pay and Google Pay (optional).** Under Settings → Payment methods → Payment method domains, add the app's domain (for example `studio.automatos.app`). The card form then offers those wallets too.
6. **Keys on Railway.**
   - In Stripe, go to Developers → API keys.
   - Copy the **secret key** (`sk_test_…`) into the API service's (`automatos-ai-api`) Variables as `STRIPE_SECRET_KEY`.
   - Copy the **publishable key** (`pk_test_…`) into the same service as `STRIPE_PUBLISHABLE_KEY`. It is safe to show in a browser; the card form needs it.
   - Check that `BILLING_OPEN` is `false` or unset.
   - Railway redeploys. Never paste the secret key into chat, a file or a ticket.
7. **Dry run.** Open a shell on the API service and preview:
   ```
   railway ssh --service automatos-ai-api
   cd /app
   python -m scripts.stripe_setup --webhook-url https://<API public host>/api/v1/billing/webhooks/stripe --dry-run
   ```
   It lists what it would create and writes nothing. `<API public host>` is the API service's public domain (Railway → the service → Settings → Networking).
8. **Real run.** Run the same command without `--dry-run`. It ends with a block of `NAME=value` lines. The script never prints the webhook signing secret. Get it in Stripe under Developers → Webhooks → the new endpoint → Signing secret → Reveal, and set it as `STRIPE_WEBHOOK_SECRET`.
9. **Values on Railway.**
   - Put every printed line on the API service's Variables: `STRIPE_PRICE_BASIC`, `STRIPE_PRICE_PRO`, `STRIPE_PRICE_BUSINESS`, `STRIPE_PRICE_TOPUP` and `STRIPE_ACCOUNT_ID`. Add `STRIPE_WEBHOOK_SECRET` from the Dashboard.
   - **Leave `BILLING_OPEN=false`.** Until it is true (9 Nov), only platform admins can buy.
   - Railway redeploys.
10. **Webhook check.** In Stripe, go to Developers → Webhooks and open the endpoint. It lists 11 events. After the tests below, every delivery should show `200`.
11. **Try it** as a platform admin. In the app, go to Settings → Billing and choose Pro. The card form opens in the tab. Use any future expiry date, any CVC and any postcode.

| Card | What happens | What to check |
|---|---|---|
| `4242 4242 4242 4242` | Pays, without leaving the page. | **Settings → Billing:** the plan is Pro and the balance shows 2,000 monthly credits.<br>**Ledger:** a "Monthly credits" row.<br>**Top-up:** buy one next; you should see +1,000 credits and a "Top-up" row. |
| `4000 0027 6000 3184` | The bank's 3-D Secure check appears over the page. | **Complete:** the same as the first card.<br>**Fail:** no plan change and no credits; the form says why. |
| `4000 0000 0000 0341` | The card saves, but every charge on it fails. | A renewal that fails (below). |

**A renewal that fails:**
1. Subscribe with `4242…`.
2. In Settings → Billing → Payment, choose **Change card** and save `…0341`.
3. Make Stripe renew now, from the Dashboard's Workbench shell or the Stripe CLI:
   `stripe post /v1/subscriptions/<sub_…> -d "billing_cycle_anchor[type]=now" -d proration_behavior=none`
   (on this API version the anchor is an object.)
4. The charge fails, and the subscription becomes past due.
   - The app marks the account past due and tells the owner.
   - Stripe retries on its schedule and, in the end, cancels the subscription.
   - The full schedule takes days. To see the end state now, cancel the subscription in the Dashboard.

If `…0341` is used straight in the card form, the first payment fails there and nothing changes in the app.

**Also try, all in Settings → Billing:**
- **Switch Own keys to Pro:** the tab shows the prorated price first. The upgrade is charged at once and the credits follow (prorated by the time left).
- **Switch Pro to Own keys:** nothing is charged; the tab says the change happens at the period end, and **Keep my plan** undoes it.
- **Cancel plan:** the plan stays until the period end, and **Resume plan** undoes it.
- **Invoices:** each one downloads as a PDF from the tab.
- **A refund:** in Stripe, go to Payments, open the payment and refund it. The ledger shows "Refund", and the credits are taken back.
- **Business:** in the operator console, choose **Offer Business** for a workspace. Its owner gets a notification and can buy Business on their Billing tab for 14 days.

## When the script stops or warns

- **`REFUSED`**
  - You ran a live key without `--live`, or `--live` with a test key.
  - Or the webhook URL is not `https://…/api/v1/billing/webhooks/stripe`.
- **`PRICE MISMATCH`**
  - A Price in Stripe charges a different amount (or currency, interval or tax behaviour) from our config. Stripe Prices can't be edited, so the script changed nothing.
  - To fix it, archive that Price (Product catalogue → the product → the price → Archive) and re-run. The new Price takes over the lookup key, and its new ID goes on Railway.
  - Customers already on the old Price stay on it until they are moved.
- **`WARNING … API version`**
  - The endpoint sends events at a different API version from the one our code pins.
  - To fix it, delete the endpoint under Developers → Webhooks, re-run, and set the new `STRIPE_WEBHOOK_SECRET`.
- **Lost the webhook secret?** Go to Developers → Webhooks → the endpoint → Signing secret → Reveal. Or delete the endpoint and re-run.

## Going live (after decision D-2, the legal entity)

1. **Activate the account.** Activate live mode on the Stripe account with the entity's details and its bank account.
2. **Live settings.** Redo steps 2–5 in live mode. Stripe keeps some settings separately for each mode, so check each one.
   - **Stripe Tax:** the real registrations (home country, EU OSS, UK).
   - **Retries and emails:** as in steps 3 and 4.
3. **Decide D-7 first** (prices including or excluding VAT). A live Price can't be changed later.
   - The default is `--tax-behavior exclusive`.
   - Add `--tax-behavior inclusive` if prices include VAT.
4. **Live keys.** Set `STRIPE_SECRET_KEY` (`sk_live_…`) and `STRIPE_PUBLISHABLE_KEY` (`pk_live_…`) on the API service.
5. **Live run.** Run the script with `--live`: first with `--dry-run`, then without it.
   ```
   python -m scripts.stripe_setup --live --webhook-url https://<API public host>/api/v1/billing/webhooks/stripe --dry-run
   ```
6. **Live values.** Replace every Stripe value on Railway with the printed live ones: the Price IDs and the account ID. Also set the live endpoint's **new** webhook signing secret, revealed in the Dashboard. Test-mode IDs don't work with a live key.
7. **Live card tests (US-012).** Run them as a platform admin, with `BILLING_OPEN` still `false`.
8. **Open (US-013).** On 9 Nov, set `BILLING_OPEN=true`.

**The consent text (D-4).**
- `BILLING_WITHDRAWAL_CONSENT_TEXT` holds counsel's EU withdrawal-consent wording.
- When it is set, the card form shows it beside a box the buyer must tick before paying, and the time they accepted is kept on the payment. When it is empty, there is no box.
- Set it on the API service once counsel signs off, before opening.

## Rolling back

- **Set `BILLING_OPEN=false`.**
  - Buying closes to everyone except platform admins.
  - Balances, plans and existing subscriptions stay as they are, and subscribers can still manage their card, invoices and plan.
  - Webhooks keep working, so renewals, refunds and cancellations are still applied.
- **Stopping renewals is a separate step.** To stop charging existing customers, cancel their subscriptions in Stripe.
