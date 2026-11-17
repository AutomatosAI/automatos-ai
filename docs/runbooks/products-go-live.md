# Products: switch on, with BudStacks as the first (PRD-274 Wave 5)

**For:** whoever holds Railway and the instance's `ORCHESTRATOR_API_KEY`. **Time:** about 30 minutes, then a day of
watching before the last step. **How products work:** [PRD-274 §5.1, §6 Wave 5 and §8](../PRDS/PRD-274-AUTOMATOS-FOR-PRODUCTS.md).

**What changes when products go on:**

- BudStacks becomes a product record. Its machine key keeps working with no change in the BudStacks app.
- Every BudStacks tenant workspace joins the product, one per tenant. From then on the **owner workspace pays for
  their platform AI**: tenant workspaces are on Basic with no credits today, so platform-key work they could not
  do before now runs on the owner's credits.
- New end users of a tenant get the product's default retention, if one is set. Existing end users keep theirs.
- Shopify does not change: it has no key on Railway, so it gets no record and its machine routes stay off (503).

**Before you start:**

1. **The spend cap is merged and deployed (#1522).** Without it, the owner's balance is the only limit on what all
   the tenants together can spend.
2. **Choose the owner workspace.** It pays for every tenant. It needs a plan with platform AI (Pro or Business; on
   Basic every tenant's platform-key work would stop) and credits to cover the tenants. Note its id.
3. **Have these to hand:** the API's address (`https://api.automatos.app`) and the instance's `ORCHESTRATOR_API_KEY`
   (Railway, service `automatos-ai-api`). The admin routes below take it as `X-API-Key`; nothing else is needed.

```bash
export API=https://api.automatos.app
export ADMIN_KEY='<ORCHESTRATOR_API_KEY>'   # never paste it into a ticket or a chat
```

## Switch on

1. **Set the variables** on the Railway service `automatos-ai-api`:
   - `PRODUCTS_ENABLED=true`
   - `BUDSTACKS_PRODUCT_OWNER_WORKSPACE_ID=<the owner workspace id>`

   Leave `BUDSTACKS_INTERNAL_API_KEY` as it is: the boot reads it once to make the record. Railway redeploys.

2. **Check the boot made the record.** In the deploy's logs, the boot leader writes:
   - `[products] budstacks is now product <id>, owned by workspace <owner>`
   - `[products] <n> of budstacks's workspaces joined product <id>`

   Then read the record (no key or hash is ever shown):

   ```bash
   curl -s "$API/api/admin/products" -H "X-API-Key: $ADMIN_KEY"
   ```

   Expect `budstacks`, with `key_prefix` `env`, `key_type` `public`, `key_scopes` `["chat"]` and
   `reuse_existing_key` `true`. Note its `id` as `PRODUCT_ID`. If there is no record: the variables are not both
   set, `PRODUCTS_ENABLED` is not `true`, or the owner id is not a live workspace; the log says which
   (`Vertical products: [...]`). That line is logged at INFO, unlike the two `[products]` lines (WARNING), so check
   the service's log level shows INFO if you don't see it.

3. **Set the product's limits.** A monthly credit cap for each tenant workspace (#1522), and, if BudStacks wants
   one, a default retention for new end users:

   ```bash
   curl -s -X PATCH "$API/api/admin/products/$PRODUCT_ID" -H "X-API-Key: $ADMIN_KEY" \
     -H "Content-Type: application/json" -d '{"workspace_monthly_credit_cap": 500}'
   ```

   (`"default_retention_days": 365` in the same body sets the retention.) BudStacks can raise or lower one tenant's
   cap later with its own key.

## Check it works

4. **A re-provision keeps the storefront's key.** Ask BudStacks to re-provision one existing tenant (or call
   `POST /api/verticals/budstacks/provision` with its key and that tenant's `external_id` and `metadata.domains`).
   Expect the same workspace id, `is_new: false`, `key_minted: false` and `api_key: null`.
5. **The storefront widget still answers.** Open one tenant's storefront and send the widget a message.
6. **The owner pays.** After a tenant's next platform-AI call, wait one meter tick (`CREDITS_METER_INTERVAL_SECONDS`,
   a minute). The owner's Billing tab shows the usage; its statement's `by_product_workspace`
   (`GET /api/v1/billing/ledger`, as the owner) lists the tenant's workspace id. A tenant's own Billing tab shows
   that its product pays, and none of the owner's figures.

## The next day: retire the environment key

7. **Merge #1521** (still a draft: mark it ready) and let it deploy. From then on the BudStacks key is only the
   record's. Repeat step 4 to check BudStacks still provisions.
8. **Remove `BUDSTACKS_INTERNAL_API_KEY`** from Railway.

## Watch

- **Errors:** the error tile's `jobs` subsystem (the meter, the daily purge of closed workspaces), and the API's
  logs: lines starting `[credits]` and `[products]`.
- **Tenants stopped at their cap:** chat turns and API calls fail with `workspace_cap_reached`. The owner is told at
  80% and at the cap of each tenant's month (its `credits_low` / `credits_exhausted` destinations).
- **The owner's balance:** its running-low and run-out notices. When it runs out, every tenant pauses together.

## Roll back

- **Before step 7:** set `PRODUCTS_ENABLED=false` and redeploy. The product routes answer 404 and the boot seeds
  nothing more. That alone does **not** move the tenants' billing back: a workspace carrying the product is still
  paid for by the owner. To undo that too, take the product off the tenants' workspaces (in a database session on
  production, after a backup):

  ```sql
  UPDATE workspaces SET settings = settings - 'product_id'
   WHERE settings->>'product_id' = '<PRODUCT_ID>';
  ```

  The record can stay; its key keeps working for BudStacks while it exists. Do not turn `PRODUCTS_ENABLED` back on
  unless you mean to re-attach the tenants: until #1521 removes it, the boot seed re-joins every BudStacks
  workspace that has no `product_id`.
- **After step 7, before step 8:** revert #1521 and redeploy. The environment key is still set, so the fallback
  works again.
- **After step 8:** set `BUDSTACKS_INTERNAL_API_KEY` again only with the same value BudStacks holds. Otherwise
  replace the product's key (`POST /api/admin/products/$PRODUCT_ID/key`, shown once) and give BudStacks the new one.
