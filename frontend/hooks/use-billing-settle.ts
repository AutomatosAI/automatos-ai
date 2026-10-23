/**
 * PRD-265 W2 — after a payment in the Billing tab, waiting for it to land.
 *
 * Stripe confirms a payment in the page at once, but the plan and the credits change when
 * Stripe's webhook reaches the API, usually within seconds. So after a payment the tab asks
 * for the plans, the summary and the ledger again every few seconds until what a payment
 * changes (the plan, the subscription, the credits) differs from how it stood when the
 * payment went through, and stops after a minute whatever happens.
 */
import { useCallback, useEffect, useState } from 'react'
import { useQueryClient, type QueryClient } from '@tanstack/react-query'
import { billingLedgerQueryKeys } from '@/hooks/use-billing-ledger'
import { billingPlansQueryKeys, type BillingPlans } from '@/hooks/use-billing-plans'
import { billingSummaryQueryKeys, type BillingSummary } from '@/hooks/use-billing-summary'

/** How often the billing reads are asked again while a payment lands. */
export const SETTLE_POLL_MS = 3_000
/** When the tab stops asking, landed or not. */
export const SETTLE_CAP_MS = 60_000

/** Ask the billing reads again: the summary, the plans and the ledger. */
export function refreshBilling(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({ queryKey: billingSummaryQueryKeys.all })
  void queryClient.invalidateQueries({ queryKey: billingPlansQueryKeys.all })
  void queryClient.invalidateQueries({ queryKey: billingLedgerQueryKeys.all })
}

/** What a payment changes, as one comparable string. */
export function billingFingerprint(plans: BillingPlans | undefined, summary: BillingSummary | undefined): string {
  const subscription = plans?.subscription
  return JSON.stringify([
    plans?.current_plan ?? null,
    subscription?.status ?? null,
    subscription?.cancel_at_period_end ?? null,
    subscription?.scheduled_change?.plan ?? null,
    summary?.plan ?? null,
    summary?.balance?.monthly ?? null,
    summary?.balance?.topup ?? null,
  ])
}

function cachedFingerprint(queryClient: QueryClient): string {
  return billingFingerprint(
    queryClient.getQueryData<BillingPlans>(billingPlansQueryKeys.all),
    queryClient.getQueryData<BillingSummary>(billingSummaryQueryKeys.all),
  )
}

/**
 * `start()` after a payment; `settling` while the tab is still waiting for it to land. The
 * tab owns one of these, so two payments in a row share one schedule.
 */
export function useBillingSettle(): { settling: boolean; start: () => void } {
  const queryClient = useQueryClient()
  const [baseline, setBaseline] = useState<string | null>(null)
  const start = useCallback(() => setBaseline(cachedFingerprint(queryClient)), [queryClient])

  useEffect(() => {
    if (baseline === null) return undefined
    refreshBilling(queryClient)
    const startedAt = Date.now()
    const timer = window.setInterval(() => {
      const landed = cachedFingerprint(queryClient) !== baseline
      if (landed || Date.now() - startedAt >= SETTLE_CAP_MS) {
        setBaseline(null)
        return
      }
      refreshBilling(queryClient)
    }, SETTLE_POLL_MS)
    return () => window.clearInterval(timer)
  }, [baseline, queryClient])

  return { settling: baseline !== null, start }
}
