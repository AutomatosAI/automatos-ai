'use client'

/**
 * PRD-265 US-011 — coming back from Stripe Checkout.
 *
 * Stripe returns the browser to `/settings?tab=billing&checkout=success` or `…=cancelled`.
 * On success the tab says the payment arrived and asks for the summary, the plans and the
 * ledger again, then a few more times over the next minute: the plan changes when Stripe's
 * webhook lands, usually within seconds. On cancel it says nothing was charged. Either way
 * the `checkout` parameter is then taken out of the address, so a reload says nothing twice.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useQueryClient, type QueryClient } from '@tanstack/react-query'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'
import { billingLedgerQueryKeys } from '@/hooks/use-billing-ledger'
import { billingPlansQueryKeys } from '@/hooks/use-billing-plans'
import { billingSummaryQueryKeys } from '@/hooks/use-billing-summary'
import { settingsHref } from '../use-settings-tab'

export const CHECKOUT_PARAM = 'checkout'
export const CHECKOUT_SUCCESS_TOAST = 'Payment received — your plan updates within a minute'
export const CHECKOUT_CANCELLED_TOAST = 'Checkout cancelled — nothing was charged'
/** When the billing reads are asked again after a payment, from the return (about a minute in all). */
export const PLAN_POLL_DELAYS_MS: readonly number[] = [5_000, 15_000, 30_000, 60_000]
/** One id, so a second effect pass replaces the toast instead of stacking another. */
const TOAST_ID = 'billing-checkout-return'

type CheckoutOutcome = 'success' | 'cancelled'

function outcomeOf(value: string | null | undefined): CheckoutOutcome | null {
  return value === 'success' || value === 'cancelled' ? value : null
}

/** Ask the billing reads again: the summary, the plans and the ledger. */
export function refreshBilling(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({ queryKey: billingSummaryQueryKeys.all })
  void queryClient.invalidateQueries({ queryKey: billingPlansQueryKeys.all })
  void queryClient.invalidateQueries({ queryKey: billingLedgerQueryKeys.all })
}

/** Say how the checkout ended, and take `?checkout=` out of the address. */
function useAnnounceOutcome(onPaid: () => void): void {
  const router = useRouter()
  const searchParams = useSearchParams()
  const outcome = outcomeOf(searchParams?.get(CHECKOUT_PARAM))
  const announced = useRef<CheckoutOutcome | null>(null)

  useEffect(() => {
    if (outcome === null || announced.current === outcome) return
    announced.current = outcome
    if (outcome === 'success') {
      toast.success(CHECKOUT_SUCCESS_TOAST, { id: TOAST_ID })
      onPaid()
    } else {
      toast.info(CHECKOUT_CANCELLED_TOAST, { id: TOAST_ID })
    }
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.delete(CHECKOUT_PARAM)
    router.replace(settingsHref(params), { scroll: false })
  }, [outcome, onPaid, router, searchParams])
}

/** Mounted by the Billing tab: handles the return from Stripe Checkout, once. */
export function useCheckoutReturn(): void {
  const queryClient = useQueryClient()
  const [polling, setPolling] = useState(false)
  const startPolling = useCallback(() => setPolling(true), [])
  useAnnounceOutcome(startPolling)

  useEffect(() => {
    if (!polling) return undefined
    refreshBilling(queryClient)
    const timers = PLAN_POLL_DELAYS_MS.map((delay) => window.setTimeout(() => refreshBilling(queryClient), delay))
    return () => timers.forEach((timer) => window.clearTimeout(timer))
  }, [polling, queryClient])
}
