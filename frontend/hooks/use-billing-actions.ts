/**
 * PRD-265 US-009 / US-011 — leaving for Stripe from the Billing tab.
 *
 * `POST /api/v1/billing/checkout` (`{kind: 'plan', plan}` or `{kind: 'topup'}`) and
 * `POST /api/v1/billing/portal` (invoices, card, cancel), owner/admin only. Both answer a
 * Stripe address, and the browser goes there with `window.location.assign`; a plan change on
 * a workspace that already subscribes answers the portal's confirm page (`flow: 'portal'`),
 * which is followed the same way. Stripe sends the browser back to
 * `/settings?tab=billing&checkout=success|cancelled` (see use-checkout-return.ts).
 *
 * A refusal is toasted in the API's own plain sentence (lib/billing-errors.ts). An answer
 * without an https address is treated as a failure: the browser is never sent anywhere else.
 */
import { useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient, type ApiHttpError } from '@/lib/api-client'
import { billingErrorMessage, isHttpsUrl } from '@/lib/billing-errors'

export const BILLING_CHECKOUT_PATH = '/api/v1/billing/checkout'
export const BILLING_PORTAL_PATH = '/api/v1/billing/portal'

export const CHECKOUT_FAILED = 'Checkout could not start. Try again in a moment.'
export const PORTAL_FAILED = 'The payment page could not open. Try again in a moment.'

export type CheckoutRequest = { kind: 'plan'; plan: string } | { kind: 'topup' }

/** What both routes answer: the Stripe page to open. */
export interface BillingRedirect {
  url: string
  flow?: 'checkout' | 'portal'
}

/** The Stripe address in an answer, or a failure in the surface's own words. */
async function stripeUrlOf(answer: Promise<BillingRedirect | null>, failure: string): Promise<string> {
  const url = (await answer)?.url
  if (!isHttpsUrl(url)) throw new Error(failure)
  return url
}

function leaveForStripe(url: string): void {
  window.location.assign(url)
}

/**
 * A mutation that ended in a redirect keeps its "busy" state; when the browser comes back
 * to this page from its back/forward cache, the buttons must work again.
 */
function useResetOnPageRestore(reset: () => void): void {
  useEffect(() => {
    const onPageShow = (event: PageTransitionEvent) => {
      if (event.persisted) reset()
    }
    window.addEventListener('pageshow', onPageShow)
    return () => window.removeEventListener('pageshow', onPageShow)
  }, [reset])
}

/** Start Stripe Checkout for a plan or a top-up, then go there. */
export function useBillingCheckout() {
  const mutation = useMutation<string, ApiHttpError, CheckoutRequest>({
    mutationFn: (request) => stripeUrlOf(apiClient.post<BillingRedirect>(BILLING_CHECKOUT_PATH, request), CHECKOUT_FAILED),
    onSuccess: leaveForStripe,
    onError: (error) => {
      toast.error(billingErrorMessage(error, CHECKOUT_FAILED))
    },
  })
  useResetOnPageRestore(mutation.reset)
  return mutation
}

/** Open Stripe's customer portal: invoices, the card on file, cancellation. */
export function useBillingPortal() {
  const mutation = useMutation<string, ApiHttpError, void>({
    mutationFn: () => stripeUrlOf(apiClient.post<BillingRedirect>(BILLING_PORTAL_PATH), PORTAL_FAILED),
    onSuccess: leaveForStripe,
    onError: (error) => {
      toast.error(billingErrorMessage(error, PORTAL_FAILED))
    },
  })
  useResetOnPageRestore(mutation.reset)
  return mutation
}

/** True while a call is in flight or the browser is already on its way to Stripe. */
export function isLeaving(mutation: { isLoading: boolean; isSuccess: boolean }): boolean {
  return mutation.isLoading || mutation.isSuccess
}
