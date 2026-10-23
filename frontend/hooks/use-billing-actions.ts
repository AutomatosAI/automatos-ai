/**
 * PRD-265 W2 — buying and cancelling from the Billing tab, without leaving it.
 *
 * `POST /api/v1/billing/checkout` (`{kind: 'plan', plan}` or `{kind: 'topup'}`, with
 * `accept_terms: true` once the buyer has ticked the terms) answers an in-page Checkout
 * Session: its client secret and Stripe's publishable key, which the tab's own payment form
 * mounts Stripe's card fields with (components/settings/billing/checkout-panel.tsx). Nothing
 * here, or anywhere in the tab, sends the browser to a Stripe page.
 *
 * `POST /api/v1/billing/subscription/cancel` ends the plan at the end of the period it is paid
 * for; `…/resume` takes that back. All three are owner/admin only. A refusal is shown in the
 * API's own plain sentence (lib/billing-errors.ts).
 */
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { formatAbsoluteDate } from '@/lib/format-relative'
import { apiClient, type ApiHttpError } from '@/lib/api-client'
import { billingErrorMessage } from '@/lib/billing-errors'
import { isClientSecret, isPublishableKey } from '@/lib/stripe'
import { billingPlansQueryKeys } from '@/hooks/use-billing-plans'

export const BILLING_CHECKOUT_PATH = '/api/v1/billing/checkout'
export const BILLING_CANCEL_PATH = '/api/v1/billing/subscription/cancel'
export const BILLING_RESUME_PATH = '/api/v1/billing/subscription/resume'

export const CHECKOUT_FAILED = 'The payment form could not open. Try again in a moment.'
export const CANCEL_FAILED = "The plan couldn't be cancelled. Try again in a moment."
export const RESUME_FAILED = "The plan couldn't be resumed. Try again in a moment."
export const RESUMED_TOAST = 'Your plan renews as usual.'

export type CheckoutRequest = { kind: 'plan'; plan: string } | { kind: 'topup' }
export type CheckoutBody = CheckoutRequest & { accept_terms?: true }

/** What the checkout route answers: an in-page Checkout Session. */
export interface CheckoutSession {
  client_secret: string
  publishable_key: string
}

export interface CancelAnswer {
  cancel_at_period_end: boolean
  ends_at: string | null
}

/** The session in an answer, or a failure: a malformed answer never reaches Stripe.js. */
function sessionOf(answer: CheckoutSession | null): CheckoutSession {
  const secret = answer?.client_secret
  const key = answer?.publishable_key
  if (!isClientSecret(secret, 'cs_') || !isPublishableKey(key)) throw new Error(CHECKOUT_FAILED)
  return { client_secret: secret, publishable_key: key }
}

/** Open a Checkout Session for a plan or a top-up; the payment form shows its errors itself. */
export function useCheckoutSession() {
  return useMutation<CheckoutSession, ApiHttpError, CheckoutBody>({
    mutationFn: async (body) => sessionOf(await apiClient.post<CheckoutSession>(BILLING_CHECKOUT_PATH, body)),
  })
}

/** "Your plan stays until 23 Nov 2026." */
export function cancelledToast(endsAt: string | null | undefined): string {
  return endsAt ? `Your plan stays until ${formatAbsoluteDate(endsAt)}.` : 'Your plan stays until the end of this period.'
}

/** End the plan when the period it is paid for ends. */
export function useCancelPlan() {
  const queryClient = useQueryClient()
  return useMutation<CancelAnswer, ApiHttpError, void>({
    mutationFn: () => apiClient.post<CancelAnswer>(BILLING_CANCEL_PATH),
    onSuccess: (answer) => {
      toast.success(cancelledToast(answer?.ends_at))
      void queryClient.invalidateQueries({ queryKey: billingPlansQueryKeys.all })
    },
    onError: (error) => {
      toast.error(billingErrorMessage(error, CANCEL_FAILED))
    },
  })
}

/** Take back a cancellation: the plan renews as usual. */
export function useResumePlan() {
  const queryClient = useQueryClient()
  return useMutation<{ cancel_at_period_end: boolean }, ApiHttpError, void>({
    mutationFn: () => apiClient.post<{ cancel_at_period_end: boolean }>(BILLING_RESUME_PATH),
    onSuccess: () => {
      toast.success(RESUMED_TOAST)
      void queryClient.invalidateQueries({ queryKey: billingPlansQueryKeys.all })
    },
    onError: (error) => {
      toast.error(billingErrorMessage(error, RESUME_FAILED))
    },
  })
}
