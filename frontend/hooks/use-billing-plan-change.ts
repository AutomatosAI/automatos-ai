/**
 * PRD-265 W2 — changing the plan of a workspace that already subscribes, in the Billing tab.
 *
 * `POST /api/v1/billing/plan-change/preview {plan}` says what the change costs before it is
 * made: an upgrade is charged now, prorated for the rest of the period; a downgrade charges
 * nothing now and takes effect when the period ends. `POST /api/v1/billing/plan-change {plan}`
 * makes it, and answers `done`, `scheduled` (a downgrade waiting for the period end) or
 * `requires_action` (the bank wants the buyer to confirm the charge: Stripe.js shows that
 * check over our page with `handleNextAction`). `POST /api/v1/billing/plan-change/cancel` takes
 * back a scheduled downgrade. All owner/admin only.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient, type ApiHttpError } from '@/lib/api-client'
import { isSaaS } from '@/lib/auth-edition'
import { billingErrorMessage, stripeErrorMessage } from '@/lib/billing-errors'
import { getStripe, isClientSecret } from '@/lib/stripe'
import { billingPlansQueryKeys } from '@/hooks/use-billing-plans'

export const PLAN_CHANGE_PREVIEW_PATH = '/api/v1/billing/plan-change/preview'
export const PLAN_CHANGE_PATH = '/api/v1/billing/plan-change'
export const PLAN_CHANGE_CANCEL_PATH = '/api/v1/billing/plan-change/cancel'

export const KEEP_PLAN_FAILED = "The plan change couldn't be undone. Try again in a moment."
export const KEPT_PLAN_TOAST = 'Your plan stays as it is.'
export const CONFIRM_FAILED = "The bank didn't confirm the payment. Nothing was changed."
/** The check runs as soon as the preview opens; nothing about it changes meanwhile. */
const PREVIEW_STALE_MS = 0

export interface PlanChangePreview {
  plan: string
  direction: 'upgrade' | 'downgrade'
  amount_due_cents: number
  tax_cents: number
  currency: string
  /** "now", or when a downgrade takes effect (the period end). */
  effective_at: string
}

export type PlanChangeResult =
  | { status: 'done' }
  | { status: 'scheduled'; effective_at: string }
  | { status: 'requires_action'; client_secret: string; publishable_key: string }

export const planChangeQueryKeys = {
  preview: (plan: string) => ['billing', 'plan-change-preview', plan] as const,
}

/** What switching to `plan` costs; asks only while a switch is being considered. */
export function usePlanChangePreview(plan: string | null) {
  return useQuery<PlanChangePreview, ApiHttpError>({
    queryKey: planChangeQueryKeys.preview(plan ?? ''),
    queryFn: () => apiClient.post<PlanChangePreview>(PLAN_CHANGE_PREVIEW_PATH, { plan }),
    enabled: isSaaS && plan !== null,
    staleTime: PREVIEW_STALE_MS,
    retry: false,
  })
}

/** Make the change. The caller decides what each answer means for the buyer. */
export function usePlanChange() {
  return useMutation<PlanChangeResult, ApiHttpError, string>({
    mutationFn: (plan) => apiClient.post<PlanChangeResult>(PLAN_CHANGE_PATH, { plan }),
  })
}

/**
 * The bank's check for a `requires_action` answer, shown by Stripe.js over our page. null when
 * the charge is confirmed; else the reason, in words for the buyer ("Your card was declined.").
 */
export async function confirmWithBank(
  result: Extract<PlanChangeResult, { status: 'requires_action' }>,
): Promise<string | null> {
  const stripe = isClientSecret(result.client_secret, 'pi_') ? await getStripe(result.publishable_key) : null
  if (!stripe) return CONFIRM_FAILED
  const outcome = await stripe.handleNextAction({ clientSecret: result.client_secret })
  return outcome.error ? stripeErrorMessage(outcome.error, CONFIRM_FAILED) : null
}

/** Take back a scheduled downgrade: the plan stays as it is. */
export function useKeepPlan() {
  const queryClient = useQueryClient()
  return useMutation<{ scheduled_change: null }, ApiHttpError, void>({
    mutationFn: () => apiClient.post<{ scheduled_change: null }>(PLAN_CHANGE_CANCEL_PATH),
    onSuccess: () => {
      toast.success(KEPT_PLAN_TOAST)
      void queryClient.invalidateQueries({ queryKey: billingPlansQueryKeys.all })
    },
    onError: (error) => {
      toast.error(billingErrorMessage(error, KEEP_PLAN_FAILED))
    },
  })
}
