'use client'

/**
 * PRD-265 W2 — switching the plan of a workspace that already subscribes.
 *
 * `open(plan)` shows the preview (./plan-change-dialog.tsx) and asks what the switch costs;
 * `confirm()` makes it. The dialog closes before anything else happens, so when the bank wants
 * the buyer to confirm the charge, its check (shown by Stripe.js over the page) is not under a
 * modal. Then, by the answer:
 *   - done: "Switched to Pro.", and the tab waits for the plan to change (`onPaid`);
 *   - scheduled: "Changes to Own keys on 23 Nov 2026. Nothing to pay now.";
 *   - requires_action: the bank's check; once it passes, "Payment received", and the tab waits.
 * Every failure is one plain sentence: the API's, or Stripe's for a declined card.
 */
import { useCallback, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { billingErrorMessage } from '@/lib/billing-errors'
import type { BillingPlan } from '@/hooks/use-billing-plans'
import {
  confirmWithBank,
  usePlanChange,
  usePlanChangePreview,
  type PlanChangeResult,
} from '@/hooks/use-billing-plan-change'
import { refreshBilling } from '@/hooks/use-billing-settle'
import { formatDay } from './billing-format'

export const PLAN_CHANGE_FAILED = "The plan couldn't be changed. Try again in a moment."
export const PAYMENT_RECEIVED = 'Payment received — your plan updates in a moment.'

export function switchedToast(plan: Pick<BillingPlan, 'name'>): string {
  return `Switched to ${plan.name}.`
}

export function scheduledToast(plan: Pick<BillingPlan, 'name'>, effectiveAt: string): string {
  return `Changes to ${plan.name} on ${formatDay(effectiveAt)}. Nothing to pay now.`
}

interface Outcomes {
  readonly onPaid: () => void
  readonly onScheduled: () => void
}

/** Say what the answer means, and start whatever comes next. */
async function finishSwitch(result: PlanChangeResult | null, plan: BillingPlan, outcomes: Outcomes): Promise<void> {
  if (result?.status === 'scheduled') {
    toast.success(scheduledToast(plan, result.effective_at))
    outcomes.onScheduled()
    return
  }
  if (result?.status === 'requires_action') {
    const refused = await confirmWithBank(result)
    if (refused) {
      toast.error(refused)
      return
    }
    toast.success(PAYMENT_RECEIVED)
    outcomes.onPaid()
    return
  }
  if (result?.status !== 'done') throw new Error(PLAN_CHANGE_FAILED)
  toast.success(switchedToast(plan))
  outcomes.onPaid()
}

export function usePlanSwitch(onPaid: () => void) {
  const queryClient = useQueryClient()
  const [target, setTarget] = useState<BillingPlan | null>(null)
  const [busy, setBusy] = useState(false)
  const preview = usePlanChangePreview(target?.key ?? null)
  const change = usePlanChange()
  const close = useCallback(() => setTarget(null), [])

  const confirm = async () => {
    if (!target) return
    const plan = target
    setBusy(true)
    try {
      const result = await change.mutateAsync(plan.key)
      setTarget(null)
      await finishSwitch(result, plan, { onPaid, onScheduled: () => refreshBilling(queryClient) })
    } catch (error) {
      toast.error(billingErrorMessage(error, PLAN_CHANGE_FAILED))
    } finally {
      setBusy(false)
    }
  }

  return { target, open: setTarget, close, preview, confirm, busy }
}
