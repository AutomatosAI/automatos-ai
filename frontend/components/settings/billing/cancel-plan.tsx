'use client'

/**
 * PRD-265 W2 — cancelling and resuming the plan, in the Billing tab (owner or admin, while
 * Stripe is billing the workspace).
 *
 * "Cancel plan" asks first ("Your plan stays until 23 Nov 2026."), then ends the plan at the
 * end of the period already paid for (`POST /api/v1/billing/subscription/cancel`). A plan that
 * is ending offers "Resume plan" (`…/resume`) instead, which keeps it renewing as usual.
 */
import { Loader2 } from 'lucide-react'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { useCancelPlan, useResumePlan } from '@/hooks/use-billing-actions'
import { hasLiveSubscription, type BillingPlans } from '@/hooks/use-billing-plans'
import { formatDay } from './billing-format'

/** What cancelling means, said before it is done. */
export function cancelNotice(periodEnd: string | null | undefined): string {
  return periodEnd
    ? `Your plan stays until ${formatDay(periodEnd)}. After that the workspace moves off it, and nothing more is charged.`
    : 'Your plan stays until the end of this period. After that the workspace moves off it, and nothing more is charged.'
}

function ResumePlanButton() {
  const resume = useResumePlan()
  return (
    <Button type="button" size="sm" variant="outline" disabled={resume.isLoading} onClick={() => resume.mutate()}>
      {resume.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
      Resume plan
    </Button>
  )
}

function CancelPlanButton({ periodEnd }: { periodEnd: string | null }) {
  const cancel = useCancelPlan()
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button type="button" size="sm" variant="outline" disabled={cancel.isLoading}>
          {cancel.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
          Cancel plan
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Cancel your plan?</AlertDialogTitle>
          <AlertDialogDescription>{cancelNotice(periodEnd)}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Keep my plan</AlertDialogCancel>
          <AlertDialogAction onClick={() => cancel.mutate()}>Cancel plan</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

export function CancelPlan({ plans }: { plans: BillingPlans }) {
  const { subscription } = plans
  if (!plans.can_manage || !hasLiveSubscription(subscription)) return null
  return subscription.cancel_at_period_end ? (
    <ResumePlanButton />
  ) : (
    <CancelPlanButton periodEnd={subscription.current_period_end} />
  )
}
