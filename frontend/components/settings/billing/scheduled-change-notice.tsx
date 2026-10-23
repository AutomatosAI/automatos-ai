'use client'

/**
 * PRD-265 W2 — a downgrade waiting for the end of the period: "Changes to Own keys on
 * 23 Nov 2026." An owner or admin can take it back with "Keep my plan"
 * (`POST /api/v1/billing/plan-change/cancel`); nothing changes until the date either way.
 */
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useKeepPlan } from '@/hooks/use-billing-plan-change'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import { formatDay } from './billing-format'

/** "Changes to Own keys on 23 Nov 2026."; null with nothing scheduled. */
export function scheduledChangeLine(plans: BillingPlans): string | null {
  const change = plans.subscription.scheduled_change
  if (!change) return null
  const name = plans.plans.find((plan) => plan.key === change.plan)?.name ?? change.plan
  return `Changes to ${name} on ${formatDay(change.effective_at)}.`
}

export function ScheduledChangeNotice({ plans }: { plans: BillingPlans }) {
  const keep = useKeepPlan()
  const line = scheduledChangeLine(plans)
  if (!line) return null
  return (
    <div
      data-testid="billing-scheduled-change"
      className="flex flex-col gap-2 rounded-xl border border-border bg-secondary/20 px-3 py-2 sm:flex-row sm:items-center sm:justify-between"
    >
      <p role="status" className="text-sm text-foreground">
        {line}
      </p>
      {plans.can_manage && (
        <Button type="button" size="sm" variant="outline" disabled={keep.isLoading} onClick={() => keep.mutate()}>
          {keep.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
          Keep my plan
        </Button>
      )}
    </div>
  )
}
