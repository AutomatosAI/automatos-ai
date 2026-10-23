'use client'

/**
 * PRD-265 US-011 / W2 — the workspace's plan as it stands: name and price, the subscription's
 * state ("Payment failed, Stripe is retrying — update your card", "Ends on …", "Active · renews
 * on …"), a downgrade waiting for the period end (./scheduled-change-notice.tsx), and, for an
 * owner or admin, "Cancel plan" / "Resume plan" (./cancel-plan.tsx). The credits by bucket sit
 * underneath (./credit-balance.tsx). The card and the invoices are in the Payment section
 * (./payment-section.tsx).
 */
import { Card } from '@/components/ui/card'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import type { BillingSummary } from '@/hooks/use-billing-summary'
import { subscriptionNotice, type NoticeTone } from './billing-format'
import { CancelPlan } from './cancel-plan'
import { CreditBalance } from './credit-balance'
import { planPriceLabel } from './plan-copy'
import { ScheduledChangeNotice } from './scheduled-change-notice'

const NOTICE_CLASSES: Readonly<Record<NoticeTone, string>> = {
  ok: 'text-sm text-foreground',
  warning: 'rounded-xl border border-warning/30 bg-warning/5 px-3 py-2 text-sm text-warning',
  muted: 'text-sm text-muted-foreground',
}
const NO_PLAN = 'No plan'

interface CurrentPlanCardProps {
  readonly plans: BillingPlans
  readonly summary: BillingSummary | undefined
}

export function CurrentPlanCard({ plans, summary }: CurrentPlanCardProps) {
  const plan = plans.plans.find((candidate) => candidate.key === plans.current_plan)
  const name = plan?.name ?? summary?.plan_name ?? NO_PLAN
  const periodEnd = plans.subscription.current_period_end ?? summary?.renewal_date ?? null
  const notice = subscriptionNotice(plans.subscription, plans.billing_open, periodEnd)

  return (
    <Card data-testid="billing-current-plan" className="space-y-4 border border-border p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Current plan</p>
          <p className="text-xl font-semibold text-foreground">{name}</p>
          {plan && <p className="text-sm text-muted-foreground">{planPriceLabel(plan)}</p>}
        </div>
        <CancelPlan plans={plans} />
      </div>
      <p
        data-testid="billing-subscription-notice"
        role={notice.tone === 'warning' ? 'alert' : undefined}
        className={NOTICE_CLASSES[notice.tone]}
      >
        {notice.text}
      </p>
      <ScheduledChangeNotice plans={plans} />
      {summary && <CreditBalance summary={summary} billingOpen={plans.billing_open} />}
    </Card>
  )
}
