'use client'

/**
 * PRD-265 US-011 — "Change plan": every plan the API lists, in its order, with the facts-sheet
 * copy (./plan-copy.ts) and the footnote on the trial, the credit unit and, while billing is
 * closed, the beta terms. Choosing a plan POSTs `{kind: 'plan', plan}` to the checkout route
 * and follows the Stripe address it answers (hooks/use-billing-actions.ts).
 */
import { useBillingCheckout, isLeaving } from '@/hooks/use-billing-actions'
import { hasLiveSubscription, type BillingPlan, type BillingPlans } from '@/hooks/use-billing-plans'
import { PlanCard, planActionOf } from './plan-card'
import { pickerNote, plansFootnote } from './plan-copy'

interface PlanPickerProps {
  readonly plans: BillingPlans
  /** The plan named in `?plan=` (Auto's onboarding link), highlighted. */
  readonly highlight: string | null
  readonly trialCredits: number | null | undefined
}

export function PlanPicker({ plans, highlight, trialCredits }: PlanPickerProps) {
  const checkout = useBillingCheckout()
  const leaving = isLeaving(checkout)
  const request = checkout.variables
  const choosing = leaving && request?.kind === 'plan' ? request.plan : null
  const subscribed = hasLiveSubscription(plans.subscription)
  const note = pickerNote(plans)
  const choose = (plan: BillingPlan) => checkout.mutate({ kind: 'plan', plan: plan.key })

  return (
    <section aria-labelledby="billing-plans-title" className="space-y-3">
      <h3 id="billing-plans-title" className="text-xs uppercase tracking-wide text-muted-foreground">
        Plans
      </h3>
      <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {plans.plans.map((plan) => (
          <PlanCard
            key={plan.key}
            plan={plan}
            billingOpen={plans.billing_open}
            action={planActionOf(plan, plans)}
            yours={plan.key === plans.current_plan}
            highlighted={plan.key === highlight}
            switching={subscribed}
            busy={choosing === plan.key}
            disabled={leaving}
            onChoose={choose}
          />
        ))}
      </ul>
      {note && <p className="text-sm text-muted-foreground">{note}</p>}
      <p className="text-xs text-muted-foreground">{plansFootnote(plans.billing_open, trialCredits)}</p>
    </section>
  )
}
