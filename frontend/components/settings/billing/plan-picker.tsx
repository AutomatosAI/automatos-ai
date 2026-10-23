'use client'

/**
 * PRD-265 US-011 / W2 — "Plans": every plan the API lists, in its order, with the facts-sheet
 * copy (./plan-copy.ts) and the footnote on the trial, the credit unit and, while billing is
 * closed, the beta terms. Choosing a plan opens the tab's own payment form (`onChoose`, see
 * ./checkout-panel.tsx); switching the plan of a workspace that already subscribes opens the
 * plan-change preview (./plan-change-dialog.tsx). The browser never leaves the page.
 */
import type { BillingPlan, BillingPlans } from '@/hooks/use-billing-plans'
import { PlanCard } from './plan-card'
import { PlanChangeDialog } from './plan-change-dialog'
import { pickerNote, plansFootnote } from './plan-copy'
import { usePlanSwitch } from './use-plan-switch'

interface PlanPickerProps {
  readonly plans: BillingPlans
  /** The plan named in `?plan=` (Auto's onboarding link), highlighted. */
  readonly highlight: string | null
  readonly trialCredits: number | null | undefined
  /** Open the payment form for a new subscription. */
  readonly onChoose: (plan: BillingPlan) => void
  /** A payment went through: wait for the plan to change. */
  readonly onPaid: () => void
}

export function PlanPicker({ plans, highlight, trialCredits, onChoose, onPaid }: PlanPickerProps) {
  const planSwitch = usePlanSwitch(onPaid)
  const note = pickerNote(plans)

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
            plans={plans}
            highlighted={plan.key === highlight}
            disabled={planSwitch.busy}
            onChoose={onChoose}
            onSwitch={planSwitch.open}
          />
        ))}
      </ul>
      {note && <p className="text-sm text-muted-foreground">{note}</p>}
      <p className="text-xs text-muted-foreground">{plansFootnote(plans.billing_open, trialCredits)}</p>
      <PlanChangeDialog
        target={planSwitch.target}
        preview={planSwitch.preview}
        busy={planSwitch.busy}
        onConfirm={planSwitch.confirm}
        onClose={planSwitch.close}
      />
    </section>
  )
}
