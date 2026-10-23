'use client'

/**
 * PRD-265 US-011 / W2 — one plan in the Billing tab's picker: name, price, the facts-sheet
 * copy, the monthly credits once billing is open, and what this person can do about it.
 *
 * The action is decided by `planActionOf`: a plan not sold here (Enterprise, or Business
 * without an open offer) offers "Contact Automatos" and never a payment form; every other plan
 * is for an owner or admin while paying is available. A workspace with no live subscription
 * "Choose"s a plan (the payment form opens in the tab); one that subscribes "Switch"es (the
 * plan-change preview opens); the plan it pays for reads "Current plan" (disabled). A plan
 * named in `?plan=` is highlighted.
 */
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { hasLiveSubscription, type BillingPlan, type BillingPlans } from '@/hooks/use-billing-plans'
import { cn } from '@/lib/utils'
import {
  CONTACT_LABEL,
  CONTACT_URL,
  isContactOnly,
  planCreditsLine,
  planDescription,
  planOfferLine,
  planPriceLabel,
} from './plan-copy'

export type PlanAction = 'contact' | 'current' | 'choose' | 'switch' | 'none'

/** What the card for `plan` offers this caller. */
export function planActionOf(plan: BillingPlan, plans: BillingPlans): PlanAction {
  if (isContactOnly(plan, plans.business_offer_until)) return 'contact'
  if (!plans.can_manage || !plans.checkout_available) return 'none'
  const subscribed = hasLiveSubscription(plans.subscription)
  if (plan.key === plans.current_plan && subscribed) return 'current'
  return subscribed ? 'switch' : 'choose'
}

interface PlanCardProps {
  readonly plan: BillingPlan
  readonly plans: BillingPlans
  readonly highlighted: boolean
  readonly disabled: boolean
  readonly onChoose: (plan: BillingPlan) => void
  readonly onSwitch: (plan: BillingPlan) => void
}

function ContactLink() {
  return (
    <a
      href={CONTACT_URL}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex h-9 items-center justify-center rounded-xl border border-border px-3 text-sm font-medium text-foreground transition-colors hover:bg-secondary/40"
    >
      {CONTACT_LABEL}
    </a>
  )
}

interface PlanActionSlotProps extends Omit<PlanCardProps, 'plans' | 'highlighted'> {
  readonly action: PlanAction
}

function PlanActionSlot({ plan, action, disabled, onChoose, onSwitch }: PlanActionSlotProps) {
  if (action === 'contact') return <ContactLink />
  if (action === 'none') return null
  if (action === 'current') {
    return (
      <Button type="button" size="sm" variant="outline" disabled>
        Current plan
      </Button>
    )
  }
  const switching = action === 'switch'
  return (
    <Button type="button" size="sm" disabled={disabled} onClick={() => (switching ? onSwitch(plan) : onChoose(plan))}>
      {switching ? `Switch to ${plan.name}` : `Choose ${plan.name}`}
    </Button>
  )
}

export function PlanCard(props: PlanCardProps) {
  const { plan, plans, highlighted } = props
  const offerUntil = plans.business_offer_until
  const credits = planCreditsLine(plan, plans.billing_open)
  const offer = planOfferLine(plan, offerUntil)
  return (
    <li
      data-testid={`billing-plan-${plan.key}`}
      data-highlighted={highlighted ? 'true' : undefined}
      aria-current={highlighted ? 'true' : undefined}
      className={cn(
        'flex flex-col gap-3 rounded-2xl border p-4',
        highlighted ? 'border-primary bg-primary/5 ring-1 ring-primary' : 'border-border bg-secondary/20',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <h4 className="text-base font-semibold text-foreground">{plan.name}</h4>
        {plan.key === plans.current_plan && <Badge variant="outline">Your plan</Badge>}
      </div>
      <p className="text-sm font-medium text-foreground">{planPriceLabel(plan)}</p>
      <p className="flex-1 text-sm text-muted-foreground">
        {planDescription(plan, plans.billing_open, isContactOnly(plan, offerUntil))}
      </p>
      {credits && <p className="text-sm text-foreground">{credits}</p>}
      {offer && <p className="text-sm font-medium text-primary">{offer}</p>}
      <div>
        <PlanActionSlot {...props} action={planActionOf(plan, plans)} />
      </div>
    </li>
  )
}
