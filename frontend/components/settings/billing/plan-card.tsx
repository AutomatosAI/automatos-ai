'use client'

/**
 * PRD-265 US-011 — one plan in the Billing tab's picker: name, price, the facts-sheet copy,
 * the monthly credits once billing is open, and what this person can do about it.
 *
 * The action is decided by `planActionOf`: an invite-only or coming-soon plan (Business,
 * Enterprise) always offers "Contact Automatos" and never checkout; every other plan offers
 * checkout only to an owner or admin while checkout is available, and the plan the workspace
 * already pays for reads "Current plan" (disabled). A plan named in `?plan=` is highlighted.
 */
import { Loader2 } from 'lucide-react'
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
  planPriceLabel,
} from './plan-copy'

export type PlanAction = 'contact' | 'current' | 'choose' | 'none'

/** What the card for `plan` offers this caller. */
export function planActionOf(plan: BillingPlan, plans: BillingPlans): PlanAction {
  if (isContactOnly(plan)) return 'contact'
  if (!plans.can_manage || !plans.checkout_available) return 'none'
  const paysForIt = plan.key === plans.current_plan && hasLiveSubscription(plans.subscription)
  return paysForIt ? 'current' : 'choose'
}

interface PlanCardProps {
  readonly plan: BillingPlan
  readonly billingOpen: boolean
  readonly action: PlanAction
  /** The workspace is on this plan (paid or not). */
  readonly yours: boolean
  readonly highlighted: boolean
  /** A plan change on a subscribed workspace confirms on Stripe's portal page. */
  readonly switching: boolean
  readonly busy: boolean
  readonly disabled: boolean
  readonly onChoose: (plan: BillingPlan) => void
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

function PlanActionSlot({ plan, action, switching, busy, disabled, onChoose }: Omit<PlanCardProps, 'billingOpen' | 'yours' | 'highlighted'>) {
  if (action === 'contact') return <ContactLink />
  if (action === 'current') {
    return (
      <Button type="button" size="sm" variant="outline" disabled>
        Current plan
      </Button>
    )
  }
  if (action === 'none') return null
  return (
    <Button type="button" size="sm" disabled={disabled} onClick={() => onChoose(plan)}>
      {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
      {switching ? `Switch to ${plan.name}` : `Choose ${plan.name}`}
    </Button>
  )
}

export function PlanCard(props: PlanCardProps) {
  const { plan, billingOpen, yours, highlighted } = props
  const credits = planCreditsLine(plan, billingOpen)
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
        {yours && <Badge variant="outline">Your plan</Badge>}
      </div>
      <p className="text-sm font-medium text-foreground">{planPriceLabel(plan)}</p>
      <p className="flex-1 text-sm text-muted-foreground">{planDescription(plan, billingOpen)}</p>
      {credits && <p className="text-sm text-foreground">{credits}</p>}
      <div>
        <PlanActionSlot {...props} />
      </div>
    </li>
  )
}
