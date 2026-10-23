'use client'

/**
 * PRD-265 US-011 / W2 — "Top up": the one-off credit pack from the plans answer's `topup`
 * block ("1,000 credits for $10, valid 12 months"). "Buy a top-up" opens the tab's own payment
 * form for `{kind: 'topup'}` (`onBuy`, ./checkout-panel.tsx). The button shows only when the
 * workspace can take a top-up (an active subscription on a plan with platform AI), the caller
 * is an owner or admin, and paying is available; otherwise the card says why not. The Billing
 * tab mounts the card only once billing is open, or for a platform admin testing before then.
 */
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import type { BillingPlans, BillingTopUpOffer } from '@/hooks/use-billing-plans'
import { formatCredits, formatUsd, validMonths } from './billing-format'
import { CHECKOUT_CLOSED_NOTE, OWNER_ONLY_NOTE } from './plan-copy'

export const TOP_UP_SPEND_NOTE = 'Top-up credits are spent after your monthly credits.'

/** "1,000 credits for $10, valid 12 months". */
export function topUpOffer(topup: BillingTopUpOffer): string {
  return `${formatCredits(topup.credits)} credits for ${formatUsd(topup.price_usd)}, valid ${validMonths(topup.valid_days)} months`
}

/** The plans that take top-ups, by display name: "Pro or Business". */
function topUpPlanNames(plans: BillingPlans): string {
  const names = plans.plans.filter((plan) => plan.platform_ai && !plan.coming_soon).map((plan) => plan.name)
  return names.length > 1 ? `${names.slice(0, -1).join(', ')} or ${names[names.length - 1]}` : names.join('')
}

/** The line under the offer: what happens to the credits, or why there is no button. */
export function topUpNote(plans: BillingPlans): string {
  if (!plans.topup.available) return `Top-ups need an active ${topUpPlanNames(plans)} subscription.`
  if (!plans.can_manage) return OWNER_ONLY_NOTE
  return plans.checkout_available ? TOP_UP_SPEND_NOTE : CHECKOUT_CLOSED_NOTE
}

export function TopUpCard({ plans, onBuy }: { plans: BillingPlans; onBuy: () => void }) {
  const canBuy = plans.topup.available && plans.can_manage && plans.checkout_available

  return (
    <Card data-testid="billing-top-up" className="space-y-2 border border-border p-5">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">Top-up</p>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <p className="text-base font-semibold text-foreground">{topUpOffer(plans.topup)}</p>
          <p className="text-sm text-muted-foreground">{topUpNote(plans)}</p>
        </div>
        {canBuy && (
          <Button type="button" size="sm" onClick={onBuy}>
            Buy a top-up
          </Button>
        )}
      </div>
    </Card>
  )
}
