'use client'

/**
 * PRD-265 W2 — what the buyer pays, from the Checkout Session itself: the subtotal, the tax
 * (Stripe Tax works it out from the billing address, so until there is one it says so), the
 * total due today and, for a plan, what each renewal costs. Stripe formats every amount in
 * the session's currency.
 */
import type { StripeCheckoutElementsValue } from '@stripe/react-stripe-js/checkout'

export const TAX_PENDING = 'Worked out from your billing address'
const TAX_READY = 'ready'

type TotalsSession = Pick<StripeCheckoutElementsValue, 'total' | 'tax' | 'recurring'>

/** "$5.80", "$5.80 included" (tax inside the price), or why there is no figure yet. */
export function taxLine(checkout: Pick<TotalsSession, 'total' | 'tax'>): string {
  if (checkout.tax?.status !== TAX_READY) return TAX_PENDING
  const { taxExclusive, taxInclusive } = checkout.total
  if (taxInclusive.minorUnitsAmount > 0 && taxExclusive.minorUnitsAmount === 0) return `${taxInclusive.amount} included`
  return taxExclusive.amount
}

/** "Then $29.00 a month." for a plan; null for a one-off purchase. */
export function renewalLine(checkout: Pick<TotalsSession, 'recurring'>): string | null {
  const recurring = checkout.recurring
  if (!recurring) return null
  const every = recurring.intervalCount > 1 ? `every ${recurring.intervalCount} ${recurring.interval}s` : `a ${recurring.interval}`
  return `Then ${recurring.dueNext.total.amount} ${every}.`
}

function TotalRow({ label, value, strong = false }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className={strong ? 'text-sm font-semibold text-foreground' : 'text-sm text-muted-foreground'}>{label}</dt>
      <dd className={strong ? 'text-sm font-semibold tabular-nums text-foreground' : 'text-sm tabular-nums text-foreground'}>{value}</dd>
    </div>
  )
}

export function CheckoutTotals({ checkout }: { checkout: TotalsSession }) {
  const renewal = renewalLine(checkout)
  return (
    <div data-testid="billing-checkout-totals" className="space-y-2 rounded-xl border border-border bg-secondary/20 px-3 py-2">
      <dl className="space-y-1">
        <TotalRow label="Subtotal" value={checkout.total.subtotal.amount} />
        <TotalRow label="Tax" value={taxLine(checkout)} />
        <TotalRow label="Total due today" value={checkout.total.total.amount} strong />
      </dl>
      {renewal && <p className="text-xs text-muted-foreground">{renewal}</p>}
    </div>
  )
}
