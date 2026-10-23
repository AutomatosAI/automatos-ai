'use client'

/**
 * PRD-265 US-011 — Settings → Billing (hosted edition only; every member sees it).
 *
 * Reads `GET /api/v1/billing/plans` (useBillingPlans), the shared credit summary
 * (useBillingSummary) and, for owners and admins, the ledger. From top to bottom:
 *   - the current plan: name, price, subscription state, renewal, "Manage payment", the
 *     credits by bucket and the paused state (./current-plan-card.tsx);
 *   - "Change plan": every plan with the public facts-sheet copy (./plan-picker.tsx);
 *   - "Top up", once billing is open or for a platform admin testing before then
 *     (./top-up-card.tsx);
 *   - the credit ledger, owner/admin only (./ledger-section.tsx).
 * Checkout and portal buttons show only where the API says the caller may use them
 * (`can_manage`, `checkout_available`). Coming back from Stripe is handled here too
 * (./use-checkout-return.ts); `?plan=` highlights a plan (Auto's onboarding link).
 */
import { useSearchParams } from 'next/navigation'
import { useBillingPlans, type BillingPlans } from '@/hooks/use-billing-plans'
import { useBillingSummary, type BillingSummary } from '@/hooks/use-billing-summary'
import { CurrentPlanCard } from './current-plan-card'
import { LedgerSection } from './ledger-section'
import { PlanPicker } from './plan-picker'
import { TopUpCard } from './top-up-card'
import { useCheckoutReturn } from './use-checkout-return'

export const PLAN_PARAM = 'plan'
export const BILLING_LEDE = "This workspace's plan, credits and payments."
export const BILLING_FAILED = 'Billing could not be loaded. Try again in a moment.'

interface BillingSectionsProps {
  readonly plans: BillingPlans
  readonly summary: BillingSummary | undefined
  readonly highlight: string | null
}

function BillingSections({ plans, summary, highlight }: BillingSectionsProps) {
  return (
    <>
      <CurrentPlanCard plans={plans} summary={summary} />
      <PlanPicker plans={plans} highlight={highlight} trialCredits={summary?.trial_credits} />
      {(plans.billing_open || plans.checkout_available) && <TopUpCard plans={plans} />}
      {plans.can_manage && <LedgerSection />}
    </>
  )
}

export function BillingTab() {
  useCheckoutReturn()
  const plans = useBillingPlans()
  const { data: summary } = useBillingSummary()
  const highlight = useSearchParams()?.get(PLAN_PARAM) ?? null

  return (
    <div data-testid="billing-tab" className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold text-foreground">Billing</h2>
        <p className="text-sm text-muted-foreground">{BILLING_LEDE}</p>
      </div>
      {plans.isLoading && <p className="text-sm text-muted-foreground">Loading billing…</p>}
      {plans.isError && !plans.data && (
        <p role="alert" className="text-sm text-destructive">
          {BILLING_FAILED}
        </p>
      )}
      {plans.data && <BillingSections plans={plans.data} summary={summary} highlight={highlight} />}
    </div>
  )
}
