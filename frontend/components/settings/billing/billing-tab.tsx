'use client'

/**
 * PRD-265 US-011 / W2 — Settings → Billing (hosted edition only; every member sees it).
 *
 * Reads `GET /api/v1/billing/plans` (useBillingPlans), the shared credit summary
 * (useBillingSummary) and, for owners and admins, the ledger. From top to bottom:
 *   - the current plan: name, price, subscription state, renewal, a scheduled downgrade,
 *     "Cancel plan" / "Resume plan", the credits by bucket and the paused state
 *     (./current-plan-card.tsx);
 *   - the payment form, when a plan or a top-up is being bought (./checkout-panel.tsx);
 *   - "Plans": every plan with the public facts-sheet copy; choosing one opens the payment
 *     form, switching one opens the plan-change preview (./plan-picker.tsx);
 *   - "Top up", once billing is open or for a platform admin testing before then
 *     (./top-up-card.tsx);
 *   - "Payment": the card on file and the invoices, owner/admin only (./payment-section.tsx);
 *   - the credit ledger, owner/admin only (./ledger-section.tsx).
 * Everything happens in this tab: no Stripe page is ever opened. After a payment the tab asks
 * for the billing reads again until the plan and the credits change
 * (hooks/use-billing-settle.ts). `?plan=` highlights a plan (Auto's onboarding link).
 */
import { useState } from 'react'
import { useSearchParams } from 'next/navigation'
import type { CheckoutRequest } from '@/hooks/use-billing-actions'
import { useBillingPlans, type BillingPlans } from '@/hooks/use-billing-plans'
import { useBillingSettle } from '@/hooks/use-billing-settle'
import { useBillingSummary, type BillingSummary } from '@/hooks/use-billing-summary'
import { CheckoutPanel, checkoutKey } from './checkout-panel'
import { CurrentPlanCard } from './current-plan-card'
import { LedgerSection } from './ledger-section'
import { PaymentSection } from './payment-section'
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
  readonly onPaid: () => void
}

function BillingSections({ plans, summary, highlight, onPaid }: BillingSectionsProps) {
  const [checkout, setCheckout] = useState<CheckoutRequest | null>(null)
  return (
    <>
      <CurrentPlanCard plans={plans} summary={summary} />
      {checkout && (
        <CheckoutPanel
          key={checkoutKey(checkout)}
          request={checkout}
          plans={plans}
          onPaid={onPaid}
          onClose={() => setCheckout(null)}
        />
      )}
      <PlanPicker
        plans={plans}
        highlight={highlight}
        trialCredits={summary?.trial_credits}
        onChoose={(plan) => setCheckout({ kind: 'plan', plan: plan.key })}
        onPaid={onPaid}
      />
      {(plans.billing_open || plans.checkout_available) && (
        <TopUpCard plans={plans} onBuy={() => setCheckout({ kind: 'topup' })} />
      )}
      {plans.can_manage && plans.subscription.has_billing_account && <PaymentSection />}
      {plans.can_manage && <LedgerSection />}
    </>
  )
}

export function BillingTab() {
  const settle = useBillingSettle()
  useCheckoutReturn(settle.start)
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
      {plans.data && <BillingSections plans={plans.data} summary={summary} highlight={highlight} onPaid={settle.start} />}
    </div>
  )
}
