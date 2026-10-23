'use client'

/**
 * PRD-265 W2 — paying for a plan or a top-up inside the Billing tab.
 *
 * The panel opens in the tab when a plan's "Choose" or "Buy a top-up" is pressed. It says what
 * is being bought, asks the buyer to accept the terms when the API has terms to accept
 * (`consent_text`), then opens a Checkout Session (`POST /api/v1/billing/checkout`, with
 * `accept_terms: true` once ticked) and mounts Stripe's card fields in our own form
 * (./checkout-form.tsx), styled from the page's theme (lib/stripe-appearance.ts). The browser
 * never leaves the page: a bank's 3-D Secure check may show over it, nothing more.
 *
 * Once the payment goes through the panel says so, and the tab waits for the plan and the
 * credits to change (`onPaid`, hooks/use-billing-settle.ts).
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { CheckoutElementsProvider } from '@stripe/react-stripe-js/checkout'
import { Loader2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { billingErrorMessage } from '@/lib/billing-errors'
import { getStripe } from '@/lib/stripe'
import { stripeAppearance } from '@/lib/stripe-appearance'
import {
  CHECKOUT_FAILED,
  useCheckoutSession,
  type CheckoutRequest,
  type CheckoutSession,
} from '@/hooks/use-billing-actions'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import { CheckoutForm } from './checkout-form'
import { planPriceLabel } from './plan-copy'
import { topUpOffer } from './top-up-card'

export const PAYMENT_RECEIVED_NOTE = 'Payment received — your plan updates in a moment.'
export const TERMS_FIRST_NOTE = 'Tick the box above to open the payment form.'
const TERMS_ID = 'billing-checkout-terms'

interface CheckoutPanelProps {
  readonly request: CheckoutRequest
  readonly plans: BillingPlans
  readonly onPaid: () => void
  readonly onClose: () => void
}

/** One key per thing being bought, so choosing another plan opens a fresh panel. */
export function checkoutKey(request: CheckoutRequest): string {
  return request.kind === 'plan' ? `plan:${request.plan}` : 'topup'
}

/** The heading and the line under it: what is being bought, for how much. */
export function checkoutSummary(request: CheckoutRequest, plans: BillingPlans): { title: string; detail: string } {
  if (request.kind === 'topup') return { title: 'Buy a top-up', detail: topUpOffer(plans.topup) }
  const plan = plans.plans.find((candidate) => candidate.key === request.plan)
  const name = plan?.name ?? request.plan
  return { title: `Subscribe to ${name}`, detail: plan ? planPriceLabel(plan) : '' }
}

function TermsCheckbox({ text, accepted, onChange }: { text: string; accepted: boolean; onChange: (accepted: boolean) => void }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-border bg-secondary/20 px-3 py-2 text-sm text-foreground">
      <Checkbox
        id={TERMS_ID}
        aria-label="I accept the terms"
        aria-describedby={`${TERMS_ID}-text`}
        checked={accepted}
        onCheckedChange={(state) => onChange(state === true)}
        className="mt-0.5"
      />
      <label id={`${TERMS_ID}-text`} htmlFor={TERMS_ID}>
        {text}
      </label>
    </div>
  )
}

/** Opens the session once the terms (if any) are accepted, and again after "Try again". */
function useSessionWhenReady(request: CheckoutRequest, needsTerms: boolean, accepted: boolean) {
  const session = useCheckoutSession()
  const { status, mutate } = session
  useEffect(() => {
    if (status !== 'idle' || (needsTerms && !accepted)) return
    mutate(needsTerms ? { ...request, accept_terms: true } : request)
  }, [status, mutate, request, needsTerms, accepted])
  return session
}

function SessionForm({ session, termsAccepted, onPaid }: { session: CheckoutSession; termsAccepted: boolean; onPaid: () => void }) {
  const stripe = getStripe(session.publishable_key)
  const options = useMemo(
    () => ({ clientSecret: session.client_secret, elementsOptions: { appearance: stripeAppearance() } }),
    [session.client_secret],
  )
  return (
    <CheckoutElementsProvider stripe={stripe} options={options}>
      <CheckoutForm termsAccepted={termsAccepted} onPaid={onPaid} />
    </CheckoutElementsProvider>
  )
}

function PanelBody({ request, plans, onPaid }: Omit<CheckoutPanelProps, 'onClose'>) {
  const terms = plans.consent_text?.trim() ?? ''
  const [accepted, setAccepted] = useState(false)
  const session = useSessionWhenReady(request, terms !== '', accepted)
  const termsAccepted = terms === '' || accepted

  return (
    <div className="space-y-4">
      {terms && <TermsCheckbox text={terms} accepted={accepted} onChange={setAccepted} />}
      {!termsAccepted && !session.data && (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">{TERMS_FIRST_NOTE}</p>
          <Button type="button" disabled>
            Pay
          </Button>
        </div>
      )}
      {session.isLoading && (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          Opening the payment form…
        </p>
      )}
      {session.isError && (
        <div className="space-y-2">
          <p role="alert" className="text-sm text-destructive">
            {billingErrorMessage(session.error, CHECKOUT_FAILED)}
          </p>
          <Button type="button" size="sm" variant="outline" onClick={() => session.reset()}>
            Try again
          </Button>
        </div>
      )}
      {session.data && <SessionForm session={session.data} termsAccepted={termsAccepted} onPaid={onPaid} />}
    </div>
  )
}

export function CheckoutPanel({ request, plans, onPaid, onClose }: CheckoutPanelProps) {
  const [paid, setPaid] = useState(false)
  const panel = useRef<HTMLDivElement>(null)
  const summary = checkoutSummary(request, plans)
  const markPaid = () => {
    setPaid(true)
    onPaid()
  }

  useEffect(() => {
    panel.current?.scrollIntoView?.({ behavior: 'smooth', block: 'start' })
  }, [])

  return (
    <Card ref={panel} data-testid="billing-checkout" className="space-y-4 border border-primary/40 p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-foreground">{summary.title}</h3>
          {summary.detail && <p className="text-sm text-muted-foreground">{summary.detail}</p>}
        </div>
        <Button type="button" size="icon" variant="ghost" aria-label="Close the payment form" onClick={onClose}>
          <X className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
      {paid ? (
        <p role="status" className="rounded-xl border border-success/30 bg-success/5 px-3 py-2 text-sm text-success">
          {PAYMENT_RECEIVED_NOTE}
        </p>
      ) : (
        <PanelBody request={request} plans={plans} onPaid={markPaid} />
      )}
    </Card>
  )
}
