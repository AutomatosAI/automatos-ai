'use client'

/**
 * PRD-265 W2 — the payment form inside the checkout panel (./checkout-panel.tsx), under
 * `CheckoutElementsProvider`.
 *
 * Stripe's billing-address and card fields (iframes, in our theme), our own optional VAT
 * number (./vat-field.tsx), the totals with tax from the session (./checkout-totals.tsx), and
 * Pay. Pay is on only when Stripe says the session can be confirmed (`canConfirm`) and the
 * terms, if any, are ticked. Paying confirms in the page (`confirm({redirect: 'if_required'})`):
 * a card is never sent anywhere, and a bank's 3-D Secure check shows over the page. A decline
 * is shown in Stripe's own words ("Your card was declined.").
 */
import { useState } from 'react'
import {
  BillingAddressElement,
  PaymentElement,
  useCheckoutElements,
  type StripeCheckoutElementsValue,
} from '@stripe/react-stripe-js/checkout'
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { stripeErrorMessage } from '@/lib/billing-errors'
import { CheckoutTotals } from './checkout-totals'
import { VatField } from './vat-field'

export const FORM_FAILED = 'The payment form could not load. Try again in a moment.'
export const PAY_FAILED = "The payment didn't go through. Nothing was charged."
export const FORM_LOADING = 'Loading the payment form…'

interface CheckoutFormProps {
  readonly termsAccepted: boolean
  readonly onPaid: () => void
}

interface CheckoutFieldsProps extends CheckoutFormProps {
  readonly checkout: StripeCheckoutElementsValue
}

/** Confirm the session in the page; null when paid, else the reason in the buyer's words. */
async function payInPage(checkout: StripeCheckoutElementsValue): Promise<string | null> {
  const result = await checkout.confirm({ redirect: 'if_required' })
  return result.type === 'success' ? null : stripeErrorMessage(result.error, PAY_FAILED)
}

function CheckoutFields({ checkout, termsAccepted, onPaid }: CheckoutFieldsProps) {
  const [country, setCountry] = useState<string | null>(null)
  const [paying, setPaying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const billingCountry = country ?? checkout.billingAddress?.address?.country ?? null

  const pay = async () => {
    setPaying(true)
    setError(null)
    try {
      const refused = await payInPage(checkout)
      if (refused) setError(refused)
      else onPaid()
    } catch (failure) {
      setError(stripeErrorMessage(failure as { message?: unknown }, PAY_FAILED))
    } finally {
      setPaying(false)
    }
  }

  return (
    <div data-testid="billing-checkout-form" className="space-y-4">
      <BillingAddressElement onChange={(event) => setCountry(event.value?.address?.country || null)} />
      <PaymentElement />
      <VatField checkout={checkout} country={billingCountry} />
      <CheckoutTotals checkout={checkout} />
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <Button type="button" className="w-full sm:w-auto" disabled={!checkout.canConfirm || !termsAccepted || paying} onClick={() => void pay()}>
        {paying && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
        Pay {checkout.total.total.amount}
      </Button>
    </div>
  )
}

export function CheckoutForm(props: CheckoutFormProps) {
  const state = useCheckoutElements()
  if (state.type === 'loading') return <p className="text-sm text-muted-foreground">{FORM_LOADING}</p>
  if (state.type === 'error') {
    return (
      <p role="alert" className="text-sm text-destructive">
        {stripeErrorMessage(state.error, FORM_FAILED)}
      </p>
    )
  }
  return <CheckoutFields checkout={state.checkout} {...props} />
}
