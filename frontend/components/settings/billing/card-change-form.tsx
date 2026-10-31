'use client'

/**
 * PRD-265 W2 — changing the card on file, inside the Billing tab.
 *
 * Stripe's card field (an iframe, in our theme) in our own form, mounted on the SetupIntent
 * the API opened (`POST /api/v1/billing/payment-method/setup`). "Save card" confirms it in the
 * page (`stripe.confirmSetup({elements, redirect: 'if_required'})`; a bank's check may show
 * over the page), then tells the API which SetupIntent to take the card from
 * (`POST /api/v1/billing/payment-method`), which makes it the workspace's default.
 */
import { useState } from 'react'
import { Elements, PaymentElement, useElements, useStripe } from '@stripe/react-stripe-js'
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { billingErrorMessage, stripeErrorMessage } from '@/lib/billing-errors'
import { getStripe } from '@/lib/stripe'
import { stripeAppearance } from '@/lib/stripe-appearance'
import { CARD_SAVE_FAILED, useSavePaymentMethod, type CardSetup } from '@/hooks/use-billing-payment'

interface CardChangeFormProps {
  readonly setup: CardSetup
  /** The form is done: the card was saved, or the buyer stopped. */
  readonly onDone: () => void
}

function CardChangeFields({ onDone }: Pick<CardChangeFormProps, 'onDone'>) {
  const stripe = useStripe()
  const elements = useElements()
  const save = useSavePaymentMethod()
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async () => {
    if (!stripe || !elements) return
    setSaving(true)
    setError(null)
    try {
      const result = await stripe.confirmSetup({ elements, redirect: 'if_required' })
      const setupIntentId = result.setupIntent?.id
      if (result.error || !setupIntentId) {
        setError(stripeErrorMessage(result.error, CARD_SAVE_FAILED))
        return
      }
      await save.mutateAsync(setupIntentId)
      onDone()
    } catch (failure) {
      setError(billingErrorMessage(failure, CARD_SAVE_FAILED))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div data-testid="billing-card-form" className="space-y-3">
      <PaymentElement />
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <Button type="button" size="sm" disabled={!stripe || !elements || saving} onClick={() => void submit()}>
          {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
          Save card
        </Button>
        <Button type="button" size="sm" variant="outline" disabled={saving} onClick={onDone}>
          Not now
        </Button>
      </div>
    </div>
  )
}

export function CardChangeForm({ setup, onDone }: CardChangeFormProps) {
  const [appearance] = useState(stripeAppearance)
  return (
    <Elements stripe={getStripe(setup.publishable_key)} options={{ clientSecret: setup.client_secret, appearance }}>
      <CardChangeFields onDone={onDone} />
    </Elements>
  )
}
