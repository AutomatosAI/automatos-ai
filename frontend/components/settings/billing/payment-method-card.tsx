'use client'

/**
 * PRD-265 W2 — the card the workspace pays with ("Visa •••• 4242 · expires 12/2028") and
 * "Change card", which opens the tab's own card form (./card-change-form.tsx) on a SetupIntent
 * the API opens for it. Owner/admin only (the Billing tab mounts it in the Payment section).
 */
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useCardSetup, usePaymentMethod, type CardOnFile } from '@/hooks/use-billing-payment'
import { CardChangeForm } from './card-change-form'

export const NO_CARD = 'No card on file.'
export const CARD_FAILED = "The card on file couldn't be loaded."

const BRAND_NAMES: Readonly<Record<string, string>> = {
  amex: 'American Express',
  diners: 'Diners Club',
  discover: 'Discover',
  jcb: 'JCB',
  mastercard: 'Mastercard',
  unionpay: 'UnionPay',
  visa: 'Visa',
}
const MONTH_DIGITS = 2

/** "Visa •••• 4242 · expires 12/2028"; "No card on file." for none. */
export function cardLabel(card: CardOnFile | null | undefined): string {
  if (!card) return NO_CARD
  const brand = BRAND_NAMES[card.brand] ?? card.brand
  const month = String(card.exp_month).padStart(MONTH_DIGITS, '0')
  return `${brand} •••• ${card.last4} · expires ${month}/${card.exp_year}`
}

function CardLine({ card }: { card: ReturnType<typeof usePaymentMethod> }) {
  if (card.isLoading) return <p className="text-sm text-muted-foreground">Loading the card…</p>
  if (card.isError) return <p role="alert" className="text-sm text-destructive">{CARD_FAILED}</p>
  return <p className="text-sm font-medium text-foreground">{cardLabel(card.data)}</p>
}

export function PaymentMethodCard() {
  const card = usePaymentMethod(true)
  const setup = useCardSetup()
  const session = setup.data

  return (
    <div data-testid="billing-card-on-file" className="space-y-3 rounded-2xl border border-border p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Card</p>
          <CardLine card={card} />
        </div>
        {!session && (
          <Button type="button" size="sm" variant="outline" disabled={setup.isLoading} onClick={() => setup.mutate()}>
            {setup.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
            {card.data ? 'Change card' : 'Add a card'}
          </Button>
        )}
      </div>
      {session && <CardChangeForm setup={session} onDone={() => setup.reset()} />}
    </div>
  )
}
