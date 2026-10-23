'use client'

/**
 * PRD-265 W2 — "Payment" on the Billing tab, for the workspace's owner and admins once it has
 * a billing account: the card on file with "Change card" (./payment-method-card.tsx) and the
 * invoices with their PDFs (./invoices-table.tsx). All of it inside the tab; there is no
 * Stripe page to go to.
 */
import { Card } from '@/components/ui/card'
import { InvoicesTable } from './invoices-table'
import { PaymentMethodCard } from './payment-method-card'

export function PaymentSection() {
  return (
    <section aria-labelledby="billing-payment-title" data-testid="billing-payment" className="space-y-3">
      <h3 id="billing-payment-title" className="text-xs uppercase tracking-wide text-muted-foreground">
        Payment
      </h3>
      <Card className="space-y-4 border border-border p-5">
        <PaymentMethodCard />
        <InvoicesTable />
      </Card>
    </section>
  )
}
