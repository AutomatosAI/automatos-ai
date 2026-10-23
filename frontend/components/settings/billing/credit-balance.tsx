'use client'

/**
 * PRD-265 US-011 — the workspace's credits on the Billing tab, from the same summary as the
 * header pill (useBillingSummary): the balance by bucket (spent in this order: trial, monthly,
 * top-up), what can be spent now, and the paused state. Work on Automatos's AI keys pauses at
 * zero; a workspace with keys of its own keeps working, so it is told that instead.
 */
import { isWorkPaused, type BillingSummary } from '@/hooks/use-billing-summary'
import { CREDIT_UNIT, creditsLabel, formatCredits } from './billing-format'

export const NOT_METERED_NOTE = "This plan isn't charged in credits."
export const OWN_KEYS_AT_ZERO_NOTE = 'No credits left. Your own AI keys keep working.'
export const LOW_NOTE = 'Credits are running low.'
const PAUSED_NOTE = "Work on Automatos's AI keys is paused: there are no credits left. Add your own AI key"
const PAUSED_WAYS_OUT_OPEN = ', top up or change plan to keep going.'
const PAUSED_WAYS_OUT_BETA = ' to keep going.'

const BUCKETS = [
  { key: 'trial', label: 'Trial' },
  { key: 'monthly', label: 'Monthly' },
  { key: 'topup', label: 'Top-up' },
] as const

/** The paused or low line, or null when there is nothing to say. */
export function balanceNote(summary: BillingSummary, billingOpen: boolean): string | null {
  if (isWorkPaused(summary)) return PAUSED_NOTE + (billingOpen ? PAUSED_WAYS_OUT_OPEN : PAUSED_WAYS_OUT_BETA)
  if (summary.paused) return OWN_KEYS_AT_ZERO_NOTE
  return summary.low ? LOW_NOTE : null
}

export function CreditBalance({ summary, billingOpen }: { summary: BillingSummary; billingOpen: boolean }) {
  if (!summary.metered) return <p className="text-sm text-muted-foreground">{NOT_METERED_NOTE}</p>
  const note = balanceNote(summary, billingOpen)
  const spendable = Math.max(0, summary.balance.spendable ?? 0)

  return (
    <div data-testid="billing-credit-balance" className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">Credits</p>
        <p className="text-sm font-medium text-foreground">{creditsLabel(spendable)} to spend</p>
      </div>
      <dl className="grid grid-cols-3 gap-3">
        {BUCKETS.map((bucket) => (
          <div key={bucket.key} className="rounded-xl border border-border bg-secondary/20 px-3 py-2">
            <dt className="text-xs text-muted-foreground">{bucket.label}</dt>
            <dd className="text-base font-semibold tabular-nums text-foreground">
              {formatCredits(summary.balance[bucket.key])}
            </dd>
          </div>
        ))}
      </dl>
      {note && (
        <p role="status" className="rounded-xl border border-warning/30 bg-warning/5 px-3 py-2 text-sm text-warning">
          {note}
        </p>
      )}
      <p className="text-xs text-muted-foreground">{CREDIT_UNIT}</p>
    </div>
  )
}
