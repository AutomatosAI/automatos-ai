'use client'

import { isWorkPaused, useBillingSummary, type BillingSummary } from '@/hooks/use-billing-summary'
import { isSaaS } from '@/lib/auth-edition'

/**
 * PRD-265 US-005 / US-008 — the credit balance chip in the header.
 *
 * Reads the workspace's credits off `GET /api/v1/billing/summary`
 * (useBillingSummary), never a second bookkeeping source. Says "420 of 500 trial
 * credits" while the trial is the only bucket with credits, else "1,500 credits";
 * goes amber when the balance runs low, and reads "Paused · 0 credits" once work
 * on Automatos's AI keys has stopped. A workspace on its own keys keeps working
 * at zero credits, so it is amber there but never "Paused". The tooltip lists the
 * buckets. Hidden in the local edition (nothing is metered), while the summary
 * loads or failed to load, and on a plan billed outside credits (`metered` false).
 *
 * Below the `sm` breakpoint the chip shows the number only (or "Paused") so the
 * phone header keeps its controls on screen; the full words stay for screen
 * readers. `className` lets the mount position it.
 */

type PillState = 'ok' | 'low' | 'paused'

const BASE_CLASSES =
  'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium whitespace-nowrap'
const STATE_CLASSES: Readonly<Record<PillState, string>> = {
  ok: 'border-border bg-muted/40 text-muted-foreground',
  low: 'border-warning/40 bg-warning/10 text-warning',
  paused: 'border-warning/40 bg-warning/10 text-warning',
}
const DOT_CLASSES: Readonly<Record<PillState, string>> = {
  ok: 'bg-success',
  low: 'bg-warning',
  paused: 'bg-warning',
}
const PAUSED_WORD = 'Paused'
const PAUSED_NOTE = "Work on Automatos's AI keys is paused."
const OWN_KEYS_NOTE = 'Your own AI keys keep working.'

function formatCount(credits: number): string {
  return credits.toLocaleString('en-US')
}

function creditsLabel(credits: number): string {
  return `${formatCount(credits)} ${credits === 1 ? 'credit' : 'credits'}`
}

function pillStateOf(summary: BillingSummary): PillState {
  if (isWorkPaused(summary)) return 'paused'
  return summary.low || summary.paused ? 'low' : 'ok'
}

/** The trial is the only bucket with credits in it. */
function onlyTrialCredits(summary: BillingSummary): boolean {
  return summary.balance.monthly <= 0 && summary.balance.topup <= 0
}

function labelOf(summary: BillingSummary, state: PillState): string {
  const spendable = Math.max(0, summary.balance.spendable ?? 0)
  if (state === 'paused') return `${PAUSED_WORD} · ${creditsLabel(spendable)}`
  if (onlyTrialCredits(summary)) {
    return `${formatCount(Math.max(0, summary.balance.trial))} of ${formatCount(summary.trial_credits)} trial credits`
  }
  return creditsLabel(spendable)
}

function shortLabelOf(summary: BillingSummary, state: PillState): string {
  return state === 'paused' ? PAUSED_WORD : formatCount(Math.max(0, summary.balance.spendable ?? 0))
}

function tooltipOf(summary: BillingSummary): string {
  const { trial, monthly, topup } = summary.balance
  const lines = [`Trial: ${creditsLabel(trial)}`, `Monthly: ${creditsLabel(monthly)}`, `Top-up: ${creditsLabel(topup)}`]
  if (summary.paused) lines.push(PAUSED_NOTE)
  if (summary.own_keys) lines.push(OWN_KEYS_NOTE)
  return lines.join('\n')
}

export function CreditBalancePill({ className = '' }: { className?: string }) {
  const { data: summary } = useBillingSummary()

  if (!isSaaS || !summary || !summary.metered) return null

  const state = pillStateOf(summary)
  return (
    <span
      data-testid="credit-balance-pill"
      data-state={state}
      title={tooltipOf(summary)}
      className={[BASE_CLASSES, STATE_CLASSES[state], className].join(' ')}
    >
      <span aria-hidden="true" className={`h-1.5 w-1.5 shrink-0 rounded-full ${DOT_CLASSES[state]}`} />
      <span className="sr-only sm:not-sr-only">{labelOf(summary, state)}</span>
      <span aria-hidden="true" className="sm:hidden">
        {shortLabelOf(summary, state)}
      </span>
    </span>
  )
}
