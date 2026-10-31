/**
 * PRD-265 US-011 — how the Billing tab writes credits, prices, dates and statuses.
 *
 * Credits are whole numbers (1 credit = 1 US cent of AI usage at Automatos prices); prices
 * are USD. Everything here is pure, so the tab's wording is tested without rendering it.
 */
import { formatAbsoluteDate } from '@/lib/format-relative'
import type { BillingSubscription } from '@/hooks/use-billing-plans'

export const CREDIT_UNIT = '1 credit = 1 US cent of AI usage.'
export const BETA_NOT_CHARGED = 'Beta pricing: not charged until billing opens.'
export const PAST_DUE_NOTICE = 'Payment failed, Stripe is retrying — update your card.'
export const INCOMPLETE_NOTICE = 'Waiting for the first payment to go through.'
export const CANCELLED_NOTICE = 'Cancelled.'
export const NO_SUBSCRIPTION_NOTICE = 'No paid plan yet.'

const USD = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 2 })
const BUCKET_LABELS: Readonly<Record<string, string>> = { trial: 'Trial', monthly: 'Monthly', topup: 'Top-up' }
const DAYS_PER_YEAR = 365
const MONTHS_PER_YEAR = 12
const CENTS_PER_UNIT = 100

export type NoticeTone = 'ok' | 'warning' | 'muted'

export interface StatusNotice {
  tone: NoticeTone
  text: string
}

/** "1,500": whole credits, thousands separated. */
export function formatCredits(credits: number): string {
  return Math.round(credits).toLocaleString('en-US')
}

/** "1 credit", "1,500 credits". */
export function creditsLabel(credits: number): string {
  return `${formatCredits(credits)} ${Math.abs(Math.round(credits)) === 1 ? 'credit' : 'credits'}`
}

/** "+500" for a grant, "-12" for a debit. */
export function signedCredits(credits: number): string {
  return credits > 0 ? `+${formatCredits(credits)}` : formatCredits(credits)
}

/** "$29". */
export function formatUsd(amount: number): string {
  return USD.format(amount)
}

/** "$13.50" for 1350 cents in "usd"; any ISO currency Stripe bills in. */
export function formatMoney(cents: number, currency: string): string {
  const code = (currency || 'usd').toUpperCase()
  try {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: code }).format(cents / CENTS_PER_UNIT)
  } catch {
    return `${(cents / CENTS_PER_UNIT).toFixed(2)} ${code}`
  }
}

/** "23 Nov 2026" (in the viewer's locale); "—" when there is no date. */
export function formatDay(iso: string | null | undefined): string {
  return formatAbsoluteDate(iso)
}

/** The ledger's date column: day and time. */
export function formatMoment(iso: string | null | undefined): string {
  if (!iso) return '—'
  const moment = new Date(iso)
  if (Number.isNaN(moment.getTime())) return '—'
  return moment.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

/** "Trial", "Monthly", "Top-up"; the raw bucket for anything else. */
export function bucketLabel(bucket: string | null | undefined): string {
  if (!bucket) return '—'
  return BUCKET_LABELS[bucket] ?? bucket
}

/** A validity in days as months: 365 → 12. */
export function validMonths(days: number): number {
  return Math.max(1, Math.round((days * MONTHS_PER_YEAR) / DAYS_PER_YEAR))
}

/** The line under the plan name: the subscription's state in plain words. */
export function subscriptionNotice(
  subscription: BillingSubscription,
  billingOpen: boolean,
  periodEnd: string | null,
): StatusNotice {
  const { status } = subscription
  if (status === 'past_due' || status === 'unpaid') return { tone: 'warning', text: PAST_DUE_NOTICE }
  if (subscription.cancel_at_period_end && periodEnd) return { tone: 'warning', text: `Ends on ${formatDay(periodEnd)}.` }
  if (status === 'active' || status === 'trialing') {
    return { tone: 'ok', text: periodEnd ? `Active · renews on ${formatDay(periodEnd)}.` : 'Active.' }
  }
  if (status === 'incomplete') return { tone: 'warning', text: INCOMPLETE_NOTICE }
  if (status === 'canceled') return { tone: 'muted', text: CANCELLED_NOTICE }
  return { tone: 'muted', text: billingOpen ? NO_SUBSCRIPTION_NOTICE : BETA_NOT_CHARGED }
}
