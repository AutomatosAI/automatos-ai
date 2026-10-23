/**
 * PRD-265 US-011 — shared answers for the Billing tab tests, shaped like the W2 API contract
 * (`GET /api/v1/billing/plans`, `/summary`, `/ledger`). Not a test file itself.
 */
import React from 'react'
import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { BillingLedger } from '@/hooks/use-billing-ledger'
import type { BillingPlan, BillingPlans, BillingSubscription } from '@/hooks/use-billing-plans'
import type { BillingSummary } from '@/hooks/use-billing-summary'

export const PERIOD_END = '2026-11-23T10:00:00+00:00'
export const CHECKOUT_URL = 'https://checkout.stripe.com/c/pay/cs_test_123'
export const PORTAL_URL = 'https://billing.stripe.com/p/session/test_123'

export const PLAN_LIST: BillingPlan[] = [
  { key: 'basic', name: 'Own keys', price_usd: 9, monthly_credits: 0, platform_ai: false, self_serve: true, coming_soon: false, seats: 5, max_agents: 20, render_minutes_month: 60, voice: false },
  { key: 'pro', name: 'Pro', price_usd: 29, monthly_credits: 2000, platform_ai: true, self_serve: true, coming_soon: false, seats: 5, max_agents: 20, render_minutes_month: 60, voice: false },
  { key: 'business', name: 'Business', price_usd: 99, monthly_credits: 8000, platform_ai: true, self_serve: false, coming_soon: false, seats: 25, max_agents: 0, render_minutes_month: 240, voice: true },
  { key: 'enterprise', name: 'Enterprise', price_usd: null, monthly_credits: null, platform_ai: true, self_serve: false, coming_soon: true },
]

export const NO_SUBSCRIPTION: BillingSubscription = {
  status: null,
  cancel_at_period_end: false,
  current_period_end: null,
  has_billing_account: false,
}

/** Beta (billing closed), an ordinary member: nothing can be bought. */
export const BETA_MEMBER: BillingPlans = {
  billing_open: false,
  checkout_available: false,
  can_manage: false,
  current_plan: 'basic',
  subscription: NO_SUBSCRIPTION,
  plans: PLAN_LIST,
  topup: { credits: 1000, price_usd: 10, valid_days: 365, available: false },
}

/** Beta, a platform admin who owns the workspace: checkout works before billing opens. */
export const BETA_ADMIN: BillingPlans = { ...BETA_MEMBER, checkout_available: true, can_manage: true }

/** Billing open, a Pro subscriber who cancelled at the period end. */
export const OPEN_PRO_CANCELLING: BillingPlans = {
  ...BETA_ADMIN,
  billing_open: true,
  current_plan: 'pro',
  subscription: { status: 'active', cancel_at_period_end: true, current_period_end: PERIOD_END, has_billing_account: true },
  topup: { ...BETA_MEMBER.topup, available: true },
}

/** Billing open, a Pro subscriber whose renewal failed. */
export const OPEN_PRO_PAST_DUE: BillingPlans = {
  ...OPEN_PRO_CANCELLING,
  subscription: { status: 'past_due', cancel_at_period_end: false, current_period_end: PERIOD_END, has_billing_account: true },
}

export const SUMMARY: BillingSummary = {
  plan: 'basic',
  plan_name: 'Own keys',
  price_usd: 9,
  price_label: 'beta',
  billing_open: false,
  status: 'active',
  renewal_date: null,
  balance: { trial: 420, monthly: 0, topup: 0, spendable: 420 },
  trial_credits: 500,
  platform_ai: false,
  metered: true,
  paused: false,
  low: false,
  own_keys: false,
}

export const PAUSED_SUMMARY: BillingSummary = {
  ...SUMMARY,
  status: 'paused',
  balance: { trial: 0, monthly: 0, topup: 0, spendable: 0 },
  paused: true,
}

export function ledgerPage(page: number, total = 2, pageSize = 25): BillingLedger {
  return {
    items: [
      { id: page * 10 + 1, at: '2026-10-22T09:30:00+00:00', kind: 'debit', bucket: 'trial', credits: -12, what: 'AI usage', model: 'claude-haiku-5-5' },
      { id: page * 10 + 2, at: '2026-10-20T08:00:00+00:00', kind: 'grant', bucket: 'trial', credits: 500, what: 'Trial credits', model: null },
    ],
    total,
    page,
    page_size: pageSize,
    usage: { since: '2026-09-22T09:30:00+00:00', by_model: [{ model: 'claude-haiku-5-5', calls: 7, credits: 80 }] },
  }
}

/** The error apiClient throws for a non-2xx answer: the status and the parsed body ride it. */
export function httpError(status: number, detail: unknown): Error {
  const message = typeof detail === 'string' ? detail : JSON.stringify(detail)
  return Object.assign(new Error(message), { status, body: { detail } })
}

export function newQueryClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { retry: false, retryDelay: 0 }, mutations: { retry: false } } })
}

export function renderWithClient(ui: React.ReactElement, client: QueryClient = newQueryClient()) {
  return { client, ...render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>) }
}

export interface BillingAnswers {
  plans: BillingPlans
  summary: BillingSummary
  ledger: (page: number) => Promise<BillingLedger>
}

/** apiClient.request stand-in: answers each billing GET from `answers` by its path. */
export function answerBillingReads(answers: BillingAnswers) {
  return async (path: string) => {
    if (path.startsWith('/api/v1/billing/plans')) return answers.plans
    if (path.startsWith('/api/v1/billing/summary')) return answers.summary
    if (path.startsWith('/api/v1/billing/ledger')) {
      const page = Number(new URLSearchParams(path.split('?')[1] ?? '').get('page') ?? '1')
      return answers.ledger(page)
    }
    throw new Error(`unexpected request ${path}`)
  }
}
