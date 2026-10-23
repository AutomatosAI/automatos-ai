/**
 * PRD-265 W2 — after a payment in the Billing tab, waiting for Stripe's webhook to land.
 *
 * `start()` asks for the plans, the summary and the ledger again at once and every few
 * seconds; it stops as soon as what a payment changes (the plan, the subscription, the
 * credits) differs from how it stood at `start()`, and after a minute whatever happens.
 */
import React from 'react'
import { act, renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SETTLE_CAP_MS, SETTLE_POLL_MS, billingFingerprint, useBillingSettle } from '@/hooks/use-billing-settle'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import type { BillingSummary } from '@/hooks/use-billing-summary'

const PLANS_KEY = ['billing', 'plans']
const SUMMARY_KEY = ['billing', 'summary']

const PLANS = {
  current_plan: 'basic',
  subscription: { status: null, cancel_at_period_end: false, current_period_end: null, has_billing_account: false },
} as unknown as BillingPlans
const SUMMARY = { plan: 'basic', balance: { trial: 100, monthly: 0, topup: 0, spendable: 100 } } as unknown as BillingSummary

function renderSettle() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  client.setQueryData(PLANS_KEY, PLANS)
  client.setQueryData(SUMMARY_KEY, SUMMARY)
  const invalidate = vi.spyOn(client, 'invalidateQueries')
  const wrapper = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const view = renderHook(() => useBillingSettle(), { wrapper })
  const plansAsked = () => invalidate.mock.calls.filter(([filters]) => JSON.stringify(filters) === JSON.stringify({ queryKey: PLANS_KEY })).length
  return { client, view, plansAsked }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('useBillingSettle', () => {
  it('asks again at once and every few seconds while nothing has changed', () => {
    const { view, plansAsked } = renderSettle()
    expect(view.result.current.settling).toBe(false)

    act(() => view.result.current.start())
    expect(view.result.current.settling).toBe(true)
    expect(plansAsked()).toBe(1)

    act(() => {
      vi.advanceTimersByTime(SETTLE_POLL_MS * 3)
    })
    expect(plansAsked()).toBe(4)
    expect(view.result.current.settling).toBe(true)
  })

  it('stops as soon as the plan has changed', () => {
    const { client, view, plansAsked } = renderSettle()
    act(() => view.result.current.start())

    client.setQueryData(PLANS_KEY, { ...PLANS, current_plan: 'pro' })
    act(() => {
      vi.advanceTimersByTime(SETTLE_POLL_MS)
    })
    expect(view.result.current.settling).toBe(false)

    const asked = plansAsked()
    act(() => {
      vi.advanceTimersByTime(SETTLE_POLL_MS * 5)
    })
    expect(plansAsked()).toBe(asked)
  })

  it('stops after a minute whatever happens', () => {
    const { view, plansAsked } = renderSettle()
    act(() => view.result.current.start())

    act(() => {
      vi.advanceTimersByTime(SETTLE_CAP_MS + SETTLE_POLL_MS)
    })
    expect(view.result.current.settling).toBe(false)
    const asked = plansAsked()
    act(() => {
      vi.advanceTimersByTime(SETTLE_CAP_MS)
    })
    expect(plansAsked()).toBe(asked)
  })
})

describe('billingFingerprint', () => {
  it('changes with the plan, the subscription and the credits a payment adds', () => {
    const before = billingFingerprint(PLANS, SUMMARY)
    expect(billingFingerprint(PLANS, SUMMARY)).toBe(before)
    expect(billingFingerprint({ ...PLANS, current_plan: 'pro' }, SUMMARY)).not.toBe(before)
    expect(
      billingFingerprint({ ...PLANS, subscription: { ...PLANS.subscription, status: 'active' } }, SUMMARY),
    ).not.toBe(before)
    expect(billingFingerprint(PLANS, { ...SUMMARY, balance: { ...SUMMARY.balance, topup: 1000 } })).not.toBe(before)
    expect(billingFingerprint(PLANS, { ...SUMMARY, balance: { ...SUMMARY.balance, trial: 90 } })).toBe(before)
  })
})
