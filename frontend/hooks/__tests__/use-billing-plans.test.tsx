/**
 * PRD-265 US-011 — useBillingPlans and useBillingLedger: the paths they ask, that neither asks
 * in the local edition (nothing is metered there; the routes answer 404), and that a 403 from
 * the ledger is not asked again. A real QueryClient; apiClient and the edition flag are stand-ins.
 */
import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { requestMock, edition } = vi.hoisted(() => ({
  requestMock: vi.fn(),
  edition: { saas: true },
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: requestMock } }))
vi.mock('@/lib/auth-edition', () => ({
  get isSaaS() {
    return edition.saas
  },
  get isLocal() {
    return !edition.saas
  },
}))

import { BILLING_PLANS_PATH, hasLiveSubscription, useBillingPlans } from '@/hooks/use-billing-plans'
import { BILLING_LEDGER_PATH, billingLedgerPath, isForbidden, useBillingLedger } from '@/hooks/use-billing-ledger'

function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, retryDelay: 0 } } })
  return ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

beforeEach(() => {
  edition.saas = true
  requestMock.mockReset()
  requestMock.mockResolvedValue({})
})

afterEach(() => {
  edition.saas = true
})

describe('useBillingPlans', () => {
  it('asks GET /api/v1/billing/plans', async () => {
    renderHook(() => useBillingPlans(), { wrapper: wrapper() })

    await waitFor(() => expect(requestMock).toHaveBeenCalledWith(BILLING_PLANS_PATH))
    expect(BILLING_PLANS_PATH).toBe('/api/v1/billing/plans')
  })

  it('never asks in the local edition', async () => {
    edition.saas = false
    renderHook(() => useBillingPlans(), { wrapper: wrapper() })
    await act(async () => {
      await Promise.resolve()
    })

    expect(requestMock).not.toHaveBeenCalled()
  })

  it('counts a subscription Stripe is still billing (including a retried renewal) as live', () => {
    const base = { cancel_at_period_end: false, current_period_end: null, has_billing_account: true }
    expect(hasLiveSubscription({ ...base, status: 'active' })).toBe(true)
    expect(hasLiveSubscription({ ...base, status: 'past_due' })).toBe(true)
    expect(hasLiveSubscription({ ...base, status: 'canceled' })).toBe(false)
    expect(hasLiveSubscription({ ...base, status: 'incomplete' })).toBe(false)
    expect(hasLiveSubscription({ ...base, status: null })).toBe(false)
    expect(hasLiveSubscription(undefined)).toBe(false)
  })
})

describe('useBillingLedger', () => {
  it('asks GET /api/v1/billing/ledger with the page and the page size', async () => {
    renderHook(() => useBillingLedger(2, 25), { wrapper: wrapper() })

    await waitFor(() => expect(requestMock).toHaveBeenCalledWith('/api/v1/billing/ledger?page=2&page_size=25'))
    expect(billingLedgerPath(1, 50)).toBe(`${BILLING_LEDGER_PATH}?page=1&page_size=50`)
  })

  it('never asks in the local edition', async () => {
    edition.saas = false
    renderHook(() => useBillingLedger(1), { wrapper: wrapper() })
    await act(async () => {
      await Promise.resolve()
    })

    expect(requestMock).not.toHaveBeenCalled()
  })

  it('does not ask again after a 403', async () => {
    requestMock.mockRejectedValue(Object.assign(new Error('Forbidden'), { status: 403 }))
    const { result } = renderHook(() => useBillingLedger(1), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(isForbidden(result.current.error)).toBe(true)
    expect(requestMock).toHaveBeenCalledTimes(1)
  })
})
