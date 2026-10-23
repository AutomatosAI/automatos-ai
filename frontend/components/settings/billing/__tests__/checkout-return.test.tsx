/**
 * PRD-265 US-011 — coming back from Stripe Checkout to /settings?tab=billing&checkout=….
 *
 * success → "Payment received — your plan updates within a minute", the summary and the plans
 * are asked again at once and a few more times over the next minute; cancelled → "Checkout
 * cancelled — nothing was charged", nothing is asked again. Either way `checkout` leaves the
 * address (router.replace), and a re-render never says it twice.
 */
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { QueryClientProvider } from '@tanstack/react-query'
import { newQueryClient } from './billing-fixtures'

const { nav, toastMock } = vi.hoisted(() => ({
  nav: { search: '', replace: vi.fn() },
  toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => '/settings',
}))
vi.mock('sonner', () => ({ toast: toastMock }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: vi.fn() } }))
vi.mock('@/lib/auth-edition', () => ({ isSaaS: true, isLocal: false }))

import {
  CHECKOUT_CANCELLED_TOAST,
  CHECKOUT_SUCCESS_TOAST,
  PLAN_POLL_DELAYS_MS,
  useCheckoutReturn,
} from '../use-checkout-return'

const SUMMARY_KEY = { queryKey: ['billing', 'summary'] }
const PLANS_KEY = { queryKey: ['billing', 'plans'] }

function renderReturn() {
  const client = newQueryClient()
  const invalidate = vi.spyOn(client, 'invalidateQueries')
  const wrapper = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const view = renderHook(() => useCheckoutReturn(), { wrapper })
  return { ...view, invalidate }
}

/** How many times the spy was called with exactly these filters. */
function callsWith(spy: { mock: { calls: unknown[][] } }, key: { queryKey: string[] }): number {
  return spy.mock.calls.filter(([filters]) => JSON.stringify(filters) === JSON.stringify(key)).length
}

beforeEach(() => {
  nav.search = ''
  nav.replace.mockReset()
  toastMock.success.mockReset()
  toastMock.info.mockReset()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('the return from Stripe Checkout (PRD-265 US-011)', () => {
  it('success: says the payment arrived, asks for the summary and plans again, and drops ?checkout=', () => {
    nav.search = 'tab=billing&checkout=success'
    const { invalidate } = renderReturn()

    expect(toastMock.success).toHaveBeenCalledTimes(1)
    expect(toastMock.success).toHaveBeenCalledWith(CHECKOUT_SUCCESS_TOAST, expect.anything())
    expect(callsWith(invalidate, SUMMARY_KEY)).toBe(1)
    expect(callsWith(invalidate, PLANS_KEY)).toBe(1)
    expect(nav.replace).toHaveBeenCalledWith('/settings?tab=billing', { scroll: false })
  })

  it('success: asks for the plans again a few times over the next minute, then stops', () => {
    vi.useFakeTimers()
    nav.search = 'tab=billing&checkout=success'
    const { invalidate } = renderReturn()
    expect(callsWith(invalidate, PLANS_KEY)).toBe(1)

    act(() => {
      vi.advanceTimersByTime(Math.max(...PLAN_POLL_DELAYS_MS))
    })
    expect(callsWith(invalidate, PLANS_KEY)).toBe(1 + PLAN_POLL_DELAYS_MS.length)
    expect(callsWith(invalidate, SUMMARY_KEY)).toBe(1 + PLAN_POLL_DELAYS_MS.length)

    act(() => {
      vi.advanceTimersByTime(5 * 60_000)
    })
    expect(callsWith(invalidate, PLANS_KEY)).toBe(1 + PLAN_POLL_DELAYS_MS.length)
  })

  it('cancelled: says nothing was charged, asks for nothing, and drops ?checkout=', () => {
    nav.search = 'tab=billing&checkout=cancelled'
    const { invalidate } = renderReturn()

    expect(toastMock.info).toHaveBeenCalledWith(CHECKOUT_CANCELLED_TOAST, expect.anything())
    expect(toastMock.success).not.toHaveBeenCalled()
    expect(invalidate).not.toHaveBeenCalled()
    expect(nav.replace).toHaveBeenCalledWith('/settings?tab=billing', { scroll: false })
  })

  it('says it once, however often the tab re-renders', () => {
    nav.search = 'tab=billing&checkout=success&plan=pro'
    const { rerender } = renderReturn()
    rerender()
    rerender()

    expect(toastMock.success).toHaveBeenCalledTimes(1)
    expect(nav.replace).toHaveBeenCalledTimes(1)
    expect(nav.replace).toHaveBeenCalledWith('/settings?tab=billing&plan=pro', { scroll: false })
  })

  it('does nothing on an ordinary visit, or with an unknown checkout value', () => {
    nav.search = 'tab=billing'
    renderReturn()
    nav.search = 'tab=billing&checkout=maybe'
    renderReturn()

    expect(toastMock.success).not.toHaveBeenCalled()
    expect(toastMock.info).not.toHaveBeenCalled()
    expect(nav.replace).not.toHaveBeenCalled()
  })
})
