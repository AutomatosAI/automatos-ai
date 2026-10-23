/**
 * PRD-265 W2 — coming back to /settings?tab=billing&checkout=return.
 *
 * Only a redirect-based payment method ever leaves the page (cards never do). On its return
 * the tab says the payment is being confirmed and starts waiting for the plan to change
 * (`onReturn`); any `checkout` parameter (a stale v1 `success` / `cancelled` too) is taken out
 * of the address with router.replace, keeping the others, and a re-render says nothing twice.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { renderHook } from '@testing-library/react'

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

import { CHECKOUT_RETURN_TOAST, useCheckoutReturn } from '../use-checkout-return'

const onReturn = vi.fn()

beforeEach(() => {
  nav.search = ''
  nav.replace.mockReset()
  toastMock.success.mockReset()
  onReturn.mockReset()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('the return to the Billing tab (PRD-265 W2)', () => {
  it('checkout=return: says the payment is being confirmed, starts waiting, and drops ?checkout=', () => {
    nav.search = 'tab=billing&checkout=return'
    renderHook(() => useCheckoutReturn(onReturn))

    expect(toastMock.success).toHaveBeenCalledTimes(1)
    expect(toastMock.success).toHaveBeenCalledWith(CHECKOUT_RETURN_TOAST, expect.anything())
    expect(onReturn).toHaveBeenCalledTimes(1)
    expect(nav.replace).toHaveBeenCalledWith('/settings?tab=billing', { scroll: false })
  })

  it('says it once, however often the tab re-renders, and keeps the other parameters', () => {
    nav.search = 'tab=billing&checkout=return&plan=pro'
    const { rerender } = renderHook(() => useCheckoutReturn(onReturn))
    rerender()
    rerender()

    expect(toastMock.success).toHaveBeenCalledTimes(1)
    expect(onReturn).toHaveBeenCalledTimes(1)
    expect(nav.replace).toHaveBeenCalledTimes(1)
    expect(nav.replace).toHaveBeenCalledWith('/settings?tab=billing&plan=pro', { scroll: false })
  })

  it('any other checkout value is only taken out of the address', () => {
    nav.search = 'tab=billing&checkout=success'
    renderHook(() => useCheckoutReturn(onReturn))

    expect(toastMock.success).not.toHaveBeenCalled()
    expect(onReturn).not.toHaveBeenCalled()
    expect(nav.replace).toHaveBeenCalledWith('/settings?tab=billing', { scroll: false })
  })

  it('does nothing on an ordinary visit', () => {
    nav.search = 'tab=billing'
    renderHook(() => useCheckoutReturn(onReturn))

    expect(toastMock.success).not.toHaveBeenCalled()
    expect(onReturn).not.toHaveBeenCalled()
    expect(nav.replace).not.toHaveBeenCalled()
  })
})
