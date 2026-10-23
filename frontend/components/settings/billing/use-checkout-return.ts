'use client'

/**
 * PRD-265 W2 — the one case where the browser comes back to the Billing tab from elsewhere.
 *
 * Cards are confirmed in the page and never leave it. A payment method that needs the bank's
 * own page (redirect-based methods only) comes back to `/settings?tab=billing&checkout=return`:
 * the tab then says the payment is being confirmed and waits for the plan and the credits to
 * change (`onReturn`, hooks/use-billing-settle.ts). Any `checkout` parameter is then taken out
 * of the address, so a reload says nothing twice.
 */
import { useEffect, useRef } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'
import { settingsHref } from '../use-settings-tab'

export const CHECKOUT_PARAM = 'checkout'
export const CHECKOUT_RETURN = 'return'
export const CHECKOUT_RETURN_TOAST = 'Thanks — your payment is being confirmed. Your plan updates in a moment.'
/** One id, so a second effect pass replaces the toast instead of stacking another. */
const TOAST_ID = 'billing-checkout-return'

/** Mounted by the Billing tab: handles `?checkout=return`, once, and drops `?checkout=`. */
export function useCheckoutReturn(onReturn: () => void): void {
  const router = useRouter()
  const searchParams = useSearchParams()
  const value = searchParams?.get(CHECKOUT_PARAM) ?? null
  const handled = useRef(false)

  useEffect(() => {
    if (value === null || handled.current) return
    handled.current = true
    if (value === CHECKOUT_RETURN) {
      toast.success(CHECKOUT_RETURN_TOAST, { id: TOAST_ID })
      onReturn()
    }
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.delete(CHECKOUT_PARAM)
    router.replace(settingsHref(params), { scroll: false })
  }, [value, onReturn, router, searchParams])
}
