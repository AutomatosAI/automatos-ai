/**
 * PRD-265 W2 — stand-ins for Stripe.js and its React bindings in the Billing tab tests. Not a
 * test file itself. A test mocks the three modules with these:
 *
 *   vi.mock('@stripe/stripe-js/pure', async () => (await import('./stripe-mocks')).stripeJsModule)
 *   vi.mock('@stripe/react-stripe-js', async () => (await import('./stripe-mocks')).reactStripeModule)
 *   vi.mock('@stripe/react-stripe-js/checkout', async () => (await import('./stripe-mocks')).checkoutModule)
 *
 * and steers them through `stripeInstance`, `checkoutState` and `providerProps`. Stripe's
 * fields become plain stand-ins: the billing address is a country <select> that reports its
 * change the way Stripe's element does.
 */
import React from 'react'
import { vi } from 'vitest'

type Fn = ReturnType<typeof vi.fn>
/** A spy whose answers a test may replace with any shape (a success, an error). */
type AnyAsync = (...args: unknown[]) => Promise<unknown>

function amount(minorUnitsAmount: number) {
  return { minorUnitsAmount, amount: `$${(minorUnitsAmount / 100).toFixed(2)}` }
}

/** The Checkout Session as `useCheckoutElements()` hands it over, with spies for its actions. */
export function checkoutSession(overrides: Record<string, unknown> = {}) {
  return {
    canConfirm: true,
    billingAddress: null,
    taxIdInfo: null,
    tax: { status: 'ready' },
    recurring: { interval: 'month', intervalCount: 1, isProrated: false, dueNext: { total: amount(2900) } },
    total: {
      subtotal: amount(2900),
      taxExclusive: amount(580),
      taxInclusive: amount(0),
      total: amount(3480),
    },
    confirm: vi.fn<AnyAsync>(async () => ({ type: 'success', session: {} })),
    updateTaxIdInfo: vi.fn<AnyAsync>(async () => ({ type: 'success', session: {} })),
    ...overrides,
  }
}

export const checkoutState: { current: unknown } = { current: { type: 'loading' } }

/** What each Stripe provider was mounted with, newest last. */
export const providerProps = { checkout: [] as unknown[], elements: [] as unknown[] }

export const stripeInstance: { confirmSetup: Fn; handleNextAction: Fn } = {
  confirmSetup: vi.fn(),
  handleNextAction: vi.fn(),
}
export const elementsInstance = { kind: 'elements' }
export const loadStripeMock = vi.fn(async (_key: string) => stripeInstance)

export function resetStripeMocks(): void {
  checkoutState.current = { type: 'success', checkout: checkoutSession() }
  providerProps.checkout = []
  providerProps.elements = []
  stripeInstance.confirmSetup.mockReset()
  stripeInstance.handleNextAction.mockReset()
  loadStripeMock.mockClear()
}

type ProviderProps = { children?: React.ReactNode } & Record<string, unknown>
type AddressChange = (event: { value: { address: { country: string } } }) => void

export const stripeJsModule = {
  loadStripe: (key: string) => loadStripeMock(key),
}

export const checkoutModule = {
  CheckoutElementsProvider: ({ children, ...props }: ProviderProps) => {
    providerProps.checkout.push(props)
    return <div data-testid="stripe-checkout-provider">{children}</div>
  },
  useCheckoutElements: () => checkoutState.current,
  PaymentElement: () => <div data-testid="stripe-payment-element" />,
  BillingAddressElement: ({ onChange }: { onChange?: AddressChange }) => (
    <select
      aria-label="Billing country"
      defaultValue=""
      onChange={(event) => onChange?.({ value: { address: { country: event.target.value } } })}
    >
      <option value="">Choose a country</option>
      <option value="US">United States</option>
      <option value="GB">United Kingdom</option>
      <option value="DE">Germany</option>
    </select>
  ),
}

export const reactStripeModule = {
  Elements: ({ children, ...props }: ProviderProps) => {
    providerProps.elements.push(props)
    return <div data-testid="stripe-elements">{children}</div>
  },
  PaymentElement: () => <div data-testid="stripe-card-element" />,
  useStripe: () => stripeInstance,
  useElements: () => elementsInstance,
}
