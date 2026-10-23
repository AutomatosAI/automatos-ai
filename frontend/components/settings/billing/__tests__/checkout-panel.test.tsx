/**
 * PRD-265 W2 — paying inside the Billing tab.
 *
 * "Choose Pro" opens the tab's own payment form: it POSTs {kind: 'plan', plan: 'pro'} to
 * /api/v1/billing/checkout and mounts Stripe's fields on the session it answers, in our theme.
 * With terms to accept, nothing is asked until the box is ticked, Pay stays off until then,
 * and the POST carries accept_terms: true. Pay confirms in the page (redirect: 'if_required'):
 * a payment says "Payment received" and the tab asks for the plans again; a decline shows
 * Stripe's own sentence. The VAT field shows for a UK or EU billing address and hands the
 * number to the session (gb_vat / eu_vat). A top-up opens the same form. Nothing ever sends
 * the browser to a Stripe page: window.location.assign is never called.
 *
 * apiClient, the edition flag, the router, sonner and Stripe's modules are stand-ins
 * (./stripe-mocks.tsx); a real QueryClient.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import {
  BETA_ADMIN,
  CHECKOUT_SECRET,
  CONSENT,
  OPEN_PRO_ACTIVE,
  PUBLISHABLE_KEY,
  SESSION,
  SUMMARY,
  answerBillingReads,
  answerBillingWrites,
  httpError,
  ledgerPage,
  renderWithClient,
} from './billing-fixtures'
import { checkoutSession, checkoutState, providerProps, resetStripeMocks, stripeInstance } from './stripe-mocks'

const { requestMock, postMock, toastMock } = vi.hoisted(() => ({
  requestMock: vi.fn(),
  postMock: vi.fn(),
  toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: requestMock, post: postMock, requestBlob: vi.fn() } }))
vi.mock('@/lib/auth-edition', () => ({ isSaaS: true, isLocal: false }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => new URLSearchParams('tab=billing'),
  usePathname: () => '/settings',
}))
vi.mock('sonner', () => ({ toast: toastMock }))
vi.mock('@stripe/stripe-js/pure', async () => (await import('./stripe-mocks')).stripeJsModule)
vi.mock('@stripe/react-stripe-js', async () => (await import('./stripe-mocks')).reactStripeModule)
vi.mock('@stripe/react-stripe-js/checkout', async () => (await import('./stripe-mocks')).checkoutModule)

import { BillingTab } from '../billing-tab'
import { PAYMENT_RECEIVED_NOTE } from '../checkout-panel'

const assign = vi.fn()
const CHECKOUT_PATH = '/api/v1/billing/checkout'

function serve(plans: BillingPlans) {
  requestMock.mockImplementation(answerBillingReads({ plans, summary: SUMMARY, ledger: async (page) => ledgerPage(page) }))
  postMock.mockImplementation(answerBillingWrites({ [CHECKOUT_PATH]: SESSION }))
}

const plansAsked = () => requestMock.mock.calls.filter(([path]) => String(path).startsWith('/api/v1/billing/plans')).length
const payButton = () => screen.getByRole('button', { name: /^Pay/ })

async function openProCheckout(plans: BillingPlans = BETA_ADMIN) {
  serve(plans)
  renderWithClient(<BillingTab />)
  fireEvent.click(await screen.findByRole('button', { name: 'Choose Pro' }))
  return screen.findByTestId('billing-checkout')
}

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  assign.mockReset()
  Object.values(toastMock).forEach((mock) => mock.mockReset())
  resetStripeMocks()
  vi.stubGlobal('location', { ...window.location, assign })
})

afterEach(() => {
  expect(assign).not.toHaveBeenCalled()
  cleanup()
  vi.unstubAllGlobals()
})

describe('the payment form inside the Billing tab (PRD-265 W2)', () => {
  it('Choose Pro opens the form in the tab and POSTs {kind: plan, plan: pro}', async () => {
    const panel = await openProCheckout()

    expect(within(panel).getByText('Subscribe to Pro')).toBeInTheDocument()
    expect(within(panel).getByText('$29 a month')).toBeInTheDocument()
    await waitFor(() => expect(postMock).toHaveBeenCalledWith(CHECKOUT_PATH, { kind: 'plan', plan: 'pro' }))
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(await screen.findByTestId('stripe-payment-element')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Billing country' })).toBeInTheDocument()
  })

  it('mounts Stripe on the session it was given, with an appearance from our theme', async () => {
    await openProCheckout()
    await screen.findByTestId('stripe-payment-element')

    const props = providerProps.checkout[providerProps.checkout.length - 1] as { stripe: Promise<unknown>; options: { clientSecret: string; elementsOptions: { appearance: { variables: Record<string, string> } } } }
    expect(props.options.clientSecret).toBe(CHECKOUT_SECRET)
    expect(props.options.elementsOptions.appearance.variables.colorPrimary).toMatch(/^#|^rgb|^hsl/)
    await expect(props.stripe).resolves.toBe(stripeInstance)
    expect(PUBLISHABLE_KEY).toMatch(/^pk_test_/)
  })

  it('shows the totals with tax from the session', async () => {
    await openProCheckout()

    const totals = await screen.findByTestId('billing-checkout-totals')
    expect(totals).toHaveTextContent('Subtotal$29.00')
    expect(totals).toHaveTextContent('Tax$5.80')
    expect(totals).toHaveTextContent('Total due today$34.80')
    expect(totals).toHaveTextContent('Then $29.00 a month.')
    expect(payButton()).toHaveTextContent('Pay $34.80')
  })

  it('Pay confirms in the page, says the payment arrived and asks for the plans again', async () => {
    await openProCheckout()
    const session = (checkoutState.current as { checkout: ReturnType<typeof checkoutSession> }).checkout
    const askedBefore = plansAsked()

    fireEvent.click(await screen.findByRole('button', { name: 'Pay $34.80' }))

    await waitFor(() => expect(session.confirm).toHaveBeenCalledWith({ redirect: 'if_required' }))
    expect(await screen.findByRole('status')).toHaveTextContent(PAYMENT_RECEIVED_NOTE)
    await waitFor(() => expect(plansAsked()).toBeGreaterThan(askedBefore))
  })

  it("shows Stripe's own sentence when the card is declined, and charges nothing", async () => {
    const declined = checkoutSession({
      confirm: vi.fn(async () => ({ type: 'error', error: { message: 'Your card was declined.', code: 'paymentFailed', paymentFailed: { declineCode: 'generic_decline' } } })),
    })
    checkoutState.current = { type: 'success', checkout: declined }
    await openProCheckout()

    fireEvent.click(await screen.findByRole('button', { name: 'Pay $34.80' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Your card was declined.')
    expect(screen.queryByText(PAYMENT_RECEIVED_NOTE)).toBeNull()
  })

  it('keeps Pay off until Stripe says the session can be confirmed', async () => {
    checkoutState.current = { type: 'success', checkout: checkoutSession({ canConfirm: false }) }
    await openProCheckout()

    expect(await screen.findByRole('button', { name: 'Pay $34.80' })).toBeDisabled()
  })

  it('with terms to accept: asks nothing and keeps Pay off until the box is ticked, then sends accept_terms', async () => {
    const panel = await openProCheckout({ ...BETA_ADMIN, consent_text: CONSENT })

    expect(within(panel).getByText(CONSENT)).toBeInTheDocument()
    expect(payButton()).toBeDisabled()
    expect(postMock).not.toHaveBeenCalled()

    fireEvent.click(within(panel).getByRole('checkbox', { name: 'I accept the terms' }))

    await waitFor(() => expect(postMock).toHaveBeenCalledWith(CHECKOUT_PATH, { kind: 'plan', plan: 'pro', accept_terms: true }))
    expect(await screen.findByRole('button', { name: 'Pay $34.80' })).toBeEnabled()

    fireEvent.click(within(panel).getByRole('checkbox', { name: 'I accept the terms' }))
    expect(screen.getByRole('button', { name: 'Pay $34.80' })).toBeDisabled()
  })

  it("shows the API's sentence when the form can't open, and tries again on request", async () => {
    serve(BETA_ADMIN)
    postMock.mockRejectedValueOnce(httpError(409, "Paid plans aren't on sale yet."))
    renderWithClient(<BillingTab />)
    fireEvent.click(await screen.findByRole('button', { name: 'Choose Pro' }))

    expect(await screen.findByRole('alert')).toHaveTextContent("Paid plans aren't on sale yet.")
    postMock.mockImplementation(answerBillingWrites({ [CHECKOUT_PATH]: SESSION }))
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))

    expect(await screen.findByTestId('stripe-payment-element')).toBeInTheDocument()
    expect(postMock).toHaveBeenCalledTimes(2)
  })

  it('never mounts Stripe on an answer that is not a Checkout Session', async () => {
    serve(BETA_ADMIN)
    postMock.mockResolvedValue({ client_secret: 'https://checkout.stripe.com/c/pay/x', publishable_key: PUBLISHABLE_KEY })
    renderWithClient(<BillingTab />)
    fireEvent.click(await screen.findByRole('button', { name: 'Choose Pro' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('The payment form could not open.')
    expect(providerProps.checkout).toHaveLength(0)
  })

  it('closes the form on request', async () => {
    await openProCheckout()

    fireEvent.click(screen.getByRole('button', { name: 'Close the payment form' }))

    expect(screen.queryByTestId('billing-checkout')).toBeNull()
  })

  it('Buy a top-up opens the same form and POSTs {kind: topup}', async () => {
    serve(OPEN_PRO_ACTIVE)
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Buy a top-up' }))

    const panel = await screen.findByTestId('billing-checkout')
    expect(within(panel).getByText('Buy a top-up')).toBeInTheDocument()
    expect(within(panel).getByText('1,000 credits for $10, valid 12 months')).toBeInTheDocument()
    await waitFor(() => expect(postMock).toHaveBeenCalledWith(CHECKOUT_PATH, { kind: 'topup' }))
  })
})

describe('the VAT number (PRD-265 W2)', () => {
  async function chooseCountry(country: string) {
    await openProCheckout()
    fireEvent.change(await screen.findByRole('combobox', { name: 'Billing country' }), { target: { value: country } })
  }

  function addVat(businessName: string, number: string) {
    fireEvent.change(screen.getByLabelText('Business name'), { target: { value: businessName } })
    fireEvent.change(screen.getByLabelText('VAT number'), { target: { value: number } })
    fireEvent.click(screen.getByRole('button', { name: 'Add VAT number' }))
  }

  const sessionActions = () => (checkoutState.current as { checkout: ReturnType<typeof checkoutSession> }).checkout

  it('hands an EU business number to the session as eu_vat', async () => {
    await chooseCountry('DE')
    addVat('Acme GmbH', 'DE123456789')

    await waitFor(() =>
      expect(sessionActions().updateTaxIdInfo).toHaveBeenCalledWith({ taxId: { type: 'eu_vat', value: 'DE123456789' }, businessName: 'Acme GmbH' }),
    )
    expect(await screen.findByText('VAT number added.')).toBeInTheDocument()
  })

  it('hands a UK number to the session as gb_vat', async () => {
    await chooseCountry('GB')
    addVat('Acme Ltd', 'GB123456789')

    await waitFor(() =>
      expect(sessionActions().updateTaxIdInfo).toHaveBeenCalledWith({ taxId: { type: 'gb_vat', value: 'GB123456789' }, businessName: 'Acme Ltd' }),
    )
  })

  it("shows Stripe's sentence for a number it can't use", async () => {
    sessionActions().updateTaxIdInfo.mockResolvedValueOnce({ type: 'error', error: { message: 'Invalid VAT number.', code: 'invalidTaxId' } })
    await chooseCountry('DE')
    addVat('Acme GmbH', 'DE0')

    expect(await screen.findByRole('alert')).toHaveTextContent('Invalid VAT number.')
  })

  it('is not shown for a billing address outside the UK and the EU', async () => {
    await chooseCountry('US')

    expect(screen.queryByTestId('billing-vat')).toBeNull()
    expect(screen.queryByLabelText('VAT number')).toBeNull()
  })
})
