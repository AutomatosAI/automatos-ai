/**
 * PRD-265 W2 — the card, the invoices and cancelling, all inside the Billing tab.
 *
 * - The card on file reads "Visa •••• 4242 · expires 12/2028"; "Change card" opens our own card
 *   form on the SetupIntent the API opens (POST …/payment-method/setup), "Save card" confirms
 *   it in the page (confirmSetup, redirect: 'if_required') and POSTs the SetupIntent's id to
 *   …/payment-method; a refused card is shown in Stripe's words and nothing is posted.
 * - The invoices list number, date, amount and status; "Download PDF" fetches the PDF through
 *   apiClient (never from Stripe) and saves it under the invoice's number.
 * - "Cancel plan" asks first ("Your plan stays until <date>"), then POSTs …/subscription/cancel;
 *   a plan that is ending offers "Resume plan" (…/resume).
 * - None of it shows to a member who cannot manage, or before there is a billing account.
 * The browser never leaves the page.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { formatAbsoluteDate } from '@/lib/format-relative'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import {
  BETA_ADMIN,
  CARD,
  INVOICES,
  OPEN_PRO_ACTIVE,
  OPEN_PRO_CANCELLING,
  PERIOD_END,
  PUBLISHABLE_KEY,
  SETUP_SECRET,
  SUMMARY,
  answerBillingReads,
  answerBillingWrites,
  ledgerPage,
  renderWithClient,
} from './billing-fixtures'
import { elementsInstance, providerProps, resetStripeMocks, stripeInstance } from './stripe-mocks'

const { requestMock, postMock, blobMock, toastMock } = vi.hoisted(() => ({
  requestMock: vi.fn(),
  postMock: vi.fn(),
  blobMock: vi.fn(),
  toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: requestMock, post: postMock, requestBlob: blobMock } }))
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
import { cancelNotice } from '../cancel-plan'
import { CARD_SAVED_TOAST } from '@/hooks/use-billing-payment'
import { RESUMED_TOAST, cancelledToast } from '@/hooks/use-billing-actions'

const SETUP_PATH = '/api/v1/billing/payment-method/setup'
const METHOD_PATH = '/api/v1/billing/payment-method'
const CANCEL_PATH = '/api/v1/billing/subscription/cancel'
const RESUME_PATH = '/api/v1/billing/subscription/resume'
const NEW_CARD = { brand: 'mastercard', last4: '4444', exp_month: 3, exp_year: 2030 }
const assign = vi.fn()

function serve(plans: BillingPlans, writes: Record<string, unknown> = {}) {
  requestMock.mockImplementation(
    answerBillingReads({ plans, summary: SUMMARY, ledger: async (page) => ledgerPage(page), card: CARD, invoices: INVOICES }),
  )
  postMock.mockImplementation(answerBillingWrites(writes))
}

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  blobMock.mockReset()
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

describe('the card on file (PRD-265 W2)', () => {
  const card = () => screen.findByTestId('billing-card-on-file')

  it('shows the brand, the last four digits and the expiry', async () => {
    serve(OPEN_PRO_ACTIVE)
    renderWithClient(<BillingTab />)

    expect(await within(await card()).findByText('Visa •••• 4242 · expires 12/2028')).toBeInTheDocument()
  })

  it('Change card: our form on the SetupIntent, confirmed in the page, then saved as the default', async () => {
    serve(OPEN_PRO_ACTIVE, {
      [SETUP_PATH]: { client_secret: SETUP_SECRET, publishable_key: PUBLISHABLE_KEY },
      [METHOD_PATH]: { payment_method: NEW_CARD },
    })
    stripeInstance.confirmSetup.mockResolvedValue({ setupIntent: { id: 'seti_1', status: 'succeeded' } })
    renderWithClient(<BillingTab />)

    fireEvent.click(await within(await card()).findByRole('button', { name: 'Change card' }))

    expect(await screen.findByTestId('stripe-card-element')).toBeInTheDocument()
    expect(postMock).toHaveBeenCalledWith(SETUP_PATH)
    const elements = providerProps.elements[providerProps.elements.length - 1] as { options: { clientSecret: string; appearance: object } }
    expect(elements.options.clientSecret).toBe(SETUP_SECRET)
    expect(elements.options.appearance).toEqual(expect.objectContaining({ variables: expect.any(Object) }))

    fireEvent.click(screen.getByRole('button', { name: 'Save card' }))

    await waitFor(() => expect(stripeInstance.confirmSetup).toHaveBeenCalledWith({ elements: elementsInstance, redirect: 'if_required' }))
    await waitFor(() => expect(postMock).toHaveBeenCalledWith(METHOD_PATH, { setup_intent_id: 'seti_1' }))
    expect(toastMock.success).toHaveBeenCalledWith(CARD_SAVED_TOAST)
    expect(await screen.findByText('Mastercard •••• 4444 · expires 03/2030')).toBeInTheDocument()
    expect(screen.queryByTestId('stripe-card-element')).toBeNull()
  })

  it("a refused card: Stripe's sentence, and nothing is saved", async () => {
    serve(OPEN_PRO_ACTIVE, { [SETUP_PATH]: { client_secret: SETUP_SECRET, publishable_key: PUBLISHABLE_KEY } })
    stripeInstance.confirmSetup.mockResolvedValue({ error: { type: 'card_error', message: 'Your card was declined.' } })
    renderWithClient(<BillingTab />)

    fireEvent.click(await within(await card()).findByRole('button', { name: 'Change card' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Save card' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Your card was declined.')
    expect(postMock).not.toHaveBeenCalledWith(METHOD_PATH, expect.anything())
  })
})

describe('the invoices (PRD-265 W2)', () => {
  it('lists number, date, amount and status, newest first', async () => {
    serve(OPEN_PRO_ACTIVE)
    renderWithClient(<BillingTab />)

    const table = await screen.findByTestId('billing-invoices-table')
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows[0]).toHaveTextContent('ABCD-0002')
    expect(rows[0]).toHaveTextContent(formatAbsoluteDate(INVOICES[0].created))
    expect(rows[0]).toHaveTextContent('$29.00')
    expect(rows[0]).toHaveTextContent('Paid')
    expect(rows[1]).toHaveTextContent('$13.50')
    expect(rows[1]).toHaveTextContent('Due')
  })

  it('Download PDF fetches the PDF through apiClient and saves it under the invoice number', async () => {
    serve(OPEN_PRO_ACTIVE)
    blobMock.mockResolvedValue(new Blob(['%PDF-1.7'], { type: 'application/pdf' }))
    const createObjectURL = vi.fn((_blob: Blob) => 'blob:invoice')
    Object.defineProperty(URL, 'createObjectURL', { value: createObjectURL, configurable: true, writable: true })
    Object.defineProperty(URL, 'revokeObjectURL', { value: vi.fn(), configurable: true, writable: true })
    const clicked: string[] = []
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicked.push(this.download)
    })
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Download PDF of invoice ABCD-0002' }))

    await waitFor(() => expect(blobMock).toHaveBeenCalledWith('/api/v1/billing/invoices/in_2/pdf'))
    await waitFor(() => expect(clicked).toEqual(['ABCD-0002.pdf']))
    expect(createObjectURL).toHaveBeenCalledTimes(1)
    expect(createObjectURL.mock.calls[0][0].type).toBe('application/pdf')
    click.mockRestore()
  })

  it("says so when a PDF can't be fetched", async () => {
    serve(OPEN_PRO_ACTIVE)
    blobMock.mockRejectedValue(Object.assign(new Error('Not Found'), { status: 404, body: { detail: 'Invoice not found.' } }))
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Download PDF of invoice ABCD-0001' }))

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith('Invoice not found.'))
  })
})

describe('cancelling and resuming the plan (PRD-265 W2)', () => {
  it('Cancel plan asks first, saying until when the plan stays, then cancels', async () => {
    serve(OPEN_PRO_ACTIVE, { [CANCEL_PATH]: { cancel_at_period_end: true, ends_at: PERIOD_END } })
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Cancel plan' }))
    const confirmation = await screen.findByRole('alertdialog')
    expect(confirmation).toHaveTextContent(cancelNotice(PERIOD_END))
    expect(confirmation).toHaveTextContent(`Your plan stays until ${formatAbsoluteDate(PERIOD_END)}.`)
    expect(postMock).not.toHaveBeenCalledWith(CANCEL_PATH)

    fireEvent.click(within(confirmation).getByRole('button', { name: 'Cancel plan' }))

    await waitFor(() => expect(postMock).toHaveBeenCalledWith(CANCEL_PATH))
    await waitFor(() => expect(toastMock.success).toHaveBeenCalledWith(cancelledToast(PERIOD_END)))
  })

  it('Keep my plan in the question cancels nothing', async () => {
    serve(OPEN_PRO_ACTIVE)
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Cancel plan' }))
    fireEvent.click(within(await screen.findByRole('alertdialog')).getByRole('button', { name: 'Keep my plan' }))

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
    expect(postMock).not.toHaveBeenCalled()
  })

  it('a plan that is ending offers Resume plan', async () => {
    serve(OPEN_PRO_CANCELLING, { [RESUME_PATH]: { cancel_at_period_end: false } })
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Resume plan' }))

    await waitFor(() => expect(postMock).toHaveBeenCalledWith(RESUME_PATH))
    await waitFor(() => expect(toastMock.success).toHaveBeenCalledWith(RESUMED_TOAST))
    expect(screen.queryByRole('button', { name: 'Cancel plan' })).toBeNull()
  })
})

describe('who sees the Payment section (PRD-265 W2)', () => {
  it('not a member who cannot manage', async () => {
    serve({ ...OPEN_PRO_ACTIVE, can_manage: false })
    renderWithClient(<BillingTab />)

    expect(await screen.findByTestId('billing-current-plan')).toBeInTheDocument()
    expect(screen.queryByTestId('billing-payment')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Cancel plan' })).toBeNull()
    expect(requestMock.mock.calls.some(([path]) => String(path).startsWith('/api/v1/billing/invoices'))).toBe(false)
  })

  it('not before the workspace has a billing account', async () => {
    serve(BETA_ADMIN)
    renderWithClient(<BillingTab />)

    expect(await screen.findByRole('button', { name: 'Choose Pro' })).toBeInTheDocument()
    expect(screen.queryByTestId('billing-payment')).toBeNull()
  })
})
