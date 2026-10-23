/**
 * PRD-265 US-009 / US-011 — leaving for Stripe from the Billing tab.
 *
 * "Choose Pro" POSTs {kind: 'plan', plan: 'pro'} to /api/v1/billing/checkout, "Buy a top-up"
 * POSTs {kind: 'topup'}, "Manage payment" POSTs /api/v1/billing/portal; each follows the
 * answered Stripe address with window.location.assign. A refusal is toasted in the API's own
 * sentence (never as JSON), and an answer that is not an https address is never followed.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import {
  BETA_ADMIN,
  CHECKOUT_URL,
  OPEN_PRO_CANCELLING,
  PORTAL_URL,
  SUMMARY,
  answerBillingReads,
  httpError,
  ledgerPage,
  renderWithClient,
} from './billing-fixtures'

const { requestMock, postMock, toastMock } = vi.hoisted(() => ({
  requestMock: vi.fn(),
  postMock: vi.fn(),
  toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: requestMock, post: postMock } }))
vi.mock('@/lib/auth-edition', () => ({ isSaaS: true, isLocal: false }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => new URLSearchParams('tab=billing'),
  usePathname: () => '/settings',
}))
vi.mock('sonner', () => ({ toast: toastMock }))

import { BillingTab } from '../billing-tab'
import { CHECKOUT_FAILED } from '@/hooks/use-billing-actions'

const assign = vi.fn()

function serve(plans: BillingPlans) {
  requestMock.mockImplementation(answerBillingReads({ plans, summary: SUMMARY, ledger: async (page) => ledgerPage(page) }))
}

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  assign.mockReset()
  toastMock.error.mockReset()
  vi.stubGlobal('location', { ...window.location, assign })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Billing tab: checkout and the portal (PRD-265 US-011)', () => {
  it('Choose Pro POSTs {kind: plan, plan: pro} and goes to the Stripe address it answers', async () => {
    serve(BETA_ADMIN)
    postMock.mockResolvedValue({ url: CHECKOUT_URL, flow: 'checkout' })
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Choose Pro' }))

    await waitFor(() => expect(assign).toHaveBeenCalledWith(CHECKOUT_URL))
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock).toHaveBeenCalledWith('/api/v1/billing/checkout', { kind: 'plan', plan: 'pro' })
    // On its way to Stripe: the plan buttons stay off, so a second click buys nothing twice.
    expect(screen.getByRole('button', { name: 'Choose Own keys' })).toBeDisabled()
  })

  it('a plan change on a subscribed workspace follows the portal confirm page the same way', async () => {
    serve(OPEN_PRO_CANCELLING)
    postMock.mockResolvedValue({ url: PORTAL_URL, flow: 'portal' })
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Switch to Own keys' }))

    await waitFor(() => expect(assign).toHaveBeenCalledWith(PORTAL_URL))
    expect(postMock).toHaveBeenCalledWith('/api/v1/billing/checkout', { kind: 'plan', plan: 'basic' })
  })

  it('Buy a top-up POSTs {kind: topup} and goes to Stripe', async () => {
    serve(OPEN_PRO_CANCELLING)
    postMock.mockResolvedValue({ url: CHECKOUT_URL, flow: 'checkout' })
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Buy a top-up' }))

    await waitFor(() => expect(assign).toHaveBeenCalledWith(CHECKOUT_URL))
    expect(postMock).toHaveBeenCalledWith('/api/v1/billing/checkout', { kind: 'topup' })
  })

  it('Manage payment POSTs /api/v1/billing/portal and goes to the portal', async () => {
    serve(OPEN_PRO_CANCELLING)
    postMock.mockResolvedValue({ url: PORTAL_URL })
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Manage payment' }))

    await waitFor(() => expect(assign).toHaveBeenCalledWith(PORTAL_URL))
    expect(postMock).toHaveBeenCalledWith('/api/v1/billing/portal')
  })

  it("toasts the API's own sentence when checkout is refused, and stays on the page", async () => {
    serve(BETA_ADMIN)
    postMock.mockRejectedValue(httpError(409, "Paid plans aren't on sale yet."))
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Choose Pro' }))

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith("Paid plans aren't on sale yet."))
    expect(assign).not.toHaveBeenCalled()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Choose Pro' })).toBeEnabled())
  })

  it('never shows a structured error as JSON', async () => {
    serve(BETA_ADMIN)
    postMock.mockRejectedValue(httpError(422, [{ loc: ['body', 'plan'], msg: 'field required' }]))
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Choose Pro' }))

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(CHECKOUT_FAILED))
    expect(assign).not.toHaveBeenCalled()
  })

  it('never follows an answer that is not an https address', async () => {
    serve(BETA_ADMIN)
    postMock.mockResolvedValue({ url: 'javascript:alert(1)' })
    renderWithClient(<BillingTab />)

    fireEvent.click(await screen.findByRole('button', { name: 'Choose Pro' }))

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(CHECKOUT_FAILED))
    expect(assign).not.toHaveBeenCalled()
  })

  it('has no Manage payment without a billing account', async () => {
    serve(BETA_ADMIN)
    renderWithClient(<BillingTab />)

    expect(await screen.findByRole('button', { name: 'Choose Pro' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Manage payment' })).toBeNull()
  })
})
