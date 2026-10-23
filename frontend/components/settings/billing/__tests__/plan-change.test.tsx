/**
 * PRD-265 W2 — switching the plan of a workspace that already subscribes, inside the tab.
 *
 * A subscriber never gets a plan checkout: "Switch to X" opens the preview
 * (POST /api/v1/billing/plan-change/preview {plan}): "You'll pay $13.50 now (prorated), then $29
 * a month." for an upgrade, "Changes to Own keys on <date>; nothing to pay now." for a
 * downgrade. Confirming POSTs /api/v1/billing/plan-change and handles each answer: done (the tab
 * asks for the plans again), scheduled (says when), requires_action (Stripe.js's
 * handleNextAction with the client secret, then the same as done; a refusal is shown in the
 * bank's words). A scheduled downgrade shows on the current plan with "Keep my plan"
 * (POST /api/v1/billing/plan-change/cancel). The browser never leaves the page.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { formatAbsoluteDate } from '@/lib/format-relative'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import type { PlanChangePreview } from '@/hooks/use-billing-plan-change'
import {
  OPEN_PRO_ACTIVE,
  PAYMENT_SECRET,
  PERIOD_END,
  PUBLISHABLE_KEY,
  SUMMARY,
  answerBillingReads,
  answerBillingWrites,
  httpError,
  ledgerPage,
  renderWithClient,
} from './billing-fixtures'
import { resetStripeMocks, stripeInstance } from './stripe-mocks'

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
import { KEPT_PLAN_TOAST } from '@/hooks/use-billing-plan-change'
import { PAYMENT_RECEIVED, scheduledToast, switchedToast } from '../use-plan-switch'
import { confirmLabel, previewSentence } from '../plan-change-dialog'

const PREVIEW_PATH = '/api/v1/billing/plan-change/preview'
const CHANGE_PATH = '/api/v1/billing/plan-change'
const KEEP_PATH = '/api/v1/billing/plan-change/cancel'
const assign = vi.fn()

/** On Own keys, subscribed: Pro is an upgrade. */
const OPEN_BASIC_ACTIVE: BillingPlans = { ...OPEN_PRO_ACTIVE, current_plan: 'basic' }
const UPGRADE: PlanChangePreview = { plan: 'pro', direction: 'upgrade', amount_due_cents: 1350, tax_cents: 0, currency: 'usd', effective_at: 'now' }
const DOWNGRADE: PlanChangePreview = { plan: 'basic', direction: 'downgrade', amount_due_cents: 0, tax_cents: 0, currency: 'usd', effective_at: PERIOD_END }

function serve(plans: BillingPlans, writes: Record<string, unknown>) {
  requestMock.mockImplementation(answerBillingReads({ plans, summary: SUMMARY, ledger: async (page) => ledgerPage(page) }))
  postMock.mockImplementation(answerBillingWrites(writes))
}

const plansAsked = () => requestMock.mock.calls.filter(([path]) => String(path).startsWith('/api/v1/billing/plans')).length

async function openSwitch(plans: BillingPlans, planName: string, writes: Record<string, unknown>) {
  serve(plans, writes)
  renderWithClient(<BillingTab />)
  fireEvent.click(await screen.findByRole('button', { name: `Switch to ${planName}` }))
  return screen.findByRole('dialog')
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

describe('switching plans inside the Billing tab (PRD-265 W2)', () => {
  it('a subscriber is offered a switch, never a plan checkout', async () => {
    serve(OPEN_BASIC_ACTIVE, {})
    renderWithClient(<BillingTab />)

    expect(await screen.findByRole('button', { name: 'Switch to Pro' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: /^Choose / })).toBeNull()
  })

  it('an upgrade charges as of the preview: its proration moment goes back with the change', async () => {
    const preview = { ...UPGRADE, proration_date: 1_790_000_000 }
    const dialog = await openSwitch(OPEN_BASIC_ACTIVE, 'Pro', { [PREVIEW_PATH]: preview, [CHANGE_PATH]: { status: 'done' } })
    await within(dialog).findByTestId('plan-change-preview')

    fireEvent.click(within(dialog).getByRole('button', { name: 'Pay $13.50 and switch' }))

    await waitFor(() => expect(postMock).toHaveBeenCalledWith(CHANGE_PATH, { plan: 'pro', proration_date: 1_790_000_000 }))
  })

  it('an upgrade: previews the prorated charge, then switches and asks for the plans again', async () => {
    const dialog = await openSwitch(OPEN_BASIC_ACTIVE, 'Pro', { [PREVIEW_PATH]: UPGRADE, [CHANGE_PATH]: { status: 'done' } })

    expect(await within(dialog).findByTestId('plan-change-preview')).toHaveTextContent("You'll pay $13.50 now (prorated), then $29 a month.")
    expect(postMock).toHaveBeenCalledWith(PREVIEW_PATH, { plan: 'pro' })
    const askedBefore = plansAsked()

    fireEvent.click(within(dialog).getByRole('button', { name: 'Pay $13.50 and switch' }))

    await waitFor(() => expect(postMock).toHaveBeenCalledWith(CHANGE_PATH, { plan: 'pro' }))
    await waitFor(() => expect(toastMock.success).toHaveBeenCalledWith(switchedToast({ name: 'Pro' })))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    await waitFor(() => expect(plansAsked()).toBeGreaterThan(askedBefore))
    expect(stripeInstance.handleNextAction).not.toHaveBeenCalled()
  })

  it('a downgrade: nothing to pay now, scheduled for the period end', async () => {
    const dialog = await openSwitch(OPEN_PRO_ACTIVE, 'Own keys', {
      [PREVIEW_PATH]: DOWNGRADE,
      [CHANGE_PATH]: { status: 'scheduled', effective_at: PERIOD_END },
    })

    expect(await within(dialog).findByTestId('plan-change-preview')).toHaveTextContent(
      `Changes to Own keys on ${formatAbsoluteDate(PERIOD_END)}; nothing to pay now.`,
    )

    fireEvent.click(within(dialog).getByRole('button', { name: 'Switch to Own keys' }))

    await waitFor(() => expect(toastMock.success).toHaveBeenCalledWith(scheduledToast({ name: 'Own keys' }, PERIOD_END)))
    expect(postMock).toHaveBeenCalledWith(CHANGE_PATH, { plan: 'basic' })
    expect(stripeInstance.handleNextAction).not.toHaveBeenCalled()
  })

  it("the bank's check: handleNextAction with the client secret, then the payment is received", async () => {
    stripeInstance.handleNextAction.mockResolvedValue({ paymentIntent: { id: 'pi_1', status: 'succeeded' } })
    const dialog = await openSwitch(OPEN_BASIC_ACTIVE, 'Pro', {
      [PREVIEW_PATH]: UPGRADE,
      [CHANGE_PATH]: { status: 'requires_action', client_secret: PAYMENT_SECRET, publishable_key: PUBLISHABLE_KEY },
    })
    await within(dialog).findByTestId('plan-change-preview')

    fireEvent.click(within(dialog).getByRole('button', { name: 'Pay $13.50 and switch' }))

    await waitFor(() => expect(stripeInstance.handleNextAction).toHaveBeenCalledWith({ clientSecret: PAYMENT_SECRET }))
    await waitFor(() => expect(toastMock.success).toHaveBeenCalledWith(PAYMENT_RECEIVED))
    // The dialog is gone before the bank's check, so nothing modal sits over it.
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it("the bank's check refused: says so in the bank's words, and nothing is received", async () => {
    stripeInstance.handleNextAction.mockResolvedValue({ error: { type: 'card_error', message: 'Your card was declined.' } })
    const dialog = await openSwitch(OPEN_BASIC_ACTIVE, 'Pro', {
      [PREVIEW_PATH]: UPGRADE,
      [CHANGE_PATH]: { status: 'requires_action', client_secret: PAYMENT_SECRET, publishable_key: PUBLISHABLE_KEY },
    })
    await within(dialog).findByTestId('plan-change-preview')

    fireEvent.click(within(dialog).getByRole('button', { name: 'Pay $13.50 and switch' }))

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith('Your card was declined.'))
    expect(toastMock.success).not.toHaveBeenCalledWith(PAYMENT_RECEIVED)
  })

  it("a refused switch is toasted in the API's sentence", async () => {
    const dialog = await openSwitch(OPEN_BASIC_ACTIVE, 'Pro', {
      [PREVIEW_PATH]: UPGRADE,
      [CHANGE_PATH]: () => {
        throw httpError(409, 'A plan change is already waiting.')
      },
    })
    await within(dialog).findByTestId('plan-change-preview')

    fireEvent.click(within(dialog).getByRole('button', { name: 'Pay $13.50 and switch' }))

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith('A plan change is already waiting.'))
  })

  it('a preview that cannot be worked out says so and offers no switch', async () => {
    const dialog = await openSwitch(OPEN_BASIC_ACTIVE, 'Pro', {
      [PREVIEW_PATH]: () => {
        throw httpError(400, 'That plan is not sold here.')
      },
    })

    expect(await within(dialog).findByRole('alert')).toHaveTextContent('That plan is not sold here.')
    expect(within(dialog).getByRole('button', { name: 'Switch to Pro' })).toBeDisabled()
    expect(postMock).not.toHaveBeenCalledWith(CHANGE_PATH, expect.anything())
  })
})

describe('a scheduled downgrade (PRD-265 W2)', () => {
  const SCHEDULED: BillingPlans = {
    ...OPEN_PRO_ACTIVE,
    subscription: { ...OPEN_PRO_ACTIVE.subscription, scheduled_change: { plan: 'basic', effective_at: PERIOD_END } },
  }

  it('shows on the current plan, and "Keep my plan" takes it back', async () => {
    serve(SCHEDULED, { [KEEP_PATH]: { scheduled_change: null } })
    renderWithClient(<BillingTab />)

    const notice = await screen.findByTestId('billing-scheduled-change')
    expect(notice).toHaveTextContent(`Changes to Own keys on ${formatAbsoluteDate(PERIOD_END)}.`)
    const askedBefore = plansAsked()

    fireEvent.click(within(notice).getByRole('button', { name: 'Keep my plan' }))

    await waitFor(() => expect(postMock).toHaveBeenCalledWith(KEEP_PATH))
    await waitFor(() => expect(toastMock.success).toHaveBeenCalledWith(KEPT_PLAN_TOAST))
    await waitFor(() => expect(plansAsked()).toBeGreaterThan(askedBefore))
  })

  it('a member who cannot manage sees it, with nothing to press', async () => {
    serve({ ...SCHEDULED, can_manage: false }, {})
    renderWithClient(<BillingTab />)

    const notice = await screen.findByTestId('billing-scheduled-change')
    expect(within(notice).queryByRole('button')).toBeNull()
  })
})

describe('the preview in words (PRD-265 W2)', () => {
  const pro = { name: 'Pro', price_usd: 29 }

  it('names the tax in a prorated charge, and the button says what is paid', () => {
    const taxed: PlanChangePreview = { ...UPGRADE, amount_due_cents: 1620, tax_cents: 270 }
    expect(previewSentence(taxed, pro)).toBe("You'll pay $16.20 now (prorated, including $2.70 tax), then $29 a month.")
    expect(confirmLabel(taxed, pro)).toBe('Pay $16.20 and switch')
  })

  it('says nothing is paid now for a downgrade, and the button only switches', () => {
    expect(confirmLabel(DOWNGRADE, { name: 'Own keys' })).toBe('Switch to Own keys')
    expect(confirmLabel(undefined, { name: 'Own keys' })).toBe('Switch to Own keys')
  })
})
