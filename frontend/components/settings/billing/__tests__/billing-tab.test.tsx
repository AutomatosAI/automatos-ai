/**
 * PRD-265 US-011 — Settings → Billing, in each state the API can put it in.
 *
 * - beta, ordinary member: the facts-sheet beta copy, no credit amounts, no buttons, no ledger;
 * - beta, platform admin: checkout buttons for Own keys and Pro, Business offers contact only;
 * - billing open, a Pro subscriber who cancelled: "Ends on <date>", "Current plan" disabled,
 *   monthly credits per plan, no beta sentences, Manage payment and Buy a top-up;
 * - billing open, renewal failed: the payment-failed notice;
 * - `?plan=` highlights that plan; the balance by bucket and the paused state.
 *
 * A real QueryClient; apiClient, the edition flag, the router and sonner are stand-ins.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, screen, within } from '@testing-library/react'
import { formatAbsoluteDate } from '@/lib/format-relative'
import type { BillingPlans } from '@/hooks/use-billing-plans'
import type { BillingSummary } from '@/hooks/use-billing-summary'
import {
  BETA_ADMIN,
  BETA_MEMBER,
  OPEN_PRO_CANCELLING,
  OPEN_PRO_PAST_DUE,
  PAUSED_SUMMARY,
  PERIOD_END,
  SUMMARY,
  answerBillingReads,
  ledgerPage,
  renderWithClient,
} from './billing-fixtures'

const { requestMock, postMock, nav, toastMock } = vi.hoisted(() => ({
  requestMock: vi.fn(),
  postMock: vi.fn(),
  nav: { search: 'tab=billing', replace: vi.fn() },
  toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: requestMock, post: postMock } }))
vi.mock('@/lib/auth-edition', () => ({ isSaaS: true, isLocal: false }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => '/settings',
}))
vi.mock('sonner', () => ({ toast: toastMock }))

import { BillingTab } from '../billing-tab'
import { CONTACT_URL } from '../plan-copy'
import { PAST_DUE_NOTICE } from '../billing-format'

function serve(plans: BillingPlans, summary: BillingSummary = SUMMARY) {
  requestMock.mockImplementation(answerBillingReads({ plans, summary, ledger: async (page) => ledgerPage(page) }))
}

const planCard = (key: string) => screen.getByTestId(`billing-plan-${key}`)
const ledgerWasAsked = () => requestMock.mock.calls.some(([path]) => String(path).startsWith('/api/v1/billing/ledger'))

beforeEach(() => {
  nav.search = 'tab=billing'
  nav.replace.mockReset()
  requestMock.mockReset()
  postMock.mockReset()
})

afterEach(() => cleanup())

describe('Billing tab (PRD-265 US-011)', () => {
  it('beta, ordinary member: the beta copy, no credit amounts, no buttons and no ledger', async () => {
    serve(BETA_MEMBER)
    renderWithClient(<BillingTab />)

    expect(await screen.findByTestId('billing-plan-pro')).toBeInTheDocument()
    expect(screen.queryAllByRole('button')).toHaveLength(0)
    expect(screen.getByText(/Beta pricing: not charged until billing opens\. During the beta every plan runs on the trial/)).toBeInTheDocument()
    expect(within(planCard('business')).getByText(/By invite during the beta: contact Automatos\./)).toBeInTheDocument()
    expect(screen.queryByText(/credits a month included/)).toBeNull()
    expect(screen.queryByTestId('billing-top-up')).toBeNull()
    expect(screen.queryByTestId('billing-ledger')).toBeNull()
    expect(ledgerWasAsked()).toBe(false)
    for (const key of ['business', 'enterprise']) {
      expect(within(planCard(key)).getByRole('link', { name: 'Contact Automatos' })).toHaveAttribute('href', CONTACT_URL)
    }
  })

  it('beta, platform admin: checkout buttons for Own keys and Pro; Business offers contact, never checkout', async () => {
    serve(BETA_ADMIN)
    renderWithClient(<BillingTab />)

    expect(await screen.findByRole('button', { name: 'Choose Own keys' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Choose Pro' })).toBeEnabled()
    expect(within(planCard('business')).queryByRole('button')).toBeNull()
    expect(within(planCard('business')).getByRole('link', { name: 'Contact Automatos' })).toBeInTheDocument()
    expect(within(planCard('enterprise')).getByText('Coming soon', { selector: 'p' })).toBeInTheDocument()
    // Still the beta copy, still no credit amounts.
    expect(screen.getByText(/Beta pricing: not charged until billing opens\. During the beta/)).toBeInTheDocument()
    expect(screen.queryByText(/credits a month included/)).toBeNull()
    // The top-up card explains; there is nothing to top up without a subscription.
    expect(screen.getByTestId('billing-top-up')).toHaveTextContent('1,000 credits for $10, valid 12 months')
    expect(screen.getByTestId('billing-top-up')).toHaveTextContent('Top-ups need an active Pro or Business subscription.')
    expect(screen.queryByRole('button', { name: 'Buy a top-up' })).toBeNull()
    // No billing account yet: no portal.
    expect(screen.queryByRole('button', { name: 'Manage payment' })).toBeNull()
    expect(await screen.findByTestId('billing-ledger-table')).toBeInTheDocument()
  })

  it('billing open, Pro subscriber who cancelled: ends on the date, current plan disabled, credits per plan', async () => {
    serve(OPEN_PRO_CANCELLING, { ...SUMMARY, plan: 'pro', plan_name: 'Pro', billing_open: true, price_label: null })
    renderWithClient(<BillingTab />)

    expect(await screen.findByTestId('billing-subscription-notice')).toHaveTextContent(`Ends on ${formatAbsoluteDate(PERIOD_END)}.`)
    expect(screen.getByRole('button', { name: 'Current plan' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Switch to Own keys' })).toBeEnabled()
    expect(within(planCard('pro')).getByText('2,000 credits a month included.')).toBeInTheDocument()
    expect(within(planCard('business')).getByText('8,000 credits a month included.')).toBeInTheDocument()
    expect(within(planCard('business')).getByText(/By invite: contact Automatos\./)).toBeInTheDocument()
    expect(screen.queryByText(/Beta pricing/)).toBeNull()
    expect(screen.getByText(/A credit is one US cent of AI usage at Automatos prices\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Manage payment' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Buy a top-up' })).toBeEnabled()
    expect(within(screen.getByTestId('billing-current-plan')).getByText('Pro')).toBeInTheDocument()
    expect(within(screen.getByTestId('billing-current-plan')).getByText('$29 a month')).toBeInTheDocument()
  })

  it('billing open, renewal failed: says the payment failed and Stripe is retrying', async () => {
    serve(OPEN_PRO_PAST_DUE)
    renderWithClient(<BillingTab />)

    const notice = await screen.findByTestId('billing-subscription-notice')
    expect(notice).toHaveTextContent(PAST_DUE_NOTICE)
    expect(notice).toHaveAttribute('role', 'alert')
    expect(screen.getByRole('button', { name: 'Manage payment' })).toBeInTheDocument()
  })

  it('billing open, a member who cannot manage: no buttons, and says who can', async () => {
    serve({ ...OPEN_PRO_CANCELLING, can_manage: false })
    renderWithClient(<BillingTab />)

    expect(await screen.findAllByText('Only a workspace owner or admin can change the plan or buy credits.')).not.toHaveLength(0)
    expect(screen.queryAllByRole('button')).toHaveLength(0)
    expect(ledgerWasAsked()).toBe(false)
  })

  it('highlights the plan named in ?plan= (Auto\'s onboarding link)', async () => {
    nav.search = 'tab=billing&plan=pro'
    serve(BETA_ADMIN)
    renderWithClient(<BillingTab />)

    expect(await screen.findByTestId('billing-plan-pro')).toHaveAttribute('data-highlighted', 'true')
    expect(planCard('basic')).not.toHaveAttribute('data-highlighted')
    expect(planCard('business')).not.toHaveAttribute('data-highlighted')
  })

  it('shows the credits by bucket, the credit unit, and the paused state', async () => {
    serve(BETA_MEMBER, { ...PAUSED_SUMMARY, balance: { trial: 0, monthly: 0, topup: 0, spendable: 0 } })
    renderWithClient(<BillingTab />)

    const balance = await screen.findByTestId('billing-credit-balance')
    expect(balance).toHaveTextContent('0 credits to spend')
    expect(within(balance).getByText('Trial')).toBeInTheDocument()
    expect(within(balance).getByText('Monthly')).toBeInTheDocument()
    expect(within(balance).getByText('Top-up')).toBeInTheDocument()
    expect(within(balance).getByRole('status')).toHaveTextContent("Work on Automatos's AI keys is paused")
    expect(balance).toHaveTextContent('1 credit = 1 US cent of AI usage.')
  })

  it('says so when the plans cannot be loaded', async () => {
    requestMock.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/v1/billing/summary')) return SUMMARY
      throw new Error('boom')
    })
    renderWithClient(<BillingTab />)

    expect(await screen.findByRole('alert')).toHaveTextContent('Billing could not be loaded. Try again in a moment.')
  })
})
