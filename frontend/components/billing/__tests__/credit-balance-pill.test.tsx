/**
 * PRD-265 US-005 / US-008 — the credit balance chip in the header.
 *
 * Reads the workspace's credits off the billing summary: the spendable balance;
 * "N of 500 trial credits" while the trial is the only bucket with credits; amber
 * when low; "Paused · N credits" once work on Automatos's keys has stopped (never
 * for a workspace on its own keys, which keeps working); a tooltip with the
 * buckets. Hidden in the local edition, while the summary loads or failed, and
 * on a plan billed outside credits. Pure/mocked: the summary hook and the
 * edition flag are stubbed.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

import type { BillingSummary } from '@/hooks/use-billing-summary'

const { summaryState, edition } = vi.hoisted(() => ({
  summaryState: { data: undefined as BillingSummary | undefined },
  edition: { saas: true },
}))

vi.mock('@/hooks/use-billing-summary', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-billing-summary')>()),
  useBillingSummary: () => summaryState,
}))
vi.mock('@/lib/auth-edition', () => ({
  get isSaaS() {
    return edition.saas
  },
  get isLocal() {
    return !edition.saas
  },
}))

import { CreditBalancePill } from '../credit-balance-pill'

const TRIAL_ONLY: BillingSummary = {
  plan: 'basic',
  plan_name: 'Own keys',
  price_usd: 9,
  price_label: 'beta',
  billing_open: false,
  status: 'active',
  renewal_date: null,
  balance: { trial: 420, monthly: 0, topup: 0, spendable: 420 },
  trial_credits: 500,
  platform_ai: false,
  metered: true,
  paused: false,
  low: false,
  own_keys: false,
}

const PRO: BillingSummary = {
  ...TRIAL_ONLY,
  plan: 'pro',
  plan_name: 'Pro',
  price_usd: 29,
  platform_ai: true,
  balance: { trial: 0, monthly: 1500, topup: 250, spendable: 1750 },
}

const PAUSED: BillingSummary = {
  ...TRIAL_ONLY,
  status: 'paused',
  balance: { trial: 0, monthly: 0, topup: 0, spendable: 0 },
  paused: true,
}

function withSummary(data: BillingSummary | undefined) {
  summaryState.data = data
}

const pill = () => screen.getByTestId('credit-balance-pill')

describe('CreditBalancePill (PRD-265 US-005 / US-008)', () => {
  beforeEach(() => {
    edition.saas = true
    withSummary(undefined)
  })

  it('shows the spendable credits once a paid bucket holds credits', () => {
    withSummary(PRO)
    render(<CreditBalancePill />)

    expect(pill()).toHaveTextContent('1,750 credits')
    expect(pill()).not.toHaveTextContent('trial')
    expect(pill()).toHaveAttribute('data-state', 'ok')
    expect(pill().className).not.toContain('warning')
  })

  it('says "N of 500 trial credits" while the trial is the only bucket with credits', () => {
    withSummary(TRIAL_ONLY)
    render(<CreditBalancePill />)

    expect(pill()).toHaveTextContent('420 of 500 trial credits')
    expect(pill()).toHaveAttribute('data-state', 'ok')
  })

  it('goes amber when the balance runs low', () => {
    withSummary({ ...TRIAL_ONLY, balance: { ...TRIAL_ONLY.balance, trial: 40, spendable: 40 }, low: true })
    render(<CreditBalancePill />)

    expect(pill()).toHaveAttribute('data-state', 'low')
    expect(pill().className).toContain('text-warning')
    expect(pill()).toHaveTextContent('40 of 500 trial credits')
  })

  it('reads "Paused · 0 credits" in amber once work on Automatos\'s keys has stopped', () => {
    withSummary(PAUSED)
    render(<CreditBalancePill />)

    expect(pill()).toHaveAttribute('data-state', 'paused')
    expect(pill()).toHaveTextContent('Paused · 0 credits')
    expect(pill().className).toContain('text-warning')
    expect(pill()).toHaveAttribute('title', expect.stringContaining("Work on Automatos's AI keys is paused."))
  })

  it('is amber but never "Paused" for a workspace on its own keys, which keeps working', () => {
    withSummary({ ...PAUSED, own_keys: true })
    render(<CreditBalancePill />)

    expect(pill()).toHaveAttribute('data-state', 'low')
    expect(pill()).not.toHaveTextContent('Paused')
    expect(pill()).toHaveTextContent('0 of 500 trial credits')
    expect(pill()).toHaveAttribute('title', expect.stringContaining('Your own AI keys keep working.'))
  })

  it('lists the buckets in its tooltip', () => {
    withSummary(PRO)
    render(<CreditBalancePill />)

    expect(pill()).toHaveAttribute('title', 'Trial: 0 credits\nMonthly: 1,500 credits\nTop-up: 250 credits')
  })

  it('is hidden on a plan billed outside credits (not metered)', () => {
    withSummary({ ...PRO, plan: 'enterprise', metered: false, balance: { ...PRO.balance, spendable: null } })
    const { container } = render(<CreditBalancePill />)

    expect(screen.queryByTestId('credit-balance-pill')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('is hidden in the local edition', () => {
    edition.saas = false
    withSummary(PRO)
    const { container } = render(<CreditBalancePill />)

    expect(screen.queryByTestId('credit-balance-pill')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('is hidden while the summary loads or when it failed to load', () => {
    withSummary(undefined)
    const { container } = render(<CreditBalancePill />)

    expect(screen.queryByTestId('credit-balance-pill')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('applies the caller-supplied className', () => {
    withSummary(PRO)
    render(<CreditBalancePill className="ml-2" />)

    expect(pill().className).toContain('ml-2')
  })
})
