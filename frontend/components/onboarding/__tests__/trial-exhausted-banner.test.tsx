/**
 * PRD-265 US-008 — the paused banner (the theme SDK's `TrialExhaustedBanner`).
 *
 * It renders on either trigger — the billing summary saying work on Automatos's
 * keys is paused (for a workspace with no keys of its own), or the typed
 * `credits_exhausted` code a refused request returns before the summary catches
 * up — and embeds the power-up card, so a paused workspace can still add a key.
 * "Top up" shows only while paid plans are open. It makes no model call: the only
 * requests reachable from it are the summary GET and the credentials POST the
 * user triggers inside the embedded card.
 *
 * The real useBillingSummary runs under a test QueryClient; apiClient, the
 * workspace provider and next/link are stubbed.
 */
import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const { requestMock, postMock, useWorkspaceMock } = vi.hoisted(() => ({
  requestMock: vi.fn(),
  postMock: vi.fn(),
  useWorkspaceMock: vi.fn(),
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: requestMock, post: postMock } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => useWorkspaceMock(),
}))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: Record<string, unknown>) => (
    <a href={href as string} {...(rest as Record<string, string>)}>
      {children as React.ReactNode}
    </a>
  ),
}))

import { TOP_UP_HREF, TrialExhaustedBanner } from '../trial-exhausted-banner'
import { BILLING_SUMMARY_PATH, type BillingSummary } from '@/hooks/use-billing-summary'
import { CREDITS_EXHAUSTED_CODE } from '@/lib/credits'

const ACTIVE: BillingSummary = {
  plan: 'pro',
  plan_name: 'Pro',
  price_usd: 29,
  price_label: 'beta',
  billing_open: false,
  status: 'active',
  renewal_date: null,
  balance: { trial: 0, monthly: 1200, topup: 0, spendable: 1200 },
  trial_credits: 500,
  platform_ai: true,
  metered: true,
  paused: false,
  low: false,
  own_keys: false,
}

const PAUSED: BillingSummary = {
  ...ACTIVE,
  status: 'paused',
  balance: { trial: 0, monthly: 0, topup: 0, spendable: 0 },
  paused: true,
}

function withSummary(summary: BillingSummary) {
  requestMock.mockResolvedValue(summary)
}

function renderBanner(errorCode?: string | null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <TrialExhaustedBanner errorCode={errorCode} />
    </QueryClientProvider>,
  )
}

/** The summary has been asked for and its answer has reached the screen. */
async function summaryAnswered() {
  await waitFor(() => expect(requestMock).toHaveBeenCalledWith(BILLING_SUMMARY_PATH))
  await act(async () => {
    await Promise.resolve()
  })
}

describe('TrialExhaustedBanner — the paused state (PRD-265 US-008)', () => {
  beforeEach(() => {
    requestMock.mockReset()
    postMock.mockReset()
    useWorkspaceMock.mockReset()
    useWorkspaceMock.mockReturnValue({ workspace: { onboarding: { stage: 'completed' } }, refreshWorkspace: vi.fn() })
  })
  afterEach(() => vi.restoreAllMocks())

  it('renders when the summary says work is paused, embedding the power-up card', async () => {
    withSummary(PAUSED)
    renderBanner()

    const banner = await screen.findByTestId('trial-exhausted-banner')
    expect(banner).toHaveTextContent('Your credits are used up')
    expect(banner).toHaveTextContent("Work on Automatos's AI keys is paused. Add your own AI key to keep going")
    // the power-up card is embedded (its key input is the way out)
    expect(screen.getByTestId('power-up-card')).toBeInTheDocument()
    expect(screen.getByTestId('power-up-key-input')).toBeInTheDocument()
  })

  it('on the Own keys plan it says the AI runs on your own keys', async () => {
    withSummary({ ...PAUSED, plan: 'basic', plan_name: 'Own keys', platform_ai: false })
    renderBanner()

    expect(await screen.findByTestId('trial-exhausted-banner')).toHaveTextContent(
      'On the Own keys plan the AI runs on your own keys. Add your Anthropic, OpenAI or OpenRouter key to keep going.',
    )
  })

  it('renders on the typed credits_exhausted code before the summary catches up', async () => {
    withSummary(ACTIVE) // the summary still says active…
    renderBanner(CREDITS_EXHAUSTED_CODE) // …but the live request was refused

    expect(screen.getByTestId('trial-exhausted-banner')).toBeInTheDocument()
    await summaryAnswered()
    expect(screen.getByTestId('trial-exhausted-banner')).toBeInTheDocument()
    expect(screen.getByTestId('power-up-card')).toBeInTheDocument()
  })

  it('does not render while work is not paused and no request was refused', async () => {
    withSummary(ACTIVE)
    const { container } = renderBanner()
    await summaryAnswered()

    expect(screen.queryByTestId('trial-exhausted-banner')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('does not render for a workspace on its own keys, which keeps working at zero credits', async () => {
    withSummary({ ...PAUSED, own_keys: true })
    const { container } = renderBanner()
    await summaryAnswered()

    expect(screen.queryByTestId('trial-exhausted-banner')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('does not render for another turn error (not a paused workspace)', async () => {
    withSummary(ACTIVE)
    renderBanner('rate_limited')
    await summaryAnswered()

    expect(screen.queryByTestId('trial-exhausted-banner')).toBeNull()
  })

  it('offers Top up only while paid plans are open', async () => {
    withSummary({ ...PAUSED, billing_open: true })
    const open = renderBanner()
    expect(await screen.findByTestId('trial-exhausted-top-up')).toHaveAttribute('href', TOP_UP_HREF)
    expect(TOP_UP_HREF).toBe('/settings?tab=billing')
    open.unmount()

    withSummary({ ...PAUSED, billing_open: false })
    renderBanner()
    await screen.findByTestId('trial-exhausted-banner')
    expect(screen.queryByTestId('trial-exhausted-top-up')).toBeNull()
    expect(screen.queryByText('Top up')).toBeNull()
  })

  it('makes no model call on render: only the summary GET and the credentials POST are reachable', async () => {
    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue(new Response('{}'))
    withSummary(PAUSED)
    renderBanner(CREDITS_EXHAUSTED_CODE)
    await summaryAnswered()

    // the one request the banner makes is the billing summary
    expect(requestMock).toHaveBeenCalledTimes(1)
    expect(requestMock).toHaveBeenCalledWith(BILLING_SUMMARY_PATH)
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(postMock).not.toHaveBeenCalled()
    // the credentials endpoint is the only other wired call, and only on user submit
    expect(screen.getByTestId('power-up-connect')).toBeInTheDocument()
  })
})
