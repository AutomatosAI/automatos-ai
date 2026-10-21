/**
 * PRD-265 US-008 — useBillingSummary: one cached answer from
 * GET /api/v1/billing/summary for every reader (header balance, paused banner,
 * the theme SDK's trial), never asked in the local edition, and asked again at
 * once when a reply lands in a chat from the background (automatos:chat-changed).
 * A real QueryClient; apiClient and the edition flag are stand-ins.
 */
import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { requestMock, edition } = vi.hoisted(() => ({
  requestMock: vi.fn(),
  edition: { saas: true },
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: requestMock } }))
vi.mock('@/lib/auth-edition', () => ({
  get isSaaS() {
    return edition.saas
  },
  get isLocal() {
    return !edition.saas
  },
}))

import { BILLING_SUMMARY_PATH, isWorkPaused, useBillingSummary, type BillingSummary } from '@/hooks/use-billing-summary'

const SUMMARY: BillingSummary = {
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

function renderSummaryHooks() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const Root = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  return renderHook(() => ({ header: useBillingSummary(), banner: useBillingSummary() }), { wrapper: Root })
}

beforeEach(() => {
  edition.saas = true
  requestMock.mockReset()
  requestMock.mockResolvedValue(SUMMARY)
})

afterEach(() => {
  edition.saas = true
})

describe('useBillingSummary', () => {
  it('asks GET /api/v1/billing/summary once and hands every reader the same answer', async () => {
    const { result } = renderSummaryHooks()

    await waitFor(() => expect(result.current.header.data).toEqual(SUMMARY))
    expect(result.current.banner.data).toBe(result.current.header.data)
    expect(requestMock).toHaveBeenCalledTimes(1)
    expect(requestMock).toHaveBeenCalledWith(BILLING_SUMMARY_PATH)
    expect(BILLING_SUMMARY_PATH).toBe('/api/v1/billing/summary')
  })

  it('never asks in the local edition, where nothing is metered', async () => {
    edition.saas = false
    const { result } = renderSummaryHooks()
    await act(async () => {
      await Promise.resolve()
    })

    expect(requestMock).not.toHaveBeenCalled()
    expect(result.current.header.data).toBeUndefined()
  })

  it('asks again when a reply lands in a chat from the background', async () => {
    renderSummaryHooks()
    await waitFor(() => expect(requestMock).toHaveBeenCalledTimes(1))

    requestMock.mockResolvedValue({ ...SUMMARY, balance: { ...SUMMARY.balance, trial: 415, spendable: 415 } })
    act(() => {
      window.dispatchEvent(new CustomEvent('automatos:chat-changed', { detail: { chat_id: 'c1' } }))
    })

    await waitFor(() => expect(requestMock).toHaveBeenCalledTimes(2))
  })
})

describe('isWorkPaused', () => {
  it('is true only when work is paused and the workspace has no keys of its own', () => {
    expect(isWorkPaused({ ...SUMMARY, paused: true })).toBe(true)
    expect(isWorkPaused({ ...SUMMARY, paused: true, own_keys: true })).toBe(false)
    expect(isWorkPaused({ ...SUMMARY, paused: false })).toBe(false)
    expect(isWorkPaused(undefined)).toBe(false)
    expect(isWorkPaused(null)).toBe(false)
  })
})
