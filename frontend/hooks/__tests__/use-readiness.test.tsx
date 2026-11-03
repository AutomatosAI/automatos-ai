/**
 * PRD-268 §6.11.3, K12 (US-C21b) — readiness on W1's routes (Check again replaces the answer; a
 * check over the limit says so), and calibration while the server offers none (US-W06: the hooks are
 * live on US-C22's routes, and CALIBRATION_ENABLED ships off): nothing offered, no answer sent.
 */
import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestMock = vi.hoisted(() => vi.fn())
const postMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    post: (...args: unknown[]) => postMock(...args),
  },
}))

import { CALIBRATION_PATH, CALIBRATION_UNAVAILABLE, useAnswerCalibration, useCalibration } from '@/hooks/use-calibration'
import { checkedRecently, readinessQueryKeys, useCheckReadiness, useReadiness } from '@/hooks/use-readiness'

function readiness(state: 'ready' | 'not_checked' | 'needs_setup', checkedAt: string) {
  return {
    v: 1,
    checked_at: checkedAt,
    items: [
      { id: 'core', label: 'Automatos', detail: 'Answering', state: 'ready', needed_by: ['brief', 'check_in'] },
      {
        id: 'web_search',
        label: 'Web search',
        detail: 'Not connected',
        state,
        fix: { label: 'Connect web search', href: '/settings?tab=integrations' },
        needed_by: ['check_in'],
      },
    ],
  }
}

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  return { client, wrapper }
}

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
})

describe('useReadiness', () => {
  it('reads Readiness@1 from W1 route', async () => {
    requestMock.mockResolvedValueOnce(readiness('needs_setup', '2026-10-30T07:00:00Z'))
    const { wrapper } = setup()
    const { result } = renderHook(() => useReadiness(), { wrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(requestMock).toHaveBeenCalledWith('/api/readiness')
    expect(result.current.data?.items.map((item) => [item.id, item.state])).toEqual([
      ['core', 'ready'],
      ['web_search', 'needs_setup'],
    ])
    expect(result.current.data?.items[1].fix).toEqual({ label: 'Connect web search', href: '/settings?tab=integrations' })
  })

  it('drops a fix that is not an app path and an item that is not one', async () => {
    const answer = readiness('needs_setup', '2026-10-30T07:00:00Z')
    requestMock.mockResolvedValueOnce({
      ...answer,
      items: [
        { ...answer.items[1], fix: { label: 'Elsewhere', href: '//evil.example/x' } },
        { ...answer.items[1], id: 'speech', fix: { label: 'Script', href: 'javascript:alert(1)' } },
        { id: 'calendar', label: 'Calendar', detail: '', state: 'broken', needed_by: [] },
      ],
    })
    const { wrapper } = setup()
    const { result } = renderHook(() => useReadiness(), { wrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data?.items.map((item) => [item.id, item.fix])).toEqual([
      ['web_search', null],
      ['speech', null],
    ])
  })
})

describe('useCheckReadiness', () => {
  it('posts Check again and shows its answer everywhere', async () => {
    requestMock.mockResolvedValueOnce(readiness('not_checked', '2026-10-30T07:00:00Z'))
    postMock.mockResolvedValueOnce(readiness('ready', '2026-10-30T07:05:00Z'))
    const { client, wrapper } = setup()
    const { result } = renderHook(() => ({ read: useReadiness(), check: useCheckReadiness() }), { wrapper })
    await waitFor(() => expect(result.current.read.isSuccess).toBe(true))

    act(() => {
      result.current.check.mutate()
    })

    await waitFor(() => expect(result.current.check.isSuccess).toBe(true))
    expect(postMock).toHaveBeenCalledWith('/api/readiness/check')
    expect(client.getQueryData(readinessQueryKeys.all)).toMatchObject({ checked_at: '2026-10-30T07:05:00Z' })
    await waitFor(() => expect(result.current.read.data?.items[1].state).toBe('ready'))
  })

  it('says when the workspace checked a moment ago, and keeps the answer shown', async () => {
    requestMock.mockResolvedValueOnce(readiness('not_checked', '2026-10-30T07:00:00Z'))
    const detail = 'Checked a moment ago. Try again in a minute.'
    postMock.mockRejectedValueOnce(Object.assign(new Error(detail), { status: 429, body: { code: 'checked_recently', detail } }))
    const { wrapper } = setup()
    const { result } = renderHook(() => ({ read: useReadiness(), check: useCheckReadiness() }), { wrapper })
    await waitFor(() => expect(result.current.read.isSuccess).toBe(true))

    act(() => {
      result.current.check.mutate()
    })

    await waitFor(() => expect(result.current.check.isError).toBe(true))
    expect(checkedRecently(result.current.check.error)).toBe(true)
    expect(result.current.read.data?.checked_at).toBe('2026-10-30T07:00:00Z')
  })

  it('does not call another failure checked recently', () => {
    expect(checkedRecently(Object.assign(new Error('Service Unavailable'), { status: 503 }))).toBe(false)
    expect(checkedRecently(null)).toBe(false)
  })
})

describe('calibration while the server offers none (CALIBRATION_ENABLED off, K15)', () => {
  const NOT_OFFERED = { available: false, offer: false, consent_line: '', status: null, calibration: null, rejected_lines: [], accepted_lines: [] }
  /** Only the calibration read was asked: nothing was answered. */
  const onlyTheRead = () => requestMock.mock.calls.every(([path]) => String(path).startsWith(CALIBRATION_PATH))

  it('offers nothing and is not available', async () => {
    requestMock.mockResolvedValue(NOT_OFFERED)
    const { wrapper } = setup()
    const { result } = renderHook(() => useCalibration(), { wrapper })

    await waitFor(() => expect(result.current.isLoading).toBe(false))
    expect(result.current).toEqual({ data: null, available: false, isLoading: false })
    expect(onlyTheRead()).toBe(true)
  })

  it('refuses an answer as not available on this server, and sends no answer', async () => {
    requestMock.mockResolvedValue(NOT_OFFERED)
    const { wrapper } = setup()
    const { result } = renderHook(() => useAnswerCalibration(), { wrapper })
    await waitFor(() => expect(requestMock).toHaveBeenCalled())

    expect(result.current.available).toBe(false)
    await expect(result.current.mutate({ calibration_id: 'c-1', answer: 'accept' })).rejects.toThrow(CALIBRATION_UNAVAILABLE)
    await expect(
      result.current.mutate({ calibration_id: 'c-1', answer: 'reject', line_id: 'l-1' }),
    ).rejects.toThrow('not available on this server')
    expect(onlyTheRead()).toBe(true)
    expect(postMock).not.toHaveBeenCalled()
  })
})
