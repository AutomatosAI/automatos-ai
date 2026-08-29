/**
 * PRD-228 US-004 — the useFleetState hook.
 *
 * Extends the existing agents hook family (no duplicate/V2 hook). Polls the
 * fleet route on the 10s mission-detail cadence and ALSO refetches immediately
 * when the board stream fans out an agent move (`automatos:board-changed`).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const getFleetStateMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({
  apiClient: { getFleetState: () => getFleetStateMock() },
}))

import { useFleetState, FLEET_POLL_MS, agentQueryKeys } from '@/hooks/use-agent-api'

function wrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children)
  }
}

function makeClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
}

const PAYLOAD = {
  version: 1,
  generated_at: '2026-08-28T12:00:00Z',
  window_hours: 24,
  cost_available: true,
  cost_source: 'llm_usage',
  agents: [],
}

describe('useFleetState — PRD-228', () => {
  beforeEach(() => {
    getFleetStateMock.mockReset().mockResolvedValue(PAYLOAD)
  })
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('polls on the 10s mission-detail cadence', () => {
    // The interval is a named constant wired into the query's refetchInterval.
    expect(FLEET_POLL_MS).toBe(10000)
  })

  it('fetches the fleet route and exposes its shape', async () => {
    const client = makeClient()
    const { result } = renderHook(() => useFleetState(), { wrapper: wrapper(client) })
    await waitFor(() => expect(result.current.data).toEqual(PAYLOAD))
    expect(getFleetStateMock).toHaveBeenCalledTimes(1)
  })

  it('refetches immediately on a board-changed event', async () => {
    const client = makeClient()
    renderHook(() => useFleetState(), { wrapper: wrapper(client) })
    await waitFor(() => expect(getFleetStateMock).toHaveBeenCalledTimes(1))

    // The board stream fans out an agent move; the fleet view refetches now,
    // not on the next poll tick.
    act(() => {
      window.dispatchEvent(new CustomEvent('automatos:board-changed', { detail: { task_id: 7 } }))
    })

    await waitFor(() => expect(getFleetStateMock).toHaveBeenCalledTimes(2))
  })

  it('the board listener is removed on unmount (no leak, no refetch after)', async () => {
    const client = makeClient()
    const { unmount } = renderHook(() => useFleetState(), { wrapper: wrapper(client) })
    await waitFor(() => expect(getFleetStateMock).toHaveBeenCalledTimes(1))
    unmount()
    act(() => {
      window.dispatchEvent(new CustomEvent('automatos:board-changed'))
    })
    // No further fetch after unmount.
    await new Promise((r) => setTimeout(r, 20))
    expect(getFleetStateMock).toHaveBeenCalledTimes(1)
  })

  it('uses a fleet query key under the agents family (no rival cache)', () => {
    expect(agentQueryKeys.fleet).toEqual(['agents', 'fleet'])
  })
})
