/**
 * F282 (night 8) — the mission page's switch calls PATCH .../settings and
 * refreshes the mission once the server confirms the change.
 */
import { describe, it, expect, vi } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const requestMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({
  apiClient: { request: (...args: unknown[]) => requestMock(...args) },
}))

import { useUpdateMissionSettings, missionQueryKeys } from '@/hooks/use-missions-api'

function wrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children)
  }
}

describe('useUpdateMissionSettings', () => {
  it('PATCHes the one setting and refreshes the mission', async () => {
    requestMock.mockResolvedValueOnce({ id: '0324', check_each_step: true })
    const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
    const invalidate = vi.spyOn(client, 'invalidateQueries')
    const { result } = renderHook(() => useUpdateMissionSettings(), { wrapper: wrapper(client) })

    act(() => {
      result.current.mutate({ id: '0324', body: { check_each_step: true } })
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(requestMock).toHaveBeenCalledWith('/api/missions/0324/settings', {
      method: 'PATCH',
      body: { check_each_step: true },
    })
    expect(invalidate).toHaveBeenCalledWith({ queryKey: missionQueryKeys.detail('0324') })
  })

  it('leaves the mutation in an error state when the mission is done', async () => {
    const refusal = Object.assign(new Error('Mission is completed; its step-check setting can no longer change.'),
                                  { status: 409 })
    requestMock.mockRejectedValueOnce(refusal)
    const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
    const { result } = renderHook(() => useUpdateMissionSettings(), { wrapper: wrapper(client) })

    act(() => {
      result.current.mutate({ id: '0324', body: { check_each_step: true } })
    })

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(result.current.error?.message).toBe(refusal.message)
  })
})
