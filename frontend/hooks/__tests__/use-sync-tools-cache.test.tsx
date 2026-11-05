/**
 * The Tools Sync button says what the sync did (4 Nov 2026).
 *
 * POST /api/tools/sync answers 200 with the job. A failed sync (Composio rejected the
 * key) used to toast "Marketplace cache sync finished", and a Sync pressed while one
 * ran started a second one. Now a failed job toasts its reason, and a running one is
 * reported instead of a second sync.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const syncMock = vi.hoisted(() => vi.fn())
const toastMock = vi.hoisted(() => Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn() }))

vi.mock('@/lib/api-client', () => ({
  apiClient: { syncToolsCache: (...args: unknown[]) => syncMock(...args) },
}))
vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ isLoaded: true, isSignedIn: true }) }))
vi.mock('sonner', () => ({ toast: toastMock }))

import {
  SYNC_ALREADY_RUNNING_TITLE,
  SYNC_FAILED_FALLBACK,
  SYNC_FAILED_TITLE,
  useSyncToolsCache,
} from '@/hooks/use-tools-api'

const KEY_REJECTED = 'Composio rejected the API key. Check COMPOSIO_KEY.'

function pressSync() {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client }, children)
  const { result } = renderHook(() => useSyncToolsCache(), { wrapper })
  act(() => {
    result.current.mutate('full')
  })
  return result
}

beforeEach(() => {
  syncMock.mockReset()
  toastMock.mockReset()
  toastMock.error.mockReset()
})

describe('useSyncToolsCache', () => {
  it("toasts the failed sync's reason instead of 'finished'", async () => {
    syncMock.mockResolvedValueOnce({ job_id: 7, status: 'failed', errors_count: 1, error: KEY_REJECTED })
    pressSync()

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(SYNC_FAILED_TITLE, { description: KEY_REJECTED }))
    expect(toastMock).not.toHaveBeenCalled()
  })

  it('falls back to where the details are when the failure has no reason for people', async () => {
    syncMock.mockResolvedValueOnce({ job_id: 8, status: 'failed', errors_count: 1, error: null })
    pressSync()

    await waitFor(() =>
      expect(toastMock.error).toHaveBeenCalledWith(SYNC_FAILED_TITLE, { description: SYNC_FAILED_FALLBACK }),
    )
  })

  it('says a sync is already running when the route returns the running job', async () => {
    syncMock.mockResolvedValueOnce({ job_id: 41, status: 'running', already_running: true })
    pressSync()

    await waitFor(() => expect(toastMock).toHaveBeenCalledTimes(1))
    expect(toastMock.mock.calls[0][0]).toBe(SYNC_ALREADY_RUNNING_TITLE)
    expect(toastMock.error).not.toHaveBeenCalled()
  })

  it('still says finished for a completed sync', async () => {
    syncMock.mockResolvedValueOnce({ job_id: 9, status: 'completed', errors_count: 0, error: null })
    pressSync()

    await waitFor(() => expect(toastMock).toHaveBeenCalledTimes(1))
    expect(toastMock.mock.calls[0][0]).toBe('Sync started/completed')
    expect(toastMock.error).not.toHaveBeenCalled()
  })
})
