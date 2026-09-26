/**
 * #1094: a refused status change says why.
 *
 * The board moves the card at once and moves it back when the server refuses the
 * change. It now also shows the server's reason, e.g. "Assign an agent first…".
 */
import { describe, it, expect, vi } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const requestMock = vi.hoisted(() => vi.fn())
const toastError = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({
  apiClient: { request: (...args: unknown[]) => requestMock(...args) },
}))
vi.mock('sonner', () => ({ toast: { error: toastError, success: vi.fn() } }))

import { useUpdateTaskStatus } from '@/hooks/use-board-tasks'

const REFUSAL = 'Assign an agent first: a ticket with no agent cannot be in progress.'

function wrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children)
  }
}

describe('useUpdateTaskStatus', () => {
  it("shows the server's reason when it refuses the move", async () => {
    requestMock.mockRejectedValueOnce(Object.assign(new Error(REFUSAL), { status: 409 }))
    const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
    const { result } = renderHook(() => useUpdateTaskStatus(), { wrapper: wrapper(client) })

    act(() => {
      result.current.mutate({ taskId: '1094', status: 'in_progress' })
    })

    await waitFor(() => expect(toastError).toHaveBeenCalledWith(REFUSAL))
  })
})
