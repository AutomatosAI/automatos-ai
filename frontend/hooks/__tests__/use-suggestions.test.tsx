/**
 * PRD-268 §6.2.5 (US-C21b) — one list of open suggestions behind every surface: asked for again on
 * each change event, Start and Hide through W1's routes, and a refused Start that keeps the card
 * with the server's reason.
 */
import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, expect, it, vi, beforeEach } from 'vitest'

const requestMock = vi.hoisted(() => vi.fn())
const postMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    post: (...args: unknown[]) => postMock(...args),
  },
}))

import {
  SUGGESTION_REFRESH_EVENTS,
  suggestionQueryKeys,
  useHideSuggestion,
  useStartSuggestion,
  useSuggestions,
} from '@/hooks/use-suggestions'
import { HIDE_REASONS } from '@/lib/proactive/types'
import { refusalOf } from '@/lib/proactive/wire'

const LIST_PATH = '/api/proactive/suggestions?status=open'
const ID = '7d1c0a52-3e0b-4c55-9a1e-2f6f0c1d9b10'
const OTHER_ID = '0b9e7c4d-1a2b-4c3d-8e9f-a0b1c2d3e4f5'

function suggestion(id: string, title: string) {
  return {
    id,
    title,
    detail: 'Three runs failed since Monday.',
    evidence: [{ type: 'ticket', id: '0419' }],
    proposed_action: { kind: 'board_task', payload: {} },
    severity: 'today',
    status: 'open',
    est_credits: null,
  }
}

const LIST = { suggestions: [suggestion(ID, 'Look at the Lisbon flights'), suggestion(OTHER_ID, 'Retry the invoice export')] }

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const invalidate = vi.spyOn(client, 'invalidateQueries')
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const listAskedAgain = () =>
    invalidate.mock.calls.filter(([filters]) => JSON.stringify(filters) === JSON.stringify({ queryKey: suggestionQueryKeys.all })).length
  return { client, wrapper, listAskedAgain }
}

function listRequests(): number {
  return requestMock.mock.calls.filter(([path]) => path === LIST_PATH).length
}

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  requestMock.mockResolvedValue(LIST)
})

describe('useSuggestions', () => {
  it('lists the open suggestions from W1 route, newest first as the server sends them', async () => {
    const { wrapper } = setup()
    const { result } = renderHook(() => useSuggestions(), { wrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(requestMock).toHaveBeenCalledWith(LIST_PATH)
    expect(result.current.data?.map((row) => row.id)).toEqual([ID, OTHER_ID])
    expect(result.current.data?.[0].title).toBe('Look at the Lisbon flights')
  })

  it('leaves out a row whose id is not a suggestion id or that has no title', async () => {
    requestMock.mockResolvedValue({
      suggestions: [suggestion(ID, 'Kept'), { ...suggestion('../../admin', 'Bad id') }, { id: OTHER_ID }],
    })
    const { wrapper } = setup()
    const { result } = renderHook(() => useSuggestions(), { wrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data?.map((row) => row.title)).toEqual(['Kept'])
  })

  it.each(SUGGESTION_REFRESH_EVENTS)('asks for the list again on %s', async (name) => {
    const { wrapper } = setup()
    const { result } = renderHook(() => useSuggestions(), { wrapper })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(listRequests()).toBe(1)

    act(() => {
      window.dispatchEvent(new CustomEvent(name, { detail: {} }))
    })

    await waitFor(() => expect(listRequests()).toBe(2))
  })

  it('stops listening once unmounted', async () => {
    const { wrapper, listAskedAgain } = setup()
    const view = renderHook(() => useSuggestions(), { wrapper })
    await waitFor(() => expect(view.result.current.isSuccess).toBe(true))
    view.unmount()

    act(() => {
      SUGGESTION_REFRESH_EVENTS.forEach((name) => window.dispatchEvent(new CustomEvent(name)))
    })

    expect(listAskedAgain()).toBe(0)
    expect(listRequests()).toBe(1)
  })
})

describe('useStartSuggestion', () => {
  it('posts Start and asks for the list again', async () => {
    postMock.mockResolvedValueOnce({ suggestion: { ...LIST.suggestions[0], status: 'started' }, opens: { type: 'board_task', board_task_id: 846 } })
    const { wrapper, listAskedAgain } = setup()
    const { result } = renderHook(() => useStartSuggestion(), { wrapper })

    act(() => {
      result.current.mutate({ id: ID })
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(postMock).toHaveBeenCalledWith(`/api/proactive/suggestions/${ID}/start`, {})
    expect(result.current.data?.opens).toEqual({ type: 'board_task', board_task_id: 846 })
    expect(listAskedAgain()).toBe(1)
  })

  it('says where the person pressed Start when the caller names it', async () => {
    postMock.mockResolvedValueOnce({ suggestion: LIST.suggestions[0], opens: { type: 'chat', suggestion_id: ID } })
    const { wrapper } = setup()
    const { result } = renderHook(() => useStartSuggestion(), { wrapper })

    act(() => {
      result.current.mutate({ id: ID, via: 'push' })
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(postMock).toHaveBeenCalledWith(`/api/proactive/suggestions/${ID}/start`, { via: 'push' })
  })

  it('keeps the card when Start is refused, with the server reason', async () => {
    const reason = "Today's spending limit is reached, so Auto can't start new work until tomorrow."
    postMock.mockRejectedValueOnce(
      Object.assign(new Error(reason), { status: 409, body: { code: 'spend_ceiling', detail: reason } }),
    )
    const { wrapper, listAskedAgain } = setup()
    const { result } = renderHook(() => ({ list: useSuggestions(), start: useStartSuggestion() }), { wrapper })
    await waitFor(() => expect(result.current.list.isSuccess).toBe(true))

    act(() => {
      result.current.start.mutate({ id: ID })
    })

    await waitFor(() => expect(result.current.start.isError).toBe(true))
    expect(refusalOf(result.current.start.error)).toEqual({ code: 'spend_ceiling', detail: reason, status: 409 })
    expect(listAskedAgain()).toBe(0)
    expect(result.current.list.data?.map((row) => row.id)).toEqual([ID, OTHER_ID])
    expect(listRequests()).toBe(1)
  })

  it('never puts an id that is not a suggestion id into the path', async () => {
    const { wrapper } = setup()
    const { result } = renderHook(() => useStartSuggestion(), { wrapper })

    act(() => {
      result.current.mutate({ id: '../../settings' })
    })

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(postMock).not.toHaveBeenCalled()
  })
})

describe('useHideSuggestion', () => {
  it.each(HIDE_REASONS)('posts Hide with the reason %s and asks for the list again', async (reason) => {
    postMock.mockResolvedValueOnce({ suggestion: { ...LIST.suggestions[0], status: 'hidden', hide_reason: reason } })
    const { wrapper, listAskedAgain } = setup()
    const { result } = renderHook(() => useHideSuggestion(), { wrapper })

    act(() => {
      result.current.mutate({ id: ID, reason })
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(postMock).toHaveBeenCalledWith(`/api/proactive/suggestions/${ID}/hide`, { reason })
    expect(listAskedAgain()).toBe(1)
  })

  it('hides with no reason when the person gives none', async () => {
    postMock.mockResolvedValueOnce({ suggestion: { ...LIST.suggestions[0], status: 'hidden' } })
    const { wrapper } = setup()
    const { result } = renderHook(() => useHideSuggestion(), { wrapper })

    act(() => {
      result.current.mutate({ id: ID })
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(postMock).toHaveBeenCalledWith(`/api/proactive/suggestions/${ID}/hide`, {})
  })
})
