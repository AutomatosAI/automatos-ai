/**
 * PRD-268 §6.4, K11 (US-C21b) — useOpenBrief posts the browser's day and remembers where the day's
 * brief is; useBrief reads that message's brief@1 part, and never opens the brief itself.
 */
import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const requestMock = vi.hoisted(() => vi.fn())
const postMock = vi.hoisted(() => vi.fn())
const workspace = vi.hoisted(() => ({ id: 'ws-1' as string | null }))

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    post: (...args: unknown[]) => postMock(...args),
  },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => (workspace.id === null ? null : { workspace: { id: workspace.id } }),
}))

import { briefQueryKeys, useBrief, useOpenBrief } from '@/hooks/use-brief'
import { browserDay, localDateOf, rememberedBrief } from '@/lib/proactive/brief-day'

const CHAT = '5b7f2c1e-8d4a-4f6b-9c3e-1a2b3c4d5e6f'
const MESSAGE = '9e8d7c6b-5a4f-4e3d-8c2b-1a0f9e8d7c6b'
const TODAY = localDateOf(new Date())

const BRIEF = {
  v: 1,
  date: TODAY,
  greeting: 'Morning, Sam.',
  prose: 'Two things need you today.',
  spoken: 'Morning, Sam. One: Lisbon flights.',
  sections: {
    needs_you: [{ id: 'grant:12', title: 'Send the Airbnb split', why: 'Waiting for your approval', display_state: 'needs_you' }],
    suggestions: [{ id: 'sugg:1', title: 'Lisbon flights', why: 'Prices dropped', action: { verb: 'start', label: 'Start' } }],
    today: [],
    since: [],
    unchecked: [],
  },
  counts: { needs_you: 1, suggestions: 1, today: 0, since: 0 },
  labels: { sections: { needs_you: 'Needs you', suggestions: 'Suggested' }, more: {} },
  state_hash: 'abc123',
  generated_at: '2026-10-30T07:00:00Z',
}

const MESSAGES = [
  { id: '11111111-2222-4333-8444-555555555555', role: 'assistant', parts: [{ type: 'text', text: 'Older' }] },
  {
    id: MESSAGE,
    role: 'assistant',
    parts: [{ type: 'text', text: BRIEF.prose }, { type: 'brief@1', brief: BRIEF }],
    source: { origin: 'daily_digest', local_date: TODAY, via: 'open' },
  },
]

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
  workspace.id = 'ws-1'
  window.localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useOpenBrief', () => {
  it("posts the browser's local date, time zone and language", async () => {
    postMock.mockResolvedValueOnce({ chat_id: CHAT, message_id: MESSAGE, posted: true })
    const { wrapper } = setup()
    const { result } = renderHook(() => useOpenBrief(), { wrapper })

    act(() => {
      result.current.mutate()
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(postMock).toHaveBeenCalledWith('/api/proactive/brief/open', browserDay())
    expect(browserDay().local_date).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    expect(result.current.data).toEqual({ chat_id: CHAT, message_id: MESSAGE, posted: true })
    expect(rememberedBrief('ws-1', TODAY)).toEqual({ chat_id: CHAT, message_id: MESSAGE })
  })

  it('remembers nothing while another open is still building the brief', async () => {
    postMock.mockResolvedValueOnce({ posted: false, pending: true })
    const { wrapper } = setup()
    const { result } = renderHook(() => useOpenBrief(), { wrapper })

    act(() => {
      result.current.mutate({ local_date: TODAY, timezone: 'Europe/Lisbon' })
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual({ posted: false, pending: true })
    expect(rememberedBrief('ws-1', TODAY)).toBeNull()
  })

  it('reads an answer whose ids are not ids as pending', async () => {
    postMock.mockResolvedValueOnce({ chat_id: '../admin', message_id: MESSAGE, posted: true })
    const { wrapper } = setup()
    const { result } = renderHook(() => useOpenBrief(), { wrapper })

    act(() => {
      result.current.mutate()
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual({ posted: false, pending: true })
  })
})

describe('useBrief', () => {
  it('reads nothing, and opens nothing, until an open has said where the brief is', async () => {
    const { client, wrapper } = setup()
    const { result } = renderHook(() => useBrief(), { wrapper })

    await waitFor(() => expect(client.getQueryState(briefQueryKeys.where('ws-1', TODAY))?.status).toBe('success'))
    expect(result.current.fetchStatus).toBe('idle')
    expect(result.current.data).toBeUndefined()
    expect(requestMock).not.toHaveBeenCalled()
    expect(postMock).not.toHaveBeenCalled()
  })

  it("reads the day's brief from its message once the open answers", async () => {
    postMock.mockResolvedValueOnce({ chat_id: CHAT, message_id: MESSAGE, posted: false })
    requestMock.mockResolvedValue(MESSAGES)
    const { client, wrapper } = setup()
    const { result } = renderHook(() => ({ brief: useBrief(), open: useOpenBrief() }), { wrapper })
    await waitFor(() => expect(client.getQueryState(briefQueryKeys.where('ws-1', TODAY))?.status).toBe('success'))

    act(() => {
      result.current.open.mutate()
    })

    await waitFor(() => expect(result.current.brief.data).not.toBeUndefined())
    expect(requestMock).toHaveBeenCalledWith(`/api/chat/${CHAT}/messages`)
    expect(result.current.brief.data?.greeting).toBe('Morning, Sam.')
    expect(result.current.brief.data?.spoken).toBe(BRIEF.spoken)
    expect(result.current.brief.data?.counts.needs_you).toBe(1)
    expect(result.current.brief.data?.sections.suggestions[0].title).toBe('Lisbon flights')
  })

  it("reads the day's brief after a reload, from where this browser remembered it", async () => {
    window.localStorage.setItem(`automatos:brief:ws-1:${TODAY}`, JSON.stringify({ chat_id: CHAT, message_id: MESSAGE }))
    requestMock.mockResolvedValue(MESSAGES)
    const { wrapper } = setup()
    const { result } = renderHook(() => useBrief(), { wrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data?.state_hash).toBe('abc123')
    expect(postMock).not.toHaveBeenCalled()
  })

  it('never reads another workspace brief', async () => {
    window.localStorage.setItem(`automatos:brief:ws-2:${TODAY}`, JSON.stringify({ chat_id: CHAT, message_id: MESSAGE }))
    const { client, wrapper } = setup()
    const { result } = renderHook(() => useBrief(), { wrapper })

    await waitFor(() => expect(client.getQueryState(briefQueryKeys.where('ws-1', TODAY))?.status).toBe('success'))
    expect(client.getQueryData(briefQueryKeys.where('ws-1', TODAY))).toBeNull()
    expect(result.current.fetchStatus).toBe('idle')
    expect(result.current.data).toBeUndefined()
    expect(requestMock).not.toHaveBeenCalled()
  })

  it('is null when the message carries no brief@1 part', async () => {
    window.localStorage.setItem(`automatos:brief:ws-1:${TODAY}`, JSON.stringify({ chat_id: CHAT, message_id: MESSAGE }))
    requestMock.mockResolvedValue([{ id: MESSAGE, role: 'assistant', parts: [{ type: 'text', text: 'Only prose' }] }])
    const { wrapper } = setup()
    const { result } = renderHook(() => useBrief(), { wrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toBeNull()
  })

  it('forgets where the brief was when its message is gone, so the next open is asked again', async () => {
    const key = `automatos:brief:ws-1:${TODAY}`
    window.localStorage.setItem(key, JSON.stringify({ chat_id: CHAT, message_id: MESSAGE }))
    requestMock.mockRejectedValue(Object.assign(new Error('Chat not found'), { status: 404, body: { detail: 'Chat not found' } }))
    const { client, wrapper } = setup()
    const { result } = renderHook(() => useBrief(), { wrapper })

    await waitFor(() => expect(client.getQueryData(briefQueryKeys.where('ws-1', TODAY))).toBeNull())
    expect(window.localStorage.getItem(key)).toBeNull()
    expect(result.current.isError).toBe(false)
  })

  it('forgets the other days when it remembers a new one', async () => {
    window.localStorage.setItem('automatos:brief:ws-1:2026-10-01', JSON.stringify({ chat_id: CHAT, message_id: MESSAGE }))
    window.localStorage.setItem('automatos:brief:ws-2:2026-10-01', JSON.stringify({ chat_id: CHAT, message_id: MESSAGE }))
    postMock.mockResolvedValueOnce({ chat_id: CHAT, message_id: MESSAGE, posted: true })
    const { wrapper } = setup()
    const { result } = renderHook(() => useOpenBrief(), { wrapper })

    act(() => {
      result.current.mutate()
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(window.localStorage.getItem('automatos:brief:ws-1:2026-10-01')).toBeNull()
    expect(window.localStorage.getItem('automatos:brief:ws-2:2026-10-01')).not.toBeNull()
  })

  it('still reads the opened brief in this tab when storage is blocked', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    postMock.mockResolvedValueOnce({ chat_id: CHAT, message_id: MESSAGE, posted: true })
    requestMock.mockResolvedValue(MESSAGES)
    const { client, wrapper } = setup()
    const { result } = renderHook(() => ({ brief: useBrief(), open: useOpenBrief() }), { wrapper })
    await waitFor(() => expect(client.getQueryState(briefQueryKeys.where('ws-1', TODAY))?.status).toBe('success'))

    act(() => {
      result.current.open.mutate()
    })

    await waitFor(() => expect(result.current.brief.data?.greeting).toBe('Morning, Sam.'))
  })
})
