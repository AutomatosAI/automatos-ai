/**
 * PRD-180 S1 (F090) — the Command Centre subscribes to the board SSE and stops
 * polling (vitest proxy for the browser AC; the loop is headless).
 *
 * Pins:
 *   1. The board query no longer sets an interval poll (`refetchInterval`) —
 *      real-time push replaces the 60s tick.
 *   2. `useBoardEventStream` opens a fetch stream to `/api/v1/tasks/stream` and
 *      invalidates the board react-query cache on each pushed `board_changed`.
 *   3. `parseSSEFrames` parses complete frames, drops heartbeat comments, and
 *      keeps a trailing partial frame in the buffer.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

import { parseSSEFrames, useBoardEventStream } from '@/hooks/use-board-event-stream'
import {
  CHAT_CHANGED_EVENT,
  rememberViewerUserId,
  resetViewerUserIdForTests,
  type ChatChangedDetail,
} from '@/lib/chat/live-events'

// ── Pure frame parser ───────────────────────────────────────────────────────

describe('parseSSEFrames', () => {
  it('parses a complete board_changed frame with JSON data', () => {
    const buffer =
      'event: board_changed\ndata: {"task_id":42,"status":"done"}\n\n'
    const { events, rest } = parseSSEFrames(buffer)
    expect(rest).toBe('')
    expect(events).toHaveLength(1)
    expect(events[0].event).toBe('board_changed')
    expect(events[0].data).toEqual({ task_id: 42, status: 'done' })
  })

  it('drops heartbeat comment frames (liveness only, no refresh)', () => {
    const { events } = parseSSEFrames(': hb\n\n')
    expect(events).toHaveLength(0)
  })

  it('keeps a trailing partial frame in the buffer for the next chunk', () => {
    const buffer =
      'event: board_changed\ndata: {"task_id":1}\n\nevent: board_changed\ndata: {"tas'
    const { events, rest } = parseSSEFrames(buffer)
    expect(events).toHaveLength(1)
    expect(rest).toContain('data: {"tas')
  })
})

// ── The subscription hook ───────────────────────────────────────────────────

// A controllable fake stream: emit() pushes an SSE chunk, close() ends it.
function makeFakeStream() {
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c
    },
  })
  return {
    body,
    emit(text: string) {
      controller?.enqueue(encoder.encode(text))
    },
    close() {
      controller?.close()
    },
  }
}

// Mock the api-client so the hook uses our fetch + a deterministic base URL.
const getAuthHeadersMock = vi.fn().mockResolvedValue({ Authorization: 'Bearer test' })
const getBaseUrlMock = vi.fn().mockReturnValue('https://api.test')
vi.mock('@/lib/api-client', () => ({
  apiClient: {
    getAuthHeaders: () => getAuthHeadersMock(),
    getBaseUrl: () => getBaseUrlMock(),
  },
}))

function wrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children)
  }
}

describe('useBoardEventStream', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    getAuthHeadersMock.mockResolvedValue({ Authorization: 'Bearer test' })
    getBaseUrlMock.mockReturnValue('https://api.test')
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('opens a fetch stream to /api/v1/tasks/stream with auth headers', async () => {
    const stream = makeFakeStream()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      body: stream.body,
    } as unknown as Response)
    vi.stubGlobal('fetch', fetchMock)

    const client = new QueryClient()
    renderHook(() => useBoardEventStream(true), { wrapper: wrapper(client) })

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('https://api.test/api/v1/tasks/stream')
    expect((init as RequestInit).headers).toMatchObject({
      Authorization: 'Bearer test',
      Accept: 'text/event-stream',
    })
    stream.close()
  })

  it('invalidates the board query cache on a pushed board_changed event', async () => {
    const stream = makeFakeStream()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      body: stream.body,
    } as unknown as Response)
    vi.stubGlobal('fetch', fetchMock)

    const client = new QueryClient()
    const invalidateSpy = vi.spyOn(client, 'invalidateQueries')

    renderHook(() => useBoardEventStream(true), { wrapper: wrapper(client) })
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    // Server pushes a real board change.
    stream.emit('event: board_changed\ndata: {"task_id":7,"status":"in_progress"}\n\n')

    await waitFor(() =>
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['board'] }),
    )
    stream.close()
  })

  it('does not subscribe when disabled', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const client = new QueryClient()
    renderHook(() => useBoardEventStream(false), { wrapper: wrapper(client) })
    // Give any async connect a tick — it must never fire.
    await new Promise((r) => setTimeout(r, 20))
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

// -- PRD-205 S7: chat payloads on the same lane -----------------------------
//
// The server wraps EVERY payload as SSE event `board_changed`; chat payloads
// are distinguished by `data.event === "chat_changed"`. The hook must route
// them to the window chat bridge (per-user filtered) and NOT refetch the
// board for them.

describe('useBoardEventStream chat_changed branch (PRD-205 S7)', () => {
  let received: string[]
  const chatListener = (event: Event) =>
    received.push((event as CustomEvent<ChatChangedDetail>).detail.chatId)

  beforeEach(() => {
    vi.restoreAllMocks()
    getAuthHeadersMock.mockResolvedValue({ Authorization: 'Bearer test' })
    getBaseUrlMock.mockReturnValue('https://api.test')
    resetViewerUserIdForTests()
    received = []
    window.addEventListener(CHAT_CHANGED_EVENT, chatListener)
  })
  afterEach(() => {
    window.removeEventListener(CHAT_CHANGED_EVENT, chatListener)
    resetViewerUserIdForTests()
    vi.unstubAllGlobals()
  })

  function setupStream() {
    const stream = makeFakeStream()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      body: stream.body,
    } as unknown as Response)
    vi.stubGlobal('fetch', fetchMock)
    const client = new QueryClient()
    const invalidateSpy = vi.spyOn(client, 'invalidateQueries')
    renderHook(() => useBoardEventStream(true), { wrapper: wrapper(client) })
    return { stream, fetchMock, invalidateSpy }
  }

  it('dispatches the window chat event and skips the board refetch', async () => {
    rememberViewerUserId(7)
    const { stream, fetchMock, invalidateSpy } = setupStream()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    stream.emit(
      'event: board_changed\ndata: {"workspace_id":"ws-1","chat_id":"chat-1","user_id":7,"event":"chat_changed"}\n\n'
    )

    await waitFor(() => expect(received).toEqual(['chat-1']))
    expect(invalidateSpy).not.toHaveBeenCalled()
    stream.close()
  })

  it('drops chat events addressed to another user', async () => {
    rememberViewerUserId(7)
    const { stream, fetchMock, invalidateSpy } = setupStream()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    stream.emit(
      'event: board_changed\ndata: {"workspace_id":"ws-1","chat_id":"chat-2","user_id":8,"event":"chat_changed"}\n\n'
    )
    // A board event afterwards proves the frame above was processed.
    stream.emit('event: board_changed\ndata: {"task_id":1,"status":"done"}\n\n')

    await waitFor(() =>
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['board'] }),
    )
    expect(received).toEqual([])
    stream.close()
  })
})

// ── The poll is gone ────────────────────────────────────────────────────────

describe('useBoardTasks polling removed (F090)', () => {
  it('use-board-tasks.ts no longer sets a refetchInterval', async () => {
    // Source-level assertion: the interval poll must be gone (real-time push
    // replaces it). Guards against a regression re-introducing the 60s tick.
    const fs = await import('node:fs')
    const path = await import('node:path')
    const src = fs.readFileSync(
      path.resolve(__dirname, '../../../hooks/use-board-tasks.ts'),
      'utf8',
    )
    expect(src).not.toContain('refetchInterval')
  })
})
