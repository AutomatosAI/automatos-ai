/**
 * PRD-268 US-W01b — where each board-stream frame goes (lib/live/route-frame.ts): board_changed invalidates the board
 * and is told as automatos:board-changed, chat_changed as automatos:chat-changed, suggestion_changed as
 * automatos:suggestion-changed (which the open suggestions list asks again on), each with the frame's data as its
 * detail; any other frame does nothing. The hook that reads the stream routes every frame here, and is at most 50
 * code lines (K4).
 */
import { readFileSync } from 'fs'
import path from 'path'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    getAuthHeaders: async () => ({ Authorization: 'Bearer test' }),
    getBaseUrl: () => 'https://api.test',
  },
}))

import { useBoardEventStream } from '@/hooks/use-board-event-stream'
import { SUGGESTION_REFRESH_EVENTS } from '@/hooks/use-suggestions'
import { FRAME_EVENTS, routeFrame, type FrameSinks } from '@/lib/live/route-frame'

const SUGGESTION_DATA = { workspace_id: 'ws-1', suggestion_id: 's-1', user_id: 3, status: 'started', event: 'suggestion_changed' }

function sinks() {
  const dispatched: Event[] = []
  const invalidateBoard = vi.fn()
  const routed: FrameSinks = { invalidateBoard, dispatch: (event) => void dispatched.push(event) }
  return { routed, dispatched, invalidateBoard }
}

const detailOf = (event: Event) => (event as CustomEvent).detail

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('routeFrame', () => {
  it('routes every frame name to its own window event, with the frame data as its detail', () => {
    const cases: Array<[string, string, unknown]> = [
      ['board_changed', 'automatos:board-changed', { task_id: 7 }],
      ['chat_changed', 'automatos:chat-changed', { chat_id: 'c-9', user_id: 3 }],
      ['suggestion_changed', 'automatos:suggestion-changed', SUGGESTION_DATA],
    ]
    for (const [name, eventName, data] of cases) {
      const { routed, dispatched } = sinks()
      routeFrame({ event: name, data }, routed)
      expect(dispatched.map((event) => event.type)).toEqual([eventName])
      expect(detailOf(dispatched[0])).toEqual(data)
    }
    expect(Object.keys(FRAME_EVENTS).sort()).toEqual(['board_changed', 'chat_changed', 'suggestion_changed'])
  })

  it('board_changed still invalidates the board, before its event; no other frame does', () => {
    const board = sinks()
    const order: string[] = []
    routeFrame(
      { event: 'board_changed', data: null },
      { invalidateBoard: () => order.push('invalidate'), dispatch: (event) => void order.push(event.type) },
    )
    expect(order).toEqual(['invalidate', 'automatos:board-changed'])
    routeFrame({ event: 'chat_changed', data: null }, board.routed)
    routeFrame({ event: 'suggestion_changed', data: SUGGESTION_DATA }, board.routed)
    expect(board.invalidateBoard).not.toHaveBeenCalled()
  })

  it('ignores any other frame, an inherited name included', () => {
    const { routed, dispatched, invalidateBoard } = sinks()
    for (const name of ['message', 'task_changed', 'toString', '__proto__', 'constructor']) {
      routeFrame({ event: name, data: {} }, routed)
    }
    expect(dispatched).toEqual([])
    expect(invalidateBoard).not.toHaveBeenCalled()
  })

  it('tells the suggestions list the event it asks again on', () => {
    expect(SUGGESTION_REFRESH_EVENTS).toContain(FRAME_EVENTS.suggestion_changed)
  })
})

describe('useBoardEventStream routes the stream through routeFrame', () => {
  function fakeStream() {
    let controller: ReadableStreamDefaultController<Uint8Array> | null = null
    const body = new ReadableStream<Uint8Array>({ start: (c) => void (controller = c) })
    return { body, emit: (text: string) => controller?.enqueue(new TextEncoder().encode(text)), close: () => controller?.close() }
  }

  it('dispatches automatos:suggestion-changed for a pushed suggestion_changed frame, without touching the board', async () => {
    const stream = fakeStream()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: stream.body } as unknown as Response))
    const client = new QueryClient()
    const invalidate = vi.spyOn(client, 'invalidateQueries')
    const heard = vi.fn()
    window.addEventListener('automatos:suggestion-changed', heard)
    const wrapper = ({ children }: { children: React.ReactNode }) => React.createElement(QueryClientProvider, { client }, children)
    const { unmount } = renderHook(() => useBoardEventStream(true), { wrapper })

    await waitFor(() => expect(fetch).toHaveBeenCalled())
    stream.emit(`event: suggestion_changed\ndata: ${JSON.stringify(SUGGESTION_DATA)}\n\n`)
    await waitFor(() => expect(heard).toHaveBeenCalledTimes(1))
    expect((heard.mock.calls[0][0] as CustomEvent).detail).toEqual(SUGGESTION_DATA)
    expect(invalidate).not.toHaveBeenCalled()

    window.removeEventListener('automatos:suggestion-changed', heard)
    unmount()
    stream.close()
  })

  it('keeps the hook at most 50 code lines (K4)', () => {
    const source = readFileSync(path.resolve(__dirname, '../../../hooks/use-board-event-stream.ts'), 'utf8')
    const lines = source.split('\n')
    const start = lines.findIndex((line) => line.startsWith('export function useBoardEventStream('))
    const end = lines.findIndex((line, index) => index > start && line === '}')
    const code = lines.slice(start, end + 1).filter((line) => line.trim() !== '' && !line.trim().startsWith('//'))
    expect(start).toBeGreaterThan(-1)
    expect(code.length).toBeLessThanOrEqual(50)
    expect(source).toContain("from '@/lib/live/route-frame'")
  })
})
