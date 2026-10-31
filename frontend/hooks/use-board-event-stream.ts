/**
 * PRD-180 S1 (F090) — real-time board subscription.
 *
 * Replaces the Command Centre's board polling: instead of refetching every 60s,
 * this subscribes to the backend's LISTEN/NOTIFY-backed SSE
 * (`GET /api/v1/tasks/stream`) and invalidates the board react-query cache the
 * moment the server pushes a `board_changed` event — so a status change shows
 * sub-second, not on a poll tick.
 *
 * Native `EventSource` can't set the `Authorization` header the hybrid auth
 * requires, so this reads the stream via `fetch` + a `ReadableStream` reader
 * (which CAN carry `apiClient.getAuthHeaders()`), parses SSE frames, and drives
 * cache invalidation. It reconnects with backoff if the stream drops, and tears
 * the connection down on unmount.
 *
 * PRD-268 US-W01b: where each frame goes (board_changed, chat_changed and
 * suggestion_changed) is lib/live/route-frame.ts; opening the stream and its
 * read loop are the module-level helpers below, so the hook stays short.
 */

import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { boardQueryKeys } from '@/hooks/use-board-tasks'
import { routeFrame, type FrameSinks, type StreamFrame } from '@/lib/live/route-frame'

const STREAM_ENDPOINT = '/api/v1/tasks/stream'
const RECONNECT_MIN_MS = 1000
const RECONNECT_MAX_MS = 30000

/**
 * Parse a chunk of accumulated SSE text into complete frames, returning the
 * frames plus any trailing partial frame still in the buffer. Pure — no I/O.
 */
export function parseSSEFrames(buffer: string): { events: StreamFrame[]; rest: string } {
  const events: StreamFrame[] = []
  // Frames are separated by a blank line. Keep the trailing (possibly partial)
  // segment in the buffer for the next chunk.
  const segments = buffer.split('\n\n')
  const rest = segments.pop() ?? ''

  for (const segment of segments) {
    const trimmed = segment.trim()
    if (!trimmed || trimmed.startsWith(':')) {
      // Blank or a heartbeat comment (`: hb`) — liveness only, no refresh.
      continue
    }
    let eventName = 'message'
    const dataLines: string[] = []
    for (const line of trimmed.split('\n')) {
      if (line.startsWith('event:')) {
        eventName = line.slice('event:'.length).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice('data:'.length).trim())
      }
    }
    let data: unknown = null
    if (dataLines.length > 0) {
      try {
        data = JSON.parse(dataLines.join('\n'))
      } catch {
        data = dataLines.join('\n')
      }
    }
    events.push({ event: eventName, data })
  }
  return { events, rest }
}

/** Open the stream with the caller's auth headers; throws when it isn't answered with a body. */
async function openStream(signal: AbortSignal): Promise<ReadableStream<Uint8Array>> {
  const headers = await apiClient.getAuthHeaders()
  const url = `${apiClient.getBaseUrl()}${STREAM_ENDPOINT}`
  const response = await fetch(url, {
    method: 'GET',
    headers: { ...headers, Accept: 'text/event-stream' },
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`board stream failed: ${response.status}`)
  }
  return response.body
}

/** Read the stream's frames, handing each to `onFrame`, until it ends or `cancelled()` says stop. */
async function readFrames(
  body: ReadableStream<Uint8Array>,
  onFrame: (frame: StreamFrame) => void,
  cancelled: () => boolean,
): Promise<void> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (!cancelled()) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const { events, rest } = parseSSEFrames(buffer)
    buffer = rest
    events.forEach(onFrame)
  }
}

/** The wait before reconnect attempt `attempt`: capped exponential backoff. */
function backoffMs(attempt: number): number {
  return Math.min(RECONNECT_MIN_MS * 2 ** attempt, RECONNECT_MAX_MS)
}

/**
 * Subscribe the current view to real-time board events. On each pushed
 * `board_changed`, the board react-query cache is invalidated so the board
 * refetches once, immediately — no interval polling. Every named frame is
 * also told to the page as a window event (lib/live/route-frame.ts).
 *
 * @param enabled — gate the subscription (e.g. only while the board is mounted).
 */
export function useBoardEventStream(enabled: boolean = true): void {
  const queryClient = useQueryClient()
  // Keep a stable ref so the effect doesn't re-subscribe when the client ref
  // identity changes across renders.
  const queryClientRef = useRef(queryClient)
  queryClientRef.current = queryClient

  useEffect(() => {
    if (!enabled || typeof window === 'undefined') return

    let cancelled = false
    let controller: AbortController | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let attempt = 0
    const sinks: FrameSinks = {
      invalidateBoard: () => void queryClientRef.current.invalidateQueries({ queryKey: boardQueryKeys.all }),
      dispatch: (event) => window.dispatchEvent(event),
    }

    const connect = async () => {
      if (cancelled) return
      controller = new AbortController()
      try {
        const body = await openStream(controller.signal)
        // Connected — reset backoff.
        attempt = 0
        await readFrames(body, (frame) => routeFrame(frame, sinks), () => cancelled)
      } catch (err) {
        // AbortError is our own teardown — not a failure to log or retry.
        if (cancelled || (err instanceof DOMException && err.name === 'AbortError')) {
          return
        }
        // Real drop — schedule a reconnect with capped exponential backoff.
      } finally {
        controller = null
      }

      if (!cancelled) {
        reconnectTimer = setTimeout(connect, backoffMs(attempt))
        attempt += 1
      }
    }

    void connect()

    return () => {
      cancelled = true
      if (controller) controller.abort()
      if (reconnectTimer) clearTimeout(reconnectTimer)
    }
  }, [enabled])
}
