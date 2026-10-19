/**
 * PRD-261 A-12h — the SDK's pages/command-center area wraps the core hooks the /command-center page
 * reads (command-center-shell.tsx): the same endpoints through the host's backend client, handed to
 * a theme as small frozen views (never react-query's result), with a theme's period, range and feed
 * filters checked before they reach a URL, and the board's event stream opened for the theme's page.
 * The backend client and fetch are mocked; react-query and the core hooks are real.
 */
import { readFileSync } from 'fs'
import path from 'path'
import type { ReactNode } from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
const listApprovalGrants = vi.hoisted(() => vi.fn())
const listWatches = vi.hoisted(() => vi.fn())
const getAuthHeaders = vi.hoisted(() => vi.fn())
const getBaseUrl = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => {
  const client = { request, listApprovalGrants, listWatches, getAuthHeaders, getBaseUrl }
  return { apiClient: client, default: client }
})

import { onHostEvent } from '../events'
import * as CommandCenter from '../pages/command-center'
import {
  type ActivityFeedFilters,
  type ActivityPeriod,
  type ScheduleRange,
  useActivityFeed,
  useActivitySchedule,
  useActivityStats,
  useBoardEventStream,
  useNeedsYou,
  useQuestions,
  useWatches,
} from '../pages/command-center'

const SDK_DIR = path.resolve(__dirname, '..')
const SURFACE_FILE = path.join(SDK_DIR, 'surface/pages-command-center.json')
const DECLARATIONS = path.resolve(SDK_DIR, '../../packages/theme-sdk/areas/pages-command-center.d.ts')
const DECLARED_VALUE = /^export declare (?:const|function) ([A-Za-z_$][\w$]*)/gm
const BASE_URL = 'https://api.test'
const STREAM_URL = `${BASE_URL}/api/v1/tasks/stream`
const QUERY_VIEW_KEYS = ['data', 'error', 'isFetching', 'isLoading', 'refetch']
const STATS = { working_now: 2, channels_live: 1, completed_today: 5, needs_attention: 3, agents_active: 4, tasks_in_queue: 6, period: '7d' }
const FEED = { items: [{ id: 'chat-1', type: 'chat', name: 'Weekly notes', status: 'completed' }], total: 1, limit: 5, offset: 0 }
const SCHEDULE = { scheduled: [{ id: 'routine-1', name: 'Morning brief', type: 'routine', next_run_at: null, frequency: 'daily' }] }
const NEEDS_YOU = { total: 1, counts: { review: 1, question: 0, approval: 0, stuck: 0, failed: 0 }, rows: { review: [], question: [], approval: [], stuck: [], failed: [] } }
const QUESTIONS = { grants: [{ id: 7, kind: 'question', status: 'pending', question_md: 'Which tone?' }] }
const WATCHES = { watches: [{ id: 'w-1', title: 'Launch post', status: 'active' }], total: 1 }
const SILENT = { log: () => undefined, warn: () => undefined, error: () => undefined }

function wrapper() {
  const client = new QueryClient({ logger: SILENT, defaultOptions: { queries: { retry: false } } })
  return function Providers({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
}

/** A stream the test writes SSE frames into. */
function fakeStream() {
  const encoder = new TextEncoder()
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c
    },
  })
  return { body, emit: (text: string) => controller?.enqueue(encoder.encode(text)), close: () => controller?.close() }
}

function sortedNames(names: Iterable<string>): string[] {
  return Array.from(new Set(names)).sort()
}

beforeEach(() => {
  request.mockReset()
  listApprovalGrants.mockReset()
  listWatches.mockReset()
  getAuthHeaders.mockReset().mockResolvedValue({ Authorization: 'Bearer test' })
  getBaseUrl.mockReset().mockReturnValue(BASE_URL)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('PRD-261 A-12h — the area is the one A-10c planned', () => {
  const planned = JSON.parse(readFileSync(SURFACE_FILE, 'utf8')) as { status: string; names: string[] }

  it('exports exactly the surface file names, built', () => {
    expect(planned.status).toBe('built')
    expect(sortedNames(Object.keys(CommandCenter))).toEqual(sortedNames(planned.names))
  })

  it('declares exactly those names in its .d.ts, as functions', () => {
    const declared = Array.from(readFileSync(DECLARATIONS, 'utf8').matchAll(DECLARED_VALUE), (match) => match[1])
    expect(sortedNames(declared)).toEqual(sortedNames(planned.names))
    for (const name of planned.names) expect(typeof (CommandCenter as unknown as Record<string, unknown>)[name], name).toBe('function')
  })
})

describe('PRD-261 A-12h — the stats and the schedule', () => {
  it('reads the stats for a known period and hands back a frozen view', async () => {
    request.mockResolvedValue(STATS)
    const { result } = renderHook(() => useActivityStats('7d'), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(STATS))
    expect(request).toHaveBeenCalledWith('/api/activity/stats?period=7d')
    expect(Object.keys(result.current).sort()).toEqual(QUERY_VIEW_KEYS)
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(result.current).toMatchObject({ isLoading: false, error: null })
  })

  it('asks for the 1d stats when the period is unset or unknown, never for what the theme wrote', async () => {
    request.mockResolvedValue(STATS)
    renderHook(() => useActivityStats(), { wrapper: wrapper() })
    renderHook(() => useActivityStats('1d&workspace=other' as unknown as ActivityPeriod), { wrapper: wrapper() })

    await waitFor(() => expect(request).toHaveBeenCalledTimes(2))
    expect(request.mock.calls.map((call) => call[0])).toEqual(['/api/activity/stats?period=1d', '/api/activity/stats?period=1d'])
  })

  it('reads the schedule for a known range, and for 7d otherwise', async () => {
    request.mockResolvedValue(SCHEDULE)
    const { result } = renderHook(() => useActivitySchedule('14d'), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual(SCHEDULE))
    expect(request).toHaveBeenCalledWith('/api/activity/schedule?range=14d')

    renderHook(() => useActivitySchedule('../x' as unknown as ScheduleRange), { wrapper: wrapper() })
    await waitFor(() => expect(request).toHaveBeenCalledWith('/api/activity/schedule?range=7d'))
  })

  it("hands back a failed read's message, not the error", async () => {
    request.mockRejectedValue(new Error('Failed to fetch activity stats'))
    const { result } = renderHook(() => useActivityStats('1d'), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.error).toBe('Failed to fetch activity stats'))
    expect(result.current.data).toBeNull()
  })
})

describe('PRD-261 A-12h — the feed', () => {
  it('puts known filters in the query string', async () => {
    request.mockResolvedValue(FEED)
    const filters: ActivityFeedFilters = { types: ['mission', 'chat'], status: 'attention', period: '30d', limit: 5, offset: 10 }
    const { result } = renderHook(() => useActivityFeed(filters), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(FEED))
    expect(request).toHaveBeenCalledWith('/api/activity/feed?type=chat%2Cmission&status=attention&period=30d&limit=5&offset=10')
  })

  it('drops unknown types and statuses, an unknown period, and paging outside the API bounds', async () => {
    request.mockResolvedValue(FEED)
    const filters = { types: ['evil'], status: 'x&y=1', period: 'forever', limit: 500, offset: -1 } as unknown as ActivityFeedFilters
    const { result } = renderHook(() => useActivityFeed(filters), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(FEED))
    expect(request).toHaveBeenCalledWith('/api/activity/feed')
  })
})

describe('PRD-261 A-12h — questions, Needs you and watches', () => {
  it('reads the open questions through the core hook', async () => {
    listApprovalGrants.mockResolvedValue(QUESTIONS)
    const { result } = renderHook(() => useQuestions(), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(QUESTIONS))
    expect(listApprovalGrants).toHaveBeenCalledWith('pending', 'question')
  })

  it('reads Needs you, and reads it again on a board change', async () => {
    request.mockResolvedValue(NEEDS_YOU)
    const { result } = renderHook(() => useNeedsYou(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual(NEEDS_YOU))
    expect(request).toHaveBeenCalledWith('/api/activity/needs-you')

    act(() => {
      window.dispatchEvent(new CustomEvent('automatos:board-changed', { detail: {} }))
    })
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2))
  })

  it('reads the live watches, and every watch only when told so with true', async () => {
    listWatches.mockResolvedValue(WATCHES)
    const { result } = renderHook(() => useWatches(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual(WATCHES))
    expect(listWatches).toHaveBeenLastCalledWith(false)

    renderHook(() => useWatches('yes' as unknown as boolean), { wrapper: wrapper() })
    renderHook(() => useWatches(true), { wrapper: wrapper() })
    await waitFor(() => expect(listWatches).toHaveBeenCalledWith(true))
    expect(listWatches.mock.calls.map((call) => call[0])).toEqual([false, false, true])
  })
})

describe('PRD-261 A-12h — the board event stream', () => {
  it("opens the board's stream and fans a pushed change out as the SDK's board-changed event", async () => {
    const stream = fakeStream()
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, body: stream.body } as unknown as Response)
    vi.stubGlobal('fetch', fetchMock)
    const seen: unknown[] = []
    const stop = onHostEvent('board-changed', (detail) => seen.push(detail))

    const { unmount } = renderHook(() => useBoardEventStream(), { wrapper: wrapper() })
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(fetchMock.mock.calls[0][0]).toBe(STREAM_URL)

    stream.emit('event: board_changed\ndata: {"task_id":42}\n\n')
    await waitFor(() => expect(seen).toEqual([{ task_id: 42 }]))
    stop()
    unmount()
    stream.close()
  })

  it('opens nothing when disabled', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    renderHook(() => useBoardEventStream(false), { wrapper: wrapper() })

    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
