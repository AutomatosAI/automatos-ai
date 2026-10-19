/**
 * PRD-261 A-12e — the SDK's pages/assignments area wraps the core hooks the /assignments Missions
 * tab and /missions/[id] read: the same endpoints through the host's backend client, handed to a
 * theme as small frozen views (never react-query's result), with a theme's inputs checked before
 * they reach a URL. The backend client is mocked; react-query and the core hooks are real.
 */
import type { ReactNode } from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
const getRecommendedAssignments = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => {
  const client = { request, getRecommendedAssignments }
  return { apiClient: client, default: client }
})

import {
  type MissionFieldOptions,
  type MissionFilters,
  useApproveMission,
  useFieldQuery,
  useMission,
  useMissionField,
  useMissions,
  usePauseMission,
  useRecommendedAssignments,
} from '../pages/assignments'
import { answer, cachedMutationData, cachedQueryData, inspectableClient } from './support/query-cache'

const MISSION_ID = '00000000-0000-4000-8000-0000000a12e1'
const LIST = { missions: [{ id: MISSION_ID, goal: 'Draft the notes', state: 'running' }], total: 1, limit: 5, offset: 0 }
const PAUSED = { id: MISSION_ID, goal: 'Draft the notes', state: 'paused' }
const QUERY_VIEW_KEYS = ['data', 'error', 'isFetching', 'isLoading', 'refetch']
const ACTION_VIEW_KEYS = ['error', 'isRunning', 'reset', 'result', 'run', 'runAsync']
const NOT_A_MISSION_ID = 'Not a mission id.'
const SILENT = { log: () => undefined, warn: () => undefined, error: () => undefined }

function wrapper() {
  const client = new QueryClient({ logger: SILENT, defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return function Providers({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
}

beforeEach(() => {
  request.mockReset()
  getRecommendedAssignments.mockReset()
})

describe('PRD-261 A-12e — queries', () => {
  it('reads the Missions list through the backend client and hands back a frozen view', async () => {
    request.mockResolvedValue(LIST)
    const { result } = renderHook(() => useMissions({ state: 'running', limit: 5 }), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(LIST))
    expect(request).toHaveBeenCalledWith('/api/missions?state=running&limit=5')
    expect(Object.keys(result.current).sort()).toEqual(QUERY_VIEW_KEYS)
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('keeps only a known state and whole-number limits in the list query string', async () => {
    request.mockResolvedValue(LIST)
    const filters = { state: 'running&admin=1', limit: 2.5, offset: -1 } as unknown as MissionFilters
    const { result } = renderHook(() => useMissions(filters), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(LIST))
    expect(request).toHaveBeenCalledWith('/api/missions')
  })

  it('reads one Mission by its id', async () => {
    request.mockResolvedValue(PAUSED)
    const { result } = renderHook(() => useMission(MISSION_ID), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(PAUSED))
    expect(request).toHaveBeenCalledWith(`/api/missions/${MISSION_ID}`)
  })

  it('asks for nothing without an id, and is not loading', () => {
    const { result } = renderHook(() => useMission(null), { wrapper: wrapper() })

    expect(request).not.toHaveBeenCalled()
    expect(result.current).toMatchObject({ data: null, isLoading: false, error: null })
  })

  it('asks for nothing with a malformed id, and says why', () => {
    const { result } = renderHook(() => useMission('../workspaces'), { wrapper: wrapper() })

    expect(request).not.toHaveBeenCalled()
    expect(result.current).toMatchObject({ data: null, isLoading: false, error: NOT_A_MISSION_ID })
  })

  it("hands back a failed read's message, not the error", async () => {
    request.mockRejectedValue(new Error('Mission not found'))
    const { result } = renderHook(() => useMission(MISSION_ID), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.error).toBe('Mission not found'))
    expect(result.current.data).toBeNull()
  })

  it("reads a Mission's field in a known scope only", async () => {
    request.mockResolvedValue({ field_id: null, backend: null, patterns: [], stability: {}, metrics: null })
    const options = { scope: 'mission&admin=1' } as unknown as MissionFieldOptions
    const { result } = renderHook(() => useMissionField(MISSION_ID, options), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).not.toBeNull())
    expect(request).toHaveBeenCalledWith(`/api/missions/${MISSION_ID}/field?scope=mission`)

    renderHook(() => useMissionField(MISSION_ID, { scope: 'workspace' }), { wrapper: wrapper() })
    await waitFor(() => expect(request).toHaveBeenCalledWith(`/api/missions/${MISSION_ID}/field?scope=workspace`))
  })

  it('asks for no field when told not to', () => {
    renderHook(() => useMissionField(MISSION_ID, { enabled: false }), { wrapper: wrapper() })
    expect(request).not.toHaveBeenCalled()
  })

  it('reads the recommendations, with the core default for a limit that is not a positive whole number', async () => {
    const answer = { items: [], workspace_count: 0, marketplace_count: 0 }
    getRecommendedAssignments.mockResolvedValue(answer)
    const { result } = renderHook(() => useRecommendedAssignments(3), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual(answer))
    expect(getRecommendedAssignments).toHaveBeenCalledWith(3)

    renderHook(() => useRecommendedAssignments(-1), { wrapper: wrapper() })
    await waitFor(() => expect(getRecommendedAssignments).toHaveBeenCalledWith(8))
  })
})

describe('PRD-261 A-12e — actions', () => {
  it('pauses a Mission through the backend client and hands back the answer', async () => {
    request.mockResolvedValue(PAUSED)
    const { result } = renderHook(() => usePauseMission(), { wrapper: wrapper() })
    expect(Object.keys(result.current).sort()).toEqual(ACTION_VIEW_KEYS)
    expect(Object.isFrozen(result.current)).toBe(true)

    act(() => result.current.run(MISSION_ID))

    await waitFor(() => expect(result.current.result).toEqual(PAUSED))
    expect(request).toHaveBeenCalledWith(`/api/missions/${MISSION_ID}/pause`, { method: 'POST' })
    expect(result.current.isRunning).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('throws a TypeError for a malformed id and sends nothing', async () => {
    const { result } = renderHook(() => usePauseMission(), { wrapper: wrapper() })

    expect(() => result.current.run('../workspaces')).toThrow(TypeError)
    await expect(result.current.runAsync('a/b')).rejects.toThrow(NOT_A_MISSION_ID)
    expect(request).not.toHaveBeenCalled()
  })

  it("hands back a failed action's message", async () => {
    request.mockRejectedValue(new Error('Mission is not running'))
    const { result } = renderHook(() => usePauseMission(), { wrapper: wrapper() })

    act(() => result.current.run(MISSION_ID))

    await waitFor(() => expect(result.current.error).toBe('Mission is not running'))
    expect(result.current.result).toBeNull()
  })

  it("approves a plan with the owner's note", async () => {
    request.mockResolvedValue(PAUSED)
    const { result } = renderHook(() => useApproveMission(), { wrapper: wrapper() })

    await act(async () => {
      await result.current.runAsync({ id: MISSION_ID, body: { note: 'Keep it short' } })
    })

    expect(request).toHaveBeenCalledWith(`/api/missions/${MISSION_ID}/approve`, {
      method: 'POST',
      body: { note: 'Keep it short' },
    })
    expect(() => result.current.run({ id: '' })).toThrow(TypeError)
  })

  it("queries a Mission's field, and refuses every run without a well-formed id", async () => {
    request.mockResolvedValue({ field_id: 'f', query: 'launch', results: [] })
    const { result } = renderHook(() => useFieldQuery(MISSION_ID), { wrapper: wrapper() })

    await act(async () => {
      await result.current.runAsync({ query: 'launch', topK: 3 })
    })
    expect(request).toHaveBeenCalledWith(`/api/missions/${MISSION_ID}/field/query`, {
      method: 'POST',
      body: JSON.stringify({ query: 'launch', top_k: 3 }),
    })

    request.mockClear()
    const refused = renderHook(() => useFieldQuery(null), { wrapper: wrapper() }).result
    await expect(refused.current.runAsync({ query: 'launch' })).rejects.toThrow(TypeError)
    expect(request).not.toHaveBeenCalled()
  })
})

describe('PRD-261 W-A4P — a theme never holds the objects the core page caches', () => {
  it("hands on a deep, frozen copy of the Missions list, so a theme's in-place change cannot reach the cache", async () => {
    request.mockResolvedValue(answer(LIST))
    const { client, wrapper: Providers } = inspectableClient()
    const { result } = renderHook(() => useMissions(), { wrapper: Providers })

    await waitFor(() => expect(result.current.data).not.toBeNull())
    const cached = cachedQueryData(client)
    expect(result.current.data).toEqual(cached)
    expect(result.current.data).not.toBe(cached)
    const missions = result.current.data?.missions as unknown as { state: string }[]
    expect(() => missions.splice(0, 1)).toThrow(TypeError)
    expect(() => {
      missions[0].state = 'cancelled'
    }).toThrow(TypeError)
    expect(cached).toEqual(LIST)
  })

  it("hands on a frozen copy of an action's answer, from run and from runAsync", async () => {
    request.mockResolvedValue(answer(PAUSED))
    const { client, wrapper: Providers } = inspectableClient()
    const { result } = renderHook(() => usePauseMission(), { wrapper: Providers })

    let resolved: unknown = null
    await act(async () => {
      resolved = await result.current.runAsync(MISSION_ID)
    })
    await waitFor(() => expect(result.current.result).toEqual(PAUSED))
    const cached = cachedMutationData(client)
    expect(result.current.result).not.toBe(cached)
    expect(resolved).not.toBe(cached)
    expect(Object.isFrozen(result.current.result)).toBe(true)
    expect(Object.isFrozen(resolved)).toBe(true)
  })
})
