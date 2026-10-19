/**
 * PRD-261 A-12d — SDK area `pages/agents`. Its hooks are the host's own (hooks/use-agent-api.ts), called
 * exactly as the core /agents page calls them (useAgents() with no options, so the roster leaves the
 * workspace's hidden system agents out), and each hands back the view every page area hands back
 * (internal/query-view.ts): `{ data, isLoading, isFetching, error, refetch }`, null data until it answers,
 * the error as its message, a refetch that returns nothing, and the data a deep, frozen copy, so a
 * theme cannot reorder or cut the roster the core page's agent pickers read from the shared cache. The area
 * imports no other core module, so it wraps no hook the page does not read and adds none. The host's hooks
 * are stood in for here, recording what they were given.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  calls: [] as { readonly hook: string; readonly args: readonly unknown[] }[],
  agents: {} as Record<string, unknown>,
  stats: {} as Record<string, unknown>,
}))

vi.mock('@/hooks/use-agent-api', () => ({
  useAgents: (...args: unknown[]) => {
    host.calls.push({ hook: 'useAgents', args })
    return host.agents
  },
  useAgentStats: (...args: unknown[]) => {
    host.calls.push({ hook: 'useAgentStats', args })
    return host.stats
  },
}))

import { useAgentStats, useAgents, type AgentStatistics, type RosterAgent } from '../pages/agents'

const AREA_SOURCE = path.resolve(__dirname, '../pages/agents.ts')
const IMPORT_SPECIFIER = /^import\b[^']*from '([^']+)'/gm
const STATUS_ERROR = 503

const ADA: RosterAgent = {
  id: 41,
  name: 'Ada',
  description: 'Writes the weekly report',
  agent_type: 'custom',
  status: 'active',
  tags: ['reports'],
  premium_icon: 'writing',
  created_at: '2026-10-01T09:00:00Z',
  updated_at: '2026-10-18T09:00:00Z',
}
const ROSTER: readonly RosterAgent[] = [ADA, { ...ADA, id: 42, name: 'Zed', status: 'inactive', tags: [] }]
const QUERY_VIEW_KEYS = ['data', 'error', 'isFetching', 'isLoading', 'refetch']

const STATS: AgentStatistics = {
  total_agents: 7,
  active_agents: 3,
  inactive_agents: 4,
  agents_by_type: { custom: 7 },
  timestamp: '2026-10-19T09:00:00Z',
}

function hostQuery(data: unknown, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    data,
    isLoading: false,
    isInitialLoading: false,
    isFetching: false,
    isError: false,
    error: null,
    status: 'success',
    refetch: vi.fn(async () => undefined),
    ...overrides,
  }
}

function callsOf(hook: string): (readonly unknown[])[] {
  return host.calls.filter((entry) => entry.hook === hook).map((entry) => entry.args)
}

afterEach(() => {
  host.calls.length = 0
})

/** The host's roster as react-query would cache it: plain objects the core page reads too. */
function cachedRoster(): RosterAgent[] {
  return ROSTER.map((agent) => ({ ...agent, tags: [...(agent.tags ?? [])] }))
}

describe('PRD-261 A-12d — useAgents', () => {
  it("reads the host's roster as the core page does: no options, so no hidden system agents", () => {
    host.agents = hostQuery(cachedRoster())
    const { result } = renderHook(() => useAgents())
    expect(callsOf('useAgents')).toEqual([[]])
    expect(result.current.data).toEqual(ROSTER)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it("hands back the page areas' frozen view, not the host's query, and the same view while nothing changes", () => {
    host.agents = hostQuery(cachedRoster())
    const { result, rerender } = renderHook(() => useAgents())
    const first = result.current
    expect(Object.isFrozen(first)).toBe(true)
    expect(first).not.toBe(host.agents)
    expect(Object.keys(first).sort()).toEqual(QUERY_VIEW_KEYS)
    rerender()
    expect(result.current).toBe(first)
  })

  it("refetches through the host's refetch and returns nothing of react-query's", () => {
    host.agents = hostQuery(cachedRoster())
    const { result } = renderHook(() => useAgents())
    expect(result.current.refetch).not.toBe(host.agents.refetch)
    expect(result.current.refetch()).toBeUndefined()
    expect(host.agents.refetch).toHaveBeenCalledTimes(1)
  })

  it("hands on a frozen copy of the roster, so a theme's in-place sort or splice cannot change the host's cache", () => {
    const cached = cachedRoster()
    host.agents = hostQuery(cached)
    const roster = renderHook(() => useAgents()).result.current.data as RosterAgent[]
    expect(roster).not.toBe(cached)
    expect(roster[0]).not.toBe(cached[0])
    expect(Object.isFrozen(roster)).toBe(true)
    expect(Object.isFrozen(roster[0].tags)).toBe(true)
    expect(() => roster.sort((a, b) => b.name.localeCompare(a.name))).toThrow(TypeError)
    expect(() => roster.splice(0, 1)).toThrow(TypeError)
    expect(() => (roster[0].tags as string[]).push('changed')).toThrow(TypeError)
    expect(cached).toEqual(ROSTER)
  })

  it('passes the first load through with null data, and a failure as its message', () => {
    host.agents = hostQuery(undefined, { isLoading: true, isInitialLoading: true, isFetching: true, status: 'loading' })
    expect(renderHook(() => useAgents()).result.current).toMatchObject({ data: null, isLoading: true, isFetching: true })

    const failure = Object.assign(new Error('Agents failed to load'), { status: STATUS_ERROR })
    host.agents = hostQuery(undefined, { isError: true, error: failure, status: 'error' })
    const failed = renderHook(() => useAgents()).result.current
    expect(failed.error).toBe('Agents failed to load')
    expect(failed.data).toBeNull()
  })
})

describe('PRD-261 A-12d — useAgentStats', () => {
  it("reads the host's agent counts with no arguments and hands a frozen copy of them back in a frozen view", () => {
    const cached = { ...STATS, agents_by_type: { ...STATS.agents_by_type } }
    host.stats = hostQuery(cached, { isFetching: true })
    const { result } = renderHook(() => useAgentStats())
    expect(callsOf('useAgentStats')).toEqual([[]])
    expect(result.current.data).toEqual(STATS)
    expect(result.current.data).not.toBe(cached)
    expect(Object.isFrozen(result.current.data?.agents_by_type)).toBe(true)
    expect(result.current.isFetching).toBe(true)
    expect(Object.keys(result.current).sort()).toEqual(QUERY_VIEW_KEYS)
    expect(Object.isFrozen(result.current)).toBe(true)
  })
})

describe('PRD-261 A-12d — the area adds no core hook', () => {
  it('imports only the agents hooks the core page reads, and the SDK views every page area shares', () => {
    const source = readFileSync(AREA_SOURCE, 'utf8')
    const specifiers = Array.from(source.matchAll(IMPORT_SPECIFIER), (match) => match[1])
    expect(specifiers).toEqual(['@/hooks/use-agent-api', '../internal/query-view'])
  })
})
