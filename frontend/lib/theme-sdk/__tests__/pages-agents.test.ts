/**
 * PRD-261 A-12d — SDK area `pages/agents`. Its hooks are the host's own (hooks/use-agent-api.ts), called
 * exactly as the core /agents page calls them (useAgents() with no options, so the roster leaves the
 * workspace's hidden system agents out), and each hands back a frozen view of the host's query: the data
 * as the backend answered, the loading and error states, and the host's own refetch. The area imports no
 * other core module, so it wraps no hook the page does not read and adds none. The host's hooks are stood
 * in for here, recording what they were given.
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

const ROSTER: readonly RosterAgent[] = [
  {
    id: 41,
    name: 'Ada',
    description: 'Writes the weekly report',
    agent_type: 'custom',
    status: 'active',
    premium_icon: 'writing',
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-10-18T09:00:00Z',
  },
]

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

describe('PRD-261 A-12d — useAgents', () => {
  it("reads the host's roster as the core page does: no options, so no hidden system agents", () => {
    host.agents = hostQuery(ROSTER)
    const { result } = renderHook(() => useAgents())
    expect(callsOf('useAgents')).toEqual([[]])
    expect(result.current.data).toBe(ROSTER)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.isError).toBe(false)
  })

  it("hands back a frozen view, not the host's query, with the host's own refetch", () => {
    host.agents = hostQuery(ROSTER)
    const { result } = renderHook(() => useAgents())
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(result.current).not.toBe(host.agents)
    expect(result.current.refetch).toBe(host.agents.refetch)
    expect(Object.keys(result.current).sort()).toEqual(['data', 'error', 'isError', 'isFetching', 'isLoading', 'refetch'])
  })

  it('passes the loading and the failed states through', () => {
    host.agents = hostQuery(undefined, { isLoading: true, isFetching: true, status: 'loading' })
    expect(renderHook(() => useAgents()).result.current).toMatchObject({ data: undefined, isLoading: true, isFetching: true })

    const failure = Object.assign(new Error('Agents failed to load'), { status: STATUS_ERROR })
    host.agents = hostQuery(undefined, { isError: true, error: failure, status: 'error' })
    const failed = renderHook(() => useAgents()).result.current
    expect(failed.isError).toBe(true)
    expect(failed.error).toBe(failure)
  })
})

describe('PRD-261 A-12d — useAgentStats', () => {
  it("reads the host's agent counts with no arguments and hands them back in a frozen view", () => {
    host.stats = hostQuery(STATS, { isFetching: true })
    const { result } = renderHook(() => useAgentStats())
    expect(callsOf('useAgentStats')).toEqual([[]])
    expect(result.current.data).toBe(STATS)
    expect(result.current.isFetching).toBe(true)
    expect(result.current.refetch).toBe(host.stats.refetch)
    expect(Object.isFrozen(result.current)).toBe(true)
  })
})

describe('PRD-261 A-12d — the area adds no core hook', () => {
  it('imports only the agents hooks the core page reads', () => {
    const source = readFileSync(AREA_SOURCE, 'utf8')
    const specifiers = Array.from(source.matchAll(IMPORT_SPECIFIER), (match) => match[1])
    expect(specifiers).toEqual(['@/hooks/use-agent-api'])
  })
})
