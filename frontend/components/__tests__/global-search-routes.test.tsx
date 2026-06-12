import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { renderHook, act } from '@testing-library/react'
import { apiClient } from '@/lib/api-client'
import { useGlobalSearch } from '@/hooks/use-global-search'

// PRD-154 S10 — global search hit /api/v1/activity/feed and /api/v1/agents,
// neither of which exist (the real routers are /api/activity and /api/agents),
// and every failure was swallowed with `catch { return [] }`, so the box went
// silently blank. Routes are corrected and failures now surface via `error`.

const HOOK = path.resolve(__dirname, '..', '..', 'hooks', 'use-global-search.ts')

describe('S10 global search — real routes', () => {
  it('targets the real, non-v1 routers', () => {
    const src = readFileSync(HOOK, 'utf8')
    expect(src).toContain('/api/activity/feed')
    expect(src).toContain('/api/agents?')
    expect(src).not.toContain('/api/v1/activity/feed')
    expect(src).not.toContain('/api/v1/agents')
  })

  it('no longer swallows errors with catch { return [] }', () => {
    const src = readFileSync(HOOK, 'utf8')
    expect(src).toContain('allSettled')
    expect(src).not.toMatch(/catch\s*\{\s*return\s*\[\]\s*\}/)
  })
})

describe('S10 global search — behaviour', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('returns live results for a seeded agent', async () => {
    vi.spyOn(apiClient, 'request').mockImplementation((url: string) => {
      if (url.startsWith('/api/agents')) {
        return Promise.resolve([{ id: 7, name: 'Seeded Agent', role: 'tester' }]) as any
      }
      return Promise.resolve([]) as any
    })

    const { result } = renderHook(() => useGlobalSearch())
    act(() => result.current.setQuery('seeded'))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300)
    })

    expect(result.current.agents).toHaveLength(1)
    expect(result.current.agents[0].label).toBe('Seeded Agent')
    expect(result.current.error).toBeNull()
  })

  it('surfaces an error when a source fails instead of blanking silently', async () => {
    vi.spyOn(apiClient, 'request').mockImplementation((url: string) => {
      if (url.startsWith('/api/agents')) return Promise.reject(new Error('boom')) as any
      return Promise.resolve([]) as any
    })

    const { result } = renderHook(() => useGlobalSearch())
    act(() => result.current.setQuery('seeded'))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300)
    })

    expect(result.current.error).toBeTruthy()
  })
})
