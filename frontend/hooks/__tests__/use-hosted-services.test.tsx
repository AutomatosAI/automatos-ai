/**
 * PRD-277 US-704 (D-12) — useHostedService.
 *
 * - local and hosted builds answer from the build: no query mounts (no QueryClientProvider needed) and nothing
 *   is asked;
 * - an enterprise build reads GET /api/hosted-services once for every reader, and answers unknown (undefined)
 *   until it lands and when it can't be read.
 */
import React from 'react'
import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))

type HookModule = typeof import('../use-hosted-services')

type Wrapper = ({ children }: { children: React.ReactNode }) => React.ReactElement

/** The hook for `edition`'s build, and a provider from the same module registry the hook now imports from. */
async function load(edition: 'local' | 'saas' | 'enterprise'): Promise<HookModule & { wrapper: Wrapper }> {
  vi.resetModules()
  vi.doMock('@/lib/edition-name', () => ({ isEnterprise: edition === 'enterprise', resolveEdition: () => edition }))
  const hook = await import('../use-hosted-services')
  const { QueryClient, QueryClientProvider } = await import('@tanstack/react-query')
  const wrapper: Wrapper = ({ children }) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
  return { ...hook, wrapper }
}

beforeEach(() => {
  api.get.mockReset()
})

afterEach(() => {
  vi.doUnmock('@/lib/edition-name')
})

describe('the local and hosted builds', () => {
  it('local answers on for every service, asking nothing and needing no provider', async () => {
    const { useHostedService } = await load('local')
    for (const service of ['themeRevocations', 'cloudSync', 'cloudItems'] as const) {
      expect(renderHook(() => useHostedService(service)).result.current).toBe(true)
    }
    expect(api.get).not.toHaveBeenCalled()
  })

  it('hosted answers off for every service, asking nothing', async () => {
    const { useHostedService } = await load('saas')
    for (const service of ['themeRevocations', 'cloudSync', 'cloudItems'] as const) {
      expect(renderHook(() => useHostedService(service)).result.current).toBe(false)
    }
    expect(api.get).not.toHaveBeenCalled()
  })
})

describe('an enterprise build', () => {
  it('reads the settings once for every reader, unknown until they land', async () => {
    api.get.mockResolvedValue({ themeRevocations: true, cloudSync: false, cloudItems: false })
    const { useHostedService, wrapper } = await load('enterprise')

    const { result } = renderHook(
      () => [useHostedService('themeRevocations'), useHostedService('cloudSync')] as const,
      { wrapper },
    )
    expect(result.current).toEqual([undefined, undefined])
    await waitFor(() => expect(result.current).toEqual([true, false]))
    expect(api.get).toHaveBeenCalledTimes(1)
    expect(api.get).toHaveBeenCalledWith('/api/hosted-services')
  })

  it('stays unknown when the settings cannot be read', async () => {
    api.get.mockRejectedValue(new Error('offline'))
    const { useHostedService, wrapper } = await load('enterprise')

    const { result } = renderHook(() => useHostedService('cloudSync'), { wrapper })
    await waitFor(() => expect(api.get).toHaveBeenCalledTimes(1))
    expect(result.current).toBeUndefined()
  })
})
