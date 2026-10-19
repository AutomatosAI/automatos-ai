/**
 * PRD-261 A-10a — SDK area `context`. Each hook hands a theme a small copy of the host's
 * provider value (never the webhook key, plan limits or settings), answers "unknown" outside a
 * provider instead of throwing, and keeps its answer's identity while nothing it reads changes.
 */
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  workspace: null as unknown,
  role: null as unknown,
  theme: {} as { forcedTheme?: string; resolvedTheme?: string },
  reducedMotion: false,
  health: { connection: 'connected', checking: false, check: () => undefined } as Record<string, unknown>,
  healthOptions: [] as unknown[],
}))

vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => host.workspace }))
vi.mock('@/contexts/role-context', () => ({ useSystemRoleOptional: () => host.role }))
vi.mock('next-themes', () => ({ useTheme: () => host.theme }))
vi.mock('@/hooks/use-media-preferences', () => ({ usePrefersReducedMotion: () => host.reducedMotion }))
vi.mock('@/hooks/use-backend-health', () => ({
  useBackendHealth: (options: unknown) => {
    host.healthOptions.push(options)
    return host.health
  },
}))

import { useEdition, useHealth, useMediaPreferences, useRole, useTone, useTrial, useWorkspace } from '../context'
import { authEdition } from '@/lib/auth-edition'

const PROVIDER_WORKSPACE = {
  id: 'ws-1',
  name: 'Acme',
  slug: 'acme',
  plan: 'pro',
  role: 'viewer',
  planLimits: { max_agents: 5, max_workflows: 5, max_documents: 5, max_members: 5 },
  webhookUrl: 'https://hooks.example.test/acme',
  webhookKey: 'not-for-themes',
  settings: { integrations: {} },
  onboarding: { stage: 'completed', trial: { granted_usd: 5, spent_usd: 1.25, state: 'warned' } },
}

afterEach(() => {
  host.workspace = null
  host.role = null
  host.theme = {}
  host.reducedMotion = false
  host.health = { connection: 'connected', checking: false, check: () => undefined }
  host.healthOptions.length = 0
})

describe('PRD-261 A-10a — useWorkspace', () => {
  it('hands over id, name, slug and plan only', () => {
    host.workspace = { workspace: PROVIDER_WORKSPACE, isLoading: false, canEdit: false }
    const { result } = renderHook(() => useWorkspace())
    expect(result.current).toEqual({ workspace: { id: 'ws-1', name: 'Acme', slug: 'acme', plan: 'pro' }, isLoading: false })
    expect(JSON.stringify(result.current)).not.toContain('not-for-themes')
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(Object.isFrozen(result.current.workspace)).toBe(true)
  })

  it('keeps the same answer across renders while the workspace is the same', () => {
    host.workspace = { workspace: PROVIDER_WORKSPACE, isLoading: false, canEdit: true }
    const { result, rerender } = renderHook(() => useWorkspace())
    const first = result.current
    host.workspace = { workspace: { ...PROVIDER_WORKSPACE }, isLoading: false, canEdit: true }
    rerender()
    expect(result.current).toBe(first)
  })

  it('answers no workspace while loading, and outside the provider', () => {
    host.workspace = { workspace: null, isLoading: true, canEdit: true }
    expect(renderHook(() => useWorkspace()).result.current).toEqual({ workspace: null, isLoading: true })
    host.workspace = null
    expect(renderHook(() => useWorkspace()).result.current).toEqual({ workspace: null, isLoading: false })
  })
})

describe('PRD-261 A-10a — useRole', () => {
  it("reads the workspace role and the platform role", () => {
    host.workspace = { workspace: PROVIDER_WORKSPACE, isLoading: false, canEdit: false }
    host.role = { systemRole: 'admin', isAdmin: true, isLoading: false }
    expect(renderHook(() => useRole()).result.current).toEqual({
      workspaceRole: 'viewer',
      systemRole: 'admin',
      isAdmin: true,
      canEdit: false,
    })
  })

  it('is no admin and may edit while unknown, as the host chrome is', () => {
    host.role = { systemRole: 'user', isAdmin: false, isLoading: true }
    expect(renderHook(() => useRole()).result.current).toEqual({
      workspaceRole: null,
      systemRole: null,
      isAdmin: false,
      canEdit: true,
    })
  })
})

describe('PRD-261 A-10a — edition, trial, health, media and tone', () => {
  it("answers the build's edition", () => {
    expect(renderHook(() => useEdition()).result.current).toBe(authEdition)
  })

  it("reads the workspace's trial ledger, or null", () => {
    host.workspace = { workspace: PROVIDER_WORKSPACE, isLoading: false, canEdit: true }
    expect(renderHook(() => useTrial()).result.current).toEqual({ state: 'warned', grantedUsd: 5, spentUsd: 1.25 })
    host.workspace = { workspace: { ...PROVIDER_WORKSPACE, onboarding: { stage: 'completed', trial: null } }, isLoading: false, canEdit: true }
    expect(renderHook(() => useTrial()).result.current).toBeNull()
  })

  it("hands over a frozen copy of the host's backend health, passing poll through", () => {
    const polled = renderHook(() => useHealth({ poll: true }))
    expect(polled.result.current).toEqual(host.health)
    expect(polled.result.current).not.toBe(host.health)
    expect(Object.isFrozen(polled.result.current)).toBe(true)
    expect(renderHook(() => useHealth()).result.current).toEqual(host.health)
    expect(host.healthOptions).toEqual([{ poll: true }, {}])
  })

  it('keeps only connection, checking, host and check, and keeps its identity while they hold', () => {
    host.health = { ...host.health, host: 'api.example.test', baseUrl: 'not-for-themes' }
    const { result, rerender } = renderHook(() => useHealth())
    expect(Object.keys(result.current).sort()).toEqual(['check', 'checking', 'connection', 'host'])
    expect(result.current.host).toBe('api.example.test')
    const first = result.current
    rerender()
    expect(result.current).toBe(first)
  })

  it('reads reduced motion', () => {
    host.reducedMotion = true
    expect(renderHook(() => useMediaPreferences()).result.current).toEqual({ reducedMotion: true })
  })

  it('prefers the forced tone, reads the chosen one, and is null before either is known', () => {
    host.theme = { forcedTheme: 'light', resolvedTheme: 'dark' }
    expect(renderHook(() => useTone()).result.current).toBe('light')
    host.theme = { resolvedTheme: 'dark' }
    expect(renderHook(() => useTone()).result.current).toBe('dark')
    host.theme = { resolvedTheme: 'system' }
    expect(renderHook(() => useTone()).result.current).toBeNull()
    host.theme = {}
    expect(renderHook(() => useTone()).result.current).toBeNull()
  })
})
