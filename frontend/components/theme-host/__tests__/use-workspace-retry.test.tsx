/**
 * PRD-261 A-19d2 (PB-8a, the W-E3 seam review) — the workspace retry is the host's: a workspace
 * that failed to load (the backend asleep) is asked for again once the backend answers GET /health,
 * once per failure, by ThemeLookRuntime's one call of ./use-workspace-retry.ts, whenever the host
 * loads or runs an installed theme (with looks, or without, when no look scope is mounted).
 * AccountLookScope no longer asks: one subscriber, never a double refetch. Classic and Studio, and a
 * theme the host stopped running, ask nothing.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useState, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const account = vi.hoisted(() => ({
  workspaceId: null as string | null,
  workspaceError: new Error('Failed to fetch workspace') as Error | null,
  workspaceLoading: false,
  refreshWorkspace: vi.fn(async (): Promise<void> => undefined),
}))

vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'user_a' } }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({
    workspace: account.workspaceId === null ? null : { id: account.workspaceId },
    isLoading: account.workspaceLoading,
    error: account.workspaceError,
    refreshWorkspace: account.refreshWorkspace,
  }),
}))

import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { ThemeLookRuntime } from '@/components/theme-host/theme-look-runtime'
import { workspaceFailed } from '@/components/theme-host/use-workspace-retry'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { BACKEND_HEALTH_QUERY_KEY } from '@/hooks/use-backend-health'
import { FIXTURE_THEME_ENTRY, FIXTURE_THEME_STYLE } from '@/lib/theme-host/__tests__/fixture-theme-entry'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import type { ThemeHostStatus } from '@/lib/theme-host/types'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'

const FRONTEND = path.resolve(__dirname, '../../..')
const ACCOUNT_LOOK_SCOPE = 'components/theme-host/account-look-scope.tsx'
const LOOK_RUNTIME = 'components/theme-host/theme-look-runtime.tsx'
/** An installed theme that fills no look editor slot: no look scope is mounted for it. */
const NO_LOOKS_ENTRY: ThemeEntry = { ...FIXTURE_THEME_ENTRY, slots: FIXTURE_THEME_ENTRY.slots.filter((slot) => slot !== 'lookEditor@1') }

let queryClient = new QueryClient()

function quietQueryClient(): QueryClient {
  const quiet = () => undefined
  return new QueryClient({ defaultOptions: { queries: { retry: false } }, logger: { log: quiet, warn: quiet, error: quiet } })
}

function hostValue(status: ThemeHostStatus): ThemeHostContextValue {
  return { status, reason: null, frame: null, slots: null, reportRenderFailure: () => undefined }
}

interface AppProps {
  readonly style: UiStyle
  readonly status: ThemeHostStatus
  readonly entry?: ThemeEntry | null
  readonly children?: ReactNode
}

/** The providers as components/providers.tsx nests them, the theme host's status given. */
function App({ style, status, entry = null, children }: AppProps) {
  const [, setTone] = useState<string | undefined>(undefined)
  return (
    <UiStyleProvider initialStyle={style} installedTheme={entry}>
      <QueryClientProvider client={queryClient}>
        <ThemeHostContext.Provider value={hostValue(status)}>
          <ThemeLookRuntime onToneChange={setTone}>{children}</ThemeLookRuntime>
        </ThemeHostContext.Provider>
      </QueryClientProvider>
    </UiStyleProvider>
  )
}

/** A theme's frame asks GET /health (useBackendHealth's query) and the backend answers, or doesn't. */
async function backendAnswers(answered: boolean): Promise<void> {
  await act(async () => {
    await queryClient
      .fetchQuery({
        queryKey: BACKEND_HEALTH_QUERY_KEY,
        queryFn: () => (answered ? Promise.resolve({ status: 'ok' }) : Promise.reject(new Error('Network error'))),
        staleTime: 0,
      })
      .catch(() => undefined)
  })
}

beforeEach(() => {
  Object.assign(account, { workspaceId: null, workspaceError: new Error('Failed to fetch workspace'), workspaceLoading: false })
  account.refreshWorkspace.mockReset()
  account.refreshWorkspace.mockResolvedValue(undefined)
  queryClient = quietQueryClient()
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.restoreAllMocks()
  window.localStorage.clear()
  document.documentElement.removeAttribute('style')
})

type WorkspaceState = NonNullable<Parameters<typeof workspaceFailed>[0]>

describe('workspaceFailed', () => {
  it('is a workspace that finished loading with none and an error, never one loading or loaded', () => {
    const failed: WorkspaceState = { workspace: null, isLoading: false, error: new Error('x'), refreshWorkspace: async () => undefined, canEdit: true }
    expect(workspaceFailed(failed)).toBe(true)
    expect(workspaceFailed({ ...failed, isLoading: true })).toBe(false)
    expect(workspaceFailed({ ...failed, error: null })).toBe(false)
    expect(workspaceFailed({ ...failed, workspace: { id: 'ws-1' } as WorkspaceState['workspace'] })).toBe(false)
    expect(workspaceFailed(null)).toBe(false)
  })
})

describe('PB-8a — the host asks for a failed workspace again, exactly once', () => {
  it.each<[string, ThemeEntry]>([
    ['an installed theme with looks', FIXTURE_THEME_ENTRY],
    ['an installed theme whose look scope is not mounted', NO_LOOKS_ENTRY],
  ])('for %s, once the backend answers', async (_label, entry) => {
    render(<App style={FIXTURE_THEME_STYLE} status="active" entry={entry} />)
    await backendAnswers(false)
    expect(account.refreshWorkspace).not.toHaveBeenCalled()
    await backendAnswers(true)
    // A second answer before the workspace arrives asks nothing more: once per failure, one subscriber.
    await backendAnswers(true)
    expect(account.refreshWorkspace).toHaveBeenCalledTimes(1)
  })

  it('while the host loads the installed theme too', async () => {
    render(<App style={FIXTURE_THEME_STYLE} status="loading" entry={FIXTURE_THEME_ENTRY} />)
    await backendAnswers(true)
    expect(account.refreshWorkspace).toHaveBeenCalledTimes(1)
  })

  it.each<UiStyle>(['studio', 'classic'])('never under %s', async (style) => {
    render(<App style={style} status="off" />)
    await backendAnswers(true)
    expect(account.refreshWorkspace).not.toHaveBeenCalled()
  })

  it('never for a workspace that loaded', async () => {
    Object.assign(account, { workspaceId: 'ws-1', workspaceError: null })
    render(<App style={FIXTURE_THEME_STYLE} status="active" entry={FIXTURE_THEME_ENTRY} />)
    await backendAnswers(true)
    expect(account.refreshWorkspace).not.toHaveBeenCalled()
  })

  it('after the host stops running the installed theme (a failed load), asks nothing', async () => {
    render(<App style={FIXTURE_THEME_STYLE} status="failed" entry={FIXTURE_THEME_ENTRY} />)
    await backendAnswers(true)
    expect(account.refreshWorkspace).not.toHaveBeenCalled()
  })
})

describe('PB-8a — where the retry lives', () => {
  it('AccountLookScope no longer calls it; ThemeLookRuntime calls it once', () => {
    const scope = readFileSync(path.join(FRONTEND, ACCOUNT_LOOK_SCOPE), 'utf8')
    const runtime = readFileSync(path.join(FRONTEND, LOOK_RUNTIME), 'utf8')
    expect(scope).not.toMatch(/useWorkspaceRetry|workspaceFailed|getQueryCache/)
    expect(runtime.match(/useHostWorkspaceRetry\(/g)).toHaveLength(1)
  })
})
