/**
 * PRD-260 F5 (Gerard, 15 Oct) — looks, the active look and the seen
 * Deliverables record are kept per signed-in user and workspace in this
 * browser. AccountLookScope, as components/providers.tsx mounts it, gives the
 * LookProvider and the shelf (useCelebrateNew) the account's scope once Clerk
 * and the workspace are known. A second account in the same browser sees
 * neither the first account's looks nor its seen record; the local edition's
 * one user (no user id) keeps its looks; what a browser kept before F5 moves
 * to the first account once and is never deleted.
 * A workspace that failed to load (the backend asleep when the page opened)
 * is asked for again once the backend answers GET /health, and until the
 * looks are read Play still paints the pre-paint script's look. PRD-261 A-19d2
 * (PB-8a): the host asks (ThemeLookRuntime's one useHostWorkspaceRetry call,
 * for a builtin theme with looks too), no longer AccountLookScope, so the
 * providers here mount the host's retry beside it.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const account = vi.hoisted(() => ({
  isLoaded: true,
  isSignedIn: true,
  userId: 'user_a' as string | null,
  workspaceId: 'ws-1' as string | null,
  workspaceLoading: false,
  workspaceError: null as Error | null,
  refreshWorkspace: vi.fn(async (): Promise<void> => undefined),
}))

vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({
    isLoaded: account.isLoaded,
    isSignedIn: account.isSignedIn,
    user: account.userId === null ? null : { id: account.userId },
  }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({
    workspace: account.workspaceId === null ? null : { id: account.workspaceId },
    isLoading: account.workspaceLoading,
    error: account.workspaceError,
    refreshWorkspace: account.refreshWorkspace,
  }),
}))

import { AccountLookScope } from '@/components/theme-host/account-look-scope'
import { useHostWorkspaceRetry } from '@/components/theme-host/use-workspace-retry'
import { LookProvider, useLook } from '@/lib/theme-host/builtin/play-look-provider'
import { LOOKS_HEADING, LooksScreen } from '@/components/looks/looks-screen'
import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { BACKEND_HEALTH_QUERY_KEY } from '@/hooks/use-backend-health'
import { useCelebrateNew } from '@/hooks/use-celebrate-new'
import type { Deliverable } from '@/hooks/use-deliverables-api'
import { clearRoot, lookNamesLeft, myLook, rootVars } from '@/lib/looks/__tests__/helpers'
import { lookDocumentVars } from '@/lib/looks/document'
import { DAYLIGHT, DEFAULT_LOOK } from '@/lib/looks/presets'
import { lookScope } from '@/lib/looks/scope'
import {
  ACTIVE_LOOK_KEY,
  LOOK_SCOPE_KEY,
  SAVED_LOOKS_KEY,
  UNSCOPED_LOOKS_MOVED_KEY,
  lookKeys,
} from '@/lib/theme-host/builtin/play-look-storage'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import { SEEN_KEY, seenKey } from '@/lib/play/seen'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'

const A = lookScope('ws-1', 'user_a')
const B = lookScope('ws-1', 'user_b')

function deliverable(id: string, createdAt: string): Deliverable {
  return { id, created_at: createdAt } as Deliverable
}

const OLD = deliverable('d-old', '2026-10-10T09:00:00+00:00')
const NEW = deliverable('d-new', '2026-10-14T09:00:00+00:00')
const OLD_ONLY: readonly Deliverable[] = [OLD]
const BOTH: readonly Deliverable[] = [NEW, OLD]
const OLD_SEEN = { ids: ['d-old'], newest: OLD.created_at }

const celebrate = vi.fn()

/** A fresh cache per test, as components/providers.tsx has one per page; no retries, no logging. */
function quietQueryClient(): QueryClient {
  const quiet = () => undefined
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
    logger: { log: quiet, warn: quiet, error: quiet },
  })
}

let queryClient = quietQueryClient()

function Probe() {
  const { look, saved, loaded, setLook, saveLook } = useLook()
  const { setStyle } = useUiStyle()
  return (
    <div>
      <p>Loaded: {loaded ? 'yes' : 'no'}</p>
      <p>Active: {look.name}</p>
      <p>Saved: {saved.map((entry) => entry.name).join(', ') || 'none'}</p>
      <button type="button" onClick={() => saveLook(myLook())}>
        Save mine
      </button>
      <button type="button" onClick={() => setLook(DAYLIGHT)}>
        Use Daylight
      </button>
      <button type="button" onClick={() => setStyle(BUILTIN_PLAY_STYLE)}>
        Go Play
      </button>
    </div>
  )
}

/** Play's shelf as far as the seen record goes: Auto celebrates a Deliverable this account hasn't seen. */
function Shelf({ items }: { readonly items: readonly Deliverable[] }) {
  const latest = useCelebrateNew(items, celebrate)
  return <p>Celebrating: {latest?.id ?? 'nothing'}</p>
}

/** The host's one workspace retry, as ThemeLookRuntime calls it under a builtin theme with looks (PB-8a). */
function HostWorkspaceRetry() {
  useHostWorkspaceRetry(true)
  return null
}

/** The providers as components/providers.tsx nests them: the account's scope above the LookProvider. */
function Providers({ style = BUILTIN_PLAY_STYLE, children }: { readonly style?: UiStyle; readonly children: ReactNode }) {
  return (
    <UiStyleProvider initialStyle={style}>
      <QueryClientProvider client={queryClient}>
        <HostWorkspaceRetry />
        <AccountLookScope>
          <LookProvider>{children}</LookProvider>
        </AccountLookScope>
      </QueryClientProvider>
    </UiStyleProvider>
  )
}

function App({ items, style }: { readonly items: readonly Deliverable[]; readonly style?: UiStyle }) {
  return (
    <Providers style={style}>
      <Probe />
      <Shelf items={items} />
    </Providers>
  )
}

function click(name: string): void {
  act(() => screen.getByRole('button', { name }).click())
}

function stored(key: string): unknown {
  const raw = window.localStorage.getItem(key)
  return raw === null ? null : JSON.parse(raw)
}

function signInAs(userId: string | null, workspaceId: string | null = 'ws-1'): void {
  account.userId = userId
  account.workspaceId = workspaceId
}

function toneCookie(): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))?.split('=')[1]
}

/** Play's frame asks GET /health (useBackendHealth's query) and the backend answers, or doesn't. */
async function backendAnswers(answered: boolean): Promise<void> {
  await act(async () => {
    await queryClient
      .fetchQuery({
        queryKey: BACKEND_HEALTH_QUERY_KEY,
        queryFn: () => (answered ? Promise.resolve({ status: 'ok' }) : Promise.reject(new Error('Network error'))),
      })
      .catch(() => undefined)
  })
}

beforeEach(() => {
  Object.assign(account, {
    isLoaded: true,
    isSignedIn: true,
    userId: 'user_a',
    workspaceId: 'ws-1',
    workspaceLoading: false,
    workspaceError: null,
  })
  account.refreshWorkspace.mockReset()
  account.refreshWorkspace.mockResolvedValue(undefined)
  queryClient = quietQueryClient()
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
  clearRoot()
  celebrate.mockClear()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.restoreAllMocks()
  window.localStorage.clear()
  clearRoot()
})

describe('a second account in the same browser (F5)', () => {
  it("sees neither the first account's looks nor its seen Deliverables record", async () => {
    const { rerender } = render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    click('Use Daylight')
    expect(screen.getByText('Saved: My look 1')).toBeInTheDocument()
    expect(stored(seenKey(A))).toEqual(OLD_SEEN)

    signInAs('user_b')
    rerender(<App items={BOTH} />)
    await waitFor(() => expect(screen.getByText('Saved: none')).toBeInTheDocument())
    expect(screen.getByText('Active: Ember')).toBeInTheDocument()
    // Had B read A's record, d-new would be newer than the newest seen: a celebration.
    expect(celebrate).not.toHaveBeenCalled()
    expect(screen.getByText('Celebrating: nothing')).toBeInTheDocument()
    expect(stored(seenKey(B))).toEqual({ ids: ['d-new', 'd-old'], newest: NEW.created_at })

    // B wrote nothing over A's keys.
    expect(stored(lookKeys(A).saved)).toEqual([myLook()])
    expect(stored(lookKeys(A).active)).toEqual(DAYLIGHT)
    expect(stored(seenKey(A))).toEqual(OLD_SEEN)
    expect(stored(lookKeys(B).saved)).toBeNull()
    expect(stored(lookKeys(B).active)).toBeNull()

    signInAs('user_a')
    rerender(<App items={BOTH} />)
    await waitFor(() => expect(screen.getByText('Saved: My look 1')).toBeInTheDocument())
    expect(screen.getByText('Active: Daylight')).toBeInTheDocument()
    await waitFor(() => expect(celebrate).toHaveBeenCalledTimes(1))
    expect(screen.getByText('Celebrating: d-new')).toBeInTheDocument()
  })

  it('keeps the same user apart in another workspace', async () => {
    const { rerender } = render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    signInAs('user_a', 'ws-2')
    rerender(<App items={OLD_ONLY} />)
    await waitFor(() => expect(screen.getByText('Saved: none')).toBeInTheDocument())
    expect(stored(lookKeys(lookScope('ws-2', 'user_a')).saved)).toBeNull()
    expect(stored(lookKeys(A).saved)).toEqual([myLook()])
  })

  it("points the pre-paint script at the account's own active look", async () => {
    const { rerender } = render(<App items={OLD_ONLY} />)
    await waitFor(() => expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBe(A))
    signInAs('user_b')
    rerender(<App items={OLD_ONLY} />)
    await waitFor(() => expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBe(B))
  })
})

describe('what this browser kept before F5', () => {
  it('moves to the first account once and is never deleted', async () => {
    window.localStorage.setItem(SAVED_LOOKS_KEY, JSON.stringify([myLook()]))
    window.localStorage.setItem(ACTIVE_LOOK_KEY, JSON.stringify(myLook()))
    window.localStorage.setItem(SEEN_KEY, JSON.stringify(OLD_SEEN))
    const { rerender } = render(<App items={BOTH} />)
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
    expect(screen.getByText('Active: My look 1')).toBeInTheDocument()
    // The record moved with them: d-new is newer than the newest this account had seen.
    await waitFor(() => expect(celebrate).toHaveBeenCalledTimes(1))
    expect(window.localStorage.getItem(UNSCOPED_LOOKS_MOVED_KEY)).toBe(A)

    signInAs('user_b')
    rerender(<App items={BOTH} />)
    await waitFor(() => expect(screen.getByText('Saved: none')).toBeInTheDocument())
    expect(screen.getByText('Active: Ember')).toBeInTheDocument()
    expect(celebrate).toHaveBeenCalledTimes(1)

    expect(stored(SAVED_LOOKS_KEY)).toEqual([myLook()])
    expect(stored(ACTIVE_LOOK_KEY)).toEqual(myLook())
    expect(stored(SEEN_KEY)).toEqual(OLD_SEEN)
  })
})

describe('the local edition', () => {
  it("keeps its one user's looks with no user id", async () => {
    signInAs(null, 'ws-local')
    const { unmount } = render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    const local = lookScope('ws-local', null)
    expect(stored(lookKeys(local).saved)).toEqual([myLook()])
    expect(stored(seenKey(local))).toEqual(OLD_SEEN)
    unmount()

    render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
  })
})

describe('before the account is known', () => {
  it("loads, moves and writes none of the account's looks while Clerk or the workspace loads, while signed out, or without a workspace", async () => {
    window.localStorage.setItem(SAVED_LOOKS_KEY, JSON.stringify([myLook()]))
    account.isLoaded = false
    const { rerender } = render(<App items={BOTH} />)
    const waiting: (() => void)[] = [
      () => Object.assign(account, { isLoaded: true, isSignedIn: false }),
      () => Object.assign(account, { isSignedIn: true, workspaceLoading: true }),
      () => Object.assign(account, { workspaceLoading: false, workspaceId: null }),
    ]
    for (const step of [() => undefined, ...waiting]) {
      step()
      rerender(<App items={BOTH} />)
      expect(screen.getByText('Loaded: no')).toBeInTheDocument()
    }
    expect(window.localStorage.getItem(UNSCOPED_LOOKS_MOVED_KEY)).toBeNull()
    expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBeNull()
    expect(stored(seenKey(A))).toBeNull()
    expect(celebrate).not.toHaveBeenCalled()

    account.workspaceId = 'ws-1'
    rerender(<App items={BOTH} />)
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
    expect(screen.getByText('Loaded: yes')).toBeInTheDocument()
  })
})

describe('a workspace that failed to load (the backend asleep when the page opened)', () => {
  beforeEach(() => {
    Object.assign(account, { workspaceId: null, workspaceError: new Error('Failed to fetch workspace') })
  })

  it("still paints Play when it is switched on before the account's looks are read: the pre-paint script's look", async () => {
    window.localStorage.setItem(LOOK_SCOPE_KEY, A)
    window.localStorage.setItem(lookKeys(A).active, JSON.stringify(DAYLIGHT))
    render(<App items={OLD_ONLY} style="studio" />)
    expect(screen.getByText('Loaded: no')).toBeInTheDocument()
    expect(lookNamesLeft()).toEqual([])

    click('Go Play')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT)))
    expect(toneCookie()).toBe('light')
    expect(screen.getByText('Loaded: no')).toBeInTheDocument()
    // Painting only reads: nothing moved, the pointer is unchanged.
    expect(window.localStorage.getItem(UNSCOPED_LOOKS_MOVED_KEY)).toBeNull()
    expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBe(A)
  })

  it('paints the default look when nothing is stored', async () => {
    render(<App items={OLD_ONLY} style="studio" />)
    click('Go Play')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    expect(screen.getByText('Loaded: no')).toBeInTheDocument()
    expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBeNull()
  })

  it('asks for the workspace again once the backend answers, and /looks then loads', async () => {
    window.localStorage.setItem(lookKeys(A).active, JSON.stringify(DAYLIGHT))
    account.refreshWorkspace.mockImplementation(async () => {
      Object.assign(account, { workspaceId: 'ws-1', workspaceError: null })
    })
    const looks = () => (
      <Providers>
        <LooksScreen />
      </Providers>
    )
    const busy = () => screen.getByRole('heading', { level: 1, name: LOOKS_HEADING }).closest('[aria-busy]')
    const { rerender } = render(looks())
    expect(busy()).not.toBeNull()

    await backendAnswers(false)
    expect(account.refreshWorkspace).not.toHaveBeenCalled()

    await backendAnswers(true)
    // A second answer before the workspace arrives asks nothing more: once per failure.
    await backendAnswers(true)
    expect(account.refreshWorkspace).toHaveBeenCalledTimes(1)

    // The WorkspaceProvider renders again with the workspace it just loaded.
    rerender(looks())
    await waitFor(() => expect(busy()).toBeNull())
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT)))

    await backendAnswers(true)
    expect(account.refreshWorkspace).toHaveBeenCalledTimes(1)
  })

  it('a workspace that loaded is never asked for again when the backend answers', async () => {
    Object.assign(account, { workspaceId: 'ws-1', workspaceError: null })
    render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    await backendAnswers(true)
    expect(account.refreshWorkspace).not.toHaveBeenCalled()
  })
})
