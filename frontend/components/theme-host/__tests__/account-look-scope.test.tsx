/**
 * PRD-261 A-19f2 — replaces components/looks/__tests__/look-scope.test.tsx's host cases (the core
 * ledger's coreReplacement). PRD-260 F5 — the looks of an installed theme with looks are kept per
 * signed-in user and workspace in this browser. AccountLookScope (components/theme-host/
 * account-look-scope.tsx), as the host's look runtime mounts it, gives the look provider the
 * account's scope once Clerk and the workspace are known: a second account in the same browser sees
 * none of the first account's looks, the same user in another workspace has their own, the local
 * edition's one user (no user id) keeps theirs, and nothing is read, moved or written before the
 * account is known. A workspace that failed to load (the backend asleep when the page opened) is
 * asked for again once the backend answers GET /health (the host's one retry,
 * components/theme-host/use-workspace-retry.ts); until the looks are read the theme still paints the
 * boot script's look. The theme is hello-a09b, its boot on the page; the scope's part in Play's
 * shelf (its seen Deliverables record) is the kit's.
 */
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
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

import { BACKEND_HEALTH_QUERY_KEY } from '@/hooks/use-backend-health'
import { lookScope } from '@/lib/look-kit/scope'
import { DUSK, HELLO, HELLO_RULES, HELLO_THEME_ID, helloThemeBoot, myLook, PAPER, putPageBoot } from '@/lib/theme-host/looks/__tests__/hello-looks'
import { clearRoot, cookieValue, dropCookies, rootVars } from '@/lib/theme-host/looks/__tests__/look-test-helpers'
import { paint } from '@/lib/theme-host/looks/paint'
import { INSTALLED_LOOK_KEY_PREFIX, lookKeys } from '@/lib/theme-host/looks/storage'
import { THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import type { ThemeHostStatus } from '@/lib/theme-host/types'
import { saveLook, useLook } from '@/lib/theme-sdk/looks'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION } from '@/lib/ui-style'

import { HELLO_LOOKS_STYLE, HELLO_LOOKS_THEME, LookAppearance, quietQueryClient } from './look-runtime-harness'

const WORKSPACE = 'ws-1'
const A = lookScope(WORKSPACE, 'user_a')
const B = lookScope(WORKSPACE, 'user_b')
const TABLE = HELLO_RULES.table

let queryClient = quietQueryClient()

/** Whether the runtime runs, and has read the account's looks: 'no runtime' until it runs at all. */
function loadState(available: boolean, loaded: boolean): string {
  if (!available) return 'no runtime'
  return loaded ? 'yes' : 'no'
}

function Probe() {
  const { available, loaded, look, saved, setLook } = useLook()
  return (
    <div>
      <p>Loaded: {loadState(available, loaded)}</p>
      <p>Look: {look?.id ?? 'none'}</p>
      <p>Saved: {saved.map((entry) => entry.name).join(', ') || 'none'}</p>
      <button type="button" onClick={() => saveLook(myLook())}>Save mine</button>
      <button type="button" onClick={() => setLook(DUSK)}>Use Dusk</button>
    </div>
  )
}

/** The app with hello-a09b as the style, the host in `status`. */
function app(status: ThemeHostStatus = 'active') {
  return (
    <LookAppearance style={HELLO_LOOKS_STYLE} status={status} entry={HELLO_LOOKS_THEME} queryClient={queryClient}>
      <Probe />
    </LookAppearance>
  )
}

function click(name: string): void {
  act(() => screen.getByRole('button', { name }).click())
}

function stored(key: string): unknown {
  const raw = window.localStorage.getItem(key)
  return raw === null ? null : JSON.parse(raw)
}

function signInAs(userId: string | null, workspaceId: string | null = WORKSPACE): void {
  account.userId = userId
  account.workspaceId = workspaceId
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

let removeBoot: () => void = () => undefined

beforeEach(() => {
  Object.assign(account, { isLoaded: true, isSignedIn: true, userId: 'user_a', workspaceId: WORKSPACE, workspaceLoading: false, workspaceError: null })
  account.refreshWorkspace.mockReset()
  account.refreshWorkspace.mockResolvedValue(undefined)
  queryClient = quietQueryClient()
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  dropCookies([THEME_TONE_COOKIE])
  clearRoot()
  removeBoot = putPageBoot(helloThemeBoot())
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  removeBoot()
  removeBoot = () => undefined
  queryClient.clear()
  vi.restoreAllMocks()
  window.localStorage.clear()
  dropCookies([THEME_TONE_COOKIE])
  clearRoot()
})

describe('a second account in the same browser (F5)', () => {
  it("sees none of the first account's looks, and the first account finds its own again", async () => {
    const { rerender } = render(app())
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    click('Use Dusk')
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()

    signInAs('user_b')
    rerender(app())
    await waitFor(() => expect(screen.getByText('Saved: none')).toBeInTheDocument())
    expect(screen.getByText(`Look: ${PAPER.id}`)).toBeInTheDocument()
    // B wrote nothing over A's keys.
    expect(stored(lookKeys(A, HELLO.keys).saved)).toEqual([myLook()])
    expect(stored(lookKeys(A, HELLO.keys).active)).toEqual(DUSK)
    expect(stored(lookKeys(B, HELLO.keys).saved)).toBeNull()
    expect(stored(lookKeys(B, HELLO.keys).active)).toBeNull()

    signInAs('user_a')
    rerender(app())
    await waitFor(() => expect(screen.getByText('Saved: My look 1')).toBeInTheDocument())
    expect(screen.getByText(`Look: ${DUSK.id}`)).toBeInTheDocument()
  })

  it('keeps the same user apart in another workspace', async () => {
    const { rerender } = render(app())
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    expect(screen.getByText('Saved: My look 1')).toBeInTheDocument()
    signInAs('user_a', 'ws-2')
    rerender(app())
    await waitFor(() => expect(screen.getByText('Saved: none')).toBeInTheDocument())
    expect(stored(lookKeys(lookScope('ws-2', 'user_a'), HELLO.keys).saved)).toBeNull()
    expect(stored(lookKeys(A, HELLO.keys).saved)).toEqual([myLook()])
  })

  it("points the next load's boot script at the account's own active look", async () => {
    const { rerender } = render(app())
    await waitFor(() => expect(window.localStorage.getItem(HELLO.keys.pointer)).toBe(A))
    signInAs('user_b')
    rerender(app())
    await waitFor(() => expect(window.localStorage.getItem(HELLO.keys.pointer)).toBe(B))
  })
})

describe('the local edition', () => {
  it("keeps its one user's looks with no user id", async () => {
    signInAs(null, 'ws-local')
    const { unmount } = render(app())
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    await waitFor(() => expect(stored(lookKeys(lookScope('ws-local', null), HELLO.keys).saved)).toEqual([myLook()]))
    unmount()

    render(app())
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
  })
})

describe('before the account is known', () => {
  it("loads, moves and writes none of the account's looks while Clerk or the workspace loads, while signed out, or without a workspace", async () => {
    window.localStorage.setItem(lookKeys(A, HELLO.keys).saved, JSON.stringify([myLook()]))
    const writes = vi.spyOn(Storage.prototype, 'setItem')
    account.isLoaded = false
    const { rerender } = render(app())
    const waiting: (() => void)[] = [
      () => Object.assign(account, { isLoaded: true, isSignedIn: false }),
      () => Object.assign(account, { isSignedIn: true, workspaceLoading: true }),
      () => Object.assign(account, { workspaceLoading: false, workspaceId: null }),
    ]
    for (const step of [() => undefined, ...waiting]) {
      step()
      rerender(app())
      expect(screen.getByText('Loaded: no')).toBeInTheDocument()
      expect(screen.getByText('Saved: none')).toBeInTheDocument()
    }
    expect(writes.mock.calls.map(([key]) => key).filter((key) => key.startsWith(INSTALLED_LOOK_KEY_PREFIX))).toEqual([])
    expect(window.localStorage.getItem(HELLO.keys.pointer)).toBeNull()

    account.workspaceId = WORKSPACE
    rerender(app())
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
    expect(screen.getByText('Loaded: yes')).toBeInTheDocument()
  })
})

describe('a workspace that failed to load (the backend asleep when the page opened)', () => {
  beforeEach(() => {
    Object.assign(account, { workspaceId: null, workspaceError: new Error('Failed to fetch workspace') })
  })

  it("still paints the theme when the host starts it before the account's looks are read: the boot script's look, and its tone", async () => {
    window.localStorage.setItem(HELLO.keys.pointer, A)
    window.localStorage.setItem(lookKeys(A, HELLO.keys).active, JSON.stringify(DUSK))
    const { rerender } = render(app('off'))
    expect(rootVars()).toEqual({})

    rerender(app('active'))
    await waitFor(() => expect(rootVars()).toEqual(paint(TABLE, DUSK)))
    await waitFor(() => expect(cookieValue(THEME_TONE_COOKIE)).toBe(themeToneCookieValue(HELLO_THEME_ID, 'dark')))
    expect(screen.getByText('Loaded: no')).toBeInTheDocument()
    // Painting only reads: the pointer is unchanged.
    expect(window.localStorage.getItem(HELLO.keys.pointer)).toBe(A)
  })

  it('paints the default look when nothing is stored', async () => {
    render(app())
    await waitFor(() => expect(rootVars()).toEqual(paint(TABLE, PAPER)))
    expect(screen.getByText('Loaded: no')).toBeInTheDocument()
    expect(window.localStorage.getItem(HELLO.keys.pointer)).toBeNull()
  })

  it('asks for the workspace again once the backend answers, and the looks then load', async () => {
    window.localStorage.setItem(lookKeys(A, HELLO.keys).active, JSON.stringify(DUSK))
    account.refreshWorkspace.mockImplementation(async () => {
      Object.assign(account, { workspaceId: WORKSPACE, workspaceError: null })
    })
    const { rerender } = render(app())
    expect(screen.getByText('Loaded: no')).toBeInTheDocument()

    await backendAnswers(false)
    expect(account.refreshWorkspace).not.toHaveBeenCalled()
    await backendAnswers(true)
    // A second answer before the workspace arrives asks nothing more: once per failure.
    await backendAnswers(true)
    expect(account.refreshWorkspace).toHaveBeenCalledTimes(1)

    // The WorkspaceProvider renders again with the workspace it just loaded.
    rerender(app())
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual(paint(TABLE, DUSK)))
    await backendAnswers(true)
    expect(account.refreshWorkspace).toHaveBeenCalledTimes(1)
  })
})
