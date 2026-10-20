/**
 * PRD-261 E3c — the host publishes whose theme state the SDK's `storage` area reads and writes
 * (components/theme-host/theme-state-scope.tsx, plan §2.2.2): `{themeId, userKey, workspaceKey}` for
 * an installed theme the host loads or runs, once its user is known (`local` in the local edition),
 * the workspace null until it loaded. Every other style and state clears it. It keeps one tree shape,
 * so the page below never remounts on a switch (M69). The SDK side (keys, values, blocked storage,
 * other tabs) is proved by lib/theme-sdk/__tests__/storage.test.tsx; the browser lane proves a real
 * package's state survives a reload (e2e/browser/specs/theme-storage.spec.ts).
 *
 * The active theme, the user, the edition and the workspace are stubbed here, so one mount can walk
 * Classic → Studio → an installed theme → Studio (a real switch to an installed theme reloads, PA-13);
 * the theme host's status comes through its real context.
 */
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'
import { useEffect, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { InstalledThemeEntryData } from '@/lib/theme-host/boot-types'
import type { ThemeEntry, UiStyle } from '@/lib/theme-host/registry-types'

interface UserStub {
  readonly isLoaded: boolean
  readonly isSignedIn: boolean | undefined
  readonly user: { readonly id: string } | null | undefined
}

interface WorkspaceStub {
  readonly workspace: { readonly id: string } | null
  readonly isLoading: boolean
  readonly error: Error | null
  readonly refreshWorkspace: () => Promise<void>
}

const harness = vi.hoisted(() => ({
  theme: null as ThemeEntry | null,
  local: false,
  user: null as UserStub | null,
  workspace: null as WorkspaceStub | null,
  pageMounts: 0,
}))

vi.mock('@/hooks/use-studio-theme', () => ({ useActiveTheme: () => harness.theme }))
vi.mock('@/lib/auth-hooks', () => ({ useUser: () => harness.user }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => harness.workspace }))
vi.mock('@/lib/auth-edition', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/auth-edition')>()),
  get isLocal() {
    return harness.local
  },
  get isSaaS() {
    return !harness.local
  },
}))

import { moduleSpecifiers } from '@/lib/__tests__/module-references'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import {
  HELLO_PACKAGE_ENTRY,
  INSTALLED_HELLO_STYLE,
  INSTALLED_PLAY_STYLE,
  PLAY_PACKAGE_ENTRY,
} from '@/lib/theme-host/__tests__/installed-entry-fixture'
import type { ThemeHostStatus } from '@/lib/theme-host/types'
import { currentThemeStateScope, resetThemeStateRuntimeForTests } from '@/lib/theme-sdk/internal/theme-state-runtime'
import { THEME_STATE_KEY_PREFIX } from '@/lib/theme-sdk/internal/theme-state-store'
import { readThemeState, subscribeThemeState, useThemeState, writeThemeState } from '@/lib/theme-sdk/storage'

import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'
import { LOCAL_EDITION_USER_KEY, ThemeStateScope } from '../theme-state-scope'

const FRONTEND = path.resolve(__dirname, '../../..')
const SCOPE_FILE = 'components/theme-host/theme-state-scope.tsx'
const LOOK_RUNTIME_FILE = 'components/theme-host/theme-look-runtime.tsx'
const PROVIDERS_FILE = 'components/providers.tsx'
/** Look code and the builtin theme's files: the scope is the host's, for every installed theme alike. */
const FORBIDDEN_DIRS = ['lib/looks', 'lib/theme-host/builtin']
const RUNTIME_MODULE = '@/lib/theme-sdk/internal/theme-state-runtime'

const USER = 'user_a'
const OTHER_USER = 'user_b'
const WORKSPACE = 'ws-1'
const KEY = 'greeting'

function entryOf(style: UiStyle, data: InstalledThemeEntryData): ThemeEntry {
  const entry = installedThemeEntry(style, data)
  if (entry === null) throw new Error(`no installed entry for ${style}`)
  return entry
}

const HELLO = entryOf(INSTALLED_HELLO_STYLE, HELLO_PACKAGE_ENTRY)
const PLAY = entryOf(INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY)
const HELLO_ID = HELLO_PACKAGE_ENTRY.themeId
const PLAY_ID = PLAY_PACKAGE_ENTRY.themeId
/** A builtin theme's entry (builtin Play's kind): the host never runs it, whatever it is called. */
const BUILTIN: ThemeEntry = { ...HELLO, kind: 'builtin' }
const NO_THEME_ID: ThemeEntry = { ...HELLO, themeId: undefined }

const SIGNED_IN: UserStub = { isLoaded: true, isSignedIn: true, user: { id: USER } }
/** lib/auth-hooks.ts in the local edition: signed in, no user. */
const LOCAL_USER: UserStub = { isLoaded: true, isSignedIn: true, user: null }
const refreshWorkspace = async (): Promise<void> => undefined
const LOADED: WorkspaceStub = { workspace: { id: WORKSPACE }, isLoading: false, error: null, refreshWorkspace }
const LOADING: WorkspaceStub = { workspace: null, isLoading: true, error: null, refreshWorkspace }
const FAILED: WorkspaceStub = { workspace: null, isLoading: false, error: new Error('Failed to fetch workspace'), refreshWorkspace }

function accountBlobKey(themeId: string, userKey: string = USER): string {
  return `${THEME_STATE_KEY_PREFIX}b:${themeId}:u:${userKey}`
}

function workspaceBlobKey(themeId: string): string {
  return `${THEME_STATE_KEY_PREFIX}b:${themeId}:w:${WORKSPACE}:u:${USER}`
}

function blob(data: Record<string, unknown>): string {
  return JSON.stringify({ v: 1, at: Date.now(), data })
}

function hostValue(status: ThemeHostStatus): ThemeHostContextValue {
  return { status, reason: status === 'failed' ? 'load' : null, frame: null, slots: null, reportRenderFailure: () => undefined }
}

/** A page below the scope: it counts its mounts, keeps a click count in its own state and reads one key. */
function Page() {
  const [clicks, setClicks] = useState(0)
  const state = useThemeState(KEY)
  useEffect(() => {
    harness.pageMounts += 1
  }, [])
  return (
    <div>
      <button type="button" onClick={() => setClicks((count) => count + 1)}>Clicks: {clicks}</button>
      <p data-testid="state">{state.ready ? `ready:${JSON.stringify(state.value)}` : 'not ready'}</p>
    </div>
  )
}

/** The theme host's context above the scope and the page below it, as components/providers.tsx nests them. */
function Host({ status }: { readonly status: ThemeHostStatus }) {
  return (
    <ThemeHostContext.Provider value={hostValue(status)}>
      <ThemeStateScope>
        <Page />
      </ThemeStateScope>
    </ThemeHostContext.Provider>
  )
}

function show(theme: ThemeEntry | null, status: ThemeHostStatus) {
  harness.theme = theme
  return render(<Host status={status} />)
}

function shownState(): HTMLElement {
  return screen.getByTestId('state')
}

/** A write inside act(), so the page's re-render it causes lands before the next assertion. */
function write(key: string, value: string, within?: 'account' | 'workspace'): ReturnType<typeof writeThemeState> {
  const results: ReturnType<typeof writeThemeState>[] = []
  act(() => {
    results.push(writeThemeState(key, value, within === undefined ? undefined : { within }))
  })
  return results[0]
}

function inOrder(source: string, order: readonly string[]): void {
  const at = order.map((tag) => source.indexOf(tag))
  expect(at.every((index) => index >= 0), order.join(' … ')).toBe(true)
  expect([...at].sort((a, b) => a - b)).toEqual(at)
}

/** A specifier as a path under frontend/ (`@/…` or relative); null for a package. */
function frontendPath(specifier: string, file: string): string | null {
  if (specifier.startsWith('@/')) return path.posix.normalize(specifier.slice(2))
  if (!specifier.startsWith('.')) return null
  const target = path.resolve(FRONTEND, path.dirname(file), specifier)
  return path.relative(FRONTEND, target).split(path.sep).join('/')
}

beforeEach(() => {
  resetThemeStateRuntimeForTests()
  window.localStorage.clear()
  window.sessionStorage.clear()
  harness.theme = null
  harness.local = false
  harness.user = SIGNED_IN
  harness.workspace = LOADED
  harness.pageMounts = 0
})

afterEach(() => {
  cleanup()
  resetThemeStateRuntimeForTests()
  window.localStorage.clear()
  window.sessionStorage.clear()
})

describe('PRD-261 E3c — the scope is published for an installed theme the host runs', () => {
  it.each<ThemeHostStatus>(['loading', 'active'])('host %s: the theme id, the user and the workspace', (status) => {
    show(HELLO, status)
    expect(currentThemeStateScope()).toEqual({ themeId: HELLO_ID, userKey: USER, workspaceKey: WORKSPACE })
    expect(shownState()).toHaveTextContent('ready:null')
    expect(write(KEY, 'hello')).toEqual({ ok: true, persisted: true })
    expect(shownState()).toHaveTextContent('ready:"hello"')
    expect(JSON.parse(window.localStorage.getItem(accountBlobKey(HELLO_ID)) ?? 'null')).toMatchObject({ v: 1, data: { [KEY]: 'hello' } })
  })

  it.each<[string, ThemeEntry | null, ThemeHostStatus]>([
    ['Classic or Studio (no registry entry)', null, 'off'],
    ['a builtin theme, even with the host running', BUILTIN, 'active'],
    ['an installed entry with no theme id', NO_THEME_ID, 'active'],
    ['an installed theme whose load failed', HELLO, 'failed'],
    ['an installed theme in safe mode', HELLO, 'safe-mode'],
    ['an installed theme the host has not started', HELLO, 'off'],
  ])('%s: no scope, so the SDK reads not ready and writes nothing', (_label, theme, status) => {
    show(theme, status)
    expect(currentThemeStateScope()).toBeNull()
    expect(readThemeState(KEY).ready).toBe(false)
    expect(write(KEY, 'hello')).toMatchObject({ ok: false, reason: 'not-ready' })
    expect(shownState()).toHaveTextContent('not ready')
    expect(window.localStorage.length).toBe(0)
  })

  it('outside the theme host (an isolated mount): no scope', () => {
    harness.theme = HELLO
    render(
      <ThemeStateScope>
        <Page />
      </ThemeStateScope>,
    )
    expect(currentThemeStateScope()).toBeNull()
    expect(shownState()).toHaveTextContent('not ready')
  })

  it.each<[string, ThemeEntry | null, ThemeHostStatus]>([
    ['a switch to Studio', null, 'off'],
    ['safe mode', HELLO, 'safe-mode'],
    ['a failed load', HELLO, 'failed'],
  ])('%s clears the scope: the page reads not ready, and the stored state stays', (_label, theme, status) => {
    const { rerender } = show(HELLO, 'active')
    write(KEY, 'hello')
    expect(shownState()).toHaveTextContent('ready:"hello"')
    harness.theme = theme
    rerender(<Host status={status} />)
    expect(currentThemeStateScope()).toBeNull()
    expect(shownState()).toHaveTextContent('not ready')
    expect(window.localStorage.getItem(accountBlobKey(HELLO_ID))).not.toBeNull()
  })

  it('leaves with the host: unmounting clears the scope', () => {
    const { unmount } = show(HELLO, 'active')
    expect(currentThemeStateScope()).not.toBeNull()
    unmount()
    expect(currentThemeStateScope()).toBeNull()
  })
})

describe('PRD-261 E3c — the workspace part', () => {
  it.each<[string, WorkspaceStub | null]>([
    ['failed to load', FAILED],
    ['still loading', LOADING],
    ['outside a WorkspaceProvider', null],
  ])('a workspace %s: account state is ready, workspace state is not', (_label, workspace) => {
    harness.workspace = workspace
    show(HELLO, 'active')
    expect(currentThemeStateScope()).toEqual({ themeId: HELLO_ID, userKey: USER, workspaceKey: null })
    expect(readThemeState(KEY).ready).toBe(true)
    expect(readThemeState(KEY, { within: 'workspace' }).ready).toBe(false)
    expect(write(KEY, 'hello', 'workspace')).toMatchObject({ ok: false, reason: 'not-ready' })
    expect(write(KEY, 'hello', 'account')).toEqual({ ok: true, persisted: true })
  })

  it('the workspace arriving makes workspace state ready, with its stored value, and tells its subscribers', () => {
    window.localStorage.setItem(workspaceBlobKey(HELLO_ID), blob({ [KEY]: 'kept in ws-1' }))
    harness.workspace = FAILED
    const { rerender } = show(HELLO, 'active')
    const listener = vi.fn()
    const unsubscribe = subscribeThemeState(KEY, listener, { within: 'workspace' })
    expect(readThemeState(KEY, { within: 'workspace' }).ready).toBe(false)

    harness.workspace = LOADED
    rerender(<Host status="active" />)
    expect(currentThemeStateScope()?.workspaceKey).toBe(WORKSPACE)
    expect(listener).toHaveBeenCalledTimes(1)
    expect(readThemeState(KEY, { within: 'workspace' })).toMatchObject({ ready: true, value: 'kept in ws-1' })
    unsubscribe()
  })
})

describe('PRD-261 E3c — each theme and each user has its own state', () => {
  it("another installed theme gets its own theme id and cannot read the first theme's keys", () => {
    const { rerender } = show(HELLO, 'active')
    write(KEY, "hello's")
    harness.theme = PLAY
    rerender(<Host status="active" />)
    expect(currentThemeStateScope()).toEqual({ themeId: PLAY_ID, userKey: USER, workspaceKey: WORKSPACE })
    expect(readThemeState(KEY)).toMatchObject({ ready: true, value: null })
    expect(shownState()).toHaveTextContent('ready:null')
    expect(window.localStorage.getItem(accountBlobKey(PLAY_ID))).toBeNull()

    harness.theme = HELLO
    rerender(<Host status="active" />)
    expect(shownState()).toHaveTextContent(`ready:"hello's"`)
  })

  it('the local edition: its one user is `local`', () => {
    harness.local = true
    harness.user = LOCAL_USER
    show(HELLO, 'active')
    expect(currentThemeStateScope()).toEqual({ themeId: HELLO_ID, userKey: LOCAL_EDITION_USER_KEY, workspaceKey: WORKSPACE })
    expect(LOCAL_EDITION_USER_KEY).toBe('local')
    write(KEY, 'local hello')
    expect(window.localStorage.getItem(accountBlobKey(HELLO_ID, 'local'))).not.toBeNull()
  })

  it.each<[string, UserStub]>([
    ['Clerk still loading', { isLoaded: false, isSignedIn: undefined, user: undefined }],
    ['signed out', { isLoaded: true, isSignedIn: false, user: null }],
    ['signed in with no user id, in the hosted edition', LOCAL_USER],
  ])('%s: no scope', (_label, user) => {
    harness.user = user
    show(HELLO, 'active')
    expect(currentThemeStateScope()).toBeNull()
    expect(shownState()).toHaveTextContent('not ready')
  })

  it("another user signs in: the scope follows, and the first user's state stays theirs", () => {
    const { rerender } = show(HELLO, 'active')
    write(KEY, "user_a's")
    harness.user = { isLoaded: true, isSignedIn: true, user: { id: OTHER_USER } }
    rerender(<Host status="active" />)
    expect(currentThemeStateScope()?.userKey).toBe(OTHER_USER)
    expect(shownState()).toHaveTextContent('ready:null')
    expect(window.localStorage.getItem(accountBlobKey(HELLO_ID))).not.toBeNull()
  })
})

describe('PRD-261 E3c — one tree shape (M69)', () => {
  it('Classic → Studio → an installed theme → Studio: the page below is never remounted', () => {
    // Classic and Studio have no registry entry, and the host is off under both.
    const { rerender } = show(null, 'off')
    fireEvent.click(screen.getByRole('button', { name: 'Clicks: 0' }))
    expect(shownState()).toHaveTextContent('not ready')
    rerender(<Host status="off" />)

    harness.theme = HELLO
    rerender(<Host status="loading" />)
    rerender(<Host status="active" />)
    expect(currentThemeStateScope()?.themeId).toBe(HELLO_ID)
    expect(shownState()).toHaveTextContent('ready:null')

    harness.theme = null
    rerender(<Host status="off" />)
    expect(currentThemeStateScope()).toBeNull()
    expect(shownState()).toHaveTextContent('not ready')
    expect(harness.pageMounts).toBe(1)
    expect(screen.getByRole('button', { name: 'Clicks: 1' })).toBeInTheDocument()
  })
})

describe('PRD-261 E3c — where the scope lives', () => {
  it('imports no look code and no builtin theme', () => {
    const specifiers = moduleSpecifiers(SCOPE_FILE, readFileSync(path.join(FRONTEND, SCOPE_FILE), 'utf8'))
    // Not vacuous: the scan reads the file's real imports.
    expect(specifiers).toContain(RUNTIME_MODULE)
    const forbidden = specifiers.filter((specifier) => {
      const target = frontendPath(specifier, SCOPE_FILE)
      return target !== null && FORBIDDEN_DIRS.some((dir) => target === dir || target.startsWith(`${dir}/`))
    })
    expect(forbidden).toEqual([])
  })

  it('the host look runtime renders every page through it, inside the theme host and the workspace', () => {
    const runtime = readFileSync(path.join(FRONTEND, LOOK_RUNTIME_FILE), 'utf8')
    // The runtime renders its children for every style (one shape, M69), and only through the scope.
    expect(runtime.split('{children}').length - 1).toBe(1)
    expect(runtime).toContain('<ThemeStateScope>{children}</ThemeStateScope>')
    inOrder(readFileSync(path.join(FRONTEND, PROVIDERS_FILE), 'utf8'), [
      '<WorkspaceProvider>',
      '<ThemeHostProvider>',
      '<ThemeLookRuntime onToneChange={setPlayTone}>',
      '</ThemeLookRuntime>',
      '</ThemeHostProvider>',
      '</WorkspaceProvider>',
    ])
  })
})
