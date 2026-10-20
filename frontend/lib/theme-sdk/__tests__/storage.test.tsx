/**
 * PRD-261 E3b — SDK area `storage` (lib/theme-sdk/storage.ts) over the theme-state runtime
 * (internal/theme-state-runtime.ts), plan §2.2.1. The runtime's scope is published here directly, as the
 * host's ThemeStateScope does (E3c, components/theme-host/theme-state-scope.tsx, proves the host side).
 * The store's own rules (keys, values, caps, eviction) are proved by theme-state-store.test.ts.
 */
import { act, renderHook } from '@testing-library/react'
import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  clearThemeStateScope,
  currentThemeStateScope,
  publishThemeStateScope,
  resetThemeStateRuntimeForTests,
  type ThemeStateRuntimeScope,
} from '../internal/theme-state-runtime'
import { THEME_STATE_KEY_PREFIX, THEME_STATE_LIMITS as STORE_LIMITS } from '../internal/theme-state-store'
import {
  readThemeState,
  removeThemeState,
  subscribeThemeState,
  THEME_STATE_LIMITS,
  useThemeState,
  writeThemeState,
  type ThemeStateOptions,
  type ThemeStateValue,
} from '../storage'

const PLAY = 'automatos.play'
const OTHER_THEME = 'acme.hello'
const SCOPE: ThemeStateRuntimeScope = { themeId: PLAY, userKey: 'user_1', workspaceKey: 'ws_1' }
const PAUSED = 'mascot-paused'
const SEEN = 'seen-3f2b8c1e-9a4d-4c2b-8e6f-1a2b3c4d5e6f'
const BROWSER_KEY = `${THEME_STATE_KEY_PREFIX}b:${PLAY}:u:user_1`
const SESSION_KEY = `${THEME_STATE_KEY_PREFIX}s:${PLAY}:u:user_1`
const WORKSPACE_KEY = `${THEME_STATE_KEY_PREFIX}b:${PLAY}:w:ws_1:u:user_1`

/** Spies on every Storage method the runtime could call, localStorage's and sessionStorage's alike. */
function spyOnStorage() {
  return [
    vi.spyOn(Storage.prototype, 'getItem'),
    vi.spyOn(Storage.prototype, 'setItem'),
    vi.spyOn(Storage.prototype, 'removeItem'),
    vi.spyOn(Storage.prototype, 'key'),
  ]
}

function refuse(): never {
  throw new DOMException('The operation is insecure.', 'SecurityError')
}

function blob(data: Record<string, unknown>): string {
  return JSON.stringify({ v: 1, at: Date.now(), data })
}

function storageEvent(key: string | null): void {
  window.dispatchEvent(new StorageEvent('storage', { key }))
}

beforeEach(() => {
  resetThemeStateRuntimeForTests()
  window.localStorage.clear()
  window.sessionStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  resetThemeStateRuntimeForTests()
})

describe('PRD-261 E3b — on the server', () => {
  it('renders useThemeState to a string without throwing, as not ready', () => {
    publishThemeStateScope(SCOPE)
    writeThemeState(PAUSED, true)
    function Probe() {
      const state = useThemeState(PAUSED)
      return createElement('p', null, `${String(state.ready)}:${String(state.value)}`)
    }
    let html = ''
    expect(() => {
      html = renderToString(createElement(Probe))
    }).not.toThrow()
    expect(html).toBe('<p>false:null</p>')
  })

  it('reads not ready, refuses writes as unavailable and subscribes to nothing without a window', () => {
    publishThemeStateScope(SCOPE)
    const storage = spyOnStorage()
    const listener = vi.fn()
    vi.stubGlobal('window', undefined)
    expect(readThemeState(PAUSED)).toMatchObject({ ready: false, value: null })
    expect(writeThemeState(PAUSED, true)).toMatchObject({ ok: false, reason: 'unavailable' })
    expect(removeThemeState(PAUSED)).toMatchObject({ ok: false, reason: 'unavailable' })
    const unsubscribe = subscribeThemeState(PAUSED, listener)
    expect(() => unsubscribe()).not.toThrow()
    vi.unstubAllGlobals()
    publishThemeStateScope({ ...SCOPE, workspaceKey: 'ws_2' })
    expect(listener).not.toHaveBeenCalled()
    storage.forEach((spy) => expect(spy).not.toHaveBeenCalled())
  })
})

describe('PRD-261 E3b — readiness and scope', () => {
  it('with no published runtime reads not ready, refuses not-ready and never touches the browser storage', () => {
    const storage = spyOnStorage()
    expect(readThemeState(PAUSED)).toMatchObject({ ready: false, value: null })
    expect(writeThemeState(PAUSED, true)).toMatchObject({ ok: false, reason: 'not-ready' })
    expect(removeThemeState(PAUSED)).toMatchObject({ ok: false, reason: 'not-ready' })
    const { result } = renderHook(() => useThemeState(PAUSED, { lifetime: 'session' }))
    expect(result.current).toMatchObject({ ready: false, value: null, persisted: false })
    expect(result.current.set(true)).toMatchObject({ ok: false, reason: 'not-ready' })
    subscribeThemeState(PAUSED, vi.fn())()
    storage.forEach((spy) => expect(spy).not.toHaveBeenCalled())
  })

  it('round-trips a value in localStorage under the theme and the account by default', () => {
    publishThemeStateScope(SCOPE)
    expect(writeThemeState(PAUSED, true)).toEqual({ ok: true, persisted: true })
    expect(readThemeState(PAUSED)).toMatchObject({ ready: true, value: true })
    expect(JSON.parse(window.localStorage.getItem(BROWSER_KEY) as string)).toMatchObject({ v: 1, data: { [PAUSED]: true } })
    expect(removeThemeState(PAUSED)).toEqual({ ok: true, persisted: true })
    expect(readThemeState(PAUSED)).toMatchObject({ ready: true, value: null })
    expect(window.localStorage.getItem(BROWSER_KEY)).toBeNull()
  })

  it('reads null for another theme, another user and, within the workspace, another workspace', () => {
    publishThemeStateScope(SCOPE)
    writeThemeState(PAUSED, true)
    writeThemeState(PAUSED, 'in ws_1', { within: 'workspace' })
    for (const other of [{ ...SCOPE, themeId: OTHER_THEME }, { ...SCOPE, userKey: 'user_2' }]) {
      publishThemeStateScope(other)
      expect(readThemeState(PAUSED)).toMatchObject({ ready: true, value: null })
      expect(readThemeState(PAUSED, { within: 'workspace' })).toMatchObject({ ready: true, value: null })
    }
    publishThemeStateScope({ ...SCOPE, workspaceKey: 'ws_2' })
    expect(readThemeState(PAUSED, { within: 'workspace' })).toMatchObject({ ready: true, value: null })
    expect(readThemeState(PAUSED).value).toBe(true)
    publishThemeStateScope(SCOPE)
    expect(readThemeState(PAUSED, { within: 'workspace' }).value).toBe('in ws_1')
    expect(window.localStorage.getItem(WORKSPACE_KEY)).toContain('in ws_1')
  })

  it('keeps account state ready while the workspace is unknown, and workspace state not ready', () => {
    publishThemeStateScope({ ...SCOPE, workspaceKey: null })
    expect(readThemeState(PAUSED).ready).toBe(true)
    expect(readThemeState(PAUSED, { within: 'workspace' }).ready).toBe(false)
    expect(writeThemeState(PAUSED, true, { within: 'workspace' })).toMatchObject({ ok: false, reason: 'not-ready' })
    expect(writeThemeState(PAUSED, true)).toEqual({ ok: true, persisted: true })
  })

  it("drops a scope only by its own cleanup, never by an older one's", () => {
    const first = publishThemeStateScope(SCOPE)
    const second = publishThemeStateScope({ ...SCOPE, themeId: OTHER_THEME })
    first()
    expect(currentThemeStateScope()?.themeId).toBe(OTHER_THEME)
    second()
    expect(currentThemeStateScope()).toBeNull()
    expect(readThemeState(PAUSED).ready).toBe(false)
  })
})

describe('PRD-261 E3b — lifetimes, values and snapshots', () => {
  it('keeps session state in sessionStorage and browser state in localStorage', () => {
    publishThemeStateScope(SCOPE)
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    writeThemeState(PAUSED, 'tab', { lifetime: 'session' })
    writeThemeState(PAUSED, 'browser')
    expect(setItem.mock.calls.map(([key]) => key)).toEqual([SESSION_KEY, BROWSER_KEY])
    expect(setItem.mock.contexts[0]).toBe(window.sessionStorage)
    expect(setItem.mock.contexts[1]).toBe(window.localStorage)
    expect(window.localStorage.getItem(SESSION_KEY)).toBeNull()
    expect(window.sessionStorage.getItem(BROWSER_KEY)).toBeNull()
    expect(readThemeState(PAUSED, { lifetime: 'session' }).value).toBe('tab')
    expect(readThemeState(PAUSED).value).toBe('browser')
  })

  it('refuses a bad key, bad options, a non-JSON value and an over-cap value without throwing, writing nothing', () => {
    publishThemeStateScope(SCOPE)
    const notJson = (() => 1) as unknown as ThemeStateValue
    const badOptions = { lifetime: 'forever' } as unknown as ThemeStateOptions
    expect(writeThemeState('Bad', true)).toMatchObject({ ok: false, reason: 'bad-key' })
    expect(writeThemeState(PAUSED, true, badOptions)).toMatchObject({ ok: false, reason: 'bad-key' })
    expect(writeThemeState(PAUSED, notJson)).toMatchObject({ ok: false, reason: 'bad-value' })
    expect(writeThemeState(PAUSED, 'x'.repeat(THEME_STATE_LIMITS.valueChars))).toMatchObject({ ok: false, reason: 'too-large' })
    expect(readThemeState('Bad')).toMatchObject({ ready: true, value: null })
    expect(readThemeState(PAUSED, badOptions)).toMatchObject({ ready: false, value: null })
    expect(window.localStorage.length).toBe(0)
    expect(THEME_STATE_LIMITS).toBe(STORE_LIMITS)
  })

  it('hands out the same frozen object for an unchanged key, and a new one once it changes', () => {
    publishThemeStateScope(SCOPE)
    writeThemeState(SEEN, { ids: ['a'] })
    const first = readThemeState(SEEN)
    expect(readThemeState(SEEN)).toBe(first)
    writeThemeState(PAUSED, true)
    expect(readThemeState(SEEN)).toBe(first)
    expect(Object.isFrozen(first)).toBe(true)
    expect(Object.isFrozen(first.value)).toBe(true)
    writeThemeState(SEEN, { ids: ['a', 'b'] })
    expect(readThemeState(SEEN)).not.toBe(first)
    expect(readThemeState(SEEN).value).toEqual({ ids: ['a', 'b'] })
    clearThemeStateScope()
    expect(readThemeState(SEEN)).toBe(readThemeState(SEEN))
  })
})

describe('PRD-261 E3b — blocked storage', () => {
  it('keeps state in memory for the page when the storage is blocked, and warns once in all', () => {
    publishThemeStateScope(SCOPE)
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(refuse)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(refuse)
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(writeThemeState(PAUSED, true)).toEqual({ ok: true, persisted: false })
    expect(readThemeState(PAUSED)).toMatchObject({ ready: true, value: true, persisted: false })
    expect(writeThemeState(SEEN, ['a'])).toEqual({ ok: true, persisted: false })
    expect(writeThemeState(PAUSED, 'tab', { lifetime: 'session' })).toEqual({ ok: true, persisted: false })
    expect(readThemeState(PAUSED, { lifetime: 'session' }).value).toBe('tab')
    expect(removeThemeState(SEEN)).toEqual({ ok: true, persisted: false })
    const { result } = renderHook(() => useThemeState(PAUSED))
    expect(result.current).toMatchObject({ ready: true, value: true, persisted: false })
    expect(warn).toHaveBeenCalledTimes(1)
  })

  it('carries what was stored into memory when the storage starts refusing writes (a full quota)', () => {
    publishThemeStateScope(SCOPE)
    writeThemeState(PAUSED, true)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(refuse)
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(writeThemeState(SEEN, ['a'])).toEqual({ ok: true, persisted: false })
    expect(readThemeState(PAUSED)).toMatchObject({ value: true, persisted: false })
    expect(readThemeState(SEEN).value).toEqual(['a'])
  })
})

describe('PRD-261 E3b — subscribers', () => {
  it('tells only the subscribers whose value another tab changed, and every one when storage is cleared', () => {
    publishThemeStateScope(SCOPE)
    writeThemeState(PAUSED, false)
    writeThemeState(SEEN, ['a'])
    const [paused, seen, session] = [vi.fn(), vi.fn(), vi.fn()]
    subscribeThemeState(PAUSED, paused)
    subscribeThemeState(SEEN, seen)
    subscribeThemeState(PAUSED, session, { lifetime: 'session' })
    window.localStorage.setItem(BROWSER_KEY, blob({ [PAUSED]: true, [SEEN]: ['a'] }))
    storageEvent(BROWSER_KEY)
    expect([paused, seen, session].map((listener) => listener.mock.calls.length)).toEqual([1, 0, 0])
    expect(readThemeState(PAUSED).value).toBe(true)
    storageEvent('unrelated-key')
    storageEvent(BROWSER_KEY)
    expect([paused, seen, session].map((listener) => listener.mock.calls.length)).toEqual([1, 0, 0])
    storageEvent(null)
    expect([paused, seen, session].map((listener) => listener.mock.calls.length)).toEqual([2, 1, 1])
  })

  it('tells the subscribers of a key written here, every subscriber on a scope change, and none once unsubscribed', () => {
    publishThemeStateScope(SCOPE)
    const [paused, seen] = [vi.fn(), vi.fn()]
    const unsubscribe = subscribeThemeState(PAUSED, paused)
    subscribeThemeState(SEEN, seen)
    writeThemeState(PAUSED, true)
    expect([paused.mock.calls.length, seen.mock.calls.length]).toEqual([1, 0])
    publishThemeStateScope({ ...SCOPE, themeId: OTHER_THEME })
    expect([paused.mock.calls.length, seen.mock.calls.length]).toEqual([2, 1])
    unsubscribe()
    clearThemeStateScope()
    expect([paused.mock.calls.length, seen.mock.calls.length]).toEqual([2, 2])
  })

  it('re-renders useThemeState when the host publishes a scope and when its key changes', () => {
    const { result } = renderHook(() => useThemeState(PAUSED))
    expect(result.current.ready).toBe(false)
    act(() => {
      publishThemeStateScope(SCOPE)
    })
    expect(result.current).toMatchObject({ ready: true, value: null, persisted: true })
    let written: unknown = null
    act(() => {
      written = result.current.set(true)
    })
    expect(written).toEqual({ ok: true, persisted: true })
    expect(result.current.value).toBe(true)
    act(() => {
      result.current.remove()
    })
    expect(result.current.value).toBeNull()
    act(() => clearThemeStateScope())
    expect(result.current.ready).toBe(false)
  })
})
