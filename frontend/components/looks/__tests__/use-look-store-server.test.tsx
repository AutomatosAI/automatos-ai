/**
 * PRD-261 A-09b — the look store with the server: the server's copy is the truth (read after this
 * browser's, so the page paints at once), this browser's copy is the cache when the server can't
 * be reached, a change is written through to both, and nothing is lost silently: a save the
 * server never got is sent on the next load, and one it refused shows the Looks page's notice.
 * An installed theme keeps its looks under automatos-theme.* keys, builtin Play under its own.
 */
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest'

import { type LookServer, type LookServerBody, LOOK_SEND_DELAY_MS, useLookStore } from '@/components/looks/use-look-store'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { memoryStorage, myLook } from '@/lib/looks/__tests__/helpers'
import { AURORA, DAYLIGHT, OBSIDIAN } from '@/lib/looks/presets'
import { lookScope } from '@/lib/looks/scope'
import { hasLookMark, installedLookTheme, lookKeys, setLookMark, type StorageSource } from '@/lib/looks/storage'
import type { Look } from '@/lib/looks/types'
import { BUILTIN_PLAY_LOOK_KEYS, BUILTIN_PLAY_LOOK_THEME } from '@/lib/theme-host/builtin/play-look-keys'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'

const WORKSPACE = '7d3c2a10-1b2c-4d5e-8f90-123456789abc'
const SCOPE = lookScope(WORKSPACE, 'user_a')
const HELLO = installedLookTheme('acme/hello')
const PLAY_KEYS = lookKeys(SCOPE, BUILTIN_PLAY_LOOK_KEYS)
const HELLO_KEYS = lookKeys(SCOPE, HELLO.keys)
/** The debounce, plus room for the promise callbacks. */
const SENT_WITHIN = { timeout: LOOK_SEND_DELAY_MS + 2000 }
/** A test that waits out the debounce more than once. */
const SLOW_TEST_MS = 10_000

interface FakeServer extends LookServer {
  readonly read: Mock<(themeId: string) => Promise<unknown>>
  readonly write: Mock<(themeId: string, body: LookServerBody) => Promise<unknown>>
}

function answer(active: Look | null, saved: readonly Look[], themeId = 'automatos/play') {
  return { themeId, looks: saved, active, defaultLook: 'ember', dropped: 0 }
}

function fakeServer(read: () => Promise<unknown> = () => Promise.resolve(answer(null, []))): FakeServer {
  return {
    read: vi.fn((_themeId: string) => read()),
    write: vi.fn((_themeId: string, body: LookServerBody) => Promise.resolve(body)),
  }
}

function refusal(status: number): Error {
  return Object.assign(new Error('That look does not fit this theme.'), { status })
}

function stored(store: Storage, key: string): unknown {
  const raw = store.getItem(key)
  return raw === null ? null : JSON.parse(raw)
}

/** A browser that already met the server once: the server's copy is the truth from the start. */
function metServer(store: Storage, keys = BUILTIN_PLAY_LOOK_KEYS): void {
  setLookMark(() => store, SCOPE, keys, 'synced', true)
}

function renderInstalled(store: Storage, server: LookServer | null, scope: string | null = SCOPE) {
  const storage: StorageSource = () => store
  return renderHook(() => useLookStore(storage, scope, { theme: HELLO, server }))
}

function renderPlay(store: Storage, server: LookServer, style: UiStyle = BUILTIN_PLAY_STYLE) {
  const storage: StorageSource = () => store
  const wrapper = ({ children }: { readonly children: ReactNode }) => (
    <UiStyleProvider initialStyle={style}>{children}</UiStyleProvider>
  )
  return renderHook(() => useLookStore(storage, SCOPE, { server }), { wrapper })
}

beforeEach(() => {
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('server first', () => {
  it("paints this browser's copy at once, then takes the server's and caches it", async () => {
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DAYLIGHT) })
    metServer(store, HELLO.keys)
    const server = fakeServer(() => Promise.resolve(answer(OBSIDIAN, [myLook()], 'acme/hello')))
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(result.current.look).toEqual(OBSIDIAN))
    expect(server.read).toHaveBeenCalledWith('acme/hello')
    expect(result.current.saved).toEqual([myLook()])
    expect(stored(store, HELLO_KEYS.active)).toEqual(OBSIDIAN)
    expect(stored(store, HELLO_KEYS.saved)).toEqual([myLook()])
    expect(server.write).not.toHaveBeenCalled()
  })

  it("checks the server's looks as it checks stored ones", async () => {
    const store = memoryStorage()
    metServer(store, HELLO.keys)
    const bad = { ...myLook({ id: 'look-2' }), glow: 99 }
    const server = fakeServer(() => Promise.resolve({ ...answer(AURORA, [myLook()]), looks: [myLook(), bad] }))
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(result.current.look).toEqual(AURORA))
    expect(result.current.saved).toEqual([myLook()])
  })
})

describe('the cache on failure', () => {
  it("keeps this browser's looks when the server can't be reached, and says nothing is lost", async () => {
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DAYLIGHT) })
    const server = fakeServer(() => Promise.reject(new TypeError('Failed to fetch')))
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    await act(async () => undefined)
    expect(result.current).toMatchObject({ look: DAYLIGHT, loaded: true, persisted: true })
    expect(stored(store, HELLO_KEYS.active)).toEqual(DAYLIGHT)
  })

  it("leaves the looks alone when the answer isn't a set of looks", async () => {
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DAYLIGHT) })
    metServer(store, HELLO.keys)
    const server = fakeServer(() => Promise.resolve({ detail: 'Something went wrong.' }))
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    await act(async () => undefined)
    expect(result.current.look).toEqual(DAYLIGHT)
    expect(stored(store, HELLO_KEYS.active)).toEqual(DAYLIGHT)
  })
})

describe('write-through', () => {
  it('writes a change here at once and sends both keys once the changes pause', async () => {
    const store = memoryStorage()
    metServer(store, HELLO.keys)
    const server = fakeServer()
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    act(() => result.current.setLook(AURORA))
    act(() => void result.current.saveLook(myLook()))
    expect(stored(store, HELLO_KEYS.active)).toEqual(AURORA)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(true)
    await waitFor(() => expect(server.write).toHaveBeenCalledTimes(1), SENT_WITHIN)
    expect(server.write).toHaveBeenCalledWith('acme/hello', { looks: [myLook()], active: AURORA })
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false))
  })

  it("keeps a save the server didn't get and sends it on the next load, instead of reading", async () => {
    const store = memoryStorage()
    metServer(store, HELLO.keys)
    const offline = fakeServer()
    offline.write.mockImplementation(() => Promise.reject(new TypeError('Failed to fetch')))
    const first = renderInstalled(store, offline)
    await waitFor(() => expect(offline.read).toHaveBeenCalled())
    act(() => first.result.current.setLook(AURORA))
    await waitFor(() => expect(offline.write).toHaveBeenCalled(), SENT_WITHIN)
    await act(async () => undefined)
    expect(first.result.current.persisted).toBe(true)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(true)
    first.unmount()

    const online = fakeServer(() => Promise.resolve(answer(OBSIDIAN, [], 'acme/hello')))
    const second = renderInstalled(store, online)
    await waitFor(() => expect(online.write).toHaveBeenCalledWith('acme/hello', { looks: [], active: AURORA }))
    expect(online.read).not.toHaveBeenCalled()
    expect(second.result.current.look).toEqual(AURORA)
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false))
  })

  // 403: no person behind the caller (themes_looks.py NO_PERSON) or no members:read; looks routes have no step-up.
  it.each([400, 403])("shows the not-kept notice when the server refuses a save (%i), and doesn't send it again", async (status) => {
    const store = memoryStorage()
    metServer(store, HELLO.keys)
    const server = fakeServer()
    server.write.mockImplementation(() => Promise.reject(refusal(status)))
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    act(() => result.current.setLook(AURORA))
    await waitFor(() => expect(result.current.persisted).toBe(false), SENT_WITHIN)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false)
    expect(stored(store, HELLO_KEYS.active)).toEqual(AURORA)
  })
})

describe('one save at a time', () => {
  /** A server whose first save waits until the test answers it. */
  function slowFirstSave(server: FakeServer): () => void {
    let answerIt: () => void = () => undefined
    server.write.mockImplementationOnce((_themeId, body) => new Promise((resolve) => {
      answerIt = () => resolve(body)
    }))
    return () => answerIt()
  }

  const pause = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, LOOK_SEND_DELAY_MS + 200)))

  it("sends a new change only once the save left from the last load has its answer, so it can't be put back", async () => {
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DAYLIGHT) })
    metServer(store, HELLO.keys)
    setLookMark(() => store, SCOPE, HELLO.keys, 'pending', true)
    const server = fakeServer()
    const answerFirst = slowFirstSave(server)
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(server.write).toHaveBeenCalledWith('acme/hello', { looks: [], active: DAYLIGHT }))
    act(() => result.current.setLook(AURORA))
    await pause()
    expect(server.write).toHaveBeenCalledTimes(1)
    await act(async () => answerFirst())
    await waitFor(() => expect(server.write).toHaveBeenCalledTimes(2))
    expect(server.write).toHaveBeenLastCalledWith('acme/hello', { looks: [], active: AURORA })
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false))
    expect(result.current.persisted).toBe(true)
  }, SLOW_TEST_MS)

  it('sends only the newest of the saves that waited, and keeps the mark until it is kept', async () => {
    const store = memoryStorage()
    metServer(store, HELLO.keys)
    const server = fakeServer()
    const answerFirst = slowFirstSave(server)
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    act(() => result.current.setLook(AURORA))
    await waitFor(() => expect(server.write).toHaveBeenCalledTimes(1), SENT_WITHIN)
    act(() => result.current.setLook(OBSIDIAN))
    await pause()
    act(() => result.current.setLook(DAYLIGHT))
    await pause()
    expect(server.write).toHaveBeenCalledTimes(1)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(true)
    await act(async () => answerFirst())
    await waitFor(() => expect(server.write).toHaveBeenCalledTimes(2))
    expect(server.write).toHaveBeenLastCalledWith('acme/hello', { looks: [], active: DAYLIGHT })
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false))
    expect(server.write).toHaveBeenCalledTimes(2)
  }, SLOW_TEST_MS)
})

describe('the first meeting with the server', () => {
  it("adds this browser's own looks to the server's, so moving to the server loses none", async () => {
    const mine = myLook({ id: 'look-7', name: 'Only here' })
    const store = memoryStorage({ [HELLO_KEYS.saved]: JSON.stringify([mine]), [HELLO_KEYS.active]: JSON.stringify(DAYLIGHT) })
    const server = fakeServer(() => Promise.resolve(answer(OBSIDIAN, [myLook()], 'acme/hello')))
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(result.current.saved).toEqual([myLook(), mine]))
    expect(result.current.look).toEqual(OBSIDIAN)
    await waitFor(() => expect(server.write).toHaveBeenCalledWith('acme/hello', { looks: [myLook(), mine], active: OBSIDIAN }))
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'synced')).toBe(true)
  })

  it('sends nothing when the server already has everything this browser kept', async () => {
    const store = memoryStorage({ [HELLO_KEYS.saved]: JSON.stringify([myLook()]) })
    const server = fakeServer(() => Promise.resolve(answer(OBSIDIAN, [myLook()], 'acme/hello')))
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(result.current.look).toEqual(OBSIDIAN))
    await act(async () => undefined)
    expect(server.write).not.toHaveBeenCalled()
  })
})

describe('which keys and which server id', () => {
  it("an installed theme uses automatos-theme.* keys and never touches Play's", async () => {
    const store = memoryStorage({ [PLAY_KEYS.active]: JSON.stringify(DAYLIGHT) })
    metServer(store, HELLO.keys)
    const server = fakeServer()
    const { result } = renderInstalled(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalledWith('acme/hello'))
    act(() => result.current.setLook(AURORA))
    expect(stored(store, HELLO_KEYS.active)).toEqual(AURORA)
    expect(stored(store, PLAY_KEYS.active)).toEqual(DAYLIGHT)
  })

  it('builtin Play, while it is the style, uses its legacy keys and its package id on the server', async () => {
    const store = memoryStorage({ [PLAY_KEYS.active]: JSON.stringify(DAYLIGHT) })
    metServer(store)
    const server = fakeServer(() => Promise.resolve(answer(DAYLIGHT, [])))
    const { result } = renderPlay(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalledWith(BUILTIN_PLAY_LOOK_THEME.themeId))
    act(() => result.current.setLook(OBSIDIAN))
    expect(stored(store, PLAY_KEYS.active)).toEqual(OBSIDIAN)
    expect(stored(store, HELLO_KEYS.active)).toBeNull()
    await waitFor(() => expect(server.write).toHaveBeenCalledWith('automatos/play', { looks: [], active: OBSIDIAN }), SENT_WITHIN)
  })

  it("asks the server nothing while Play isn't the style, or with no workspace in the scope", async () => {
    const store = memoryStorage({ [PLAY_KEYS.active]: JSON.stringify(DAYLIGHT) })
    const studio = fakeServer()
    const { result } = renderPlay(store, studio, 'studio')
    await waitFor(() => expect(result.current.loaded).toBe(true))
    const noWorkspace = fakeServer()
    const local = renderInstalled(store, noWorkspace, lookScope(null, null))
    await waitFor(() => expect(local.result.current.loaded).toBe(true))
    await act(async () => undefined)
    expect(result.current.look).toEqual(DAYLIGHT)
    expect(studio.read).not.toHaveBeenCalled()
    expect(noWorkspace.read).not.toHaveBeenCalled()
  })

  it('reads the server once for a theme built on every render', async () => {
    const store = memoryStorage()
    metServer(store, HELLO.keys)
    const server = fakeServer(() => Promise.resolve(answer(OBSIDIAN, [], 'acme/hello')))
    const storage: StorageSource = () => store
    const { result, rerender } = renderHook(() =>
      useLookStore(storage, SCOPE, { theme: installedLookTheme('acme/hello'), server }),
    )
    await waitFor(() => expect(result.current.look).toEqual(OBSIDIAN))
    rerender()
    await act(async () => undefined)
    expect(server.read).toHaveBeenCalledTimes(1)
  })

  it('keeps the looks in this browser only when there is no server', async () => {
    const store = memoryStorage()
    const { result } = renderInstalled(store, null)
    await waitFor(() => expect(result.current.loaded).toBe(true))
    act(() => result.current.setLook(AURORA))
    expect(stored(store, HELLO_KEYS.active)).toEqual(AURORA)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false)
  })
})
