/**
 * PRD-261 A-19d1 — ported from components/looks/__tests__/use-look-store-server.test.tsx to the
 * generic look runtime, on hello-a09b's data: the look store with the server. The server's copy is
 * the truth (read after this browser's, so the page paints at once), this browser's copy is the cache
 * when the server can't be reached, a change is written through to both (PUT /api/themes/looks/<id>),
 * and nothing is lost silently: a save the server never got is sent on the next load, and one it
 * refused says the looks aren't kept. A theme keeps its looks under its automatos-theme.* keys, and
 * the server hears its theme id.
 */
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import { LOOK_SEND_DELAY_MS, type LookServer, type LookServerBody } from '@/lib/theme-host/looks/server'
import type { ThemeLook } from '@/lib/theme-host/looks/spec'
import { useLookStore } from '@/lib/theme-host/looks/store'
import { hasLookMark, lookKeys, lookTheme, setLookMark, type LookTheme, type StorageSource } from '@/lib/theme-host/looks/storage'

import { DUSK, HELLO, HELLO_RULES, HELLO_THEME_ID, INK, memoryStorage, myLook, PAPER } from './hello-looks'

const WORKSPACE = '7d3c2a10-1b2c-4d5e-8f90-123456789abc'
const SCOPE = lookScope(WORKSPACE, 'user_a')
const OTHER = lookTheme('acme/hello', HELLO_RULES)
const HELLO_KEYS = lookKeys(SCOPE, HELLO.keys)
const OTHER_KEYS = lookKeys(SCOPE, OTHER.keys)
/** The debounce, plus room for the promise callbacks. */
const SENT_WITHIN = { timeout: LOOK_SEND_DELAY_MS + 2000 }
/** A test that waits out the debounce more than once. */
const SLOW_TEST_MS = 10_000

interface FakeServer extends LookServer {
  readonly read: Mock<(themeId: string) => Promise<unknown>>
  readonly write: Mock<(themeId: string, body: LookServerBody) => Promise<unknown>>
}

function answer(active: ThemeLook | null, saved: readonly ThemeLook[], themeId = HELLO_THEME_ID) {
  return { themeId, looks: saved, active, defaultLook: PAPER, dropped: 0 }
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
function metServer(store: Storage, theme: LookTheme = HELLO): void {
  setLookMark(() => store, SCOPE, theme.keys, 'synced', true)
}

function renderStore(store: Storage, server: LookServer | null, scope: string | null = SCOPE, theme: LookTheme | null = HELLO) {
  const storage: StorageSource = () => store
  return renderHook(() => useLookStore(storage, scope, theme, { server }))
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('server first', () => {
  it("paints this browser's copy at once, then takes the server's and caches it", async () => {
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DUSK) })
    metServer(store)
    const server = fakeServer(() => Promise.resolve(answer(INK, [myLook()])))
    const { result } = renderStore(store, server)
    await waitFor(() => expect(result.current.look).toEqual(INK))
    expect(server.read).toHaveBeenCalledWith(HELLO_THEME_ID)
    expect(result.current.saved).toEqual([myLook()])
    expect(stored(store, HELLO_KEYS.active)).toEqual(INK)
    expect(stored(store, HELLO_KEYS.saved)).toEqual([myLook()])
    expect(server.write).not.toHaveBeenCalled()
  })

  it("checks the server's looks as it checks stored ones", async () => {
    const store = memoryStorage()
    metServer(store)
    const bad = { ...myLook({ id: 'look-2' }), glow: 99 }
    const server = fakeServer(() => Promise.resolve({ ...answer(DUSK, [myLook()]), looks: [myLook(), bad] }))
    const { result } = renderStore(store, server)
    await waitFor(() => expect(result.current.look).toEqual(DUSK))
    expect(result.current.saved).toEqual([myLook()])
  })
})

describe('the cache on failure', () => {
  it("keeps this browser's looks when the server can't be reached, and says nothing is lost", async () => {
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DUSK) })
    const server = fakeServer(() => Promise.reject(new TypeError('Failed to fetch')))
    const { result } = renderStore(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    await act(async () => undefined)
    expect(result.current).toMatchObject({ look: DUSK, loaded: true, persisted: true })
    expect(stored(store, HELLO_KEYS.active)).toEqual(DUSK)
  })

  it("leaves the looks alone when the answer isn't a set of looks", async () => {
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DUSK) })
    metServer(store)
    const server = fakeServer(() => Promise.resolve({ detail: 'Something went wrong.' }))
    const { result } = renderStore(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    await act(async () => undefined)
    expect(result.current.look).toEqual(DUSK)
    expect(stored(store, HELLO_KEYS.active)).toEqual(DUSK)
  })
})

describe('write-through', () => {
  it('writes a change here at once and sends both keys once the changes pause', async () => {
    const store = memoryStorage()
    metServer(store)
    const server = fakeServer()
    const { result } = renderStore(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    act(() => void result.current.setLook(INK))
    act(() => void result.current.saveLook(myLook()))
    expect(stored(store, HELLO_KEYS.active)).toEqual(INK)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(true)
    await waitFor(() => expect(server.write).toHaveBeenCalledTimes(1), SENT_WITHIN)
    expect(server.write).toHaveBeenCalledWith(HELLO_THEME_ID, { looks: [myLook()], active: INK })
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false))
  })

  it("keeps a save the server didn't get and sends it on the next load, instead of reading", async () => {
    const store = memoryStorage()
    metServer(store)
    const offline = fakeServer()
    offline.write.mockImplementation(() => Promise.reject(new TypeError('Failed to fetch')))
    const first = renderStore(store, offline)
    await waitFor(() => expect(offline.read).toHaveBeenCalled())
    act(() => void first.result.current.setLook(INK))
    await waitFor(() => expect(offline.write).toHaveBeenCalled(), SENT_WITHIN)
    await act(async () => undefined)
    expect(first.result.current.persisted).toBe(true)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(true)
    first.unmount()

    const online = fakeServer(() => Promise.resolve(answer(DUSK, [])))
    const second = renderStore(store, online)
    await waitFor(() => expect(online.write).toHaveBeenCalledWith(HELLO_THEME_ID, { looks: [], active: INK }))
    expect(online.read).not.toHaveBeenCalled()
    expect(second.result.current.look).toEqual(INK)
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false))
  })

  // 403: no person behind the caller (themes_looks.py NO_PERSON) or no members:read; looks routes have no step-up.
  it.each([400, 403])("says the looks aren't kept when the server refuses a save (%i), and doesn't send it again", async (status) => {
    const store = memoryStorage()
    metServer(store)
    const server = fakeServer()
    server.write.mockImplementation(() => Promise.reject(refusal(status)))
    const { result } = renderStore(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    act(() => void result.current.setLook(INK))
    await waitFor(() => expect(result.current.persisted).toBe(false), SENT_WITHIN)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false)
    expect(stored(store, HELLO_KEYS.active)).toEqual(INK)
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
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DUSK) })
    metServer(store)
    setLookMark(() => store, SCOPE, HELLO.keys, 'pending', true)
    const server = fakeServer()
    const answerFirst = slowFirstSave(server)
    const { result } = renderStore(store, server)
    await waitFor(() => expect(server.write).toHaveBeenCalledWith(HELLO_THEME_ID, { looks: [], active: DUSK }))
    act(() => void result.current.setLook(INK))
    await pause()
    expect(server.write).toHaveBeenCalledTimes(1)
    await act(async () => answerFirst())
    await waitFor(() => expect(server.write).toHaveBeenCalledTimes(2))
    expect(server.write).toHaveBeenLastCalledWith(HELLO_THEME_ID, { looks: [], active: INK })
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false))
    expect(result.current.persisted).toBe(true)
  }, SLOW_TEST_MS)

  it('sends only the newest of the saves that waited, and keeps the mark until it is kept', async () => {
    const store = memoryStorage()
    metServer(store)
    const server = fakeServer()
    const answerFirst = slowFirstSave(server)
    const { result } = renderStore(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalled())
    act(() => void result.current.setLook(INK))
    await waitFor(() => expect(server.write).toHaveBeenCalledTimes(1), SENT_WITHIN)
    act(() => void result.current.setLook(DUSK))
    await pause()
    act(() => void result.current.setLook(PAPER))
    await pause()
    expect(server.write).toHaveBeenCalledTimes(1)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(true)
    await act(async () => answerFirst())
    await waitFor(() => expect(server.write).toHaveBeenCalledTimes(2))
    expect(server.write).toHaveBeenLastCalledWith(HELLO_THEME_ID, { looks: [], active: PAPER })
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false))
    expect(server.write).toHaveBeenCalledTimes(2)
  }, SLOW_TEST_MS)
})

describe('the first meeting with the server', () => {
  it("adds this browser's own looks to the server's, so moving to the server loses none", async () => {
    const mine = myLook({ id: 'look-7', name: 'Only here' })
    const store = memoryStorage({ [HELLO_KEYS.saved]: JSON.stringify([mine]), [HELLO_KEYS.active]: JSON.stringify(DUSK) })
    const server = fakeServer(() => Promise.resolve(answer(INK, [myLook()])))
    const { result } = renderStore(store, server)
    await waitFor(() => expect(result.current.saved).toEqual([myLook(), mine]))
    expect(result.current.look).toEqual(INK)
    await waitFor(() => expect(server.write).toHaveBeenCalledWith(HELLO_THEME_ID, { looks: [myLook(), mine], active: INK }))
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'synced')).toBe(true)
  })

  it('sends nothing when the server already has everything this browser kept', async () => {
    const store = memoryStorage({ [HELLO_KEYS.saved]: JSON.stringify([myLook()]) })
    const server = fakeServer(() => Promise.resolve(answer(INK, [myLook()])))
    const { result } = renderStore(store, server)
    await waitFor(() => expect(result.current.look).toEqual(INK))
    await act(async () => undefined)
    expect(server.write).not.toHaveBeenCalled()
  })
})

describe('which keys and which server id', () => {
  it("a theme uses its own automatos-theme.* keys and server id, and never touches another theme's", async () => {
    const store = memoryStorage({ [OTHER_KEYS.active]: JSON.stringify(DUSK) })
    metServer(store)
    const server = fakeServer()
    const { result } = renderStore(store, server)
    await waitFor(() => expect(server.read).toHaveBeenCalledWith(HELLO_THEME_ID))
    act(() => void result.current.setLook(INK))
    expect(stored(store, HELLO_KEYS.active)).toEqual(INK)
    expect(stored(store, OTHER_KEYS.active)).toEqual(DUSK)
    await waitFor(() => expect(server.write).toHaveBeenCalledWith(HELLO_THEME_ID, { looks: [], active: INK }), SENT_WITHIN)
    expect(server.write).not.toHaveBeenCalledWith(OTHER.themeId, expect.anything())
  })

  it('asks the server nothing with no workspace in the scope', async () => {
    const store = memoryStorage({ [lookKeys(lookScope(null, null), HELLO.keys).active]: JSON.stringify(DUSK) })
    const server = fakeServer()
    const { result } = renderStore(store, server, lookScope(null, null))
    await waitFor(() => expect(result.current.loaded).toBe(true))
    await act(async () => undefined)
    expect(result.current.look).toEqual(DUSK)
    expect(server.read).not.toHaveBeenCalled()
  })

  it('asks the server nothing while no theme with looks runs', async () => {
    const store = memoryStorage({ [HELLO_KEYS.active]: JSON.stringify(DUSK) })
    metServer(store)
    const server = fakeServer()
    const { result } = renderStore(store, server, SCOPE, null)
    await act(async () => undefined)
    expect(result.current).toMatchObject({ look: null, loaded: false })
    expect(server.read).not.toHaveBeenCalled()
    expect(server.write).not.toHaveBeenCalled()
  })

  it('reads the server once for a theme built on every render', async () => {
    const store = memoryStorage()
    metServer(store)
    const server = fakeServer(() => Promise.resolve(answer(INK, [])))
    const storage: StorageSource = () => store
    const { result, rerender } = renderHook(() => useLookStore(storage, SCOPE, lookTheme(HELLO_THEME_ID, HELLO_RULES), { server }))
    await waitFor(() => expect(result.current.look).toEqual(INK))
    rerender()
    await act(async () => undefined)
    expect(server.read).toHaveBeenCalledTimes(1)
  })

  it('keeps the looks in this browser only when there is no server', async () => {
    const store = memoryStorage()
    const { result } = renderStore(store, null)
    await waitFor(() => expect(result.current.loaded).toBe(true))
    act(() => void result.current.setLook(INK))
    expect(stored(store, HELLO_KEYS.active)).toEqual(INK)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false)
  })
})
