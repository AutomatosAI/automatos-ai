/**
 * PRD-261 A-19d1 — the generic look store's own rules, on hello-a09b's data, with no server: every
 * look handed in passes the theme's checks first (a look outside the knobs is refused, nothing
 * changes), the theme's presets cannot be saved over, renamed or deleted, the store hands out only
 * the running theme's looks, another tab's change arrives through storage (M25), and with no theme
 * with looks the store is inert.
 */
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import { LOOKS_NOT_AVAILABLE, NOT_SAVED, PRESET_REFUSED, useLookStore } from '@/lib/theme-host/looks/store'
import { lookKeys, lookTheme, type LookTheme, type StorageSource } from '@/lib/theme-host/looks/storage'

import { DUSK, HELLO, HELLO_RULES, INK, memoryStorage, myLook, PAPER } from './hello-looks'

const SCOPE = lookScope('ws-1', 'user_a')
const KEYS = lookKeys(SCOPE, HELLO.keys)
const OTHER = lookTheme('acme/hello', HELLO_RULES)

function stored(store: Storage, key: string): unknown {
  const raw = store.getItem(key)
  return raw === null ? null : JSON.parse(raw)
}

function renderStore(store: Storage, theme: LookTheme | null = HELLO, scope: string | null = SCOPE) {
  const storage: StorageSource = () => store
  return renderHook(({ given }: { readonly given: LookTheme | null }) => useLookStore(storage, scope, given, { server: null }), {
    initialProps: { given: theme },
  })
}

async function loadedStore(store: Storage, theme: LookTheme | null = HELLO) {
  const rendered = renderStore(store, theme)
  await waitFor(() => expect(rendered.result.current.loaded).toBe(true))
  return rendered
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('before the looks are read', () => {
  it("answers the theme's default look, nothing saved, not loaded, until the account is known", () => {
    const { result } = renderStore(memoryStorage({ [KEYS.active]: JSON.stringify(DUSK) }), HELLO, null)
    expect(result.current).toMatchObject({ look: PAPER, saved: [], loaded: false, scope: null })
  })
})

describe('a look handed in', () => {
  it("is refused when it is outside the theme's knobs, and nothing changes", async () => {
    const store = memoryStorage()
    const { result } = await loadedStore(store)
    let answer: ReturnType<typeof result.current.setLook> | undefined
    act(() => {
      answer = result.current.setLook({ ...DUSK, ground: 'graphite' })
    })
    expect(answer).toEqual({ ok: false, errors: ['"ground" must be one of its choices: paper, ink, dusk.'] })
    act(() => {
      answer = result.current.saveLook({ ...myLook(), glow: 99 })
    })
    expect(answer?.ok).toBe(false)
    expect(result.current).toMatchObject({ look: PAPER, saved: [] })
    expect(store.getItem(KEYS.active)).toBeNull()
    expect(store.getItem(KEYS.saved)).toBeNull()
  })

  it("is kept as the theme's checks keep it (trimmed, only its fields)", async () => {
    const store = memoryStorage()
    const { result } = await loadedStore(store)
    act(() => void result.current.saveLook({ ...myLook(), name: '  Night  ' }))
    expect(result.current.saved).toEqual([myLook({ name: 'Night' })])
    expect(stored(store, KEYS.saved)).toEqual([myLook({ name: 'Night' })])
  })
})

describe("the theme's presets", () => {
  it('cannot be saved over, renamed or deleted', async () => {
    const store = memoryStorage()
    const { result } = await loadedStore(store)
    act(() => void result.current.setLook(DUSK))
    expect(result.current.saveLook({ ...DUSK, glow: 0 })).toBe(PRESET_REFUSED)
    expect(result.current.renameLook('dusk', 'Midnight')).toBe(PRESET_REFUSED)
    act(() => result.current.deleteLook('dusk'))
    expect(result.current).toMatchObject({ look: DUSK, saved: [] })
    expect(stored(store, KEYS.active)).toEqual(DUSK)
    expect(store.getItem(KEYS.saved)).toBeNull()
  })

  it('a saved look can be renamed and deleted; deleting the active one goes back to the default', async () => {
    const store = memoryStorage()
    const { result } = await loadedStore(store)
    act(() => void result.current.saveLook(myLook()))
    act(() => void result.current.setLook(myLook()))
    expect(result.current.renameLook('my-look-2', 'Nope')).toBe(NOT_SAVED)
    act(() => void result.current.renameLook('my-look-1', 'Sunset'))
    expect(result.current.look?.name).toBe('Sunset')
    expect(result.current.renameLook('my-look-1', '   ')).toMatchObject({ ok: false })
    act(() => result.current.deleteLook('my-look-1'))
    expect(result.current).toMatchObject({ look: PAPER, saved: [] })
    expect(stored(store, KEYS.active)).toEqual(PAPER)
  })
})

describe('which looks it hands out', () => {
  it("only the running theme's: a switch reads the other theme's own, and back", async () => {
    const store = memoryStorage({ [KEYS.active]: JSON.stringify(DUSK), [lookKeys(SCOPE, OTHER.keys).active]: JSON.stringify(INK) })
    const { result, rerender } = await loadedStore(store)
    expect(result.current.look).toEqual(DUSK)
    rerender({ given: OTHER })
    await waitFor(() => expect(result.current.look).toEqual(INK))
    rerender({ given: HELLO })
    await waitFor(() => expect(result.current.look).toEqual(DUSK))
  })

  it("takes another tab's change to this scope's looks (M25)", async () => {
    const store = memoryStorage()
    const { result } = await loadedStore(store)
    store.setItem(KEYS.active, JSON.stringify(INK))
    act(() => {
      window.dispatchEvent(new StorageEvent('storage', { key: KEYS.active }))
    })
    expect(result.current.look).toEqual(INK)
  })
})

describe('with no theme with looks', () => {
  it('is inert: no look, nothing read or written, every action refused', async () => {
    const store = memoryStorage({ [KEYS.active]: JSON.stringify(DUSK) })
    const getItem = vi.spyOn(store, 'getItem')
    const setItem = vi.spyOn(store, 'setItem')
    const { result } = renderStore(store, null)
    await act(async () => undefined)
    expect(result.current).toMatchObject({ look: null, saved: [], loaded: false, scope: null })
    expect(result.current.setLook(DUSK)).toBe(LOOKS_NOT_AVAILABLE)
    expect(result.current.saveLook(myLook())).toBe(LOOKS_NOT_AVAILABLE)
    expect(result.current.renameLook('my-look-1', 'x')).toBe(LOOKS_NOT_AVAILABLE)
    expect(getItem).not.toHaveBeenCalled()
    expect(setItem).not.toHaveBeenCalled()
  })

  it('forgets what it read when switched off, and reads again when switched back on', async () => {
    const store = memoryStorage({ [KEYS.active]: JSON.stringify(DUSK) })
    const { result, rerender } = await loadedStore(store)
    rerender({ given: null })
    await waitFor(() => expect(result.current.look).toBeNull())
    store.setItem(KEYS.active, JSON.stringify(INK))
    rerender({ given: HELLO })
    await waitFor(() => expect(result.current.look).toEqual(INK))
  })
})
