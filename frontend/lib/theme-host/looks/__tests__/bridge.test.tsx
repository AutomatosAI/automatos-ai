/**
 * PRD-261 A-19d2 — the SDK's `looks` area on the generic look runtime (./bridge.tsx), hello-a09b's
 * boot on the page: useLook() reads the runtime through context and the plain functions reach it as
 * the current one; every look a theme hands in passes the theme's checks first (a look outside its
 * knobs, or over a preset, is refused and never kept); what the SDK hands out is a frozen copy;
 * Export writes the look as its checks keep it and Import gives a fresh id. While no installed theme
 * with looks runs, the bridge hands on whatever runtime is above it (none, in the host's tree) and
 * registers nothing.
 */
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import { ThemeLookProvider } from '@/lib/theme-host/looks/provider'
import { lookKeys, type StorageSource } from '@/lib/theme-host/looks/storage'
import { lookFile } from '@/lib/theme-host/looks/transfer'
import { currentLookRuntime, LookRuntimeContext, registerLookRuntime, type HostLookRuntime } from '@/lib/theme-sdk/internal/look-runtime'
import { exportLook, importLook, listLooks, saveLook, useLook, type HostLook, type HostLookResult, type HostLookState } from '@/lib/theme-sdk/looks'

import { DUSK, HELLO, HELLO_RULES, HELLO_THEME_ID, helloThemeBoot, memoryStorage, myLook, PAPER, putPageBoot } from './hello-looks'

const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a')
const KEYS = lookKeys(SCOPE, HELLO.keys)
const FORGED = { ...myLook(), accent: 'url(https://evil.example/x.png)' } as unknown as HostLook

const seen: { current: HostLookState | null } = { current: null }

function Probe() {
  const state = useLook()
  seen.current = state
  return (
    <div>
      <p>Looks: {state.available ? 'available' : 'none'}</p>
      <p>Look: {state.look?.id ?? 'none'}</p>
      <p>Saved: {state.saved.map((look) => look.name).join(', ') || 'none'}</p>
    </div>
  )
}

function stored(store: Storage, key: string): unknown {
  const raw = store.getItem(key)
  return raw === null ? null : JSON.parse(raw)
}

function renderRuntime(store: Storage, enabled = true) {
  const storage: StorageSource = () => store
  return render(
    <ThemeLookProvider themeId={HELLO_THEME_ID} enabled={enabled} tonePolicy="from-look" scope={SCOPE} storage={storage} server={null}>
      <Probe />
    </ThemeLookProvider>,
  )
}

async function loadedRuntime(store: Storage = memoryStorage()) {
  const rendered = renderRuntime(store)
  await waitFor(() => expect(seen.current?.loaded).toBe(true))
  return rendered
}

function call(action: () => HostLookResult): HostLookResult | undefined {
  const answer: { result?: HostLookResult } = {}
  act(() => {
    answer.result = action()
  })
  return answer.result
}

let removeBoot: () => void = () => undefined

beforeEach(() => {
  seen.current = null
  removeBoot = putPageBoot(helloThemeBoot())
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  removeBoot()
  vi.restoreAllMocks()
  document.documentElement.removeAttribute('style')
})

describe("useLook on the generic runtime", () => {
  it("hands out the theme's look as a frozen copy, with the tone its look forces", async () => {
    await loadedRuntime()
    expect(screen.getByText('Looks: available')).toBeInTheDocument()
    const state = seen.current
    expect(state).toMatchObject({ available: true, look: PAPER, saved: [], loaded: true, persisted: true, tone: 'light' })
    expect(Object.isFrozen(state)).toBe(true)
    expect(Object.isFrozen(state?.look)).toBe(true)
  })

  it("refuses a look outside the theme's knobs and keeps the active look; takes a valid one", async () => {
    const store = memoryStorage()
    await loadedRuntime(store)
    const refused = call(() => seen.current?.setLook({ ...DUSK, ground: 'graphite' } as HostLook) ?? { ok: false, errors: [] })
    expect(refused?.ok).toBe(false)
    expect(screen.getByText('Look: paper')).toBeInTheDocument()
    expect(call(() => seen.current?.setLook(DUSK) ?? { ok: false, errors: [] })).toMatchObject({ ok: true, look: { id: 'dusk' } })
    expect(await screen.findByText('Look: dusk')).toBeInTheDocument()
    await waitFor(() => expect(stored(store, KEYS.active)).toEqual(DUSK))
    expect(seen.current?.tone).toBe('dark')
  })

  it('renames and deletes a saved look, never a preset', async () => {
    await loadedRuntime(memoryStorage({ [KEYS.saved]: JSON.stringify([myLook()]) }))
    expect(screen.getByText('Saved: My look 1')).toBeInTheDocument()
    expect(call(() => seen.current?.renameLook('paper', 'Mine') ?? { ok: false, errors: [] })?.ok).toBe(false)
    expect(call(() => seen.current?.renameLook('my-look-1', 'Night') ?? { ok: false, errors: [] })?.ok).toBe(true)
    expect(await screen.findByText('Saved: Night')).toBeInTheDocument()
    act(() => seen.current?.deleteLook('my-look-1'))
    expect(await screen.findByText('Saved: none')).toBeInTheDocument()
  })
})

describe('the plain functions on the generic runtime', () => {
  it("save a checked look, list it, and refuse a forged one or a preset's id", async () => {
    const store = memoryStorage()
    await loadedRuntime(store)
    expect(call(() => saveLook(FORGED))?.ok).toBe(false)
    expect(call(() => saveLook({ ...myLook(), id: PAPER.id }))?.ok).toBe(false)
    expect(call(() => saveLook(myLook()))).toMatchObject({ ok: true, look: { id: 'my-look-1' } })
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
    await waitFor(() => expect(listLooks().map((look) => look.id)).toEqual(['my-look-1']))
    expect(Object.isFrozen(listLooks()[0])).toBe(true)
    await waitFor(() => expect(stored(store, KEYS.saved)).toEqual([myLook()]))
  })

  it("export the look as its checks keep it, and import one under a fresh id", async () => {
    await loadedRuntime()
    expect(exportLook(DUSK)).toEqual(lookFile(HELLO_RULES, DUSK))
    expect(exportLook(FORGED)).toBeNull()
    const imported = importLook(JSON.stringify(DUSK))
    expect(imported).toMatchObject({ ok: true, look: { name: DUSK.name } })
    expect(imported.ok && imported.look.id).not.toBe(DUSK.id)
    expect(importLook('not json').ok).toBe(false)
  })
})

describe('while no installed theme with looks runs here', () => {
  /** A runtime mounted above this one: the bridge hands it on unchanged rather than hide it. */
  function fakeRuntime(): HostLookRuntime {
    const refused: HostLookResult = { ok: false, errors: ['Not this runtime.'] }
    const state: HostLookState = {
      available: true,
      look: null,
      saved: [],
      loaded: true,
      persisted: true,
      tone: null,
      setLook: () => refused,
      deleteLook: () => undefined,
      renameLook: () => refused,
    }
    return { state, saveLook: () => refused, exportLook: () => null, importLook: () => refused }
  }

  it('hands on the runtime above it and registers nothing', async () => {
    const above = fakeRuntime()
    const unregister = registerLookRuntime(above)
    render(
      <LookRuntimeContext.Provider value={above}>
        <ThemeLookProvider themeId={null} enabled={false} tonePolicy="user" scope={SCOPE} server={null}>
          <Probe />
        </ThemeLookProvider>
      </LookRuntimeContext.Provider>,
    )
    expect(seen.current).toBe(above.state)
    expect(currentLookRuntime()).toBe(above)
    unregister()
  })

  it('answers no looks with nothing above it', () => {
    renderRuntime(memoryStorage(), false)
    expect(screen.getByText('Looks: none')).toBeInTheDocument()
    expect(currentLookRuntime()).toBeNull()
    expect(saveLook(myLook()).ok).toBe(false)
  })
})
