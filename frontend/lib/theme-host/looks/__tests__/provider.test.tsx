/**
 * PRD-261 A-19d2 — the generic look runtime's provider, on hello-a09b's boot as the page carries it:
 * under an installed theme with looks it paints the account's active look on <html> from the
 * theme's own table (the boot script's look until the account's looks are read), puts
 * data-theme-motion and --theme-motion there and never data-pl-motion, and points the next load's
 * boot script at the account's look (F5). It runs only the looks of a boot of its own theme. Off,
 * a failed load or another style takes the table's variables and the motion marks off; while the
 * host loads or runs an installed theme (keepFirstPaint) the boot's first paint stays. A look set
 * here repaints at once and is kept, also on blocked storage (on the default look); another tab's
 * change arrives, its tone with it, and is never written back (M25).
 *
 * PRD-261 A-08c: a page that got the base boot (the server's boot fetch missed its 300 ms, was
 * refused or failed) runs the looks of the install's boot the browser fetched instead: the table's
 * look is painted and the account's looks are read from the server. A page whose boot names the
 * theme never fetches; when the browser's fetch fails too, nothing is painted and one warning says so.
 */
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import { baseThemeBoot, THEME_BOOT_ELEMENT_ID, type ThemeBoot } from '@/lib/theme-host/boot'
import { createInstalledBootSource } from '@/lib/theme-host/boot-client'
import { THEME_MOTION_ATTRIBUTE, THEME_MOTION_VAR } from '@/lib/theme-host/looks/motion'
import { paint, paintNames } from '@/lib/theme-host/looks/paint'
import { ThemeLookProvider, type ThemeLookProviderProps } from '@/lib/theme-host/looks/provider'
import type { LookServer, LookServerBody } from '@/lib/theme-host/looks/server'
import { lookKeys, type StorageSource } from '@/lib/theme-host/looks/storage'
import { clearThemeToneCookie, THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import { importLook, saveLook, useLook, type HostLookState } from '@/lib/theme-sdk/looks'

import { runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'
import { blockedStorage, DUSK, HELLO, HELLO_ENTRY, HELLO_RULES, HELLO_THEME_ID, helloThemeBoot, INK, memoryStorage, myLook, PAPER, putPageBoot } from './hello-looks'
import { cookieValue } from './look-test-helpers'

const TABLE = HELLO_RULES.table
const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a')
const OTHER_SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_b')
/** A theme's own motion mark (Play's name), not a theme-neutral one: the generic runtime never sets it. */
const THEME_OWN_MOTION_ATTRIBUTE = 'data-pl-motion'
const REDUCED_MOTION = '(prefers-reduced-motion: reduce)'
/** A value another look runtime puts on one of the table's names. */
const ANOTHER_RUNTIMES_VALUE = '#123456'
const KEYS = lookKeys(SCOPE, HELLO.keys)
/** A saved look another tab kept, under an id a fresh one must not take. */
const THEIRS = myLook({ id: 'look-1', name: 'Theirs' })
/** The install hello-a09b runs as, and its style. */
const INSTALL_ID = '6e1f3b8f-2d5c-4fab-a4e7-9a2b3d4f5c6e'
const INSTALLED_STYLE = `theme:install:${INSTALL_ID}` as const
const BOOT_LOG_PREFIX = '[theme-boot]'

function root(): HTMLElement {
  return document.documentElement
}

function painted(): Readonly<Record<string, string>> {
  const entries = paintNames(TABLE).map((name): [string, string] => [name, root().style.getPropertyValue(name).trim()])
  return Object.fromEntries(entries.filter(([, value]) => value !== ''))
}

function motion(): { readonly attribute: string | null; readonly variable: string } {
  return { attribute: root().getAttribute(THEME_MOTION_ATTRIBUTE), variable: root().style.getPropertyValue(THEME_MOTION_VAR) }
}

/** What useLook() last handed the theme below the provider. */
const seen: { current: HostLookState | null } = { current: null }

function Probe() {
  const state = useLook()
  seen.current = state
  return <p>Looks: {state.available ? state.look?.id : 'none'}</p>
}

/** One storage source per test: the provider reads again whenever its source changes. */
function sourceOf(initial: Readonly<Record<string, string>> = {}): { readonly store: Storage; readonly storage: StorageSource } {
  const store = memoryStorage(initial)
  return { store, storage: () => store }
}

function providerFor(storage: StorageSource, fields: Partial<ThemeLookProviderProps> = {}) {
  return (
    <ThemeLookProvider themeId={HELLO_THEME_ID} enabled tonePolicy="from-look" scope={SCOPE} storage={storage} server={null} {...fields}>
      <Probe />
    </ThemeLookProvider>
  )
}

function stored(store: Storage, key: string): unknown {
  const raw = store.getItem(key)
  return raw === null ? null : JSON.parse(raw)
}

/** What another tab does: writes the key itself, and the browser tells this tab with a storage event (key null: cleared). */
function fromAnotherTab(key: string | null): void {
  act(() => {
    window.dispatchEvent(new StorageEvent('storage', { key }))
  })
}

function stubReducedMotion(reduced: boolean): void {
  const noop = () => undefined
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: reduced && query === REDUCED_MOTION,
    media: query,
    onchange: null,
    addEventListener: noop,
    removeEventListener: noop,
    addListener: noop,
    removeListener: noop,
    dispatchEvent: () => false,
  }))
}

interface FakeLookServer extends LookServer {
  readonly read: Mock<(themeId: string) => Promise<unknown>>
  readonly write: Mock<(themeId: string, body: LookServerBody) => Promise<unknown>>
}

/** The server's copy of the account's looks: none kept yet. */
function emptyLookServer(): FakeLookServer {
  return {
    read: vi.fn((themeId: string): Promise<unknown> => Promise.resolve({ themeId, looks: [], active: null, defaultLook: PAPER, dropped: 0 })),
    write: vi.fn((_themeId: string, body: LookServerBody): Promise<unknown> => Promise.resolve(body)),
  }
}

/** The warnings the browser's boot fetch logged (boot-client.ts). */
function bootWarnings(): unknown[][] {
  return vi.mocked(console.warn).mock.calls.filter(([prefix]) => prefix === BOOT_LOG_PREFIX)
}

let removeBoot: () => void = () => undefined

/** The root layout's JSON element with this text as it is (a page whose boot isn't JSON). */
function putRawBoot(text: string): () => void {
  const element = document.createElement('script')
  element.type = 'application/json'
  element.id = THEME_BOOT_ELEMENT_ID
  element.textContent = text
  document.head.appendChild(element)
  return () => element.remove()
}

function pageBoot(boot: ThemeBoot | string): void {
  removeBoot = typeof boot === 'string' ? putRawBoot(boot) : putPageBoot(boot)
}

beforeEach(() => {
  seen.current = null
  root().removeAttribute('style')
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  removeBoot()
  removeBoot = () => undefined
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  root().removeAttribute('style')
  root().removeAttribute(THEME_OWN_MOTION_ATTRIBUTE)
  clearThemeToneCookie()
})

describe('under an installed theme with looks', () => {
  it('puts data-theme-motion and --theme-motion on <html>, and never data-pl-motion', async () => {
    pageBoot(helloThemeBoot())
    render(providerFor(sourceOf().storage))
    await waitFor(() => expect(motion()).toEqual({ attribute: 'full', variable: '1' }))
    expect(root().hasAttribute(THEME_OWN_MOTION_ATTRIBUTE)).toBe(false)
  })

  it('marks reduced motion when the user asks for it', async () => {
    stubReducedMotion(true)
    pageBoot(helloThemeBoot())
    render(providerFor(sourceOf().storage))
    await waitFor(() => expect(motion()).toEqual({ attribute: 'reduce', variable: '0' }))
    expect(root().hasAttribute(THEME_OWN_MOTION_ATTRIBUTE)).toBe(false)
  })

  it("paints the boot script's look until the account is known, then the account's own, and points the next load at it", async () => {
    pageBoot(helloThemeBoot())
    const { store, storage } = sourceOf({
      [HELLO.keys.pointer]: OTHER_SCOPE,
      [lookKeys(OTHER_SCOPE, HELLO.keys).active]: JSON.stringify(INK),
      [lookKeys(SCOPE, HELLO.keys).active]: JSON.stringify(DUSK),
    })
    const { rerender } = render(providerFor(storage, { scope: null }))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, INK)))
    expect(store.getItem(HELLO.keys.pointer)).toBe(OTHER_SCOPE)

    rerender(providerFor(storage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, DUSK)))
    expect(await screen.findByText('Looks: dusk')).toBeInTheDocument()
    await waitFor(() => expect(store.getItem(HELLO.keys.pointer)).toBe(SCOPE))
  })

  it("paints the theme's default look for an account with none stored, and writes no look", async () => {
    pageBoot(helloThemeBoot())
    const { store, storage } = sourceOf()
    render(providerFor(storage))
    expect(await screen.findByText(`Looks: ${PAPER.id}`)).toBeInTheDocument()
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    expect(store.getItem(KEYS.active)).toBeNull()
    expect(store.getItem(KEYS.saved)).toBeNull()
  })
})

describe('a look set here', () => {
  it('repaints <html> at once and is kept', async () => {
    pageBoot(helloThemeBoot())
    const { store, storage } = sourceOf()
    render(providerFor(storage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    act(() => void seen.current?.setLook(DUSK))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, DUSK)))
    expect(stored(store, KEYS.active)).toEqual(DUSK)
  })

  it("runs on the theme's default look with storage blocked, still switches looks, and says they aren't kept", async () => {
    pageBoot(helloThemeBoot())
    render(providerFor(blockedStorage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    act(() => void seen.current?.setLook(DUSK))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, DUSK)))
    expect(seen.current?.persisted).toBe(false)
    expect(console.warn).toHaveBeenCalled()
  })
})

describe('another tab (M25)', () => {
  it('takes the saved looks another tab stored, so an import keeps both under different ids', async () => {
    pageBoot(helloThemeBoot())
    const { store, storage } = sourceOf()
    render(providerFor(storage))
    expect(await screen.findByText(`Looks: ${PAPER.id}`)).toBeInTheDocument()
    store.setItem(KEYS.saved, JSON.stringify([THEIRS]))
    fromAnotherTab(KEYS.saved)
    expect(seen.current?.saved).toEqual([THEIRS])
    const imported = importLook(JSON.stringify(THEIRS))
    if (!imported.ok) throw new Error(imported.errors.join(' '))
    act(() => void saveLook(imported.look))
    await waitFor(() => expect(seen.current?.saved.map((look) => look.id)).toEqual([THEIRS.id, imported.look.id]))
    expect(imported.look.id).not.toBe(THEIRS.id)
    const kept = stored(store, KEYS.saved) as readonly { readonly id: string }[]
    expect(kept.map((look) => look.id)).toEqual([THEIRS.id, imported.look.id])
    expect(kept[0]).toEqual(THEIRS)
  })

  it('follows the active look another tab stored, and the tone cookie with it', async () => {
    pageBoot(helloThemeBoot())
    const { store, storage } = sourceOf()
    const onToneChange = vi.fn()
    render(providerFor(storage, { onToneChange }))
    await waitFor(() => expect(cookieValue(THEME_TONE_COOKIE)).toBe(themeToneCookieValue(HELLO_THEME_ID, 'light')))
    store.setItem(KEYS.active, JSON.stringify(INK))
    fromAnotherTab(KEYS.active)
    expect(await screen.findByText(`Looks: ${INK.id}`)).toBeInTheDocument()
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, INK)))
    await waitFor(() => expect(cookieValue(THEME_TONE_COOKIE)).toBe(themeToneCookieValue(HELLO_THEME_ID, 'dark')))
    expect(onToneChange).toHaveBeenLastCalledWith('dark')
  })

  it('never writes back what it took, so two tabs cannot ping-pong', async () => {
    pageBoot(helloThemeBoot())
    const { store, storage } = sourceOf()
    render(providerFor(storage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    const setItem = vi.spyOn(store, 'setItem')
    store.setItem(KEYS.active, JSON.stringify(INK))
    fromAnotherTab(KEYS.active)
    store.setItem(KEYS.saved, JSON.stringify([THEIRS]))
    fromAnotherTab(KEYS.saved)
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, INK)))
    expect(seen.current?.saved).toEqual([THEIRS])
    const keys = setItem.mock.calls.map(([key]) => key)
    // Only the other tab's own two writes: nothing written back.
    expect(keys.filter((key) => key === KEYS.active || key === KEYS.saved)).toEqual([KEYS.active, KEYS.saved])
  })

  it("ignores another key's event, and takes a cleared storage as nothing stored", async () => {
    pageBoot(helloThemeBoot())
    const { store, storage } = sourceOf({ [KEYS.active]: JSON.stringify(DUSK) })
    render(providerFor(storage))
    expect(await screen.findByText(`Looks: ${DUSK.id}`)).toBeInTheDocument()
    const otherKey = lookKeys(OTHER_SCOPE, HELLO.keys).active
    store.setItem(otherKey, JSON.stringify(INK))
    fromAnotherTab(otherKey)
    expect(screen.getByText(`Looks: ${DUSK.id}`)).toBeInTheDocument()
    store.clear()
    fromAnotherTab(null)
    expect(await screen.findByText(`Looks: ${PAPER.id}`)).toBeInTheDocument()
  })
})

describe('the looks it runs are its own theme boot', () => {
  it.each<[string, ThemeBoot | string | null]>([
    ["another theme's boot", helloThemeBoot({ entry: { ...HELLO_ENTRY, themeId: 'acme/other' } })],
    ['a base boot and no install to fetch its boot for (installId null)', helloThemeBoot({ entry: null })],
    ['a boot with no presets', helloThemeBoot({ looks: null })],
    ['no boot element at all', null],
    ['a boot that is not JSON', '{not json'],
  ])('with %s: no looks, nothing painted, no motion marks', async (_label, boot) => {
    if (boot !== null) pageBoot(boot)
    render(providerFor(sourceOf().storage))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(painted()).toEqual({})
    expect(motion()).toEqual({ attribute: null, variable: '' })
  })
})

describe('a page that got the base boot (the server missed its boot budget or was refused)', () => {
  it("runs the looks of the install's boot the browser fetched: the table's default look painted, the account's looks read", async () => {
    pageBoot(baseThemeBoot(INSTALLED_STYLE))
    const bootSource = vi.fn(async () => helloThemeBoot())
    const server = emptyLookServer()
    render(providerFor(sourceOf().storage, { installId: INSTALL_ID, bootSource, server }))
    expect(await screen.findByText(`Looks: ${PAPER.id}`)).toBeInTheDocument()
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    await waitFor(() => expect(server.read).toHaveBeenCalledWith(HELLO_THEME_ID))
    expect(bootSource.mock.calls).toEqual([[INSTALL_ID]])
    expect(motion()).toEqual({ attribute: 'full', variable: '1' })
  })

  it('a page whose boot names the theme never asks for it again', async () => {
    pageBoot(helloThemeBoot())
    const bootSource = vi.fn(async () => helloThemeBoot())
    render(providerFor(sourceOf().storage, { installId: INSTALL_ID, bootSource }))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    expect(bootSource).not.toHaveBeenCalled()
  })

  it("when the browser's fetch fails too: no looks, nothing painted, one warning", async () => {
    pageBoot(baseThemeBoot(INSTALLED_STYLE))
    const fetchBoot = vi.fn(() => Promise.reject(new Error('Failed to fetch')))
    render(providerFor(sourceOf().storage, { installId: INSTALL_ID, bootSource: createInstalledBootSource(fetchBoot) }))
    await waitFor(() => expect(bootWarnings()).toHaveLength(1))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(painted()).toEqual({})
    expect(motion()).toEqual({ attribute: null, variable: '' })
    expect(fetchBoot).toHaveBeenCalledTimes(1)
  })

  it('when the fetched boot names no theme either (a base boot): no looks, nothing painted', async () => {
    pageBoot(baseThemeBoot(INSTALLED_STYLE))
    const bootSource = vi.fn(async () => baseThemeBoot(INSTALLED_STYLE))
    render(providerFor(sourceOf().storage, { installId: INSTALL_ID, bootSource }))
    await waitFor(() => expect(bootSource).toHaveBeenCalledTimes(1))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(painted()).toEqual({})
  })
})

describe('switched off', () => {
  it("takes the table's variables and the motion marks off when the host stops running the theme", async () => {
    pageBoot(helloThemeBoot())
    const { storage } = sourceOf()
    const { rerender } = render(providerFor(storage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    rerender(providerFor(storage, { enabled: false }))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(painted()).toEqual({})
    expect(motion()).toEqual({ attribute: null, variable: '' })
  })

  it('a theme left in place (its id gone with the style) takes its variables and the motion marks off', async () => {
    pageBoot(helloThemeBoot())
    const { storage } = sourceOf()
    const { rerender } = render(providerFor(storage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    rerender(providerFor(storage, { themeId: null, enabled: false }))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    await waitFor(() => expect(painted()).toEqual({}))
    expect(motion()).toEqual({ attribute: null, variable: '' })
  })

  it("holds the theme's table into the commit where the host stops running it (its status trails the style)", async () => {
    pageBoot(helloThemeBoot())
    const { storage } = sourceOf()
    const { rerender } = render(providerFor(storage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    rerender(providerFor(storage, { themeId: null, enabled: false, keepFirstPaint: true }))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(painted()).toEqual(paint(TABLE, PAPER))
    rerender(providerFor(storage, { themeId: null, enabled: false, keepFirstPaint: false }))
    await waitFor(() => expect(painted()).toEqual({}))
    expect(motion()).toEqual({ attribute: null, variable: '' })
  })

  it('clears a theme it left once: what another runtime paints on those names afterwards stays', async () => {
    pageBoot(helloThemeBoot())
    const { storage } = sourceOf()
    const { rerender } = render(providerFor(storage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    rerender(providerFor(storage, { themeId: null, enabled: false }))
    await waitFor(() => expect(painted()).toEqual({}))
    const [name] = paintNames(TABLE)
    root().style.setProperty(name, ANOTHER_RUNTIMES_VALUE)
    rerender(providerFor(storage, { themeId: null, enabled: false, keepFirstPaint: true }))
    rerender(providerFor(storage, { themeId: null, enabled: false }))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(root().style.getPropertyValue(name)).toBe(ANOTHER_RUNTIMES_VALUE)
  })

  it("leaves the boot's first paint while the host loads or runs the installed theme (keepFirstPaint)", async () => {
    const boot = helloThemeBoot()
    runThemeBoot(boot)
    pageBoot(boot)
    render(providerFor(sourceOf().storage, { enabled: false, keepFirstPaint: true }))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(painted()).toEqual(paint(TABLE, PAPER))
    expect(motion().attribute).toBeNull()
  })

  it('a failed load from the start takes the first paint off', async () => {
    const boot = helloThemeBoot()
    runThemeBoot(boot)
    pageBoot(boot)
    render(providerFor(sourceOf().storage, { enabled: false }))
    await waitFor(() => expect(painted()).toEqual({}))
  })

  it('no theme with looks (themeId null) reads no looks and paints nothing', async () => {
    pageBoot(helloThemeBoot())
    const { store, storage } = sourceOf({ [lookKeys(SCOPE, HELLO.keys).active]: JSON.stringify(DUSK) })
    const reads = vi.spyOn(store, 'getItem')
    render(providerFor(storage, { themeId: null }))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(reads).not.toHaveBeenCalled()
    expect(painted()).toEqual({})
  })

  it('unmounting takes the look and the motion marks off', async () => {
    pageBoot(helloThemeBoot())
    const { unmount } = render(providerFor(sourceOf().storage))
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
    unmount()
    expect(painted()).toEqual({})
    expect(motion()).toEqual({ attribute: null, variable: '' })
  })
})
