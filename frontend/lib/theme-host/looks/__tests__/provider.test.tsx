/**
 * PRD-261 A-19d2 — the generic look runtime's provider, on hello-a09b's boot as the page carries it:
 * under an installed theme with looks it paints the account's active look on <html> from the
 * theme's own table (the boot script's look until the account's looks are read), puts
 * data-theme-motion and --theme-motion there and never data-pl-motion, and points the next load's
 * boot script at the account's look (F5). It runs only the looks of a boot of its own theme. Off,
 * a failed load or another style takes the table's variables and the motion marks off; while the
 * host loads or runs an installed theme (keepFirstPaint) the boot's first paint stays.
 */
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import { THEME_BOOT_ELEMENT_ID, type ThemeBoot } from '@/lib/theme-host/boot'
import { THEME_MOTION_ATTRIBUTE, THEME_MOTION_VAR } from '@/lib/theme-host/looks/motion'
import { paint, paintNames } from '@/lib/theme-host/looks/paint'
import { ThemeLookProvider, type ThemeLookProviderProps } from '@/lib/theme-host/looks/provider'
import { lookKeys, type StorageSource } from '@/lib/theme-host/looks/storage'
import { useLook } from '@/lib/theme-sdk/looks'

import { runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'
import { DUSK, HELLO, HELLO_ENTRY, HELLO_RULES, HELLO_THEME_ID, helloThemeBoot, INK, memoryStorage, PAPER, putPageBoot } from './hello-looks'

const TABLE = HELLO_RULES.table
const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a')
const OTHER_SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_b')
/** Builtin Play's own motion mark (its look provider sets it), never set by the generic runtime. */
const BUILTIN_MOTION_ATTRIBUTE = 'data-pl-motion'
const REDUCED_MOTION = '(prefers-reduced-motion: reduce)'

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

function Probe() {
  const { available, look } = useLook()
  return <p>Looks: {available ? look?.id : 'none'}</p>
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
  root().removeAttribute(BUILTIN_MOTION_ATTRIBUTE)
})

describe('under an installed theme with looks', () => {
  it('puts data-theme-motion and --theme-motion on <html>, and never data-pl-motion', async () => {
    pageBoot(helloThemeBoot())
    render(providerFor(sourceOf().storage))
    await waitFor(() => expect(motion()).toEqual({ attribute: 'full', variable: '1' }))
    expect(root().hasAttribute(BUILTIN_MOTION_ATTRIBUTE)).toBe(false)
  })

  it('marks reduced motion when the user asks for it', async () => {
    stubReducedMotion(true)
    pageBoot(helloThemeBoot())
    render(providerFor(sourceOf().storage))
    await waitFor(() => expect(motion()).toEqual({ attribute: 'reduce', variable: '0' }))
    expect(root().hasAttribute(BUILTIN_MOTION_ATTRIBUTE)).toBe(false)
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

  it("paints the theme's default look for an account with none stored", async () => {
    pageBoot(helloThemeBoot())
    render(providerFor(sourceOf().storage))
    expect(await screen.findByText(`Looks: ${PAPER.id}`)).toBeInTheDocument()
    await waitFor(() => expect(painted()).toEqual(paint(TABLE, PAPER)))
  })
})

describe('the looks it runs are its own theme boot', () => {
  it.each<[string, ThemeBoot | string | null]>([
    ["another theme's boot", helloThemeBoot({ entry: { ...HELLO_ENTRY, themeId: 'acme/other' } })],
    ['a base boot with no entry', helloThemeBoot({ entry: null })],
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
