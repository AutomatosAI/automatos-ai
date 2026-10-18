/**
 * PRD-261 A-01, A-04a — the load of an installed theme (sources.ts installedThemeSource over the
 * loader's steps), with every outside call injected. Each step that fails returns its reason and
 * stops the load (an SDK mismatch never requests the entry), a request or import that never
 * answers fails once the timeout passes, a look failure undoes the theme's registration, and the
 * ok result's cleanup removes the look and unregisters, once. The registration host takes the
 * frame, the other slots, pages and routes by their allowed names only.
 * jsdom loads no stylesheets, so each test settles the theme's <link> itself.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { THEME_HOST_VERSION } from '../constants'
import { registerTheme } from '../loader'
import type { SlotId } from '../registry-types'
import { installedThemeSource } from '../sources'
import type { LoaderDeps, ThemeHostGlobal, ThemeLoadResult, ThemeRegistrationHost } from '../types'

const TIMEOUT_MS = 1_000
const HTML_CLASS = 'hello-spike'
const STYLE_LINK = 'link[data-theme-host-style]'
const ACCENT = '--hello-accent'
const INSTALL_ID = '00000000-0000-4000-8000-00000000e2e1'
const OTHER_INSTALL_ID = '00000000-0000-4000-8000-000000000000'
const DIGEST = '0123456789abcdef'.repeat(4)
const FILES_ROOT = `/api/themes/files/${DIGEST}/`

const MANIFEST = {
  schema: 'automatos-theme-spike/v1',
  id: 'automatos-e2e/hello',
  name: 'Hello theme',
  version: '0.0.1',
  engines: { sdk: '^1.0.0' },
  base: 'studio',
  htmlClass: HTML_CLASS,
  entry: 'entry.js',
  styleSheet: 'theme.css',
  preset: 'presets/default.look.json',
  slots: ['frame@1'],
}
const PRESET = { vars: { [ACCENT]: '#2f6fed', '--hello-ink': '#1a1814' } }
const HOST: ThemeHostGlobal = Object.freeze({ sdkVersion: '1.0.0', modules: Object.freeze({}) })

/** The GET /api/themes/active answer for the install, carrying `manifest`. */
function activeWith(manifest: unknown, theme: Record<string, unknown> = {}) {
  return { choice: INSTALL_ID, theme: { installId: INSTALL_ID, digest: DIGEST, manifest, ...theme } }
}

function Frame() {
  return null
}

function themeModule(unregister: () => void = vi.fn()) {
  const register = vi.fn((host: ThemeRegistrationHost) => host.registerSlot('frame@1', Frame))
  return { register, unregister, module: { default: { register, unregister } } }
}

function depsWith(overrides: Partial<LoaderDeps> = {}): LoaderDeps {
  return {
    exposeHost: () => HOST,
    fetchActive: vi.fn(async () => activeWith(MANIFEST)),
    fetchJson: vi.fn(async () => PRESET),
    importModule: vi.fn(async () => themeModule().module),
    doc: document,
    timeoutMs: TIMEOUT_MS,
    ...overrides,
  }
}

function loadInstalled(deps: LoaderDeps): Promise<ThemeLoadResult> {
  return installedThemeSource(INSTALL_ID).load(deps)
}

/** Fires `load` or `error` on the theme's stylesheet as soon as it is appended. */
function settleStylesheets(event: 'load' | 'error'): MutationObserver {
  const observer = new MutationObserver(() => {
    document.querySelectorAll(STYLE_LINK).forEach((link) => link.dispatchEvent(new Event(event)))
  })
  observer.observe(document.head, { childList: true })
  return observer
}

function reasonOf(result: ThemeLoadResult): string | null {
  return result.ok ? null : result.reason
}

function lookIsGone(): boolean {
  const root = document.documentElement
  return !root.classList.contains(HTML_CLASS) && document.querySelector(STYLE_LINK) === null && root.style.getPropertyValue(ACCENT) === ''
}

let observer: MutationObserver | null = null

afterEach(() => {
  observer?.disconnect()
  observer = null
  vi.useRealTimers()
  vi.restoreAllMocks()
  document.querySelectorAll(STYLE_LINK).forEach((link) => link.remove())
  document.documentElement.className = ''
  document.documentElement.removeAttribute('style')
})

describe('PRD-261 A-04a — an installed theme fails to load with a reason, in order', () => {
  it("maps a throw while sharing the host's modules to 'load', before any request", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const deps = depsWith({ exposeHost: () => { throw new Error('a second React') } })
    expect(reasonOf(await loadInstalled(deps))).toBe('load')
    expect(deps.fetchActive).not.toHaveBeenCalled()
    expect(deps.importModule).not.toHaveBeenCalled()
  })

  it("maps an active-theme request that fails to 'manifest'", async () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const deps = depsWith({ fetchActive: vi.fn(async () => { throw new Error('HTTP 404') }) })
    expect(reasonOf(await loadInstalled(deps))).toBe('manifest')
    expect(error).toHaveBeenCalledWith('[theme-host]', 'manifest', expect.any(Error))
    expect(deps.importModule).not.toHaveBeenCalled()
  })

  it("maps no active theme, another install, a bad digest or a refused manifest to 'manifest'", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const answers = [
      { choice: 'studio', theme: null },
      activeWith(MANIFEST, { installId: OTHER_INSTALL_ID }),
      activeWith(MANIFEST, { digest: 'not-a-digest' }),
      activeWith({ ...MANIFEST, entry: '../elsewhere/entry.js' }),
      activeWith({ ...MANIFEST, htmlClass: 'studio' }),
    ]
    for (const answer of answers) {
      const deps = depsWith({ fetchActive: vi.fn(async () => answer) })
      expect(reasonOf(await loadInstalled(deps)), JSON.stringify(answer)).toBe('manifest')
      expect(deps.importModule).not.toHaveBeenCalled()
    }
  })

  it("refuses an SDK mismatch as 'engines' and never requests the entry", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const mismatched = { ...MANIFEST, engines: { sdk: '^2.0.0' } }
    const deps = depsWith({ fetchActive: vi.fn(async () => activeWith(mismatched)) })
    expect(reasonOf(await loadInstalled(deps))).toBe('engines')
    expect(deps.importModule).not.toHaveBeenCalled()
  })

  it("maps an entry that fails to import, or has no register, to 'load'", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const rejects = depsWith({ importModule: vi.fn(async () => { throw new Error('deliberate load failure') }) })
    expect(reasonOf(await loadInstalled(rejects))).toBe('load')
    const noRegister = depsWith({ importModule: vi.fn(async () => ({ default: { unregister() {} } })) })
    expect(reasonOf(await loadInstalled(noRegister))).toBe('load')
    const noDefault = depsWith({ importModule: vi.fn(async () => ({})) })
    expect(reasonOf(await loadInstalled(noDefault))).toBe('load')
  })

  it("maps an entry that never settles to 'load' once the timeout passes", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    vi.useFakeTimers()
    const pending = loadInstalled(depsWith({ importModule: () => new Promise<unknown>(() => undefined) }))
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS + 1)
    expect(reasonOf(await pending)).toBe('load')
  })

  it("maps an active-theme request that never answers to 'manifest' once the timeout passes", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    vi.useFakeTimers()
    const deps = depsWith({ fetchActive: () => new Promise<unknown>(() => undefined) })
    const pending = loadInstalled(deps)
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS + 1)
    expect(reasonOf(await pending)).toBe('manifest')
    expect(deps.importModule).not.toHaveBeenCalled()
  })

  it("maps a preset request that never answers to 'look' and unregisters the theme", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    vi.useFakeTimers()
    const theme = themeModule()
    const stalledPreset = () => new Promise<unknown>(() => undefined)
    const pending = loadInstalled(depsWith({ fetchJson: stalledPreset, importModule: vi.fn(async () => theme.module) }))
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS + 1)
    expect(reasonOf(await pending)).toBe('look')
    expect(theme.unregister).toHaveBeenCalledTimes(1)
    expect(lookIsGone()).toBe(true)
  })

  it("maps a register that throws, or registers no frame, to 'register'", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const throws = { default: { register() { throw new Error('deliberate register failure') } } }
    expect(reasonOf(await loadInstalled(depsWith({ importModule: vi.fn(async () => throws) })))).toBe('register')
    const noFrame = { default: { register() {} } }
    expect(reasonOf(await loadInstalled(depsWith({ importModule: vi.fn(async () => noFrame) })))).toBe('register')
    const otherSlot = { default: { register(host: ThemeRegistrationHost) { host.registerSlot('sidebar@1' as SlotId, Frame) } } }
    expect(reasonOf(await loadInstalled(depsWith({ importModule: vi.fn(async () => otherSlot) })))).toBe('register')
  })

  it('unregisters a theme whose register throws, or registers no frame', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const unregisterThrown = vi.fn()
    const throws = { default: { register() { throw new Error('deliberate register failure') }, unregister: unregisterThrown } }
    expect(reasonOf(await loadInstalled(depsWith({ importModule: vi.fn(async () => throws) })))).toBe('register')
    expect(unregisterThrown).toHaveBeenCalledTimes(1)
    const unregisterNoFrame = vi.fn()
    const noFrame = { default: { register() {}, unregister: unregisterNoFrame } }
    expect(reasonOf(await loadInstalled(depsWith({ importModule: vi.fn(async () => noFrame) })))).toBe('register')
    expect(unregisterNoFrame).toHaveBeenCalledTimes(1)
  })

  it("maps a stylesheet that fails to 'look', unregisters the theme and leaves nothing on <html>", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    observer = settleStylesheets('error')
    const theme = themeModule()
    const result = await loadInstalled(depsWith({ importModule: vi.fn(async () => theme.module) }))
    expect(reasonOf(result)).toBe('look')
    expect(theme.unregister).toHaveBeenCalledTimes(1)
    expect(lookIsGone()).toBe(true)
  })

  it("maps a preset with a refused value to 'look'", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const badPreset = { vars: { [ACCENT]: 'url(x)' } }
    const deps = depsWith({ fetchJson: vi.fn(async () => badPreset) })
    expect(reasonOf(await loadInstalled(deps))).toBe('look')
    expect(lookIsGone()).toBe(true)
  })
})

describe('PRD-261 A-04a — an installed theme on success', () => {
  it("reads its files from the install's folder, by its digest", async () => {
    observer = settleStylesheets('load')
    const deps = depsWith()
    expect((await loadInstalled(deps)).ok).toBe(true)
    expect(deps.importModule).toHaveBeenCalledWith(`${window.location.origin}${FILES_ROOT}entry.js`)
    expect(deps.fetchJson).toHaveBeenCalledWith(`${window.location.origin}${FILES_ROOT}presets/default.look.json`)
  })

  it('returns the frame with the class, the stylesheet and the variables applied', async () => {
    observer = settleStylesheets('load')
    const theme = themeModule()
    const result = await loadInstalled(depsWith({ importModule: vi.fn(async () => theme.module) }))
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.frame).toBe(Frame)
    expect(result.ref).toBe(`install:${INSTALL_ID}`)
    expect(result.manifest?.id).toBe(MANIFEST.id)
    const host = theme.register.mock.calls[0]?.[0]
    expect(theme.register).toHaveBeenCalledTimes(1)
    expect(host?.sdkVersion).toBe('1.0.0')
    expect(host?.hostVersion).toBe(THEME_HOST_VERSION)
    expect(Object.isFrozen(host)).toBe(true)
    expect(document.documentElement.classList.contains(HTML_CLASS)).toBe(true)
    expect(document.querySelector(STYLE_LINK)?.getAttribute('href')).toBe(`${window.location.origin}${FILES_ROOT}theme.css`)
    expect(document.documentElement.style.getPropertyValue(ACCENT)).toBe('#2f6fed')
  })

  it('cleans up the look and unregisters once, however often cleanup runs', async () => {
    observer = settleStylesheets('load')
    const theme = themeModule()
    const result = await loadInstalled(depsWith({ importModule: vi.fn(async () => theme.module) }))
    if (!result.ok) throw new Error(`expected ok, got ${result.reason}`)
    result.cleanup()
    expect(lookIsGone()).toBe(true)
    expect(theme.unregister).toHaveBeenCalledTimes(1)
    result.cleanup()
    expect(theme.unregister).toHaveBeenCalledTimes(1)
  })

  it('logs an unregister that throws and still removes the look', async () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    observer = settleStylesheets('load')
    const theme = themeModule(vi.fn(() => { throw new Error('deliberate unregister failure') }))
    const result = await loadInstalled(depsWith({ importModule: vi.fn(async () => theme.module) }))
    if (!result.ok) throw new Error(`expected ok, got ${result.reason}`)
    expect(() => result.cleanup()).not.toThrow()
    expect(lookIsGone()).toBe(true)
    expect(error).toHaveBeenCalledWith('[theme-host]', 'cleanup', expect.any(Error))
  })
})

describe('PRD-261 A-04a — registerTheme: what the host accepts', () => {
  const Page = () => null
  const LAZY_LIKE = { $$typeof: Symbol.for('react.lazy') }

  function registering(register: (host: ThemeRegistrationHost) => void) {
    return { register, unregister: vi.fn() }
  }

  it('keeps the frame, the other slots, the Q3 pages and the /x routes', () => {
    const registrations = registerTheme(
      registering((host) => {
        host.registerSlot('frame@1', Frame)
        host.registerSlot('lookEditor@1', LAZY_LIKE)
        host.registerPage('/chat', Page)
        host.registerPage('/missions/[id]', Page)
        host.registerRoute('hello', Page)
        host.registerRoute('hello/world', Page)
      }),
      '1.0.0',
    )
    expect(registrations.frame).toBe(Frame)
    expect([...registrations.slots.keys()]).toEqual(['frame@1', 'lookEditor@1'])
    expect([...registrations.pages.keys()]).toEqual(['/chat', '/missions/[id]'])
    expect([...registrations.routes.keys()]).toEqual(['hello', 'hello/world'])
  })

  const refused: ReadonlyArray<{ readonly label: string; readonly extra: (host: ThemeRegistrationHost) => void }> = [
    { label: 'a page outside the Q3 list', extra: (host) => host.registerPage('/settings', Page) },
    { label: 'a route that leaves /x', extra: (host) => host.registerRoute('../admin', Page) },
    { label: 'a route with an upper-case or empty segment', extra: (host) => host.registerRoute('Hello//x', Page) },
    { label: 'a page that is not a component', extra: (host) => host.registerPage('/chat', 'not a component') },
    { label: 'the same slot twice', extra: (host) => host.registerSlot('frame@1', Frame) },
  ]

  it.each(refused)('refuses $label and unregisters the theme', ({ extra }) => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const theme = registering((host) => {
      host.registerSlot('frame@1', Frame)
      extra(host)
    })
    expect(() => registerTheme(theme, '1.0.0')).toThrow()
    expect(theme.unregister).toHaveBeenCalledTimes(1)
  })
})
