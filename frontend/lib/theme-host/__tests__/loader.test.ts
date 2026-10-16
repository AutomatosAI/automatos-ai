/**
 * PRD-261 A-01 — the loader, with every outside call injected. Each step that fails returns its
 * reason and stops the load (an SDK mismatch never requests the entry), a request or import that
 * never answers fails once the timeout passes, a look failure undoes the theme's registration,
 * and the ok result's cleanup removes the look and unregisters, once.
 * jsdom loads no stylesheets, so each test settles the theme's <link> itself.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { THEME_SPIKE_MANIFEST_PATH } from '../constants'
import { loadSpikeTheme, type LoaderDeps } from '../loader'
import type { ThemeHostGlobal, ThemeLoadResult, ThemeRegistrationHost } from '../types'

const TIMEOUT_MS = 1_000
const HTML_CLASS = 'hello-spike'
const STYLE_LINK = 'link[data-theme-host-style]'
const ACCENT = '--hello-accent'

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
    fetchJson: vi.fn(async (url: string) => (url === THEME_SPIKE_MANIFEST_PATH ? MANIFEST : PRESET)),
    importModule: vi.fn(async () => themeModule().module),
    doc: document,
    timeoutMs: TIMEOUT_MS,
    ...overrides,
  }
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

describe('PRD-261 A-01 — loadSpikeTheme fails with a reason, in order', () => {
  it("maps a throw while sharing the host's modules to 'load', before any request", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const deps = depsWith({ exposeHost: () => { throw new Error('a second React') } })
    expect(reasonOf(await loadSpikeTheme(deps))).toBe('load')
    expect(deps.fetchJson).not.toHaveBeenCalled()
    expect(deps.importModule).not.toHaveBeenCalled()
  })

  it("maps an unreadable theme.json to 'manifest'", async () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const deps = depsWith({ fetchJson: vi.fn(async () => { throw new Error('404') }) })
    expect(reasonOf(await loadSpikeTheme(deps))).toBe('manifest')
    expect(error).toHaveBeenCalledWith('[theme-host]', 'manifest', expect.any(Error))
  })

  it("refuses an SDK mismatch as 'engines' and never requests the entry", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const mismatched = { ...MANIFEST, engines: { sdk: '^2.0.0' } }
    const deps = depsWith({ fetchJson: vi.fn(async (url: string) => (url === THEME_SPIKE_MANIFEST_PATH ? mismatched : PRESET)) })
    expect(reasonOf(await loadSpikeTheme(deps))).toBe('engines')
    expect(deps.importModule).not.toHaveBeenCalled()
  })

  it("maps an entry that fails to import, or has no register, to 'load'", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const rejects = depsWith({ importModule: vi.fn(async () => { throw new Error('deliberate load failure') }) })
    expect(reasonOf(await loadSpikeTheme(rejects))).toBe('load')
    const noRegister = depsWith({ importModule: vi.fn(async () => ({ default: { unregister() {} } })) })
    expect(reasonOf(await loadSpikeTheme(noRegister))).toBe('load')
    const noDefault = depsWith({ importModule: vi.fn(async () => ({})) })
    expect(reasonOf(await loadSpikeTheme(noDefault))).toBe('load')
  })

  it("maps an entry that never settles to 'load' once the timeout passes", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    vi.useFakeTimers()
    const pending = loadSpikeTheme(depsWith({ importModule: () => new Promise<unknown>(() => undefined) }))
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS + 1)
    expect(reasonOf(await pending)).toBe('load')
  })

  it("maps a theme.json request that never answers to 'manifest' once the timeout passes", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    vi.useFakeTimers()
    const deps = depsWith({ fetchJson: () => new Promise<unknown>(() => undefined) })
    const pending = loadSpikeTheme(deps)
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS + 1)
    expect(reasonOf(await pending)).toBe('manifest')
    expect(deps.importModule).not.toHaveBeenCalled()
  })

  it("maps a preset request that never answers to 'look' and unregisters the theme", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    vi.useFakeTimers()
    const theme = themeModule()
    const stalledPreset = (url: string) => (url === THEME_SPIKE_MANIFEST_PATH ? Promise.resolve(MANIFEST) : new Promise<unknown>(() => undefined))
    const pending = loadSpikeTheme(depsWith({ fetchJson: stalledPreset, importModule: vi.fn(async () => theme.module) }))
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS + 1)
    expect(reasonOf(await pending)).toBe('look')
    expect(theme.unregister).toHaveBeenCalledTimes(1)
    expect(lookIsGone()).toBe(true)
  })

  it("maps a register that throws, or registers no frame, to 'register'", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const throws = { default: { register() { throw new Error('deliberate register failure') } } }
    expect(reasonOf(await loadSpikeTheme(depsWith({ importModule: vi.fn(async () => throws) })))).toBe('register')
    const noFrame = { default: { register() {} } }
    expect(reasonOf(await loadSpikeTheme(depsWith({ importModule: vi.fn(async () => noFrame) })))).toBe('register')
    const otherSlot = { default: { register(host: ThemeRegistrationHost) { host.registerSlot('sidebar@1', Frame) } } }
    expect(reasonOf(await loadSpikeTheme(depsWith({ importModule: vi.fn(async () => otherSlot) })))).toBe('register')
  })

  it("maps a stylesheet that fails to 'look', unregisters the theme and leaves nothing on <html>", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    observer = settleStylesheets('error')
    const theme = themeModule()
    const result = await loadSpikeTheme(depsWith({ importModule: vi.fn(async () => theme.module) }))
    expect(reasonOf(result)).toBe('look')
    expect(theme.unregister).toHaveBeenCalledTimes(1)
    expect(lookIsGone()).toBe(true)
  })

  it("maps a preset with a refused value to 'look'", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const badPreset = { vars: { [ACCENT]: 'url(x)' } }
    const deps = depsWith({ fetchJson: vi.fn(async (url: string) => (url === THEME_SPIKE_MANIFEST_PATH ? MANIFEST : badPreset)) })
    expect(reasonOf(await loadSpikeTheme(deps))).toBe('look')
    expect(lookIsGone()).toBe(true)
  })
})

describe('PRD-261 A-01 — loadSpikeTheme on success', () => {
  it('returns the frame with the class, the stylesheet and the variables applied', async () => {
    observer = settleStylesheets('load')
    const theme = themeModule()
    const result = await loadSpikeTheme(depsWith({ importModule: vi.fn(async () => theme.module) }))
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.frame).toBe(Frame)
    expect(theme.register).toHaveBeenCalledTimes(1)
    expect(theme.register.mock.calls[0]?.[0].sdkVersion).toBe('1.0.0')
    expect(Object.isFrozen(theme.register.mock.calls[0]?.[0])).toBe(true)
    expect(document.documentElement.classList.contains(HTML_CLASS)).toBe(true)
    expect(document.querySelector(STYLE_LINK)?.getAttribute('href')).toBe(`${window.location.origin}/theme-spike/hello/theme.css`)
    expect(document.documentElement.style.getPropertyValue(ACCENT)).toBe('#2f6fed')
  })

  it('cleans up the look and unregisters once, however often cleanup runs', async () => {
    observer = settleStylesheets('load')
    const theme = themeModule()
    const result = await loadSpikeTheme(depsWith({ importModule: vi.fn(async () => theme.module) }))
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
    const result = await loadSpikeTheme(depsWith({ importModule: vi.fn(async () => theme.module) }))
    if (!result.ok) throw new Error(`expected ok, got ${result.reason}`)
    expect(() => result.cleanup()).not.toThrow()
    expect(lookIsGone()).toBe(true)
    expect(error).toHaveBeenCalledWith('[theme-host]', 'cleanup', expect.any(Error))
  })
})
