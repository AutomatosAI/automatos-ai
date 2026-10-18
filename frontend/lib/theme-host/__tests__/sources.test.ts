/**
 * PRD-261 A-04a — where a theme comes from. A style names an installed theme's source only for
 * `theme:install:<uuid>`; Classic, Studio and builtin Play name none (builtin Play keeps its
 * compiled path until A-04b). A builtin source loads its compiled module and registers its frame
 * as soon as the import resolves, with no manifest, no look and no host module global. A 404 from
 * GET /api/themes/active means no active install: no theme to load, and no failure.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ getActiveTheme: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { getActiveTheme: api.getActiveTheme } }))

import {
  builtinThemeSource,
  defaultLoaderDeps,
  fetchActiveInstall,
  installedThemeSource,
  installIdOf,
  loadThemeForStyle,
  sourceForStyle,
} from '../sources'
import type { LoaderDeps, ThemeHostGlobal, ThemeLoadResult, ThemeRegistrationHost } from '../types'
import { UI_STYLES, type UiStyle } from '@/lib/ui-style'

const INSTALL_ID = '00000000-0000-4000-8000-00000000e2e1'
const INSTALLED_STYLE: UiStyle = `theme:install:${INSTALL_ID}`
const TIMEOUT_MS = 1_000
const HOST: ThemeHostGlobal = Object.freeze({ sdkVersion: '1.0.0', modules: Object.freeze({}) })

function Frame() {
  return null
}

function depsWith(overrides: Partial<LoaderDeps> = {}): LoaderDeps {
  return {
    exposeHost: vi.fn(() => HOST),
    fetchActive: vi.fn(async () => ({ choice: 'studio', theme: null })),
    fetchJson: vi.fn(async () => ({})),
    importModule: vi.fn(async () => ({})),
    doc: document,
    timeoutMs: TIMEOUT_MS,
    ...overrides,
  }
}

function reasonOf(result: ThemeLoadResult): string | null {
  return result.ok ? null : result.reason
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  api.getActiveTheme.mockReset()
})

/** The error apiClient throws for a response that is not ok: the status rides it. */
function httpError(status: number): Error {
  return Object.assign(new Error(`HTTP ${status}`), { status })
}

describe('PRD-261 A-04a — GET /api/themes/active through the named apiClient method', () => {
  it("is the loader's active-theme request, and hands back the server's answer", async () => {
    const answer = { choice: 'studio', theme: null }
    api.getActiveTheme.mockResolvedValue(answer)
    expect(defaultLoaderDeps().fetchActive).toBe(fetchActiveInstall)
    await expect(fetchActiveInstall()).resolves.toBe(answer)
    expect(api.getActiveTheme).toHaveBeenCalledTimes(1)
  })

  it('reads a 404 as no active install', async () => {
    api.getActiveTheme.mockRejectedValue(httpError(404))
    await expect(fetchActiveInstall()).resolves.toBeNull()
  })

  it('rethrows any other failure', async () => {
    const failure = httpError(500)
    api.getActiveTheme.mockRejectedValue(failure)
    await expect(fetchActiveInstall()).rejects.toBe(failure)
  })

  it('loads nothing and reports no failure for no active install (a stale theme:install cookie)', async () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    api.getActiveTheme.mockRejectedValue(httpError(404))
    const deps = depsWith({ fetchActive: fetchActiveInstall })
    const result = await installedThemeSource(INSTALL_ID).load(deps)
    expect(result).toEqual({ ok: false, reason: null, detail: expect.stringContaining('404') })
    expect(deps.fetchJson).not.toHaveBeenCalled()
    expect(deps.importModule).not.toHaveBeenCalled()
    expect(error).not.toHaveBeenCalled()
  })
})

describe('PRD-261 A-04a — the source a style names', () => {
  it('names the install for an installed theme style', () => {
    expect(installIdOf(INSTALLED_STYLE)).toBe(INSTALL_ID)
    expect(sourceForStyle(INSTALLED_STYLE)?.ref).toBe(`install:${INSTALL_ID}`)
  })

  it.each(UI_STYLES)('names no source under %s', (style) => {
    expect(installIdOf(style)).toBeNull()
    expect(sourceForStyle(style)).toBeNull()
  })

  it("fails a style with no source as 'load', and reaches for nothing", async () => {
    const deps = depsWith()
    const result = await loadThemeForStyle('studio', deps)
    expect(reasonOf(result)).toBe('load')
    expect(deps.exposeHost).not.toHaveBeenCalled()
    expect(deps.fetchActive).not.toHaveBeenCalled()
  })

  it('reads the active install for an installed theme style', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const deps = depsWith()
    expect(reasonOf(await loadThemeForStyle(INSTALLED_STYLE, deps))).toBe('manifest')
    expect(deps.fetchActive).toHaveBeenCalledTimes(1)
  })
})

describe('PRD-261 A-04a — builtinThemeSource', () => {
  it('registers the frame as soon as the compiled module resolves, with no manifest and no host global', async () => {
    const register = vi.fn((host: ThemeRegistrationHost) => host.registerSlot('frame@1', Frame))
    const unregister = vi.fn()
    const deps = depsWith()
    const source = builtinThemeSource('builtin:hello', async () => ({ default: { register, unregister } }))
    const result = await source.load(deps)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.ref).toBe('builtin:hello')
    expect(result.frame).toBe(Frame)
    expect(result.manifest).toBeNull()
    expect(register).toHaveBeenCalledTimes(1)
    expect(deps.exposeHost).not.toHaveBeenCalled()
    expect(deps.fetchActive).not.toHaveBeenCalled()
    expect(deps.importModule).not.toHaveBeenCalled()
    result.cleanup()
    result.cleanup()
    expect(unregister).toHaveBeenCalledTimes(1)
  })

  it("fails a module that will not import, or has no definition, as 'load'", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const rejects = builtinThemeSource('builtin:hello', async () => { throw new Error('chunk failed') })
    expect(reasonOf(await rejects.load(depsWith()))).toBe('load')
    const empty = builtinThemeSource('builtin:hello', async () => ({}))
    expect(reasonOf(await empty.load(depsWith()))).toBe('load')
  })

  it("fails a module that never resolves as 'load' once the timeout passes", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    vi.useFakeTimers()
    const stalled = builtinThemeSource('builtin:hello', () => new Promise<unknown>(() => undefined))
    const pending = stalled.load(depsWith())
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS + 1)
    expect(reasonOf(await pending)).toBe('load')
  })

  it("fails a register that registers no frame as 'register', and unregisters", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const unregister = vi.fn()
    const source = builtinThemeSource('builtin:hello', async () => ({ default: { register() {}, unregister } }))
    expect(reasonOf(await source.load(depsWith()))).toBe('register')
    expect(unregister).toHaveBeenCalledTimes(1)
  })
})
