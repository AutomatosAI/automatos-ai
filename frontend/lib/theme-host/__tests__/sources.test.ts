/**
 * PRD-261 A-04a — where a theme comes from. A style names an installed theme's source only for
 * `theme:install:<uuid>`; Classic, Studio and every other style name none (no theme is compiled
 * into the app since A-19f1). When GET /api/themes/active names no theme for the install (a 404,
 * a 200 with `theme: null`, a `disable` revocation or another install) there is no theme to load,
 * and no failure. The loader's steps (import, timeout, register) are held in ./loader.test.ts.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ getActiveTheme: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { getActiveTheme: api.getActiveTheme } }))

import { SDK_VERSION } from '../sdk'
import {
  defaultLoaderDeps,
  fetchActiveInstall,
  installedThemeSource,
  loadThemeForStyle,
  rememberActiveDigest,
  revocationDisableOf,
  sourceForStyle,
} from '../sources'
import { installIdOfStyle } from '../registry'
import { THEME_DIGEST_COOKIE } from '../types'
import type { LoaderDeps, ThemeHostGlobal, ThemeLoadResult } from '../types'
import { UI_STYLES, type UiStyle } from '@/lib/ui-style'

const INSTALL_ID = '00000000-0000-4000-8000-00000000e2e1'
const INSTALLED_STYLE: UiStyle = `theme:install:${INSTALL_ID}`
const TIMEOUT_MS = 1_000
const HOST: ThemeHostGlobal = Object.freeze({ sdkVersion: SDK_VERSION, modules: Object.freeze({}) })

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

const OTHER_INSTALL_ID = '00000000-0000-4000-8000-00000000e2e2'
const REVOKED = { installId: INSTALL_ID, level: 'disable', reason: 'withdrawn' }

describe('PRD-261 A-07b seam — a 200 answer that names no theme for this install', () => {
  it.each([
    ['no choice, or the install was removed', { choice: 'studio', theme: null }, 'no active install'],
    ['the workspace default, another install', { choice: OTHER_INSTALL_ID, theme: { installId: OTHER_INSTALL_ID } }, OTHER_INSTALL_ID],
  ])('loads nothing and reports no failure for %s', async (_case, answer, detail) => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    api.getActiveTheme.mockResolvedValue(answer)
    const deps = depsWith({ fetchActive: fetchActiveInstall })
    const result = await installedThemeSource(INSTALL_ID).load(deps)
    expect(result).toEqual({ ok: false, reason: null, detail: expect.stringContaining(detail) })
    expect(result).not.toHaveProperty('revoked')
    expect(deps.fetchJson).not.toHaveBeenCalled()
    expect(deps.importModule).not.toHaveBeenCalled()
    expect(error).not.toHaveBeenCalled()
  })

  it('loads nothing and reports no failure for a disable revocation, and hands its reason on (C-05l)', async () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    api.getActiveTheme.mockResolvedValue({ choice: 'studio', theme: null, revoked: REVOKED })
    const deps = depsWith({ fetchActive: fetchActiveInstall })
    const result = await installedThemeSource(INSTALL_ID).load(deps)
    expect(result).toEqual({
      ok: false, reason: null, detail: expect.stringContaining('revoked'), revoked: { level: 'disable', reason: 'withdrawn' },
    })
    expect(deps.fetchJson).not.toHaveBeenCalled()
    expect(deps.importModule).not.toHaveBeenCalled()
    expect(error).not.toHaveBeenCalled()
  })

  it("reads only this install's disable revocation from the answer", () => {
    const answer = { choice: 'studio', theme: null, revoked: REVOKED }
    expect(revocationDisableOf(answer, INSTALL_ID)).toEqual({ level: 'disable', reason: 'withdrawn' })
    expect(revocationDisableOf(answer, OTHER_INSTALL_ID)).toBeNull()
    expect(revocationDisableOf({ ...answer, revoked: { ...REVOKED, level: 'warn' } }, INSTALL_ID)).toBeNull()
    expect(revocationDisableOf({ ...answer, theme: { installId: INSTALL_ID } }, INSTALL_ID)).toBeNull()
    expect(revocationDisableOf({ choice: 'studio', theme: null }, INSTALL_ID)).toBeNull()
    expect(revocationDisableOf({ ...answer, revoked: { level: 'disable' } }, INSTALL_ID)).toEqual({ level: 'disable', reason: '' })
  })
})

const ACTIVE_DIGEST = 'c'.repeat(64)

describe('PRD-261 A-08c — the active digest left for the boot cache of the next request', () => {
  afterEach(() => {
    document.cookie = `${THEME_DIGEST_COOKIE}=; Path=/; Max-Age=0`
  })

  it('hands on the digest the active-theme answer names for this install, and nothing for another', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const rememberDigest = vi.fn()
    const active = { choice: INSTALL_ID, theme: { installId: INSTALL_ID, digest: ACTIVE_DIGEST, manifest: {} } }
    await installedThemeSource(INSTALL_ID).load(depsWith({ fetchActive: vi.fn(async () => active), rememberDigest }))
    expect(rememberDigest).toHaveBeenCalledWith(INSTALL_ID, ACTIVE_DIGEST)
    rememberDigest.mockClear()
    const other = { choice: OTHER_INSTALL_ID, theme: { installId: OTHER_INSTALL_ID, digest: ACTIVE_DIGEST } }
    await installedThemeSource(INSTALL_ID).load(depsWith({ fetchActive: vi.fn(async () => other), rememberDigest }))
    expect(rememberDigest).not.toHaveBeenCalled()
  })

  it('writes `<installId>.<digest>` to the cookie the root layout reads, by default', () => {
    expect(defaultLoaderDeps().rememberDigest).toBe(rememberActiveDigest)
    rememberActiveDigest(INSTALL_ID, ACTIVE_DIGEST)
    expect(document.cookie.split('; ')).toContain(`${THEME_DIGEST_COOKIE}=${INSTALL_ID}.${ACTIVE_DIGEST}`)
  })
})

describe('PRD-261 A-04a — the source a style names', () => {
  it('names the install for an installed theme style', () => {
    expect(installIdOfStyle(INSTALLED_STYLE)).toBe(INSTALL_ID)
    expect(sourceForStyle(INSTALLED_STYLE)?.ref).toBe(`install:${INSTALL_ID}`)
  })

  it.each(UI_STYLES)('names no source under %s', (style) => {
    expect(installIdOfStyle(style)).toBeNull()
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
    const deps = depsWith({ fetchActive: vi.fn(async () => ({ choice: INSTALL_ID, theme: { installId: INSTALL_ID } })) })
    expect(reasonOf(await loadThemeForStyle(INSTALLED_STYLE, deps))).toBe('manifest')
    expect(deps.fetchActive).toHaveBeenCalledTimes(1)
  })
})
