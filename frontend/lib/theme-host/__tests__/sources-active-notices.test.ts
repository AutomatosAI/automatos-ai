/**
 * PRD-261 seam (W-A2) — what the active-theme answer says besides the install, and files that
 * cannot be reached:
 * - `optIn.required` (private code the member has not opted in to, Q6) fails the load as
 *   `opt-in` before any theme file is asked for (the files proxy would 404 each one);
 * - a `warn` revocation loads the theme and hands its reason on (ThemeLoadOk.revoked); a
 *   `disable` one is no theme at all (sources.test.ts);
 * - a 5xx or a network failure on the entry or the preset fails as `unavailable`, which the
 *   failure ledger never counts.
 * jsdom loads no stylesheets, so the successful loads settle the theme's <link> themselves.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { countsTowardRevert } from '../failure-ledger'
import { settleLoad, step } from '../loader'
import { SDK_VERSION } from '../sdk'
import {
  defaultLoaderDeps,
  importInstalledEntry,
  installedThemeSource,
  isUnavailableStatus,
  optInRequired,
  revocationWarningOf,
} from '../sources'
import type { LoaderDeps, ThemeFailureReason, ThemeHostGlobal, ThemeLoadOk, ThemeLoadResult, ThemeRegistrationHost } from '../types'

const TIMEOUT_MS = 1_000
const STYLE_LINK = 'link[data-theme-host-style]'
const INSTALL_ID = '00000000-0000-4000-8000-0000000a2b01'
const DIGEST = 'd'.repeat(64)
const ENTRY_URL = `/api/themes/files/${DIGEST}/entry.js`
const PRESET_URL = `/api/themes/files/${DIGEST}/presets/default.look.json`
const HOST: ThemeHostGlobal = Object.freeze({ sdkVersion: SDK_VERSION, modules: Object.freeze({}) })
const MANIFEST = {
  schema: 'automatos-theme-spike/v1',
  id: 'automatos-e2e/hello',
  name: 'Hello theme',
  version: '0.0.1',
  engines: { sdk: '^1.0.0' },
  base: 'studio',
  htmlClass: 'hello-notices',
  entry: 'entry.js',
  styleSheet: 'theme.css',
  preset: 'presets/default.look.json',
  slots: ['frame@1'],
}
const PRESET = { vars: { '--hello-accent': '#2f6fed' } }
const WARN = { level: 'warn', reason: 'A security fix is out: update soon.' }

function Frame() {
  return null
}

const themeModule = { default: { register: (host: ThemeRegistrationHost) => host.registerSlot('frame@1', Frame) } }

function activeWith(theme: Record<string, unknown> = {}) {
  return { choice: INSTALL_ID, theme: { installId: INSTALL_ID, digest: DIGEST, manifest: MANIFEST, ...theme } }
}

function depsWith(answer: unknown): LoaderDeps {
  return {
    exposeHost: () => HOST,
    fetchActive: vi.fn(async () => answer),
    fetchJson: vi.fn(async () => PRESET),
    importModule: vi.fn(async () => themeModule),
    doc: document,
    timeoutMs: TIMEOUT_MS,
  }
}

/** Fires `load` on the theme's stylesheet as soon as it is appended. */
function settleStylesheets(): MutationObserver {
  const observer = new MutationObserver(() => {
    document.querySelectorAll(STYLE_LINK).forEach((link) => link.dispatchEvent(new Event('load')))
  })
  observer.observe(document.head, { childList: true })
  return observer
}

/** The reason a step fails with, run the way the loader runs it (step + settleLoad). */
async function reasonAt(reason: ThemeFailureReason, run: () => Promise<unknown>): Promise<ThemeFailureReason | null> {
  const result: ThemeLoadResult = await settleLoad(() => step(reason, run) as Promise<ThemeLoadOk>)
  return result.ok ? null : result.reason
}

const importFails = async (): Promise<unknown> => {
  throw new TypeError('Failed to fetch dynamically imported module')
}

let observer: MutationObserver | null = null

afterEach(() => {
  observer?.disconnect()
  observer = null
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  document.querySelectorAll(STYLE_LINK).forEach((link) => link.remove())
  document.documentElement.className = ''
  document.documentElement.removeAttribute('style')
})

describe('the opt-in the active-theme answer asks for', () => {
  it('reads optIn.required, and nothing else, as needing an opt-in', () => {
    expect(optInRequired(activeWith({ optIn: { required: true, wording: 'x', packageId: 'p' } }))).toBe(true)
    expect(optInRequired(activeWith({ optIn: { required: false } }))).toBe(false)
    expect(optInRequired(activeWith())).toBe(false)
    expect(optInRequired({ choice: 'studio', theme: null })).toBe(false)
  })

  it("fails as 'opt-in', asks for no theme file, and never counts toward the revert", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const deps = depsWith(activeWith({ optIn: { required: true, wording: 'Run its code?', packageId: 'p' } }))
    const result = await installedThemeSource(INSTALL_ID).load(deps)
    expect(result.ok ? null : result.reason).toBe('opt-in')
    expect(deps.importModule).not.toHaveBeenCalled()
    expect(deps.fetchJson).not.toHaveBeenCalled()
    expect(countsTowardRevert('opt-in')).toBe(false)
  })
})

describe('a warn revocation of the running version', () => {
  it('reads only the warn level, with its reason', () => {
    expect(revocationWarningOf(activeWith({ revoked: WARN }))).toEqual(WARN)
    expect(revocationWarningOf(activeWith({ revoked: { level: 'warn' } }))).toEqual({ level: 'warn', reason: '' })
    expect(revocationWarningOf(activeWith({ revoked: { level: 'disable', reason: 'gone' } }))).toBeNull()
    expect(revocationWarningOf(activeWith({ revoked: null }))).toBeNull()
  })

  it('loads the theme and hands the warning on for the notice', async () => {
    observer = settleStylesheets()
    const result = await installedThemeSource(INSTALL_ID).load(depsWith(activeWith({ revoked: WARN })))
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.revoked).toEqual(WARN)
    expect(result.frame).toBe(Frame)
    result.cleanup()
  })

  it('hands on no warning without one', async () => {
    observer = settleStylesheets()
    const result = await installedThemeSource(INSTALL_ID).load(depsWith(activeWith({ revoked: null })))
    expect(result.ok && result.revoked == null).toBe(true)
    if (result.ok) result.cleanup()
  })
})

describe('theme files that cannot be reached', () => {
  it('reads a 5xx or no answer as unavailable, and nothing else', () => {
    for (const status of [500, 502, 503, null]) expect(isUnavailableStatus(status)).toBe(true)
    for (const status of [200, 206, 401, 403, 404]) expect(isUnavailableStatus(status)).toBe(false)
  })

  it("fails a failed entry import as 'unavailable' when the entry answers 5xx or the probe gets no answer", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    for (const status of [500, 502, 503, null]) {
      expect(await reasonAt('load', () => importInstalledEntry(ENTRY_URL, importFails, async () => status))).toBe('unavailable')
    }
  })

  it("fails the preset as 'unavailable' on a 5xx or a network failure", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    for (const status of [502, 503]) {
      vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status })))
      expect(await reasonAt('look', () => defaultLoaderDeps().fetchJson(PRESET_URL))).toBe('unavailable')
    }
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('network down')
    }))
    expect(await reasonAt('look', () => defaultLoaderDeps().fetchJson(PRESET_URL))).toBe('unavailable')
  })

  it('never counts toward the revert', () => {
    expect(countsTowardRevert('unavailable')).toBe(false)
  })
})

describe('a real automatos-theme/v1 package (the hello fixture hello.autotheme is built from)', () => {
  const packageDir = path.resolve(__dirname, '../../../../orchestrator/tests/fixtures/themes/hello')
  const readJson = (file: string): Record<string, unknown> =>
    JSON.parse(readFileSync(path.join(packageDir, file), 'utf8')) as Record<string, unknown>
  const root = `${window.location.origin}/api/themes/files/${DIGEST}/`

  it('loads from the package layout: dist/entry.js, dist/theme.css and its default look', async () => {
    observer = settleStylesheets()
    const deps: LoaderDeps = {
      ...depsWith(activeWith({ manifest: readJson('theme.json'), defaultLook: 'default' })),
      fetchJson: vi.fn(async () => readJson('looks/presets/default.look.json')),
    }
    const result = await installedThemeSource(INSTALL_ID).load(deps)
    expect(result.ok).toBe(true)
    expect(deps.importModule).toHaveBeenCalledWith(`${root}dist/entry.js`)
    expect(deps.fetchJson).toHaveBeenCalledWith(`${root}looks/presets/default.look.json`)
    expect(document.querySelector(STYLE_LINK)?.getAttribute('href')).toBe(`${root}dist/theme.css`)
    expect(document.documentElement.classList.contains('hello')).toBe(true)
    if (result.ok) result.cleanup()
  })
})
