/**
 * PRD-261 A-08b — a refused session fails an installed theme's load as `auth`, not as the theme's
 * own failure: a 401 or 403 on GET /api/themes/active (apiClient's 401 error carries no status, so
 * its message is read), on the preset (a same-origin file) or on the entry. import() reports no
 * status, so after a failed entry import one byte of the entry is fetched again to read it. Any
 * other failure keeps its reason ('manifest', 'load', 'look'), except a 5xx or no answer from the
 * files proxy, which is 'unavailable' (sources-active-notices.test.ts).
 */
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest'

const api = vi.hoisted(() => ({ getActiveTheme: vi.fn() }))
const native = vi.hoisted(() => ({ importThemeModule: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { getActiveTheme: api.getActiveTheme } }))
vi.mock('../native-import', () => ({ importThemeModule: native.importThemeModule }))

import { settleLoad, step } from '../loader'
import {
  defaultLoaderDeps,
  fetchActiveInstall,
  importInstalledEntry,
  installedThemeSource,
  isAuthRefusal,
  themeFileStatus,
} from '../sources'
import type { LoaderDeps, ThemeFailureReason, ThemeHostGlobal, ThemeLoadOk, ThemeLoadResult } from '../types'

const INSTALL_ID = '00000000-0000-4000-8000-0000000a08b0'
const ENTRY_URL = `/api/themes/files/${'c'.repeat(64)}/dist/entry.js`
const PRESET_URL = `/api/themes/files/${'c'.repeat(64)}/looks/presets/default.look.json`
const HOST: ThemeHostGlobal = Object.freeze({ sdkVersion: '1.0.0', modules: Object.freeze({}) })
/** What apiClient throws for a 401 (api-client.ts httpErrorOf): a message naming the sign-in, no status. */
const API_CLIENT_401 = new Error('HTTP 401: Unauthorized (missing/invalid Clerk token). Make sure you are signed in.')

function httpError(status: number): Error {
  return Object.assign(new Error(`HTTP ${status}`), { status })
}

function depsWith(overrides: Partial<LoaderDeps> = {}): LoaderDeps {
  return {
    exposeHost: vi.fn(() => HOST),
    fetchActive: vi.fn(async () => ({ choice: 'studio', theme: null })),
    fetchJson: vi.fn(async () => ({})),
    importModule: vi.fn(async () => ({})),
    doc: document,
    timeoutMs: 1_000,
    ...overrides,
  }
}

/** The reason a step fails with, run the way the loader runs it (step + settleLoad). */
async function reasonAt(reason: ThemeFailureReason, run: () => Promise<unknown>): Promise<ThemeFailureReason | null> {
  const result: ThemeLoadResult = await settleLoad(() => step(reason, run) as Promise<ThemeLoadOk>)
  return result.ok ? null : result.reason
}

type FetchCall = (url: string, init?: RequestInit) => Promise<Response>

function stubFetchStatus(status: number): Mock<FetchCall> {
  const fetchMock = vi.fn<FetchCall>(async () => new Response('{}', { status }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  api.getActiveTheme.mockReset()
  native.importThemeModule.mockReset()
})

describe('PRD-261 A-08b — isAuthRefusal', () => {
  it.each([
    ['a 401', httpError(401), true],
    ['a 403', httpError(403), true],
    ["apiClient's 401, which has no status", API_CLIENT_401, true],
    ['a 404', httpError(404), false],
    ['a 500', httpError(500), false],
    ['an error with no status', new Error('boom'), false],
    ['not an error', 'HTTP 401', false],
  ])('reads %s', (_case, error, refused) => {
    expect(isAuthRefusal(error)).toBe(refused)
  })
})

describe('PRD-261 A-08b — GET /api/themes/active refused', () => {
  it.each([
    ["apiClient's 401", API_CLIENT_401],
    ['a 403', httpError(403)],
  ])("fails the load as 'auth' on %s, and requests no theme file", async (_case, error) => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    api.getActiveTheme.mockRejectedValue(error)
    const deps = depsWith({ fetchActive: fetchActiveInstall })
    const result = await installedThemeSource(INSTALL_ID).load(deps)
    expect(result.ok ? null : result.reason).toBe('auth')
    expect(deps.importModule).not.toHaveBeenCalled()
    expect(deps.fetchJson).not.toHaveBeenCalled()
  })

  it("keeps 'manifest' for any other failure of the read", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    api.getActiveTheme.mockRejectedValue(httpError(500))
    const result = await installedThemeSource(INSTALL_ID).load(depsWith({ fetchActive: fetchActiveInstall }))
    expect(result.ok ? null : result.reason).toBe('manifest')
  })
})

describe('PRD-261 A-08b — the entry refused', () => {
  const importFails = async (): Promise<unknown> => {
    throw new TypeError('Failed to fetch dynamically imported module')
  }

  it("fails as 'auth' when the import fails and the entry answers 401 or 403", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    expect(await reasonAt('load', () => importInstalledEntry(ENTRY_URL, importFails, async () => 401))).toBe('auth')
    expect(await reasonAt('load', () => importInstalledEntry(ENTRY_URL, importFails, async () => 403))).toBe('auth')
  })

  it("keeps 'load' when the entry answers anything else (a 5xx or no answer is 'unavailable')", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    for (const status of [206, 200, 404]) {
      expect(await reasonAt('load', () => importInstalledEntry(ENTRY_URL, importFails, async () => status))).toBe('load')
    }
  })

  it('probes nothing when the import succeeds', async () => {
    const probe = vi.fn(async () => 401)
    const loaded = { default: { register() {} } }
    await expect(importInstalledEntry(ENTRY_URL, async () => loaded, probe)).resolves.toBe(loaded)
    expect(probe).not.toHaveBeenCalled()
  })

  it("is the default loader's import: the browser's import(), then the status probe", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    native.importThemeModule.mockRejectedValue(new TypeError('Failed to fetch dynamically imported module'))
    const fetchMock = stubFetchStatus(401)
    expect(await reasonAt('load', () => defaultLoaderDeps().importModule(ENTRY_URL))).toBe('auth')
    expect(native.importThemeModule).toHaveBeenCalledWith(ENTRY_URL)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('PRD-261 A-08b — themeFileStatus', () => {
  it('asks for one byte on the same origin and returns the status', async () => {
    const fetchMock = stubFetchStatus(401)
    await expect(themeFileStatus(ENTRY_URL)).resolves.toBe(401)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe(ENTRY_URL)
    expect(init?.credentials).toBe('same-origin')
    expect(init?.headers).toEqual({ Range: 'bytes=0-0' })
  })

  it('returns null, and logs, when the probe itself fails', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('network down')
    }))
    await expect(themeFileStatus(ENTRY_URL)).resolves.toBeNull()
    expect(logged).toHaveBeenCalled()
  })
})

describe('PRD-261 A-08b — the preset refused', () => {
  it("fails as 'auth' on a 401 or 403, 'unavailable' on a 5xx, and keeps 'look' otherwise", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    for (const [status, reason] of [[401, 'auth'], [403, 'auth'], [404, 'look'], [500, 'unavailable']] as const) {
      stubFetchStatus(status)
      expect(await reasonAt('look', () => defaultLoaderDeps().fetchJson(PRESET_URL))).toBe(reason)
    }
  })
})
