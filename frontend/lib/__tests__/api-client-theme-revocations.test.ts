/**
 * PRD-261 W-C4 opening commit (C-05j, plan §2.4): the routes W-C3 wired (api/admin_theme_revocations.py,
 * api/admin_theme_revocation_sync.py, admin_themes.py's review-package, themes_import.py's signature
 * part) are named apiClient methods with literal paths, so the route-contract check sees them and the
 * three W-C4 lanes never edit api-client.ts. Each method is pinned to its path, its HTTP method and its
 * body; every (method, path) pair is checked against orchestrator/reports/route-manifest.json; and the
 * answer types are held to the orchestrator's dicts, read as text (as api-client-theme-signature.test.ts
 * does): a key the Python adds or drops fails here, and the typed samples fail tsc on a key the type
 * lacks or misses.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  apiClient,
  type InstalledTheme,
  type ThemeReviewPackageLink,
  type ThemeRevocationRow,
  type ThemeRevocationSyncStatus,
} from '@/lib/api-client'

const ORCHESTRATOR = path.resolve(__dirname, '..', '..', '..', 'orchestrator')
const MANIFEST = path.join(ORCHESTRATOR, 'reports', 'route-manifest.json')
const THEMES_PY = path.join(ORCHESTRATOR, 'services', 'themes')
const REVOCATION_ID = '7a0c2f4e-5b1d-4e8a-9c3f-2d6b8e1a4c70'
const PACKAGE_ID = '6f1c1d7e-1d1b-4a8e-9a55-3c1f7f2b9a10'
const DIGEST = 'a'.repeat(64)
const KEY_ID = 'ed25519:0123456789abcdef'
const WHEN = '2026-10-23T09:00:00+00:00'

type Call = [string, RequestInit?]

const ROW: ThemeRevocationRow = {
  id: REVOCATION_ID, digest: DIGEST, themeId: null, versionRange: null, level: 'disable', reason: 'Steals tokens.',
  origin: 'local', createdAt: WHEN, liftedAt: null,
}
const SYNC: ThemeRevocationSyncStatus = {
  enabled: true, lastAttemptAt: WHEN, lastSuccessAt: WHEN, lastError: null, feedGeneratedAt: WHEN, rows: 2,
  revokedKeys: [KEY_ID], stale: false,
}
const LINK: ThemeReviewPackageLink = { url: 'https://storage.example/review.autotheme', digest: DIGEST, expiresAt: WHEN }
const INSTALL_SIGNATURE: Pick<InstalledTheme, 'signature'> = { signature: { state: 'verified', keyId: KEY_ID } }

function spyRequest() {
  return vi.spyOn(apiClient, 'request').mockResolvedValue({} as never)
}

/** Every W-C4 method once (8 calls). */
async function callEveryMethod(): Promise<void> {
  await apiClient.getThemeRevocations()
  await apiClient.createThemeRevocation({ digest: DIGEST, level: 'disable', reason: 'Steals tokens.' })
  await apiClient.liftThemeRevocation(REVOCATION_ID)
  await apiClient.getThemeRevocationSync()
  await apiClient.pullThemeRevocations()
  await apiClient.getThemeReviewPackage(PACKAGE_ID)
  await apiClient.importTheme(new File(['zip'], 'hello.autotheme'))
  await apiClient.importTheme(new File(['zip'], 'hello.autotheme'), new File(['{}'], 'hello.sig.json'))
}

/** `{param}` segments match any one segment; the query string is not part of the route. */
function manifestPattern(routePath: string): RegExp {
  const segments = routePath.split('/').map((segment) =>
    segment.startsWith('{') && segment.endsWith('}') ? '[^/]+' : segment.replace(/[.*+?^$()|[\]\\]/g, '\\$&'),
  )
  return new RegExp(`^${segments.join('/')}$`)
}

/** A function's source, from its `def` line to the next statement at its own indent or less. */
function pythonFunction(source: string, name: string): string {
  const head = new RegExp(`^( *)def ${name}\\(`, 'm').exec(source)
  expect(head, `def ${name} not found`).not.toBeNull()
  const rest = source.slice(head!.index)
  const next = new RegExp(`\\n {0,${head![1].length}}[A-Za-z_@]`).exec(rest.slice(1))
  return next === null ? rest : rest.slice(0, next.index + 1)
}

/** The keys of the dict a function's (last) `return {...}` builds. */
function returnedKeys(body: string): string[] {
  const returned = /return (\{[^{}]*\})\s*$/.exec(body.trimEnd())
  expect(returned, 'no `return {...}` at the end of the function').not.toBeNull()
  return Array.from(returned![1].matchAll(/"(\w+)":/g), (match) => match[1]).sort()
}

const keysOf = (sample: object): string[] => Object.keys(sample).sort()
const readThemesModule = (name: string): string => readFileSync(path.join(THEMES_PY, name), 'utf8')

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.spyOn(console, 'log').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('apiClient revocation methods (C-05e, C-05h)', () => {
  it('lists with GET (with an optional limit), and reads the sync status with GET', async () => {
    const spy = spyRequest()
    await apiClient.getThemeRevocations()
    await apiClient.getThemeRevocations(50)
    await apiClient.getThemeRevocationSync()
    expect(spy.mock.calls).toEqual([
      ['/api/admin/themes/revocations'],
      ['/api/admin/themes/revocations?limit=50'],
      ['/api/admin/themes/revocations/sync'],
    ])
  })

  it('creates with the body the route takes: a digest, or a theme id with an optional range', async () => {
    const spy = spyRequest()
    await apiClient.createThemeRevocation({ digest: DIGEST, level: 'disable', reason: 'Steals tokens.' })
    await apiClient.createThemeRevocation({ themeId: 'acme/ink', versionRange: '^1.0.0', level: 'warn', reason: 'Update.' })
    await apiClient.createThemeRevocation({ themeId: 'acme/ink', level: 'warn', reason: 'Every version.' })
    const post = (body: object) => ['/api/admin/themes/revocations', { method: 'POST', body: JSON.stringify(body) }]
    expect(spy.mock.calls).toEqual([
      post({ digest: DIGEST, level: 'disable', reason: 'Steals tokens.' }),
      post({ themeId: 'acme/ink', versionRange: '^1.0.0', level: 'warn', reason: 'Update.' }),
      post({ themeId: 'acme/ink', level: 'warn', reason: 'Every version.' }),
    ])
  })

  it('lifts and pulls with POST and no body; the revocation id is one encoded segment', async () => {
    const spy = spyRequest()
    await apiClient.liftThemeRevocation(REVOCATION_ID)
    await apiClient.liftThemeRevocation('../sync')
    await apiClient.pullThemeRevocations()
    expect(spy.mock.calls).toEqual([
      [`/api/admin/themes/revocations/${REVOCATION_ID}/lift`, { method: 'POST' }],
      ['/api/admin/themes/revocations/..%2Fsync/lift', { method: 'POST' }],
      ['/api/admin/themes/revocations/sync/pull', { method: 'POST' }],
    ])
  })

  it("carries Check now's 502 to the caller with the pull's own error and status", async () => {
    const detail = "The revocation feed couldn't be reached (timed out)."
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail }), { status: 502 })))
    const error = await apiClient.pullThemeRevocations().then(() => null, (thrown: { message: string; status?: number }) => thrown)
    expect(error?.message).toBe(detail)
    expect(error?.status).toBe(502)
  })
})

describe('apiClient review job inputs and the signed import (C-03c, C-05d)', () => {
  it('reads the review package link with GET under the package id', async () => {
    const spy = spyRequest()
    await apiClient.getThemeReviewPackage(PACKAGE_ID)
    expect(spy).toHaveBeenCalledWith(`/api/admin/themes/${PACKAGE_ID}/review-package`)
  })

  it('sends the .sig.json as the `signature` part beside `file`, and no such part without one', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ duplicate: false }), { status: 201 }))
    vi.stubGlobal('fetch', fetchMock)
    await apiClient.importTheme(new File(['zip'], 'hello.autotheme'), new File(['{}'], 'hello.sig.json'))
    await apiClient.importTheme(new File(['zip'], 'hello.autotheme'))

    const imports = (fetchMock.mock.calls as unknown as [string, RequestInit][])
      .filter(([url]) => url.endsWith('/api/themes/import'))
    expect(imports.map(([, init]) => init.method)).toEqual(['POST', 'POST'])
    const forms = imports.map(([, init]) => init.body as FormData)
    expect((forms[0].get('file') as File).name).toBe('hello.autotheme')
    expect((forms[0].get('signature') as File).name).toBe('hello.sig.json')
    expect((forms[1].get('file') as File).name).toBe('hello.autotheme')
    expect(forms[1].get('signature')).toBeNull()
  })
})

describe('the route manifest serves every W-C4 call', () => {
  it('matches each (method, path) to a route W-C3 registered', async () => {
    const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8')) as { routes: { method: string; path: string }[] }
    const spy = spyRequest()
    await callEveryMethod()

    const calls = spy.mock.calls as unknown as Call[]
    expect(calls).toHaveLength(8)
    const unserved = calls
      .map(([endpoint, init]) => ({ method: init?.method ?? 'GET', path: endpoint.split('?')[0] }))
      .filter((call) => !manifest.routes.some((route) =>
        route.method === call.method && manifestPattern(route.path).test(call.path)))
    expect(unserved).toEqual([])
  })
})

describe("the answer types are the orchestrator's dicts", () => {
  it("ThemeRevocationRow's keys are revocation_admin.revocation_view's", () => {
    expect(keysOf(ROW)).toEqual(returnedKeys(pythonFunction(readThemesModule('revocation_admin.py'), 'revocation_view')))
  })

  it("ThemeRevocationSyncStatus's keys are revocation_pull.sync_status's", () => {
    expect(keysOf(SYNC)).toEqual(returnedKeys(pythonFunction(readThemesModule('revocation_pull.py'), 'sync_status')))
  })

  it("ThemeReviewPackageLink's keys are review_package.review_link's", () => {
    expect(keysOf(LINK)).toEqual(returnedKeys(pythonFunction(readThemesModule('review_package.py'), 'review_link')))
  })

  it('an installed row carries the signature view (updates.install_row, C-05d)', () => {
    expect(pythonFunction(readThemesModule('updates.py'), 'install_row')).toMatch(/"signature": signature_view\(/)
    expect(keysOf(INSTALL_SIGNATURE)).toEqual(['signature'])
  })
})
