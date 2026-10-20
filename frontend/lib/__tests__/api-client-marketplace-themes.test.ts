/**
 * PRD-261 W-C2 opening commit (plan §2.2, §2.4): the Marketplace theme routes W-C1 shipped
 * (orchestrator/api/marketplace_themes.py, marketplace_themes_submit.py, admin_themes.py) are
 * named apiClient methods with literal paths, so the route-contract check sees them and the
 * three UI lanes (C-01b Publish theme, C-03b admin queue, C-04b the Themes tab) never edit
 * api-client.ts. Each method is pinned to its path, its HTTP method and its body, and every
 * (method, path) pair is checked against orchestrator/reports/route-manifest.json, which the
 * route-contract check does not do (it compares paths only).
 */
import { readFileSync } from 'fs'
import path from 'path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type ApiHttpError } from '@/lib/api-client'

const MANIFEST = path.resolve(__dirname, '..', '..', '..', 'orchestrator', 'reports', 'route-manifest.json')
const LISTING_ID = 'acme.hello'
const PACKAGE_ID = '6f1c1d7e-1d1b-4a8e-9a55-3c1f7f2b9a10'
const REQUEST_ID = '0b9e3c55-8d0e-4c64-9d3c-5a2a1e7c4f21'

type Call = [string, RequestInit?]

function spyRequest() {
  return vi.spyOn(apiClient, 'request').mockResolvedValue({} as never)
}

/** Every Marketplace theme method once (11 calls). */
async function callEveryMethod(): Promise<void> {
  await apiClient.getThemeListings()
  await apiClient.getThemeListing(LISTING_ID)
  await apiClient.installThemeListing(LISTING_ID)
  await apiClient.submitTheme(new File(['zip'], 'hello.autotheme'))
  await apiClient.getPendingThemes()
  await apiClient.getThemeReport(PACKAGE_ID)
  await apiClient.addThemeReportNote(PACKAGE_ID, { summary: 'Outcome: pass', outcome: 'pass' })
  await apiClient.approveTheme(PACKAGE_ID)
  await apiClient.rejectTheme(PACKAGE_ID, 'Copies another theme.')
  await apiClient.deactivateTheme(PACKAGE_ID, 'Breaks the chat page.')
  await apiClient.storeThemeSignature(PACKAGE_ID, REQUEST_ID, 'c2lnbmF0dXJl')
}

/** `{param}` segments match any one segment; the query string is not part of the route. */
function manifestPattern(routePath: string): RegExp {
  const segments = routePath.split('/').map((segment) =>
    segment.startsWith('{') && segment.endsWith('}') ? '[^/]+' : segment.replace(/[.*+?^$()|[\]\\]/g, '\\$&'),
  )
  return new RegExp(`^${segments.join('/')}$`)
}

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.spyOn(console, 'log').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('apiClient Marketplace theme methods (C-04a, C-01a)', () => {
  it('reads the listings and one listing with GET, the listing id as one encoded segment', async () => {
    const spy = spyRequest()
    await apiClient.getThemeListings()
    await apiClient.getThemeListing(LISTING_ID)
    await apiClient.getThemeListing('acme/../admin')
    expect(spy.mock.calls).toEqual([
      ['/api/marketplace/themes'],
      ['/api/marketplace/themes/acme.hello'],
      ['/api/marketplace/themes/acme%2F..%2Fadmin'],
    ])
  })

  it('installs the newest compatible version with an empty body, or the version asked for', async () => {
    const spy = spyRequest()
    await apiClient.installThemeListing(LISTING_ID)
    await apiClient.installThemeListing(LISTING_ID, PACKAGE_ID)
    expect(spy.mock.calls).toEqual([
      ['/api/marketplace/themes/acme.hello/install', { method: 'POST', body: '{}' }],
      ['/api/marketplace/themes/acme.hello/install', { method: 'POST', body: JSON.stringify({ packageId: PACKAGE_ID }) }],
    ])
  })

  it('submits the .autotheme to fetch as multipart form data under "file"', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ duplicate: false }), { status: 201 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(apiClient.submitTheme(new File(['zip'], 'hello.autotheme'))).resolves.toEqual({ duplicate: false })

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url.endsWith('/api/marketplace/themes/submit')).toBe(true)
    expect(init.method).toBe('POST')
    expect(init.body).toBeInstanceOf(FormData)
    expect(((init.body as FormData).get('file') as File).name).toBe('hello.autotheme')
  })

  it('carries the install refusal (409) to the caller with its message and status', async () => {
    const detail = 'This workspace has acme/hello 1.0.0 installed. Update it from Settings, Appearance, Themes.'
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail }), { status: 409 })))

    const error = await apiClient.installThemeListing(LISTING_ID).then(
      () => null,
      (thrown: ApiHttpError) => thrown,
    )
    expect(error?.message).toBe(detail)
    expect(error?.status).toBe(409)
  })
})

describe('apiClient admin theme review methods (C-03a)', () => {
  it('reads the queue (with an optional limit) and one report with GET', async () => {
    const spy = spyRequest()
    await apiClient.getPendingThemes()
    await apiClient.getPendingThemes(20)
    await apiClient.getThemeReport(PACKAGE_ID)
    expect(spy.mock.calls).toEqual([
      ['/api/admin/themes/pending'],
      ['/api/admin/themes/pending?limit=20'],
      [`/api/admin/themes/${PACKAGE_ID}/report`],
    ])
  })

  it('POSTs the note, approve, reject, deactivate and signature with the bodies the routes accept', async () => {
    const spy = spyRequest()
    await apiClient.addThemeReportNote(PACKAGE_ID, { summary: 'Outcome: pass', outcome: 'pass' })
    await apiClient.approveTheme(PACKAGE_ID)
    await apiClient.rejectTheme(PACKAGE_ID, 'Copies another theme.')
    await apiClient.deactivateTheme(PACKAGE_ID, 'Breaks the chat page.')
    await apiClient.storeThemeSignature(PACKAGE_ID, REQUEST_ID, 'c2lnbmF0dXJl')
    const base = `/api/admin/themes/${PACKAGE_ID}`
    expect(spy.mock.calls).toEqual([
      [`${base}/report-note`, { method: 'POST', body: JSON.stringify({ summary: 'Outcome: pass', outcome: 'pass' }) }],
      [`${base}/approve`, { method: 'POST' }],
      [`${base}/reject`, { method: 'POST', body: JSON.stringify({ reason: 'Copies another theme.' }) }],
      [`${base}/deactivate`, { method: 'POST', body: JSON.stringify({ reason: 'Breaks the chat page.' }) }],
      [`${base}/signature`, { method: 'POST', body: JSON.stringify({ requestId: REQUEST_ID, signature: 'c2lnbmF0dXJl' }) }],
    ])
  })

  it('encodes a package id that is not a plain uuid', async () => {
    const spy = spyRequest()
    await apiClient.approveTheme('../pending')
    expect(spy).toHaveBeenCalledWith('/api/admin/themes/..%2Fpending/approve', { method: 'POST' })
  })
})

describe('the route manifest serves every Marketplace theme call', () => {
  it('matches each (method, path) to a route W-C1 registered', async () => {
    const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8')) as { routes: { method: string; path: string }[] }
    const spy = spyRequest()
    await callEveryMethod()

    const calls = spy.mock.calls as unknown as Call[]
    expect(calls).toHaveLength(11)
    const unserved = calls
      .map(([endpoint, init]) => ({ method: init?.method ?? 'GET', path: endpoint.split('?')[0] }))
      .filter((call) => !manifest.routes.some((route) =>
        route.method === call.method && manifestPattern(route.path).test(call.path)))
    expect(unserved).toEqual([])
  })
})
