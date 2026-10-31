/**
 * PRD-261 M-04 (plan §2.6, §2.2.11): GET /api/themes/mobile/{install_id}, the member's theme as data
 * for the phone (orchestrator/api/themes_mobile.py), is the named apiClient method getMobileTheme with
 * a literal path, so the route-contract check (scripts/check-route-contract.js, which reads only
 * api-client.ts) sees it. Pinned: the path and the GET, the install id as one encoded segment, the
 * answer and a 404 carried to the caller, the (method, path) pair in
 * orchestrator/reports/route-manifest.json, and a method body that goes through this.request, never a
 * raw fetch. PRD-271 C03: the answer carries the theme's phone layout, its contract, tone and slots.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type ApiHttpError, type MobileTheme } from '@/lib/api-client'

const API_CLIENT = path.resolve(__dirname, '..', 'api-client.ts')
const MANIFEST = path.resolve(__dirname, '..', '..', '..', 'orchestrator', 'reports', 'route-manifest.json')
const ROUTE = '/api/themes/mobile/{install_id}'
const INSTALL_ID = '6f1c1d7e-1d1b-4a8e-9a55-3c1f7f2b9a10'
const DIGEST = 'a'.repeat(64)
const FILES = `/api/themes/files/${DIGEST}`

const ANSWER: MobileTheme = {
  schema: 1,
  installId: INSTALL_ID,
  themeId: 'acme/harbour',
  version: '1.1.0',
  mobile: {
    source: 'fromLooks',
    tone: 'dark',
    tokens: { light: null, dark: { '--background': '222 47% 8%', '--foreground': '240 4.76% 95.88%' } },
    fonts: { sans: `${FILES}/assets/fonts/mobile/harbour-sans.ttf` },
    mascot: {
      mode: 'clips',
      states: [{ name: 'idle', clip: `${FILES}/assets/mascot/idle.mp4`, still: `${FILES}/assets/mascot/idle.png` }],
    },
    layout: {
      contract: 'hud@1',
      tone: 'dark',
      slots: { bg: '#030507', grid: 14, scan: 'on', accent: '#3FE0FF', accentInk: '#021116', glow: 22, auto: 'on' },
    },
  },
}

type Call = [string, RequestInit?]

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

/** `{param}` segments match any one segment; the query string is not part of the route. */
function manifestPattern(routePath: string): RegExp {
  const segments = routePath.split('/').map((segment) =>
    segment.startsWith('{') && segment.endsWith('}') ? '[^/]+' : segment.replace(/[.*+?^$()|[\]\\]/g, '\\$&'),
  )
  return new RegExp(`^${segments.join('/')}$`)
}

/** The source of one apiClient method, from its signature to its closing brace. */
function methodSource(name: string): string {
  const source = readFileSync(API_CLIENT, 'utf8')
  const start = source.indexOf(`  async ${name}(`)
  expect(start).toBeGreaterThan(-1)
  return source.slice(start, source.indexOf('\n  }\n', start))
}

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.spyOn(console, 'log').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('apiClient.getMobileTheme (M-04)', () => {
  it('GETs /api/themes/mobile/<id> through apiClient.request, the id as one encoded segment', async () => {
    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue(ANSWER as never)
    await expect(apiClient.getMobileTheme(INSTALL_ID)).resolves.toEqual(ANSWER)
    await apiClient.getMobileTheme('../files')
    expect(spy.mock.calls).toEqual([[`/api/themes/mobile/${INSTALL_ID}`], ['/api/themes/mobile/..%2Ffiles']])
  })

  it('hands the answer back as the server sent it, over a GET', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, ANSWER))
    vi.stubGlobal('fetch', fetchMock)

    await expect(apiClient.getMobileTheme(INSTALL_ID)).resolves.toEqual(ANSWER)

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url.endsWith(`/api/themes/mobile/${INSTALL_ID}`)).toBe(true)
    expect(init.method ?? 'GET').toBe('GET')
  })

  it("carries the route's one 404 to the caller with its status and message", async () => {
    const detail = "That theme isn't installed in any workspace you belong to."
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(404, { detail })))

    const error = await apiClient.getMobileTheme(INSTALL_ID).then(
      () => null,
      (thrown: ApiHttpError) => thrown,
    )
    expect(error?.status).toBe(404)
    expect(error?.message).toBe(detail)
  })

  it('names the path literally in this.request and never calls fetch itself', () => {
    const body = methodSource('getMobileTheme')
    expect(body).toContain('this.request<MobileTheme>(`/api/themes/mobile/${encodeURIComponent(installId)}`)')
    expect(body).not.toMatch(/\bfetch\(/)
  })
})

describe('the route manifest serves getMobileTheme', () => {
  it('lists GET /api/themes/mobile/{install_id} once, and the call matches it', async () => {
    const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8')) as {
      route_count: number
      routes: { method: string; path: string }[]
    }
    expect(manifest.route_count).toBe(manifest.routes.length)
    expect(manifest.routes.filter((route) => route.path === ROUTE)).toEqual([{ method: 'GET', path: ROUTE }])

    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue(ANSWER as never)
    await apiClient.getMobileTheme(INSTALL_ID)
    const [[endpoint, init]] = spy.mock.calls as unknown as Call[]
    const served = manifest.routes.filter(
      (route) => route.method === (init?.method ?? 'GET') && manifestPattern(route.path).test(endpoint),
    )
    expect(served).toEqual([{ method: 'GET', path: ROUTE }])
  })
})
