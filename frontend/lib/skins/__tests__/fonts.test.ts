/**
 * PRD-259 S-004a — a skin's woff2 faces: fetched through the API client's seam with its auth
 * headers, handed to FontFace as bytes under Studio's own family name, and removed again; a
 * face the browser refuses, or a file the route won't serve, leaves the base font.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerSkinFonts, skinAssetPath, skinFontFiles } from '@/lib/skins/fonts'

const BASE = 'http://api.test'
const ID = '0f8c2b6e-4d1a-4c3e-9b7a-2e5d6f8a1c90'
const AUTH = { Authorization: 'Bearer test-token' }

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    getBaseUrl: () => 'http://api.test',
    getAuthHeaders: async () => ({ Authorization: 'Bearer test-token' }),
  },
}))

interface FakeFace {
  family: string
  source: unknown
}

const made: FakeFace[] = []
const refuse = new Set<string>()
const fontSet = { add: vi.fn(), delete: vi.fn() }

class FakeFontFace implements FakeFace {
  constructor(public family: string, public source: unknown) {
    made.push(this)
  }

  load(): Promise<this> {
    return refuse.has(this.family) ? Promise.reject(new Error('refused')) : Promise.resolve(this)
  }
}

type FetchFn = (url: string, init?: RequestInit) => Promise<unknown>

function serve(status: number) {
  const bytes = new ArrayBuffer(8)
  return vi.fn<FetchFn>(async () => ({ ok: status < 400, status, blob: async () => ({ arrayBuffer: async () => bytes }) }))
}

const SKIN = {
  version: '1.2.0',
  fonts: { sans: 'assets/fonts/Brand_Sans.woff2', serif: 'newsreader', mono: 'assets/fonts/Brand-Mono.woff2' },
}

beforeEach(() => {
  made.length = 0
  refuse.clear()
  fontSet.add.mockClear()
  fontSet.delete.mockClear()
  vi.stubGlobal('FontFace', FakeFontFace)
  Object.defineProperty(document, 'fonts', { value: fontSet, configurable: true })
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('registerSkinFonts', () => {
  it('fetches each woff2 with the auth headers and registers it under skin-<id>-<slot>', async () => {
    const fetchMock = serve(200)
    vi.stubGlobal('fetch', fetchMock)
    const remove = await registerSkinFonts(SKIN, ID)
    expect(fetchMock.mock.calls.map(([url, init]) => [url, init])).toEqual([
      [`${BASE}/api/skins/${ID}/assets/assets/fonts/Brand_Sans.woff2?v=1.2.0`, { headers: AUTH }],
      [`${BASE}/api/skins/${ID}/assets/assets/fonts/Brand-Mono.woff2?v=1.2.0`, { headers: AUTH }],
    ])
    expect(made.map((face) => face.family)).toEqual([`skin-${ID}-sans`, `skin-${ID}-mono`])
    expect(made.every((face) => face.source instanceof ArrayBuffer)).toBe(true)
    expect(fontSet.add).toHaveBeenCalledTimes(2)
    remove()
    expect(fontSet.delete).toHaveBeenCalledTimes(2)
  })

  it('leaves a slot on its base font when the browser refuses the face: logged, not thrown', async () => {
    vi.stubGlobal('fetch', serve(200))
    refuse.add(`skin-${ID}-sans`)
    await expect(registerSkinFonts(SKIN, ID)).resolves.toBeTypeOf('function')
    expect(fontSet.add).toHaveBeenCalledTimes(1)
    expect(console.warn).toHaveBeenCalled()
  })

  it('registers nothing when the skin changed while the files were loading', async () => {
    vi.stubGlobal('fetch', serve(200))
    const controller = new AbortController()
    const pending = registerSkinFonts(SKIN, ID, controller.signal)
    controller.abort()
    await pending
    expect(fontSet.add).not.toHaveBeenCalled()
  })

  it('registers nothing when the route will not serve the file', async () => {
    vi.stubGlobal('fetch', serve(404))
    await registerSkinFonts(SKIN, ID)
    expect(made).toEqual([])
    expect(fontSet.add).not.toHaveBeenCalled()
  })

  it('fetches nothing for shipped faces only, or a bad skin id', async () => {
    const fetchMock = serve(200)
    vi.stubGlobal('fetch', fetchMock)
    await registerSkinFonts({ version: '1.0.0', fonts: { sans: 'geist' } }, ID)
    await registerSkinFonts(SKIN, 'x"]{}')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('skin font helpers', () => {
  it('lists only woff2 slots, and versions the asset path', () => {
    expect(skinFontFiles(SKIN)).toEqual([
      { slot: 'sans', file: 'assets/fonts/Brand_Sans.woff2' },
      { slot: 'mono', file: 'assets/fonts/Brand-Mono.woff2' },
    ])
    expect(skinAssetPath(ID, 'assets/fonts/a.woff2', '1.0.0-rc.1')).toBe(`/api/skins/${ID}/assets/assets/fonts/a.woff2?v=1.0.0-rc.1`)
  })
})
