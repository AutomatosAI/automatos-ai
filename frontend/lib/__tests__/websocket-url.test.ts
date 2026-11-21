// @vitest-environment node
/**
 * PRD-279 US-401 — the WebSocket address from the page.
 *
 *   1. With a WebSocket address set, that address, in every edition (as before).
 *   2. With an absolute API address, that address with `ws` and `/ws`, in every edition (as before).
 *   3. With neither, the enterprise edition uses the page's own host and scheme (`wss` on an https
 *      page); local and saas answer '' exactly as before. With no page (the server), '' everywhere.
 *
 * The address is worked out when WEBSOCKET_CONFIG.URL is read, not when lib/api-config.ts loads:
 * this file runs on Node, where there is no window, and the module loads without one.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getWebSocketUrl, type PageAddress } from '@/lib/websocket-url'

const HTTPS_PAGE: PageAddress = { protocol: 'https:', host: 'automatos.example.test' }
const HTTP_PAGE: PageAddress = { protocol: 'http:', host: 'localhost:3000' }
const EDITIONS = [true, false] as const

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.resetModules()
})

describe('the WebSocket address, as before, when one is set', () => {
  it.each(EDITIONS)('uses NEXT_PUBLIC_WS_URL itself (enterprise: %s)', (enterprise) => {
    expect(getWebSocketUrl('wss://ws.example.test/ws', 'https://api.example.test', HTTPS_PAGE, enterprise)).toBe(
      'wss://ws.example.test/ws',
    )
  })

  it.each(EDITIONS)('works it out from an absolute API address (enterprise: %s)', (enterprise) => {
    expect(getWebSocketUrl(undefined, 'http://localhost:8000', HTTPS_PAGE, enterprise)).toBe('ws://localhost:8000/ws')
    // Exactly the old helper's answer for an https API: its `ws` replaced the whole `https`.
    expect(getWebSocketUrl('', 'https://api.example.test', HTTP_PAGE, enterprise)).toBe('ws://api.example.test/ws')
  })

  it('keeps the old answer for a relative API address outside enterprise', () => {
    expect(getWebSocketUrl(undefined, '/backend', HTTPS_PAGE, false)).toBe('/backend/ws')
  })
})

describe('the third case: neither set', () => {
  it('uses the page’s host with wss on an https page, in enterprise', () => {
    expect(getWebSocketUrl(undefined, undefined, HTTPS_PAGE, true)).toBe('wss://automatos.example.test/ws')
    expect(getWebSocketUrl('', '', HTTPS_PAGE, true)).toBe('wss://automatos.example.test/ws')
  })

  it('uses ws on an http page, keeping the port, in enterprise', () => {
    expect(getWebSocketUrl(undefined, undefined, HTTP_PAGE, true)).toBe('ws://localhost:3000/ws')
  })

  it('takes the page, not a relative API address, in enterprise', () => {
    expect(getWebSocketUrl(undefined, '/backend', HTTPS_PAGE, true)).toBe('wss://automatos.example.test/ws')
  })

  it("answers '' in local and saas, as before", () => {
    expect(getWebSocketUrl(undefined, undefined, HTTPS_PAGE, false)).toBe('')
  })

  it.each(EDITIONS)("answers '' with no page, as on the server (enterprise: %s)", (enterprise) => {
    expect(getWebSocketUrl(undefined, undefined, null, enterprise)).toBe('')
  })
})

describe('WEBSOCKET_CONFIG.URL is worked out when read', () => {
  async function config(edition: string) {
    vi.resetModules()
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
    vi.stubEnv('NEXT_PUBLIC_WS_URL', '')
    vi.stubEnv('NEXT_PUBLIC_API_URL', '')
    return (await import('@/lib/api-config')).WEBSOCKET_CONFIG
  }

  it("loads on the server with no page and answers '' there", async () => {
    expect((await config('enterprise')).URL).toBe('')
  })

  it('answers from the page once there is one, in enterprise', async () => {
    const websocket = await config('enterprise')
    vi.stubGlobal('window', { location: HTTPS_PAGE })
    expect(websocket.URL).toBe('wss://automatos.example.test/ws')
  })

  it("still answers '' on a page in local and saas with nothing set", async () => {
    for (const edition of ['local', 'saas']) {
      const websocket = await config(edition)
      vi.stubGlobal('window', { location: HTTPS_PAGE })
      expect(websocket.URL, edition).toBe('')
      vi.unstubAllGlobals()
    }
  })

  it('answers the set address in local, as before', async () => {
    const websocket = await config('local')
    vi.stubEnv('NEXT_PUBLIC_WS_URL', 'ws://localhost:8000/ws')
    expect(websocket.URL).toBe('ws://localhost:8000/ws')
  })
})
