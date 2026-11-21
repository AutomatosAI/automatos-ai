/**
 * PRD-277 US-505 — the API address a person copies from a page (the Playbook webhook address):
 * the page's own origin in enterprise, today's value, byte for byte, in local and saas. Each case
 * loads the helper as that edition's build loads it (NEXT_PUBLIC_AUTH_EDITION read when
 * lib/edition-name.ts loads).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

const BUILD_API = 'https://api.example.test'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

async function shownIn(edition: string, buildApi: string): Promise<(typeof import('@/lib/api-address-shown'))['apiAddressShown']> {
  vi.resetModules()
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  vi.stubEnv('NEXT_PUBLIC_API_URL', buildApi)
  return (await import('@/lib/api-address-shown')).apiAddressShown
}

describe('apiAddressShown', () => {
  it('is the page’s own origin in enterprise', async () => {
    const shown = await shownIn('enterprise', '')
    expect(shown()).toBe(window.location.origin)
    expect(shown(undefined, { origin: 'https://automatos.example.test' })).toBe('https://automatos.example.test')
  })

  it("is '' in enterprise with no page, as on the server", async () => {
    expect((await shownIn('enterprise', ''))(undefined, null)).toBe('')
  })

  it.each(['local', 'saas'])('is the build’s API address in %s, as before', async (edition) => {
    expect((await shownIn(edition, BUILD_API))(undefined, { origin: 'https://other.example.test' })).toBe(BUILD_API)
  })

  it.each(['local', 'saas'])('falls back to http://localhost:8000 in %s with no build address, as before', async (edition) => {
    expect((await shownIn(edition, ''))()).toBe('http://localhost:8000')
  })
})
