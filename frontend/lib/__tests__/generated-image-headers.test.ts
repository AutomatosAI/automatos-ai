import type { NextRequest } from 'next/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GET } from '@/app/api/generated-images/[id]/route'
import { imageResponseHeaders } from '../generated-image-headers'

// F179: a public generated image never renders as a page on the app's origin.
const CSP = "default-src 'none'; sandbox"
const ID = '11111111-2222-4333-8444-555555555555'

describe('imageResponseHeaders', () => {
  it.each(['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'])(
    'shows a %s inline, unsniffed and sandboxed',
    (type) => {
      expect(imageResponseHeaders(type)).toMatchObject({
        'Content-Type': type,
        'Content-Disposition': 'inline',
        'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': CSP,
      })
    }
  )

  it.each(['image/svg+xml', 'text/html; charset=utf-8', 'application/pdf', 'text/csv'])(
    'downloads a %s instead of rendering it',
    (type) => {
      expect(imageResponseHeaders(type)).toMatchObject({
        'Content-Disposition': 'attachment',
        'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': CSP,
      })
    }
  )
})

describe('GET /api/generated-images/[id]', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const get = (id: string) =>
    GET(new Request(`http://app.test/api/generated-images/${id}`) as unknown as NextRequest, {
      params: Promise.resolve({ id }),
    })

  it('serves a stored page as a sandboxed download', async () => {
    const backend = vi.fn().mockResolvedValue(
      new Response('<script>alert(1)</script>', { headers: { 'Content-Type': 'text/html' } })
    )
    vi.stubGlobal('fetch', backend)

    const response = await get(ID)

    expect(response.status).toBe(200)
    expect(response.headers.get('content-disposition')).toBe('attachment')
    expect(response.headers.get('x-content-type-options')).toBe('nosniff')
    expect(response.headers.get('content-security-policy')).toBe(CSP)
    expect(backend.mock.calls[0][0]).toMatch(new RegExp(`/api/generated-images/${ID}$`))
  })

  it('shows an image inline', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('png', { headers: { 'Content-Type': 'image/png' } })
    ))

    const response = await get(ID)

    expect(response.headers.get('content-type')).toBe('image/png')
    expect(response.headers.get('content-disposition')).toBe('inline')
    expect(response.headers.get('content-security-policy')).toBe(CSP)
  })

  // An id is lower-case hex; ID has no letters, so its upper-case form is itself.
  it.each(['../../health', '..%2F..%2Fhealth', 'not-an-id', 'abcdef12-2222-4333-8444-555555555555'.toUpperCase()])(
    'never forwards %s to the backend',
    async (id) => {
      const backend = vi.fn()
      vi.stubGlobal('fetch', backend)

      expect((await get(id)).status).toBe(404)
      expect(backend).not.toHaveBeenCalled()
    }
  )
})
