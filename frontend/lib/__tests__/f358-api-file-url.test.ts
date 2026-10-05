/**
 * F352/F358: a file path from a tool result is fetched with the Authorization header only when
 * it is a path on the API's own origin. Glued on as `${base}${path}`, ".evil.example/x" or
 * "@evil.example/x" would change the host and send the token there.
 */
import { describe, expect, it } from 'vitest'

import { apiFileUrl } from '@/lib/api-file-url'

const BASE = 'https://api.automatos.app'

describe('apiFileUrl', () => {
  it('takes a path on the API', () => {
    expect(apiFileUrl(BASE, '/api/documents/generated/20261005_Quay_letter.pdf')).toBe(
      'https://api.automatos.app/api/documents/generated/20261005_Quay_letter.pdf',
    )
    expect(apiFileUrl(BASE, '/api/documents/42/download?x=1')).toBe(
      'https://api.automatos.app/api/documents/42/download?x=1',
    )
  })

  it.each([
    '.evil.example/steal',          // https://api.automatos.app.evil.example
    '@evil.example/steal',          // https://api.automatos.app@evil.example
    ':8443@evil.example/steal',
    '//evil.example/steal',
    '/\\evil.example/steal',
    'https://evil.example/steal',
    'api/documents/generated/x.pdf', // not rooted
    '',
  ])('refuses %j, which would leave the API', (path) => {
    expect(apiFileUrl(BASE, path)).toBeNull()
  })

  it('keeps a same-origin API (no base) on the page origin', () => {
    const url = apiFileUrl('', '/api/documents/generated/x.pdf')
    expect(url).toBe(`${window.location.origin}/api/documents/generated/x.pdf`)
  })
})
