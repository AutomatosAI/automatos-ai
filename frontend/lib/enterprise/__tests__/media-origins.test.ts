// @vitest-environment node
/**
 * PRD-279 US-402 — MEDIA_ORIGINS, read when the enterprise server starts, is checked before any
 * origin reaches the Content-Security-Policy: an origin and nothing else, https unless it is a local
 * development address, a wildcard only as today's hosted default has one, and nothing that could end
 * a directive. One bad origin refuses the whole value, and the refusal names the setting, never
 * the value.
 */
import { describe, expect, it } from 'vitest'

import {
  MAX_MEDIA_ORIGINS,
  MAX_MEDIA_ORIGINS_LENGTH,
  MEDIA_ORIGINS_SETTING,
  isMediaOrigin,
  parseMediaOrigins,
} from '@/lib/enterprise/media-origins'

describe('isMediaOrigin', () => {
  it.each([
    'https://media.example.test',
    'https://media.example.test:9443',
    'https://*.amazonaws.com',
    'https://storage.example.test',
    'https://10.0.0.5',
    'http://localhost:9000',
    'http://127.0.0.1:9000',
    'http://localhost',
  ])('takes %s', (origin) => {
    expect(isMediaOrigin(origin)).toBe(true)
  })

  it.each([
    ['http on a host that is not local', 'http://media.example.test'],
    ['a wildcard over a bare suffix', 'https://*.com'],
    ['a wildcard alone', 'https://*'],
    ['a wildcard in the middle', 'https://media.*.example.test'],
    ['a wildcard on a local address', 'http://*.localhost'],
    ['a bare wildcard', '*'],
    ['a path', 'https://media.example.test/bucket'],
    ['a query', 'https://media.example.test?x=1'],
    ['a user name', 'https://user@media.example.test'],
    ['a keyword', "'unsafe-inline'"],
    ['a scheme source', 'data:'],
    ['another scheme', 'javascript:alert(1)'],
    ['a directive break', 'https://media.example.test;script-src'],
    ['a quote', "https://media.example.test'"],
    ['port zero', 'https://media.example.test:0'],
    ['a port out of range', 'https://media.example.test:70000'],
    ['an empty label', 'https://media..example.test'],
    ['a label starting with a hyphen', 'https://-media.example.test'],
    ['a trailing dot', 'https://media.example.test.'],
  ])('refuses %s', (_why, origin) => {
    expect(isMediaOrigin(origin)).toBe(false)
  })
})

describe('parseMediaOrigins', () => {
  it('reads unset and blank as no origin', () => {
    expect(parseMediaOrigins(undefined)).toEqual({ ok: true, origins: [] })
    expect(parseMediaOrigins('  ,  ')).toEqual({ ok: true, origins: [] })
  })

  it('splits on commas and spaces, and drops one trailing slash', () => {
    expect(parseMediaOrigins('https://media.example.test/, https://*.amazonaws.com  http://localhost:9000')).toEqual({
      ok: true,
      origins: ['https://media.example.test', 'https://*.amazonaws.com', 'http://localhost:9000'],
    })
  })

  it('takes at most MAX_MEDIA_ORIGINS origins in at most MAX_MEDIA_ORIGINS_LENGTH characters', () => {
    const list = (count: number) => Array.from({ length: count }, (_, n) => `https://m${n}.example.test`).join(' ')
    expect(parseMediaOrigins(list(MAX_MEDIA_ORIGINS)).ok).toBe(true)
    expect(parseMediaOrigins(list(MAX_MEDIA_ORIGINS + 1)).ok).toBe(false)
    expect(parseMediaOrigins('https://media.example.test'.padEnd(MAX_MEDIA_ORIGINS_LENGTH, ' ')).ok).toBe(true)
    expect(parseMediaOrigins('https://media.example.test'.padEnd(MAX_MEDIA_ORIGINS_LENGTH + 1, ' ')).ok).toBe(false)
  })

  it('checks the wildcard for shape only: a two-label public suffix passes (the setting says to name the narrowest host)', () => {
    expect(isMediaOrigin('https://*.co.uk')).toBe(true)
  })

  it('refuses the whole value for one bad origin, naming the setting and never the value', () => {
    const value = 'https://media.example.test https://evil.example.test;script-src'
    const parsed = parseMediaOrigins(value)
    expect(parsed.ok).toBe(false)
    if (parsed.ok) return
    expect(parsed.error).toContain(MEDIA_ORIGINS_SETTING)
    expect(parsed.error).not.toContain('evil.example.test')
    expect(parsed.error).not.toContain('media.example.test')
  })
})
