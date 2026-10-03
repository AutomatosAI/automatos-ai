/**
 * F237 (PRD-251B, TESTER build 8): presigned media — a Socials template's thumbnail, a post's files —
 * loads from object storage's public origin, so the app's Content-Security-Policy allows it in
 * img-src and media-src. The origins come from NEXT_PUBLIC_MEDIA_ORIGINS at build time (MinIO's
 * http://localhost:9000 locally, AWS S3 by the Dockerfile's default); anything that is not a bare
 * origin is left out, so the policy is never widened by a path, a keyword or a stray quote.
 */
import { createRequire } from 'module'
import { afterEach, describe, expect, it } from 'vitest'

const load = createRequire(import.meta.url)
const { mediaOrigins } = load('../../csp-sources.js') as { mediaOrigins: (raw: unknown) => string[] }
const CONFIG = load.resolve('../../next.config.js')
const BEFORE = process.env.NEXT_PUBLIC_MEDIA_ORIGINS

async function policyWith(origins: string | undefined): Promise<Record<string, string>> {
  if (origins === undefined) delete process.env.NEXT_PUBLIC_MEDIA_ORIGINS
  else process.env.NEXT_PUBLIC_MEDIA_ORIGINS = origins
  delete load.cache[CONFIG]
  const config = load(CONFIG) as { headers: () => Promise<Array<{ headers: Array<{ key: string; value: string }> }>> }
  const all = (await config.headers()).flatMap((rule) => rule.headers)
  const csp = all.find((header) => header.key === 'Content-Security-Policy')?.value ?? ''
  return Object.fromEntries(csp.split('; ').map((part) => [part.split(' ')[0], part]))
}

afterEach(() => {
  if (BEFORE === undefined) delete process.env.NEXT_PUBLIC_MEDIA_ORIGINS
  else process.env.NEXT_PUBLIC_MEDIA_ORIGINS = BEFORE
  delete load.cache[CONFIG]
})

describe('the media origins', () => {
  it('keeps bare origins only', () => {
    expect(mediaOrigins('http://localhost:9000, https://*.amazonaws.com  https://cdn.example.com/')).toEqual([
      'http://localhost:9000', 'https://*.amazonaws.com', 'https://cdn.example.com',
    ])
    expect(mediaOrigins("'unsafe-inline' * https://x.test/path data: javascript:alert(1)")).toEqual([])
    expect(mediaOrigins(undefined)).toEqual([])
  })
})

describe('the Content-Security-Policy', () => {
  it('lets the browser load presigned media from the local MinIO origin', async () => {
    const policy = await policyWith('http://localhost:9000')
    expect(policy['img-src']).toContain('http://localhost:9000')
    expect(policy['media-src']).toBe("media-src 'self' blob: http://localhost:9000")
  })

  it('takes the hosted default, and adds nothing when none is set', async () => {
    expect((await policyWith('https://*.amazonaws.com'))['img-src']).toContain('https://*.amazonaws.com')
    const bare = await policyWith(undefined)
    expect(bare['img-src']).toBe(
      "img-src 'self' data: blob: https://*.clerk.accounts.dev https://img.clerk.com https://*.googleusercontent.com https://logos.composio.dev",
    )
    expect(bare['media-src']).toBe("media-src 'self' blob:")
  })
})
