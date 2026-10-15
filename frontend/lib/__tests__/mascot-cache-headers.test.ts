/**
 * PRD-260 (P260-FX-15, M75): Auto's clips and stills (public/auto/) are cached by the browser.
 * Next serves a public file with Cache-Control max-age=0 unless next.config.js headers() names
 * it, and then every state swap waits a revalidation round trip before the next clip plays.
 * Loads next.config.js the way csp-media-sources.test.ts does. Not `immutable`: the file names
 * carry no version.
 */
import { createRequire } from 'module'
import { describe, expect, it } from 'vitest'

import { MASCOT_DIR } from '@/lib/mascot/media'

type HeaderRule = { source: string; headers: Array<{ key: string; value: string }> }

const load = createRequire(import.meta.url)
const config = load('../../next.config.js') as { headers: () => Promise<HeaderRule[]> }

const MAX_AGE = /(?:^|,)\s*max-age=(\d+)/
const STALE_WHILE_REVALIDATE = /(?:^|,)\s*stale-while-revalidate=(\d+)/

/** The number a Cache-Control directive carries, or 0 without it. */
function seconds(directive: RegExp, cacheControl: string): number {
  const match = directive.exec(cacheControl)
  return match ? Number(match[1]) : 0
}

async function mascotCacheControl(): Promise<string> {
  const rules = await config.headers()
  const rule = rules.find((candidate) => candidate.source.startsWith(`${MASCOT_DIR}/`))
  return rule?.headers.find((header) => header.key === 'Cache-Control')?.value ?? ''
}

describe("Auto's clips in next.config.js headers()", () => {
  it(`caches ${MASCOT_DIR}/ for a while (max-age > 0), then revalidates in the background`, async () => {
    const cacheControl = await mascotCacheControl()
    expect(seconds(MAX_AGE, cacheControl)).toBeGreaterThan(0)
    expect(seconds(STALE_WHILE_REVALIDATE, cacheControl)).toBeGreaterThan(0)
    expect(cacheControl).toMatch(/^public,/)
  })

  it('never marks them immutable, since a changed clip keeps its file name', async () => {
    expect(await mascotCacheControl()).not.toContain('immutable')
  })
})
