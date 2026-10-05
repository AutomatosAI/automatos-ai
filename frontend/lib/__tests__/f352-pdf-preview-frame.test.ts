/**
 * F352 (night 10b): the Deliverables side panel showed a broken-page icon for a PDF. The panel
 * fetches the file with the Bearer header and frames it as a blob: URL (useAuthenticatedBlobUrl in
 * FilePreview), so the API's own frame headers never apply; the app's Content-Security-Policy did
 * the blocking, because frame-src listed only 'self' (which never matches blob:) and object-src was
 * 'none' (the blob document inherits the policy, and the browser's PDF viewer is a plugin).
 *
 * These tests pin the policy the preview needs, and that nothing else was widened: no wildcard,
 * no data:, the page itself still cannot be framed by anyone.
 */
import { createRequire } from 'module'
import { describe, expect, it } from 'vitest'

const load = createRequire(import.meta.url)

async function directives(): Promise<Record<string, string>> {
  const config = load('../../next.config.js') as {
    headers: () => Promise<Array<{ headers: Array<{ key: string; value: string }> }>>
  }
  const all = (await config.headers()).flatMap((rule) => rule.headers)
  const csp = all.find((header) => header.key === 'Content-Security-Policy')?.value ?? ''
  return Object.fromEntries(csp.split('; ').map((part) => [part.split(' ')[0], part]))
}

describe('F352: a PDF preview can be framed as a blob: URL', () => {
  it('lets the app frame a blob: URL it made, beside its existing frame sources', async () => {
    expect((await directives())['frame-src']).toBe(
      "frame-src 'self' blob: https://*.clerk.accounts.dev https://challenges.cloudflare.com",
    )
  })

  it('lets the PDF viewer run in that blob: document and nowhere else', async () => {
    expect((await directives())['object-src']).toBe('object-src blob:')
  })

  it('still refuses to let any other site frame the app', async () => {
    const policy = await directives()
    expect(policy['frame-ancestors']).toBe("frame-ancestors 'none'")
    expect(policy['frame-src']).not.toMatch(/\s(\*|data:|https?:)(\s|$)/)
  })
})
