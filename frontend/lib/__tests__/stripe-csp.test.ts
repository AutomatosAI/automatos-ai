/**
 * PRD-265 W2: payments happen inside Settings → Billing, with Stripe's card fields as iframes
 * in our own form. The Content-Security-Policy must let Stripe.js load, its frames (the card
 * and address fields, the 3-D Secure check from hooks.stripe.com, Link) render, its API and
 * the address autocomplete be called, and Link's card art show; exactly the sources Stripe's
 * CSP guide lists, and nothing wider. The app itself still cannot be framed by anyone.
 *
 * Loads next.config.js the way csp-media-sources.test.ts does.
 */
import { createRequire } from 'module'
import { describe, expect, it } from 'vitest'

const load = createRequire(import.meta.url)
const { STRIPE_CSP } = load('../../csp-sources.js') as {
  STRIPE_CSP: { script: string; frame: string; connect: string; img: string }
}

async function directives(): Promise<Record<string, string[]>> {
  const config = load('../../next.config.js') as {
    headers: () => Promise<Array<{ headers: Array<{ key: string; value: string }> }>>
  }
  const all = (await config.headers()).flatMap((rule) => rule.headers)
  const csp = all.find((header) => header.key === 'Content-Security-Policy')?.value ?? ''
  return Object.fromEntries(
    csp.split('; ').map((part) => {
      const [name, ...sources] = part.split(/\s+/)
      return [name, sources.filter(Boolean)]
    }),
  )
}

describe("the Content-Security-Policy for Stripe's in-page payment forms", () => {
  it('names the sources from Stripe\'s CSP guide, directive by directive', () => {
    expect(STRIPE_CSP.script.split(' ')).toEqual(['https://js.stripe.com', 'https://*.js.stripe.com', 'https://maps.googleapis.com'])
    expect(STRIPE_CSP.frame.split(' ')).toEqual([
      'https://js.stripe.com',
      'https://*.js.stripe.com',
      'https://hooks.stripe.com',
      'https://link.com',
      'https://*.link.com',
    ])
    expect(STRIPE_CSP.connect.split(' ')).toEqual(['https://api.stripe.com', 'https://maps.googleapis.com', 'https://link.com', 'https://*.link.com'])
    expect(STRIPE_CSP.img.split(' ')).toEqual(['https://*.link.com'])
  })

  it('lets Stripe.js load, its frames render, its API answer and Link show its card art', async () => {
    const policy = await directives()
    expect(policy['script-src']).toEqual(expect.arrayContaining(STRIPE_CSP.script.split(' ')))
    expect(policy['frame-src']).toEqual(expect.arrayContaining(STRIPE_CSP.frame.split(' ')))
    expect(policy['connect-src']).toEqual(expect.arrayContaining(STRIPE_CSP.connect.split(' ')))
    expect(policy['img-src']).toEqual(expect.arrayContaining(STRIPE_CSP.img.split(' ')))
  })

  it('widens nothing: no bare wildcard or scheme, and the app still cannot be framed', async () => {
    const policy = await directives()
    for (const name of ['script-src', 'frame-src', 'img-src']) {
      expect(policy[name]).not.toContain('*')
      expect(policy[name]).not.toContain('https:')
    }
    expect(policy['frame-ancestors']).toEqual(["'none'"])
    expect(policy['form-action']).toEqual(["'self'"])
  })
})
