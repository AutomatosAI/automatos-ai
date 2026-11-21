// @vitest-environment node
/**
 * PRD-277 US-502 — the enterprise middleware: a page asked for with no session cookie goes to
 * /sign-in and comes back to where it was going; the sign-in page, the public routes and `/api/…`
 * are left alone (the API, and the web app's own route handlers, answer for themselves). A cookie
 * is let through without being judged: the API judges every request.
 */
import { NextRequest } from 'next/server'
import { describe, expect, it } from 'vitest'

import { enterpriseMiddleware } from '@/lib/enterprise/session-middleware'
import { SESSION_COOKIE_NAME } from '@/lib/enterprise/session-transport'

const PUBLIC = new Set(['/sign-in', '/verify', '/go/pair'])
const middleware = enterpriseMiddleware((request) => PUBLIC.has(request.nextUrl.pathname))

function visit(href: string, cookie?: string): Response {
  const headers: Record<string, string> = cookie ? { cookie } : {}
  return middleware(new NextRequest(new URL(href, 'https://automatos.example.test'), { headers }))
}

const passedThrough = (answer: Response) => answer.headers.get('x-middleware-next') === '1'

describe('PRD-277 US-502 — the enterprise middleware', () => {
  it('sends a page with no session to sign in, bringing the person back', () => {
    const answer = visit('/chat/abc?tab=files')
    expect(answer.status).toBe(307)
    const location = new URL(answer.headers.get('location') ?? '')
    expect(location.pathname).toBe('/sign-in')
    expect(location.searchParams.get('return_to')).toBe('/chat/abc?tab=files')
    expect(location.origin).toBe('https://automatos.example.test')
  })

  it('lets a page through when the browser holds a session cookie', () => {
    expect(passedThrough(visit('/chat', `${SESSION_COOKIE_NAME}=${'s'.repeat(40)}`))).toBe(true)
  })

  it.each(['/sign-in', '/verify', '/go/pair', '/api/chat', '/api/themes/files/x', '/api'])('leaves %s alone', (href) => {
    expect(passedThrough(visit(href))).toBe(true)
  })

  it('does not take an empty cookie for a session', () => {
    expect(visit('/settings', `${SESSION_COOKIE_NAME}=`).status).toBe(307)
  })
})
