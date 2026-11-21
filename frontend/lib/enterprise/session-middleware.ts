/**
 * PRD-277 US-502 — the enterprise middleware (middleware.ts's third branch).
 *
 * A page asked for with no session cookie is sent to the sign-in page, which brings the person
 * back to it afterwards. That is a convenience only: the cookie's presence proves nothing (it may
 * have ended, or be made up), and every request is judged by the API, which reads the cookie
 * itself. So nothing else is decided here.
 *
 * Never redirected: the public routes (the sign-in page among them), and `/api/…` — the web app's
 * own route handlers (chat, workflow stream, theme files) forward the cookie and answer 401 for
 * themselves, which a fetch or a module import can read; a redirect would not be.
 *
 * PRD-279 US-402: every answer it gives, passed through or redirected, carries the
 * Content-Security-Policy with the media origins the running server was given
 * (content-security-policy.ts); the build sets the policy only where this never runs.
 */
import { NextResponse, type NextRequest } from 'next/server'

import { withRunningPolicy } from './content-security-policy'
import { SESSION_COOKIE_NAME } from './session-transport'

export const SIGN_IN_PAGE = '/sign-in'
const API_PREFIX = '/api/'

/** Is `path` one the middleware leaves alone in enterprise? */
export function isLeftAlone(path: string, isPublic: (request: NextRequest) => boolean, request: NextRequest): boolean {
  return path === '/api' || path.startsWith(API_PREFIX) || isPublic(request)
}

/** The answer for `request`: through, or to sign in and back. */
function sessionAnswer(request: NextRequest, isPublic: (request: NextRequest) => boolean): NextResponse {
  const { pathname, search } = request.nextUrl
  if (isLeftAlone(pathname, isPublic, request)) return NextResponse.next()
  if (request.cookies.get(SESSION_COOKIE_NAME)?.value) return NextResponse.next()
  const signIn = new URL(SIGN_IN_PAGE, request.nextUrl)
  signIn.search = ''
  signIn.searchParams.set('return_to', `${pathname}${search}`)
  return NextResponse.redirect(signIn)
}

/** The enterprise middleware, given the app's public routes. */
export function enterpriseMiddleware(isPublic: (request: NextRequest) => boolean) {
  return (request: NextRequest) => withRunningPolicy(sessionAnswer(request, isPublic))
}
