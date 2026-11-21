// PRD-261 B-03j (PB-29): where this web app's API is, for the theme CLI. `automatos-theme install --to
// http://localhost:3000` (PRD §8 item 7) reads this file to find the API base, so a theme author gives the
// URL they open, not the local edition's API port. It answers exactly {schema, apiUrl} and nothing else:
// NEXT_PUBLIC_API_URL is already public (the client bundle inlines it). Next.js inlines NEXT_PUBLIC_* into
// the server bundle too, so a built image answers with the value it was BUILT with (frontend/Dockerfile takes
// it as a build ARG), not the container's runtime env: changing it at deploy time needs a rebuild, as for
// every other NEXT_PUBLIC_API_URL reader here. Only `next dev` and the tests see a per-call value.
// middleware.ts's matcher skips a path with an extension, so no middleware, Clerk's in SaaS, runs for it in
// either edition (__tests__/route.test.ts holds the matcher to that).
// PRD-277 US-505: the enterprise build has no API address (the API answers on the web app's own host, §5.3),
// so there the answer is the origin the request came in on, from its Host and X-Forwarded-Proto
// (lib/enterprise/request-origin.ts, a plain host name and a valid port only), and it says it varies by both.
import { isEnterprise } from '@/lib/edition-name'
import { hostFileOrigin } from '@/lib/enterprise/request-origin'

export const dynamic = 'force-dynamic'

const HOST_FILE_SCHEMA = 1
const NO_STORE = 'no-store'
/** Enterprise only: the answer depends on these request headers, so a cache must not share it across them. */
const VARIES_BY = { Vary: 'Host, X-Forwarded-Proto' }
const UNSET =
  'This web app has no NEXT_PUBLIC_API_URL set, so it cannot say where its API is. ' +
  'Set it to the API URL and rebuild, or give the CLI the API URL itself.'
const NO_HOST =
  'This web app could not read its own address from the request, so it cannot say where its API is. ' +
  'Give the CLI the API URL itself: it is this web app\'s own address.'

/** The API's address: the build's NEXT_PUBLIC_API_URL, or in enterprise this request's own origin. */
function apiUrlFor(request: Request): string | null | undefined {
  return isEnterprise ? hostFileOrigin(request.headers) : process.env.NEXT_PUBLIC_API_URL
}

/** The answer's headers: its type, never cached, and in enterprise what it varies by. */
function answerHeaders(contentType: string): Record<string, string> {
  const headers = { 'Content-Type': contentType, 'Cache-Control': NO_STORE }
  return isEnterprise ? { ...headers, ...VARIES_BY } : headers
}

export function GET(request: Request): Response {
  const apiUrl = apiUrlFor(request)
  if (!apiUrl) {
    return new Response(isEnterprise ? NO_HOST : UNSET, {
      status: 500,
      headers: answerHeaders('text/plain; charset=utf-8'),
    })
  }
  return new Response(JSON.stringify({ schema: HOST_FILE_SCHEMA, apiUrl }), {
    status: 200,
    headers: answerHeaders('application/json'),
  })
}
