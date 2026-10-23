// PRD-261 B-03j (PB-29): where this web app's API is, for the theme CLI. `automatos-theme install --to
// http://localhost:3000` (PRD §8 item 7) reads this file to find the API base, so a theme author gives the
// URL they open, not the local edition's API port. It answers exactly {schema, apiUrl} and nothing else:
// NEXT_PUBLIC_API_URL is already public (the client bundle inlines it). Next.js inlines NEXT_PUBLIC_* into
// the server bundle too, so a built image answers with the value it was BUILT with (frontend/Dockerfile takes
// it as a build ARG), not the container's runtime env: changing it at deploy time needs a rebuild, as for
// every other NEXT_PUBLIC_API_URL reader here. Only `next dev` and the tests see a per-call value.
// middleware.ts's matcher skips a path with an extension, so no middleware, Clerk's in SaaS, runs for it in
// either edition (__tests__/route.test.ts holds the matcher to that).
export const dynamic = 'force-dynamic'

const HOST_FILE_SCHEMA = 1
const NO_STORE = 'no-store'
const UNSET =
  'This web app has no NEXT_PUBLIC_API_URL set, so it cannot say where its API is. ' +
  'Set it to the API URL and rebuild, or give the CLI the API URL itself.'

export function GET(): Response {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL
  if (!apiUrl) {
    return new Response(UNSET, {
      status: 500,
      headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': NO_STORE },
    })
  }
  return new Response(JSON.stringify({ schema: HOST_FILE_SCHEMA, apiUrl }), {
    status: 200,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': NO_STORE },
  })
}
