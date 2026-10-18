import type { Route } from '@playwright/test'

// PRD-261 A-04a: answers, with JSON, a route the page calls on the backend's origin. The app runs
// on :3000 and calls :8000, so a cross-origin read needs CORS headers on the answer; a preflight
// (OPTIONS) that reaches the route is answered here too, so it never depends on the real backend.

const DEFAULT_APP_URL = 'http://localhost:3000'
const NO_CONTENT = 204
const ALLOWED_METHODS = 'GET, PUT, POST, DELETE, OPTIONS'

async function corsHeaders(route: Route): Promise<Record<string, string>> {
  const origin = (await route.request().headerValue('origin')) ?? new URL(process.env.E2E_APP_URL ?? DEFAULT_APP_URL).origin
  return { 'access-control-allow-origin': origin, 'access-control-allow-credentials': 'true', vary: 'Origin' }
}

/** Fulfils `route` with `body` as JSON and `status`, or answers its preflight. */
export async function fulfillBackendJson(route: Route, status: number, body: unknown): Promise<void> {
  const request = route.request()
  const cors = await corsHeaders(route)
  if (request.method() === 'OPTIONS') {
    const asked = (await request.headerValue('access-control-request-headers')) ?? ''
    await route.fulfill({
      status: NO_CONTENT,
      headers: { ...cors, 'access-control-allow-methods': ALLOWED_METHODS, 'access-control-allow-headers': asked },
    })
    return
  }
  await route.fulfill({
    status,
    contentType: 'application/json',
    headers: { ...cors, 'cache-control': 'no-store' },
    body: JSON.stringify(body),
  })
}
