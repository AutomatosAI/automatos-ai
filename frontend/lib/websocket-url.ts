/**
 * PRD-279 US-401 — the web app's WebSocket address, worked out in one place.
 *
 *   1. NEXT_PUBLIC_WS_URL set: that address, as before.
 *   2. Else an absolute API address (NEXT_PUBLIC_API_URL): that address with `ws` for its scheme
 *      and `/ws` after it, as before (`https://api` gives `ws://api/ws`, exactly what
 *      lib/api-config.ts gave).
 *   3. Else, in the enterprise edition, the page's own host and scheme: `wss://<host>/ws` on an
 *      https page, `ws://<host>/ws` on an http one. The web app and the API share one host there
 *      (PRD-277 §5.3), and the enterprise build carries neither address (enterprise-build.js).
 *
 * Outside enterprise the third case answers '' as before, so local and saas are unchanged. The
 * address is worked out when it is asked for, not when a module loads: on the server there is no
 * page, and the answer there is '' (a socket is only ever opened in the browser).
 */
import { isEnterprise } from '@/lib/edition-name'

/** The part of the page's address the third case reads (`window.location`). */
export interface PageAddress {
  readonly protocol: string
  readonly host: string
}

const ABSOLUTE_HTTP = /^https?:\/\//i
const WS_PATH = '/ws'

/** The WebSocket address for these settings and this page (null: there is no page, as on the server). */
export function getWebSocketUrl(
  wsUrl: string | undefined,
  apiUrl: string | undefined,
  page: PageAddress | null,
  enterprise: boolean = isEnterprise,
): string {
  if (wsUrl) return wsUrl
  const api = apiUrl || ''
  if (api && (!enterprise || ABSOLUTE_HTTP.test(api))) return api.replace(/^https?/, 'ws') + WS_PATH
  if (!enterprise || page === null) return ''
  const scheme = page.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${page.host}${WS_PATH}`
}

/** The address for this build, on this page (`''` on the server). */
export function currentWebSocketUrl(): string {
  const page = typeof window === 'undefined' ? null : window.location
  return getWebSocketUrl(process.env.NEXT_PUBLIC_WS_URL, process.env.NEXT_PUBLIC_API_URL, page)
}
