/**
 * PRD-277 US-502 — the enterprise edition's session on a request made outside apiClient.
 *
 * A few screens call `fetch` themselves (an upload, a form post). In enterprise the API reads the
 * session cookie and, on a method that changes something, also wants the cross-site header
 * (session-transport.ts); a request without them is refused. EnterpriseSessionProvider installs
 * this wrapper over `window.fetch` once, and takes it off when it unmounts. It is never installed
 * in `local` or `saas` (`installSessionFetch` does nothing there).
 *
 * For a request whose address, resolved against the page, has the page's own origin:
 *   - an unsafe method (POST, PUT, PATCH, DELETE) gets the cross-site header unless it has one;
 *   - a path under `/api` gets the stored workspace header unless it has one;
 *   - such a request is sent with credentials (a caller's own `credentials: 'omit'` stays);
 *   - a 401 from a path under `/api` is reported as apiClient reports one (lib/api-unauthorized.ts).
 * Everything else is passed to the browser's fetch exactly as given: another origin (an absolute
 * address, a protocol-relative one, a `Request` for one), and a safe method to a path outside
 * `/api`. A `Request` is never changed: a new one is built from it with the added headers.
 */
import { reportUnauthorized } from '@/lib/api-unauthorized'
import { isEnterprise } from '@/lib/edition-name'

import { SESSION_CSRF_HEADER, SESSION_CSRF_VALUE, isUnsafeMethod } from './session-transport'

const WORKSPACE_HEADER = 'X-Workspace-ID'
const WORKSPACE_KEY = 'last_active_workspace'
const API_PATH = '/api'
const UNAUTHORIZED = 401

type Fetch = typeof fetch

/** Each wrapper installed, and how to take it off: a wrapper is never wrapped again. */
const restores = new WeakMap<Fetch, () => void>()

/** What the wrapper needs of the page: its fetch, its address and its storage. */
export interface FetchHost {
  fetch: Fetch
  readonly location: { readonly href: string; readonly origin: string }
  readonly localStorage?: Pick<Storage, 'getItem'>
}

function isApiPath(pathname: string): boolean {
  return pathname === API_PATH || pathname.startsWith(`${API_PATH}/`)
}

function storedWorkspace(host: FetchHost): string | null {
  try {
    return host.localStorage?.getItem(WORKSPACE_KEY) ?? null
  } catch (error) {
    console.warn('[session] the stored workspace could not be read', error)
    return null
  }
}

function resolved(input: RequestInfo | URL, host: FetchHost): URL | null {
  const address = input instanceof Request ? input.url : String(input)
  try {
    return new URL(address, host.location.href)
  } catch {
    return null
  }
}

/** The headers to send: the caller's own, with the session's added where absent. */
function sessionHeaders(base: HeadersInit | undefined, unsafe: boolean, api: boolean, host: FetchHost): Headers {
  const headers = new Headers(base)
  if (unsafe && !headers.has(SESSION_CSRF_HEADER)) headers.set(SESSION_CSRF_HEADER, SESSION_CSRF_VALUE)
  const workspaceId = api && !headers.has(WORKSPACE_HEADER) ? storedWorkspace(host) : null
  if (workspaceId) headers.set(WORKSPACE_HEADER, workspaceId)
  return headers
}

async function reportIfUnauthorized(response: Response): Promise<void> {
  if (response.status !== UNAUTHORIZED) return
  let body: unknown
  try {
    body = await response.clone().json()
  } catch {
    body = undefined
  }
  reportUnauthorized(body)
}

/** The wrapper over `original`: the session on the page's own requests, nothing on any other. */
export function sessionFetch(original: Fetch, host: FetchHost): Fetch {
  return async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = resolved(input, host)
    if (url === null || url.origin !== host.location.origin) return original(input, init)
    const request = input instanceof Request ? input : null
    const method = init?.method ?? request?.method ?? 'GET'
    const unsafe = isUnsafeMethod(method)
    const api = isApiPath(url.pathname)
    if (!unsafe && !api) return original(input, init)
    const headers = sessionHeaders(init?.headers ?? request?.headers, unsafe, api, host)
    const credentials: RequestCredentials = (init?.credentials ?? request?.credentials) === 'omit' ? 'omit' : 'include'
    const response = request
      ? await original(new Request(request, { ...init, headers, credentials }))
      : await original(input, { ...init, headers, credentials })
    if (api) await reportIfUnauthorized(response)
    return response
  }
}

/**
 * Put the wrapper over `host.fetch` in enterprise; the returned function takes it off again (only
 * while it is still the one in place). Asked again while it is in place, it returns that same
 * function (React's strict mode renders twice). Outside enterprise, or with no fetch, nothing is touched.
 */
export function installSessionFetch(host: FetchHost | undefined, enterprise: boolean = isEnterprise): () => void {
  if (!enterprise || !host || typeof host.fetch !== 'function') return () => {}
  const original = host.fetch
  const installed = restores.get(original)
  if (installed) return installed
  // The browser's fetch must be called on the window it came from.
  const wrapped = sessionFetch(original.bind(host), host)
  const restore = () => {
    if (host.fetch === wrapped) host.fetch = original
  }
  restores.set(wrapped, restore)
  host.fetch = wrapped
  return restore
}
