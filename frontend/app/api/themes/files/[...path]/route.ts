/**
 * PRD-261 A-08b — an installed theme's files on the app's own origin.
 *
 * GET /api/themes/files/<digest>/<path> forwards to the backend's files route (A-08a) and hands
 * back its answer. Why a proxy and not apiClient: the browser asks for these files itself, with a
 * native import() of the entry, the module graph behind it and a <video src> for a clip, and none
 * of those can carry a Bearer token or an X-Workspace-ID header, or take a cross-origin answer
 * the import map's integrity pins. So this handler is the one place their session is turned into
 * a backend call: in SaaS it reads the Clerk session (the middleware lets this path through to
 * here, `isPublicRoute`, so that signed out is a 401 from this handler and not Clerk's 404) and
 * sends its token as `Bearer`; in the local edition it sends none and the backend serves the
 * local operator. It is a same-origin file proxy, not the backend client, so it fetches directly.
 *
 * Only a package digest and plain file-name segments reach the backend (F179, as
 * app/api/generated-images/[id]/route.ts does). `Range` and `If-None-Match` go forward, so a
 * clip plays over 206 (Safari asks for ranges) and a cached module revalidates with 304; the
 * answer keeps only its content and caching headers (never `Set-Cookie`). A request made to
 * register a service worker (`Service-Worker: script`) is refused here, as the backend refuses it.
 */
import { auth } from '@clerk/nextjs/server'

import { isSaaS } from '@/lib/auth-edition'
import { THEME_DIGEST_PATTERN, THEME_FILES_PATH } from '@/lib/theme-host/constants'

export const dynamic = 'force-dynamic'

const BACKEND_URL = // Server-side: BACKEND_INTERNAL_URL (container DNS, local edition) beats the
// browser-facing NEXT_PUBLIC_API_URL — inside the frontend container 'localhost'
// is the frontend itself. Unset in SaaS ⇒ identical to before. (PRD-209)
process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/** One file-name segment, as a package may name it (orchestrator/services/themes/package.py NAME). */
const FILE_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
/** The deepest file a package holds: source/, seven folders and the file (package.py FILE_KINDS). */
const MAX_FILE_SEGMENTS = 9
const SERVICE_WORKER_HEADER = 'service-worker'
const SERVICE_WORKER_SCRIPT = 'script'

const FORWARDED_REQUEST_HEADERS = ['range', 'if-none-match'] as const
const PASSED_RESPONSE_HEADERS = ['content-type', 'content-range', 'accept-ranges', 'cache-control', 'etag'] as const
/** The answers handed back with the backend's headers; 200 and 206 also with its body. */
const PASSED_STATUSES: ReadonlySet<number> = new Set([200, 206, 304, 416])
const BODY_STATUSES: ReadonlySet<number> = new Set([200, 206])
/** The refusals handed back with their status and a fixed text, never the backend's body. */
const REFUSAL_TEXT: Readonly<Record<number, string>> = {
  401: 'Sign in again to load this theme.',
  403: 'Theme files can’t be loaded this way.',
  404: 'That theme file isn’t available.',
  503: 'Theme files can’t be read right now. Try again in a moment.',
}
const BACKEND_FAILED = 'The theme file could not be fetched.'

/** As the backend sends on every file (A-08a): never sniffed, and a file opened as a page runs nothing. */
const SERVING_HEADERS: Readonly<Record<string, string>> = {
  'X-Content-Type-Options': 'nosniff',
  'Content-Security-Policy': "default-src 'none'; sandbox",
}

type RouteContext = { params: Promise<{ path: string[] }> }
type Credential = { readonly ok: true; readonly authorization: string | null } | { readonly ok: false }

function refusal(status: number, text: string): Response {
  return new Response(text, {
    status,
    headers: { ...SERVING_HEADERS, 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' },
  })
}

/** The backend path for a digest and plain file segments, or null for anything else. */
function backendFilePath(segments: readonly string[] | undefined): string | null {
  if (!segments || segments.length < 2 || segments.length > MAX_FILE_SEGMENTS + 1) return null
  const [digest, ...file] = segments
  if (!THEME_DIGEST_PATTERN.test(digest)) return null
  if (!file.every((segment) => FILE_SEGMENT.test(segment))) return null
  return `${THEME_FILES_PATH}${digest}/${file.join('/')}`
}

/** SaaS: the Clerk session's token, or not ok when signed out. Local: no token at all. */
async function backendCredential(): Promise<Credential> {
  if (!isSaaS) return { ok: true, authorization: null }
  try {
    const session = await auth()
    if (!session.userId) return { ok: false }
    const token = await session.getToken()
    return token ? { ok: true, authorization: `Bearer ${token}` } : { ok: false }
  } catch (error) {
    console.error('[theme-files] could not read the session', error)
    return { ok: false }
  }
}

function forwardedHeaders(request: Request, authorization: string | null): Headers {
  const headers = new Headers()
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name)
    if (value !== null) headers.set(name, value)
  }
  if (authorization !== null) headers.set('Authorization', authorization)
  return headers
}

/** The response headers handed back: the passed ones, the length unless it was re-encoded, and the serving ones. */
function passedHeaders(backend: Response): Headers {
  const headers = new Headers(SERVING_HEADERS)
  for (const name of PASSED_RESPONSE_HEADERS) {
    const value = backend.headers.get(name)
    if (value !== null) headers.set(name, value)
  }
  const length = backend.headers.get('content-length')
  if (length !== null && backend.headers.get('content-encoding') === null) headers.set('content-length', length)
  return headers
}

function discardBody(backend: Response): void {
  backend.body?.cancel().catch((error: unknown) => console.error('[theme-files] could not discard a body', error))
}

/** The backend's answer as the browser gets it. */
function handedBack(backend: Response): Response {
  const { status } = backend
  if (PASSED_STATUSES.has(status)) {
    const body = BODY_STATUSES.has(status) ? backend.body : null
    if (body === null) discardBody(backend)
    return new Response(body, { status, headers: passedHeaders(backend) })
  }
  discardBody(backend)
  const text = REFUSAL_TEXT[status]
  if (text !== undefined) return refusal(status, text)
  console.error('[theme-files] the backend answered', status)
  return refusal(502, BACKEND_FAILED)
}

export async function GET(request: Request, { params }: RouteContext): Promise<Response> {
  if (request.headers.get(SERVICE_WORKER_HEADER)?.trim().toLowerCase() === SERVICE_WORKER_SCRIPT) {
    return refusal(403, REFUSAL_TEXT[403])
  }
  const filePath = backendFilePath((await params).path)
  if (filePath === null) return refusal(404, REFUSAL_TEXT[404])
  const credential = await backendCredential()
  if (!credential.ok) return refusal(401, REFUSAL_TEXT[401])
  try {
    const backend = await fetch(`${BACKEND_URL}${filePath}`, {
      headers: forwardedHeaders(request, credential.authorization),
      cache: 'no-store',
      redirect: 'manual',
      signal: request.signal,
    })
    return handedBack(backend)
  } catch (error) {
    console.error('[theme-files] the backend could not be reached', error)
    return refusal(502, BACKEND_FAILED)
  }
}
