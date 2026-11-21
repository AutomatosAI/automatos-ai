/**
 * The full URL of a file route on the API, or null when the path is not one (F352/F358).
 *
 * Files behind auth (a generated document, a brand file, a Deliverable's preview) are
 * fetched with the API client's Authorization header. Their paths often come from a tool
 * result or an agent's answer, so a path is taken only when it is a path on the API's own
 * origin: it starts with a single "/", and the URL it makes has the API's origin. Glued on
 * as `${base}${path}`, ".evil.example/x" or "@evil.example/x" would change the host and
 * send the token there.
 */
import { isEnterprise } from '@/lib/edition-name'

const FALLBACK_ORIGIN = 'http://localhost'

function pageOrigin(): string {
  return typeof window === 'undefined' ? FALLBACK_ORIGIN : window.location.origin
}

export function apiFileUrl(base: string, path: string): string | null {
  if (typeof path !== 'string' || !path.startsWith('/') || path.startsWith('//') || path.includes('\\')) {
    return null
  }
  try {
    const expected = new URL(base || '/', pageOrigin())
    const full = new URL(`${base}${path}`, pageOrigin())
    return full.origin === expected.origin ? full.toString() : null
  } catch {
    return null
  }
}

/**
 * The credentials a file read sends (FilePreview). An absolute (presigned) address sends them as
 * before. A file on the API sends none in saas and local, where the bearer token says who is asking;
 * in enterprise the session cookie says it (PRD-277 US-502), so it goes with them.
 */
export function apiFileCredentials(isAbsolute: boolean, enterprise: boolean = isEnterprise): RequestCredentials {
  return isAbsolute || enterprise ? 'include' : 'omit'
}

/** A path the API does not serve: what a guarded fetch reports instead of sending the token. */
export const NOT_AN_API_FILE = 'Not a file on this workspace'
