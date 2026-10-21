/**
 * PRD-266 US-018d — what a pairing link carries, checked before /go/pair shows any of it.
 *
 * The pairing QR is a universal link, https://<web>/go/pair?api=<api_url>&code=<code> (PRD §4.3).
 * The code is shown only when it is a pairing code (8 characters of the pairing alphabet, optionally
 * written XXXX-XXXX, in any case) and the server only as the host of an http(s) URL. Anything else
 * is an incomplete link, and the page shows neither value.
 */

/** No 0/O/1/I: a code is read off a screen and typed once (services/cli_host_service.py's alphabet). */
export const PAIRING_CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
/** The query names of the pairing link. */
export const PAIR_LINK_PARAMS = { code: 'code', server: 'api' } as const

const CODE_HALF = `[${PAIRING_CODE_ALPHABET}]{4}`
// No `u` flag: case-insensitive matching then folds ASCII letters only, so no other character
// (the long s, the Kelvin sign) can stand in for a letter of the alphabet.
const PAIRING_CODE_FORMAT = new RegExp(`^(${CODE_HALF})-?(${CODE_HALF})$`, 'i')
const SERVER_PROTOCOLS: readonly string[] = ['http:', 'https:']

/** A pairing link whose two values both passed their checks. */
export interface PairLink {
  /** The code as the app asks for it: XXXX-XXXX, upper case. */
  readonly code: string
  /** The server's host (and port, when it has one), never the whole address. */
  readonly serverHost: string
}

/** The code in its XXXX-XXXX form, or null when it is not a pairing code. */
export function parsePairingCode(raw: string | null): string | null {
  const match = raw === null ? null : PAIRING_CODE_FORMAT.exec(raw)
  return match ? `${match[1]}-${match[2]}`.toUpperCase() : null
}

function toUrl(raw: string): URL | null {
  try {
    return new URL(raw)
  } catch {
    // Not an absolute URL: the caller reads null as an incomplete link.
    return null
  }
}

/** The host of an http(s) URL, or null for anything else (javascript:, a bare word, no host). */
export function parseServerHost(raw: string | null): string | null {
  const url = raw === null ? null : toUrl(raw)
  if (!url || !SERVER_PROTOCOLS.includes(url.protocol)) return null
  return url.host || null
}

/** Both values of the link, or null when either is missing or malformed (or there is no query). */
export function readPairLink(params: Pick<URLSearchParams, 'get'> | null): PairLink | null {
  const code = parsePairingCode(params?.get(PAIR_LINK_PARAMS.code) ?? null)
  const serverHost = parseServerHost(params?.get(PAIR_LINK_PARAMS.server) ?? null)
  return code && serverHost ? { code, serverHost } : null
}
