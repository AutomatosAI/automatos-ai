/**
 * PRD-279 US-402 — the enterprise image's media origins, read from MEDIA_ORIGINS when the server
 * runs (never a NEXT_PUBLIC_* variable: those are fixed when the image is built).
 *
 * The browser loads presigned media (Socials thumbnails, a post's files) from object storage's
 * public origin, which is the install's own. The origins go into the Content-Security-Policy's
 * img-src and media-src, so each one is checked strictly before it is used:
 *
 *   - an origin and nothing else: scheme, host and an optional port (one trailing slash, as a value
 *     copied from a browser carries, is dropped);
 *   - `https`, or `http` only for a local development address (`localhost`, `127.0.0.1`);
 *   - a wildcard only as today's hosted default has one, a leading `*.` over a domain of at least two
 *     labels (`https://*.amazonaws.com`), never over a bare suffix and never on a local address;
 *   - nothing that could end a directive or add a keyword: no quote, semicolon, comma inside a
 *     token, space, path, query or user name gets past the patterns below.
 *
 * Origins are separated by commas or spaces; unset or blank means none. One bad origin makes the
 * whole value bad: the server refuses to start (start-up.ts), and the message names the setting,
 * never its value.
 */

/** The setting's name. */
export const MEDIA_ORIGINS_SETTING = 'MEDIA_ORIGINS'

export const MEDIA_ORIGINS_REFUSED =
  `${MEDIA_ORIGINS_SETTING} must list origins separated by commas or spaces, each https://<host>[:port] ` +
  'or https://*.<domain>.<suffix>[:port], or http://localhost[:port] or http://127.0.0.1[:port] for local development.'

const LABEL = '[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?'
const PORT = '(?::(\\d{1,5}))?'
/** https, a host of one or more labels, or `*.` over at least two. */
const HTTPS_ORIGIN = new RegExp(`^https://(?:\\*\\.${LABEL}\\.${LABEL}(?:\\.${LABEL})*|${LABEL}(?:\\.${LABEL})*)${PORT}$`, 'i')
/** http on a local development address only. */
const LOCAL_ORIGIN = new RegExp(`^http://(?:localhost|127\\.0\\.0\\.1)${PORT}$`, 'i')
const MAX_PORT = 65535

export type MediaOrigins =
  | { readonly ok: true; readonly origins: readonly string[] }
  | { readonly ok: false; readonly error: string }

function hasValidPort(match: RegExpExecArray): boolean {
  const port = match[1]
  return port === undefined || (Number(port) >= 1 && Number(port) <= MAX_PORT)
}

/** True when `token` is one origin the policy may name. */
export function isMediaOrigin(token: string): boolean {
  const match = HTTPS_ORIGIN.exec(token) ?? LOCAL_ORIGIN.exec(token)
  return match !== null && hasValidPort(match)
}

/** The origins in `raw`, or the refusal when any one of them is not an origin the policy may name. */
export function parseMediaOrigins(raw: string | undefined): MediaOrigins {
  const tokens = String(raw ?? '')
    .split(/[\s,]+/)
    .filter(Boolean)
    .map((token) => (token.endsWith('/') ? token.slice(0, -1) : token))
  if (!tokens.every(isMediaOrigin)) return { ok: false, error: MEDIA_ORIGINS_REFUSED }
  return { ok: true, origins: tokens }
}
