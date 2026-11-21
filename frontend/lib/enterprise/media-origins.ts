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
 * Name the narrowest host that works: the storage's own public host (`https://media.example.com`)
 * rather than a wildcard. The wildcard form is checked only for shape: it cannot tell a public
 * suffix of two labels (`*.co.uk`) from an organisation's own domain, so `https://*.co.uk` would
 * pass and allow every site under it.
 *
 * Origins are separated by commas or spaces; unset or blank means none. At most
 * MAX_MEDIA_ORIGINS of them, in at most MAX_MEDIA_ORIGINS_LENGTH characters. One bad origin, or
 * too many, makes the whole value bad: the server refuses to start (start-up.ts), and the message
 * names the setting, never its value.
 */

/** The setting's name. */
export const MEDIA_ORIGINS_SETTING = 'MEDIA_ORIGINS'

/**
 * Bounds on the value. An install names its storage's public host, perhaps a CDN in front of it
 * and a second bucket: 16 leaves room for that many times over. Both img-src and media-src carry
 * the list, so the policy grows by twice the value; 1024 characters keep the whole policy (about
 * 1 KB without media origins) with the other headers inside a 4 KB response-header buffer, the
 * smaller of nginx's `proxy_buffer_size` defaults (4k|8k), beyond which nginx calls the answer invalid.
 */
export const MAX_MEDIA_ORIGINS = 16
export const MAX_MEDIA_ORIGINS_LENGTH = 1024

export const MEDIA_ORIGINS_REFUSED =
  `${MEDIA_ORIGINS_SETTING} must list at most ${MAX_MEDIA_ORIGINS} origins, in at most ${MAX_MEDIA_ORIGINS_LENGTH} ` +
  'characters, separated by commas or spaces, each https://<host>[:port] or https://*.<domain>.<suffix>[:port], ' +
  'or http://localhost[:port] or http://127.0.0.1[:port] for local development.'

/** One DNS label: letters, digits and inner hyphens, at most 63 characters. */
export const HOST_LABEL = '[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?'
/** A host of one or more labels, with no empty label and no trailing dot. */
export const HOST_NAME = `${HOST_LABEL}(?:\\.${HOST_LABEL})*`
const PORT = '(?::(\\d{1,5}))?'
/** https, a host of one or more labels, or `*.` over at least two. */
const HTTPS_ORIGIN = new RegExp(`^https://(?:\\*\\.${HOST_LABEL}\\.${HOST_NAME}|${HOST_NAME})${PORT}$`, 'i')
/** http on a local development address only. */
const LOCAL_ORIGIN = new RegExp(`^http://(?:localhost|127\\.0\\.0\\.1)${PORT}$`, 'i')
const MAX_PORT = 65535

export type MediaOrigins =
  | { readonly ok: true; readonly origins: readonly string[] }
  | { readonly ok: false; readonly error: string }

/** True when `port` (the digits after `:`, or none) is 1 to 65535. */
export function isValidPort(port: string | undefined): boolean {
  return port === undefined || (Number(port) >= 1 && Number(port) <= MAX_PORT)
}

/** True when `token` is one origin the policy may name. */
export function isMediaOrigin(token: string): boolean {
  const match = HTTPS_ORIGIN.exec(token) ?? LOCAL_ORIGIN.exec(token)
  return match !== null && isValidPort(match[1])
}

/** The origins in `raw`, or the refusal when it is too long, lists too many, or any one is not an origin the policy may name. */
export function parseMediaOrigins(raw: string | undefined): MediaOrigins {
  const value = String(raw ?? '')
  if (value.length > MAX_MEDIA_ORIGINS_LENGTH) return { ok: false, error: MEDIA_ORIGINS_REFUSED }
  const tokens = value
    .split(/[\s,]+/)
    .filter(Boolean)
    .map((token) => (token.endsWith('/') ? token.slice(0, -1) : token))
  if (tokens.length > MAX_MEDIA_ORIGINS || !tokens.every(isMediaOrigin)) return { ok: false, error: MEDIA_ORIGINS_REFUSED }
  return { ok: true, origins: tokens }
}
