/**
 * PRD-277 US-505 — the origin a request came in on, for the enterprise `/automatos-host.json`,
 * whose answer tells the theme CLI where the API is: in this edition the API answers on the web
 * app's own host (§5.3), and the build carries no address.
 *
 * Read from the Host header and the first X-Forwarded-Proto value (the proxy in front of the web
 * app names the scheme the person used; http or https only, else https). X-Forwarded-Host is never
 * read: Host is what the ingress routed on. The host must be a plain host name: labels of letters,
 * digits and inner hyphens, no empty label, no trailing dot, and a port, if any, from 1 to 65535;
 * anything else gives no origin, never one built from it. Stricter than the Marketplace share
 * card's reader (lib/marketplace/share-card.ts), which the other editions use and this leaves alone.
 */
import { HOST_NAME, isValidPort } from './media-origins'

type HeaderReader = Pick<Headers, 'get'>

const HOST = new RegExp(`^${HOST_NAME}(?::(\\d{1,5}))?$`, 'i')
const WEB_PROTOCOLS: ReadonlySet<string> = new Set(['http', 'https'])
const DEFAULT_PROTOCOL = 'https'

/** The request's origin, or null when its Host is not a plain host name with a valid port. */
export function hostFileOrigin(headers: HeaderReader): string | null {
  const host = headers.get('host')?.trim() ?? ''
  const match = HOST.exec(host)
  if (match === null || !isValidPort(match[1])) return null
  const forwarded = headers.get('x-forwarded-proto')?.split(',')[0]?.trim().toLowerCase()
  const protocol = forwarded && WEB_PROTOCOLS.has(forwarded) ? forwarded : DEFAULT_PROTOCOL
  return `${protocol}://${host.toLowerCase()}`
}
