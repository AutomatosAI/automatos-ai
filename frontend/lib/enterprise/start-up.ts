/**
 * PRD-277 US-505, PRD-279 US-402 — what the enterprise web server needs from its environment
 * before it serves anything, since none of it is fixed in the image:
 *
 *   - MEDIA_ORIGINS: the media origins for the Content-Security-Policy (media-origins.ts). Unset or
 *     blank is allowed (no media origin); anything that is not a list of origins is refused.
 *   - BACKEND_INTERNAL_URL: where the web app's own server code (the chat, workflow stream, theme
 *     files and generated images routes, the theme boot, the Marketplace share card) reaches the
 *     API inside the cluster. The browser never sees it. Without it those fall back to
 *     http://localhost:8000, which inside the web app's container is the web app itself, so it is
 *     required: an http or https address.
 *
 * The answer names the setting, never its value. instrumentation.ts runs this through
 * start-up-node.ts once, when the Node server starts, in the enterprise build only.
 */
import { parseMediaOrigins } from './media-origins'

export const BACKEND_INTERNAL_URL_SETTING = 'BACKEND_INTERNAL_URL'

export const BACKEND_INTERNAL_URL_REFUSED =
  `${BACKEND_INTERNAL_URL_SETTING} must be the API's http or https address inside the cluster ` +
  "(the web app's server routes call the API there)."

const SERVER_PROTOCOLS: ReadonlySet<string> = new Set(['http:', 'https:'])

/** What start-up reads: the two settings, as the environment holds them. */
export interface StartUpSettings {
  readonly MEDIA_ORIGINS?: string
  readonly BACKEND_INTERNAL_URL?: string
}

function isServerAddress(raw: string | undefined): boolean {
  try {
    const url = new URL(String(raw ?? '').trim())
    return SERVER_PROTOCOLS.has(url.protocol) && url.hostname !== '' && url.username === '' && url.password === ''
  } catch {
    return false
  }
}

/** Why the enterprise web server must not start with `settings`, or null when it may. */
export function enterpriseStartUpProblem(settings: StartUpSettings): string | null {
  const media = parseMediaOrigins(settings.MEDIA_ORIGINS)
  if (!media.ok) return media.error
  if (!isServerAddress(settings.BACKEND_INTERNAL_URL)) return BACKEND_INTERNAL_URL_REFUSED
  return null
}
