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
 *     required: an origin and nothing else, `http` or `https`, a host and an optional port. No
 *     trailing slash and no path, since the routes append `/api/…` to it as it is.
 *   - ENTERPRISE_CALLS_CLOUD_SYNC (PRD-277 D-12): whether the policy allows Automatos Cloud's
 *     screenshots (cloud-sync-setting.ts). Unset or blank is off; anything but the API's switch
 *     forms is refused.
 *
 * The answer names the setting, never its value. instrumentation.ts runs this through
 * start-up-node.ts once, when the Node server starts, in the enterprise build only.
 */
import { parseCloudSync } from './cloud-sync-setting'
import { HOST_NAME, isValidPort, parseMediaOrigins } from './media-origins'

export const BACKEND_INTERNAL_URL_SETTING = 'BACKEND_INTERNAL_URL'

export const BACKEND_INTERNAL_URL_REFUSED =
  `${BACKEND_INTERNAL_URL_SETTING} must be the API's origin inside the cluster, http(s)://<host>[:port] ` +
  "with no path and no trailing slash (the web app's server routes call the API there)."

/** http or https, a host name (a cluster service name has one label), an optional port; nothing after it. */
const SERVER_ORIGIN = new RegExp(`^https?://${HOST_NAME}(?::(\\d{1,5}))?$`, 'i')

/** What start-up reads: the three settings, as the environment holds them. */
export interface StartUpSettings {
  readonly MEDIA_ORIGINS?: string
  readonly BACKEND_INTERNAL_URL?: string
  readonly ENTERPRISE_CALLS_CLOUD_SYNC?: string
}

function isServerOrigin(raw: string | undefined): boolean {
  const match = SERVER_ORIGIN.exec(String(raw ?? ''))
  return match !== null && isValidPort(match[1])
}

/** Why the enterprise web server must not start with `settings`, or null when it may. */
export function enterpriseStartUpProblem(settings: StartUpSettings): string | null {
  const media = parseMediaOrigins(settings.MEDIA_ORIGINS)
  if (!media.ok) return media.error
  if (!isServerOrigin(settings.BACKEND_INTERNAL_URL)) return BACKEND_INTERNAL_URL_REFUSED
  const cloudSync = parseCloudSync(settings.ENTERPRISE_CALLS_CLOUD_SYNC)
  if (!cloudSync.ok) return cloudSync.error
  return null
}
