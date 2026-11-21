/**
 * PRD-277 D-12 (#1605): whether the enterprise web server may let the browser load a Cloud
 * listing's screenshots from Automatos Cloud, read from ENTERPRISE_CALLS_CLOUD_SYNC when the server
 * runs (never a NEXT_PUBLIC_* variable: those are fixed when the image is built).
 *
 * The API reads the same setting to decide whether the install syncs with Automatos Cloud
 * (orchestrator/core/editions.py); the chart gives both containers the one value, from
 * edition.enterprise.hostedServices.cloudSync. It is read the way the API reads it: trimmed and
 * lower-cased, on for 1, true, yes or on, off for 0, false, no, off or unset. Any other value is
 * refused, and the server refuses to start (start-up.ts), naming the setting, never its value.
 */

/** The setting's name: the API's own. */
export const CLOUD_SYNC_SETTING = 'ENTERPRISE_CALLS_CLOUD_SYNC'

const SWITCH_ON: ReadonlySet<string> = new Set(['1', 'true', 'yes', 'on'])
const SWITCH_OFF: ReadonlySet<string> = new Set(['', '0', 'false', 'no', 'off'])

export const CLOUD_SYNC_REFUSED =
  `${CLOUD_SYNC_SETTING} must be true or false (or 1/0, yes/no, on/off), as the API reads it: ` +
  "the web app allows Automatos Cloud's screenshots in its content security policy only while Cloud sync is on."

export type CloudSync = { readonly ok: true; readonly on: boolean } | { readonly ok: false; readonly error: string }

/** Whether `raw` switches Cloud sync on, or the refusal when it is not a switch form. */
export function parseCloudSync(raw: string | undefined): CloudSync {
  const value = String(raw ?? '').trim().toLowerCase()
  if (SWITCH_ON.has(value)) return { ok: true, on: true }
  if (SWITCH_OFF.has(value)) return { ok: true, on: false }
  return { ok: false, error: CLOUD_SYNC_REFUSED }
}
