/**
 * PRD-277 US-603 — the page a phone's sign-in waits on (app/sign-in/phone), its calls and its words.
 *
 * A phone signs in at the API's own start address in its system browser. After the organisation's
 * identity provider, the API makes no code: it binds what waits to this browser with a cookie scripts
 * cannot read and sends the browser here, with nothing in the address. This page says who is signing
 * in, to which app, on which phone, and asks the person to continue only if they started it from the
 * app. Only Continue makes the one-time code; the page then opens the app link the API answers.
 *
 *   GET  /api/v1/auth/oidc/phone/waiting   {person, app, platform, removed_before}, or 404 nothing_waiting
 *   POST /api/v1/auth/oidc/phone/continue  {app_link}, or 404 nothing_waiting, 409 device_limit_reached
 *   POST /api/v1/auth/oidc/phone/cancel    204
 *
 * All through apiClient, which in enterprise sends the cookie and, on a POST, the cross-site header
 * the API asks for (session-transport.ts); the browser adds the page's own Origin. A refusal the API
 * makes before any of this (`?error=<code>` on this page: no workspace, several, the device cap) is
 * one sentence of PHONE_REFUSALS; any other code reads as the general one, never the code itself.
 */
import { apiClient } from '@/lib/api-client'

export const PHONE_WAITING_PATH = '/api/v1/auth/oidc/phone/waiting'
export const PHONE_CONTINUE_PATH = '/api/v1/auth/oidc/phone/continue'
export const PHONE_CANCEL_PATH = '/api/v1/auth/oidc/phone/cancel'
export const PHONE_ERROR_PARAM = 'error'
/** The only app link the page will open (the API's phone_sign_in.APP_LINKS). */
export const APP_LINK_PREFIX = 'automatos://go/signed-in?'

export const NOTHING_WAITING_CODE = 'nothing_waiting'
const NOT_FOUND = 404

/** GET /api/v1/auth/oidc/phone/waiting. */
export interface PhoneWaiting {
  readonly person: string | null
  readonly app: string
  readonly platform: string | null
  readonly removed_before: boolean
}

export const PHONE_TITLE = 'Sign in on your phone'
export const PHONE_CONTINUE = 'Continue'
export const PHONE_CANCEL = 'Cancel'
export const PHONE_ONLY_IF_YOU_STARTED = 'Continue only if you started this from the app.'
export const PHONE_REMOVED_BEFORE = 'This phone was removed from Automatos before. Continuing adds it again.'
export const PHONE_NOTHING_WAITING =
  'There is no phone sign-in waiting here. It took too long, or it already finished. Start again from the app.'
export const PHONE_CANCELLED = 'Signing in on your phone was cancelled. You can close this window.'
export const PHONE_FAILED = 'Something went wrong. Try again in a moment.'

const DEVICES: Readonly<Record<string, string>> = { ios: 'iPhone', android: 'Android phone' }

/** "You are signing in to the <app> app on this <device> as <person>." */
export function phoneSigningInAs(waiting: PhoneWaiting): string {
  const device = (waiting.platform && DEVICES[waiting.platform]) || 'phone'
  const who = waiting.person ? `as ${waiting.person}` : "with your organisation's account"
  return `You are signing in to the ${waiting.app} app on this ${device} ${who}.`
}

export function phoneOpening(app: string): string {
  return `Opening ${app}. You can close this window once it opens.`
}

/** What the page says for each refusal the API sends here once the person is known. */
export const PHONE_REFUSALS: Readonly<Record<string, string>> = {
  no_workspace:
    'You signed in, but you are not in a workspace yet, so this phone cannot be added. ' +
    'Ask an administrator to add you to a workspace, then start again from the app.',
  several_workspaces:
    'You are in more than one workspace, so the app cannot sign in on its own. Pair this phone with a code ' +
    'instead, if your organisation allows it: in Automatos in your browser, open Settings, then ' +
    'Phones & devices, then Pair a phone.',
  device_limit_reached:
    'You already have the most phones one person can have here. Remove one in Automatos in your browser ' +
    '(Settings, then Phones & devices), then start again from the app.',
}

/** The sentence for a `?error=` code, the general one for a code it does not know, or null for none. */
export function phoneRefusalMessage(code: string | null | undefined): string | null {
  if (!code) return null
  return Object.prototype.hasOwnProperty.call(PHONE_REFUSALS, code) ? PHONE_REFUSALS[code] : PHONE_FAILED
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/** The API's refusal code in an apiClient error (`{detail: {code}}`), or null. */
export function refusalCodeOf(error: unknown): string | null {
  const body = isRecord(error) ? error.body : null
  const detail = isRecord(body) ? body.detail : null
  return isRecord(detail) && typeof detail.code === 'string' ? detail.code : null
}

/** What waits in this browser, or null when nothing does; any other failure is thrown. */
export async function fetchPhoneWaiting(): Promise<PhoneWaiting | null> {
  try {
    return await apiClient.get<PhoneWaiting>(PHONE_WAITING_PATH)
  } catch (error) {
    if (isRecord(error) && error.status === NOT_FOUND) return null
    throw error
  }
}

/** Continue: the app link carrying the one-time code, only when it is the app's own link. */
export async function continuePhoneSignIn(): Promise<string> {
  const answer = await apiClient.post<{ app_link?: unknown }>(PHONE_CONTINUE_PATH)
  const link = answer?.app_link
  if (typeof link !== 'string' || !link.startsWith(APP_LINK_PREFIX)) throw new Error('not the app link')
  return link
}

export function cancelPhoneSignIn(): Promise<unknown> {
  return apiClient.post(PHONE_CANCEL_PATH)
}
