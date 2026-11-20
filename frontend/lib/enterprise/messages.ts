/**
 * PRD-277 Wave 5 — every sentence the enterprise web app shows for a sign-in or session code, in
 * one place. The codes are the API's (orchestrator/core/auth/sign_in_flow.py REFUSAL_CODES, the
 * session lane's `session_ended` and `workspace_required`, step-up's `step_up_required`).
 *
 * A refusal never says whether a person is known: an unknown person, a deactivated one and an
 * email another account holds all read the same, because the API sends each of them the same code.
 */

export const SIGN_IN_ERROR_PARAM = 'error'

export const SIGN_IN_REFUSALS: Readonly<Record<string, string>> = {
  sign_in_expired: 'Signing in took too long, or was started in another tab. Start again.',
  sign_in_refused: "Signing in didn't finish. Try again, and ask your administrator if it keeps happening.",
  sign_in_not_allowed:
    "Your organisation's sign-in worked, but this account isn't allowed into Automatos. Ask your administrator for access.",
  step_up_refused: "We couldn't confirm it's you, so nothing was changed. Sign in again, then try once more.",
  step_up_no_auth_time:
    "Admin actions can't be confirmed: your organisation's sign-in did not say when you last signed in. " +
    'An administrator needs to set the identity provider to send auth_time in its ID token ' +
    "(Microsoft Entra ID: add the auth_time optional claim; Google: enable auth_time in the client's settings; " +
    'Amazon Cognito: use managed login).',
  identity_provider_unavailable: "Your organisation's sign-in can't be reached right now. Try again in a few minutes.",
  sign_in_failed: "Signing in didn't finish. Try again.",
}

/** What any other code reads as: never the code itself, never anything the URL can make up. */
export const SIGN_IN_REFUSAL_FALLBACK = SIGN_IN_REFUSALS.sign_in_failed

/** The sentence for a `?error=` code, or null when there is none. */
export function signInRefusalMessage(code: string | null | undefined): string | null {
  if (!code) return null
  return Object.prototype.hasOwnProperty.call(SIGN_IN_REFUSALS, code) ? SIGN_IN_REFUSALS[code] : SIGN_IN_REFUSAL_FALLBACK
}

export const SESSION_ENDED_CODE = 'session_ended'
export const SESSION_ENDED_TITLE = 'Your session ended'
export const SESSION_ENDED_BODY =
  'Sign in again to carry on. Anything you were typing in the chat box is kept for when you come back.'
export const SIGN_IN_AGAIN = 'Sign in again'

export const WAITING_TITLE = 'You are signed in'
export const WAITING_BODY = 'You are not in a workspace yet. Ask an administrator to add you.'
export const SIGN_OUT = 'Sign out'

export const UNREACHABLE_TITLE = "Automatos can't be reached"
export const UNREACHABLE_BODY = 'We could not check your sign-in. Try again in a moment.'
export const TRY_AGAIN = 'Try again'
export const LOADING = 'Checking your sign-in…'

export const PROFILE_MANAGED_NOTE =
  "Your account is managed by your organisation's identity provider. Change your name or email there; they update here the next time you sign in."

export const STEP_UP_REQUIRED_CODE = 'step_up_required'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/** Whether a 401's parsed body is the session lane's `session_ended`. */
export function isSessionEnded(body: unknown): boolean {
  if (!isRecord(body)) return false
  const detail = body.detail
  return (isRecord(detail) ? detail.code : body.code) === SESSION_ENDED_CODE
}
