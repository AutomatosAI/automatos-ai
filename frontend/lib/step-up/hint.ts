/**
 * PRD-261 A-07c/A-07d — the backend's step-up refusal, as the browser sees it.
 *
 * `orchestrator/core/auth/step_up.py` answers a sensitive theme route with HTTP 403 and one of
 * two bodies:
 *
 * - `step_up_required`: Clerk's `reverificationError({ level: 'first_factor', afterMinutes })`
 *   with `clerk_error` at the TOP level, plus `code` and a plain `detail` (`step_up_body`).
 *   Clerk's `useReverification` recognises it with `isReverificationHint`
 *   (`@clerk/shared` `authorization-errors`), opens its reverification and retries once.
 * - `step_up_unavailable`: an `HTTPException`, so `{ detail: { code, message } }`. The caller is
 *   an API key or anonymous: nobody can sign in again, so no prompt can help.
 *
 * `apiClient` attaches the parsed body to the error it throws as `body` (lib/api-client.ts,
 * `httpErrorOf`), which is how these readers find it. Pure functions, no Clerk import, so the
 * local edition (no Clerk mount) can load this file.
 */

export const STEP_UP_REQUIRED_CODE = 'step_up_required'
export const STEP_UP_UNAVAILABLE_CODE = 'step_up_unavailable'

/** Clerk's runtime error code when the person closes the reverification (`@clerk/shared` error). */
export const STEP_UP_CANCELLED_CODE = 'reverification_cancelled'

export const STEP_UP_FAILED_MESSAGE = "We couldn't confirm it's you. Sign in again, then try once more."

// @clerk/shared authorization-errors: REVERIFICATION_REASON and the error type.
const REVERIFICATION_REASON = 'reverification-error'
const FORBIDDEN_TYPE = 'forbidden'
const FORBIDDEN_STATUS = 403

/** The `step_up_required` body (a Clerk reverification hint). */
export interface StepUpHint {
  readonly clerk_error: {
    readonly type: 'forbidden'
    readonly reason: 'reverification-error'
    readonly metadata?: {
      readonly reverification?: { readonly level?: string; readonly afterMinutes?: number }
    }
  }
  readonly code?: string
  readonly detail?: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/**
 * Clerk's `isReverificationHint`, mirrored: `@clerk/shared` is not a direct dependency of the
 * frontend, and `@clerk/nextjs` does not re-export it. The test pins this to Clerk's own.
 */
export function isStepUpHint(body: unknown): body is StepUpHint {
  if (!isRecord(body)) return false
  const clerkError = body.clerk_error
  return isRecord(clerkError) && clerkError.type === FORBIDDEN_TYPE && clerkError.reason === REVERIFICATION_REASON
}

/** The parsed body of a 403 `apiClient` error, or null for any other error. */
function forbiddenBodyOf(error: unknown): unknown {
  if (!isRecord(error) || error.status !== FORBIDDEN_STATUS) return null
  return 'body' in error ? error.body : null
}

/** The hint a thrown `apiClient` error carries: a 403 whose body is Clerk's hint; else null. */
export function stepUpHintOf(error: unknown): StepUpHint | null {
  const body = forbiddenBodyOf(error)
  return isStepUpHint(body) ? body : null
}

/** True for `step_up_unavailable`: a key or anonymous caller, whom no prompt can confirm. */
export function isStepUpUnavailable(error: unknown): boolean {
  const body = forbiddenBodyOf(error)
  if (!isRecord(body)) return false
  const detail = body.detail
  const code = isRecord(detail) ? detail.code : body.code
  return code === STEP_UP_UNAVAILABLE_CODE
}

/** True when the person closed Clerk's reverification (Clerk's `isReverificationCancelledError`). */
export function isStepUpCancelled(error: unknown): boolean {
  return isRecord(error) && error.code === STEP_UP_CANCELLED_CODE
}

/**
 * The action was refused again after a reverification: the one retry did not satisfy step-up.
 * It carries the hint as `hint`, not as `body`, so no outer wrapper reads it as a fresh refusal.
 */
export class StepUpFailedError extends Error {
  readonly hint: StepUpHint

  constructor(hint: StepUpHint) {
    super(STEP_UP_FAILED_MESSAGE)
    this.name = 'StepUpFailedError'
    this.hint = hint
  }
}
