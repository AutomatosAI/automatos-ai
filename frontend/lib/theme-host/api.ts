'use client'

/**
 * PRD-261 A-07c/A-07d — calling a theme route that may ask the person to confirm it's them.
 *
 * `orchestrator/core/auth/step_up.py` answers a sensitive theme route (import, opt-in, default,
 * update, rollback, settings) with HTTP 403 and one of two bodies:
 *
 * - `step_up_required`: Clerk's `reverificationError({ level: 'first_factor', afterMinutes })`
 *   with `clerk_error` at the TOP level, plus `code` and a plain `detail` (`step_up_body`).
 * - `step_up_unavailable`: an `HTTPException`, so `{ detail: { code, message } }`. The caller is
 *   an API key or anonymous: nobody can sign in again, so no prompt can help.
 *
 * `apiClient` attaches the parsed body of a non-2xx answer to the error it throws as `body`
 * (lib/api-client.ts `httpErrorOf`, plan §2.2). `useStepUp` wraps an `apiClient` method in
 * Clerk's `useReverification` with a fetcher that RETURNS `err.body` when it is Clerk's hint:
 * Clerk's `createReverificationHandler` calls the fetcher and, on `isReverificationHint(result)`,
 * opens its reverification (or calls `onNeedsReverification`), waits, and calls the fetcher ONE
 * more time (`@clerk/shared` react/index.mjs:1529-1576). A hint left after that retry is thrown
 * as `StepUpFailedError`; a closed prompt rejects with Clerk's `reverification_cancelled` error
 * (`isStepUpCancelled`); any other error is thrown as the method threw it.
 *
 * Local edition: a no-op. There is no Clerk mount (`useReverification` would throw) and the
 * backend's step-up passes every local request (`step_up.py`, `_is_local_edition`). `isSaaS` is a
 * build-time constant, so each build has one implementation and the rules of hooks hold (the
 * lib/auth-hooks.ts pattern).
 *
 * Moved here from the W-A2 opening commit's lib/step-up/ (15266f0f7), as plan §2.6 A-07d names.
 */

import { useCallback } from 'react'
import { useReverification } from '@clerk/nextjs'

import { isSaaS } from '@/lib/auth-edition'

export const STEP_UP_REQUIRED_CODE = 'step_up_required'
export const STEP_UP_UNAVAILABLE_CODE = 'step_up_unavailable'

/** Clerk's runtime error code when the person closes the reverification (`@clerk/shared` error). */
export const STEP_UP_CANCELLED_CODE = 'reverification_cancelled'

export const STEP_UP_FAILED_MESSAGE = "We couldn't confirm it's you. Sign in again, then try once more."
export const STEP_UP_CANCELLED_MESSAGE = 'Nothing was changed: the confirmation was closed.'
const FALLBACK_MESSAGE = 'Something went wrong. Try again.'
/** A 401: the session ended. apiClient's own 401 text is for developers (missing Clerk token). */
export const SIGNED_OUT_MESSAGE = 'Your session has ended. Sign in again, then try once more.'
const UNAUTHORIZED_STATUS = 401
/** apiClient throws a 401 without `status`; its message opens with this (lib/api-client.ts httpErrorOf). */
const UNAUTHORIZED_MESSAGE_PREFIX = `HTTP ${UNAUTHORIZED_STATUS}:`

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

/** Whether `error` is apiClient's 401 (signed out, or the session token expired). */
function isUnauthorized(error: unknown): boolean {
  if (isRecord(error) && error.status === UNAUTHORIZED_STATUS) return true
  return error instanceof Error && error.message.startsWith(UNAUTHORIZED_MESSAGE_PREFIX)
}

/**
 * The sentence a theme route's refusal (or a step-up flow's: role change, invitation, API key)
 * shows: sign in again on a 401, `detail` (string or `{message}`), else the error's own.
 */
export function themeErrorMessage(error: unknown): string {
  if (isStepUpCancelled(error)) return STEP_UP_CANCELLED_MESSAGE
  if (isUnauthorized(error)) return SIGNED_OUT_MESSAGE
  const body = isRecord(error) && 'body' in error ? error.body : null
  const detail = isRecord(body) ? body.detail : null
  if (typeof detail === 'string' && detail) return detail
  if (isRecord(detail) && typeof detail.message === 'string') return detail.message
  return error instanceof Error && error.message ? error.message : FALLBACK_MESSAGE
}

/** An action that may meet a step-up refusal: usually one `apiClient` theme method. */
export type StepUpAction<Args extends unknown[], Result> = (...args: Args) => Promise<Result>

/** Clerk's options: `onNeedsReverification` replaces Clerk's own modal with the caller's UI. */
export type StepUpOptions = NonNullable<Parameters<typeof useReverification>[1]>

/** The fetcher Clerk wraps: a step-up refusal comes back as Clerk's hint (`err.body`); other errors are rethrown. */
export function asReverificationFetcher<Args extends unknown[], Result>(
  action: StepUpAction<Args, Result>,
): (...args: Args) => Promise<unknown> {
  return async (...args: Args) => {
    try {
      return await action(...args)
    } catch (error) {
      const hint = stepUpHintOf(error)
      if (hint) return hint
      throw error
    }
  }
}

/** Clerk's answer after its one retry: a hint left means step-up refused again. */
export function settleStepUp<Result>(result: unknown): Result {
  if (isStepUpHint(result)) throw new StepUpFailedError(result)
  return result as Result
}

function useClerkStepUp<Args extends unknown[], Result>(
  action: StepUpAction<Args, Result>,
  options?: StepUpOptions,
): StepUpAction<Args, Result> {
  const enhanced = useReverification(asReverificationFetcher(action), options)
  return useCallback(
    async (...args: Args) => {
      const result: unknown = await enhanced(...args)
      return settleStepUp<Result>(result)
    },
    [enhanced],
  )
}

function useLocalStepUp<Args extends unknown[], Result>(
  action: StepUpAction<Args, Result>,
  _options?: StepUpOptions,
): StepUpAction<Args, Result> {
  return action
}

/**
 * `const putSettings = useStepUp(apiClient.putThemeSettings.bind(apiClient))`, then
 * `await putSettings(changes)`. SaaS: Clerk asks the person to confirm it's them and the action
 * runs once more; local: the action itself.
 */
export const useStepUp: <Args extends unknown[], Result>(
  action: StepUpAction<Args, Result>,
  options?: StepUpOptions,
) => StepUpAction<Args, Result> = isSaaS ? useClerkStepUp : useLocalStepUp
