'use client'

/**
 * PRD-261 A-07c/A-07d — `useStepUp`: run an `apiClient` action that a step-up route may refuse.
 *
 * SaaS: the action is wrapped in Clerk's `useReverification`. When the backend answers 403 with
 * its reverification hint (lib/step-up/hint.ts), the wrapped fetcher hands Clerk the hint as a
 * value, which is how Clerk reads it: `createReverificationHandler` calls the fetcher, and on
 * `isReverificationHint(result)` opens the reverification (or calls `onNeedsReverification`),
 * waits for it, and calls the fetcher ONE more time (`@clerk/shared` react/index.mjs:1529-1576).
 * A hint left after that retry is thrown as `StepUpFailedError`; a closed prompt rejects with
 * Clerk's `reverification_cancelled` error (`isStepUpCancelled`); any other error is thrown as
 * the action threw it.
 *
 * Local edition: a no-op. There is no Clerk mount (`useReverification` would throw) and the
 * backend's step-up passes every local request (`step_up.py`, `_is_local_edition`).
 *
 * `isSaaS` is a build-time constant, so each build has one implementation and the rules of
 * hooks hold (the lib/auth-hooks.ts pattern). Clerk's hook is read only when it is called.
 */

import { useCallback } from 'react'
import { useReverification } from '@clerk/nextjs'

import { isSaaS } from '@/lib/auth-edition'

import { isStepUpHint, stepUpHintOf, StepUpFailedError } from './hint'

/** An action that may meet a step-up refusal: usually one `apiClient` theme method. */
export type StepUpAction<Args extends unknown[], Result> = (...args: Args) => Promise<Result>

/** Clerk's options: `onNeedsReverification` replaces Clerk's own modal with the caller's UI. */
export type StepUpOptions = NonNullable<Parameters<typeof useReverification>[1]>

/** The fetcher Clerk wraps: a step-up refusal comes back as Clerk's hint; other errors are rethrown. */
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
