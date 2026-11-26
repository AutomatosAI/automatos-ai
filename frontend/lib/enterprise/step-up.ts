'use client'

/**
 * PRD-277 US-406, US-504 — step-up in enterprise: the person confirms it's them by signing in
 * again at the organisation's identity provider.
 *
 * The API refuses a sensitive action from a sign-in older than its window with
 *   403 {code: "step_up_required", detail, step_up: {start: "/api/v1/auth/oidc/start?step_up=1", max_age_minutes}}
 * (orchestrator/core/auth/step_up.py `sign_in_again_body`). Then the browser goes to that start
 * address, with the page it is on as `return_to`; the API's rule decides whether that path may be
 * used. The chat box's unsent text is kept first (kept-chat-draft.ts). Nothing about the action is
 * kept: on return the person does it again, now with a fresh sign-in.
 *
 * Only the API's own start route is followed: a `start` naming anything else is not a step-up the
 * web app knows, and the refusal is thrown as it came.
 */
import { useCallback } from 'react'

import { SIGN_IN_START_PATH, currentPath, signInStartUrl } from './auth-api'
import { keepChatDraft } from './kept-chat-draft'
import { STEP_UP_REQUIRED_CODE } from './messages'

const FORBIDDEN_STATUS = 403

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/** The API's start address in a `step_up_required` refusal, when it is the API's own start route; else null. */
export function signInAgainStartOf(error: unknown): string | null {
  if (!isRecord(error) || error.status !== FORBIDDEN_STATUS || !isRecord(error.body)) return null
  const body = error.body
  if (body.code !== STEP_UP_REQUIRED_CODE || !isRecord(body.step_up)) return null
  const start = body.step_up.start
  if (typeof start !== 'string') return null
  const path = start.split('?')[0]
  return path === SIGN_IN_START_PATH ? start : null
}

/** Where the browser goes: a seam so tests can watch it. */
export type Navigate = (url: string) => void

const assignLocation: Navigate = (url) => window.location.assign(url)

/** Keep the chat draft and send the browser to sign in again; the returned promise never settles. */
export function signInAgainFor<Result>(start: string, navigate: Navigate = assignLocation): Promise<Result> {
  keepChatDraft()
  navigate(signInStartUrl(currentPath(), start))
  return new Promise<Result>(() => {})
}

/**
 * `useStepUp`'s enterprise implementation (lib/theme-host/api.ts): the action as it is, except
 * that a step-up refusal sends the browser to sign in again.
 */
export function useEnterpriseStepUp<Args extends unknown[], Result>(
  action: (...args: Args) => Promise<Result>,
  _options?: unknown,
): (...args: Args) => Promise<Result> {
  return useCallback(
    async (...args: Args) => {
      try {
        return await action(...args)
      } catch (error) {
        const start = signInAgainStartOf(error)
        if (start === null) throw error
        return signInAgainFor<Result>(start)
      }
    },
    [action],
  )
}
