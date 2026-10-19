/**
 * PRD-261 A-07c/A-07d — the step-up refusal readers (lib/step-up/hint.ts).
 *
 * The mirror of Clerk's `isReverificationHint` must agree with Clerk's own on every body, and
 * the codes must be the backend's (orchestrator/core/auth/step_up.py, read here as text, so a
 * rename on either side fails this test).
 */
import { readFileSync } from 'fs'
import path from 'path'

import { isReverificationHint, reverificationError } from '@clerk/shared/authorization-errors'
import { ClerkRuntimeError, isReverificationCancelledError } from '@clerk/shared/error'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiClient } from '@/lib/api-client'
import {
  STEP_UP_REQUIRED_CODE,
  STEP_UP_UNAVAILABLE_CODE,
  STEP_UP_CANCELLED_CODE,
  isStepUpCancelled,
  isStepUpHint,
  isStepUpUnavailable,
  stepUpHintOf,
} from '@/lib/step-up/hint'

const STEP_UP_PY = path.resolve(__dirname, '..', '..', '..', '..', 'orchestrator', 'core', 'auth', 'step_up.py')

// step_up.py step_up_body(10).
const STEP_UP_BODY = {
  clerk_error: {
    type: 'forbidden',
    reason: 'reverification-error',
    metadata: { reverification: { level: 'first_factor', afterMinutes: 10 } },
  },
  code: 'step_up_required',
  detail: "Confirm it's you to continue: this action needs a sign-in from the last 10 minutes.",
}

// step_up.py _refuse_without_person: an HTTPException, so the code nests under detail.
const UNAVAILABLE_BODY = {
  detail: { code: 'step_up_unavailable', message: 'This action needs a person signed in to Automatos; keys cannot confirm it.' },
}

function apiError(status: number, body?: unknown): Error {
  const extra = body === undefined ? { status } : { status, body }
  return Object.assign(new Error('refused'), extra)
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('isStepUpHint', () => {
  const bodies: Array<[string, unknown]> = [
    ['the backend step-up body', STEP_UP_BODY],
    ["Clerk's own reverificationError()", reverificationError({ level: 'first_factor', afterMinutes: 10 })],
    ['the step_up_unavailable body', UNAVAILABLE_BODY],
    ['a hint nested under detail (HTTPException shape)', { detail: STEP_UP_BODY }],
    ['another forbidden reason', { clerk_error: { type: 'forbidden', reason: 'other' } }],
    ['another type', { clerk_error: { type: 'unauthorized', reason: 'reverification-error' } }],
    ['a string clerk_error', { clerk_error: 'forbidden' }],
    ['null', null],
    ['a string', 'reverification-error'],
    ['undefined', undefined],
  ]

  it.each(bodies)('agrees with Clerk isReverificationHint on %s', (_name, body) => {
    expect(isStepUpHint(body)).toBe(Boolean(isReverificationHint(body)))
  })

  it('recognises the backend body', () => {
    expect(isStepUpHint(STEP_UP_BODY)).toBe(true)
    expect(isStepUpHint(UNAVAILABLE_BODY)).toBe(false)
  })
})

describe('the codes are the backend codes', () => {
  const source = readFileSync(STEP_UP_PY, 'utf8')

  it.each([
    ['STEP_UP_CODE', STEP_UP_REQUIRED_CODE],
    ['STEP_UP_UNAVAILABLE_CODE', STEP_UP_UNAVAILABLE_CODE],
    ['REVERIFICATION_REASON', 'reverification-error'],
    ['REVERIFICATION_ERROR_TYPE', 'forbidden'],
  ])('step_up.py %s is %s', (name, value) => {
    expect(source).toContain(`${name} = "${value}"`)
  })
})

describe('stepUpHintOf', () => {
  it('reads the hint from a 403 apiClient error', () => {
    expect(stepUpHintOf(apiError(403, STEP_UP_BODY))).toEqual(STEP_UP_BODY)
  })

  it.each([
    ['a 409 with a hint body', apiError(409, STEP_UP_BODY)],
    ['a 403 without a body', apiError(403)],
    ['a 403 for another reason', apiError(403, { detail: 'Not a member of this workspace.' })],
    ['step_up_unavailable', apiError(403, UNAVAILABLE_BODY)],
    ['a plain Error', new Error('network down')],
    ['a string', 'refused'],
  ])('is null for %s', (_name, error) => {
    expect(stepUpHintOf(error)).toBeNull()
  })

  it('reads the hint end to end from a real apiClient call (the body rides the error)', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const response = new Response(JSON.stringify(STEP_UP_BODY), {
      status: 403,
      headers: { 'Content-Type': 'application/json' },
    })
    vi.stubGlobal('fetch', vi.fn(async () => response))

    const error = await apiClient.putThemeSettings({ membersMayChoose: false }).catch((e: unknown) => e)

    expect(isReverificationHint((error as { body?: unknown }).body)).toBe(true)
    expect(stepUpHintOf(error)).toEqual(STEP_UP_BODY)
  })
})

describe('isStepUpUnavailable', () => {
  it('is true for the nested HTTPException body and for a top-level code', () => {
    expect(isStepUpUnavailable(apiError(403, UNAVAILABLE_BODY))).toBe(true)
    expect(isStepUpUnavailable(apiError(403, { code: STEP_UP_UNAVAILABLE_CODE }))).toBe(true)
  })

  it('is false for the reverification hint, other statuses and other errors', () => {
    expect(isStepUpUnavailable(apiError(403, STEP_UP_BODY))).toBe(false)
    expect(isStepUpUnavailable(apiError(400, UNAVAILABLE_BODY))).toBe(false)
    expect(isStepUpUnavailable(new Error('x'))).toBe(false)
  })
})

describe('isStepUpCancelled', () => {
  it("agrees with Clerk's isReverificationCancelledError", () => {
    const cancelled = new ClerkRuntimeError('User cancelled attempted verification', { code: STEP_UP_CANCELLED_CODE })
    const other = new ClerkRuntimeError('Something else', { code: 'network_error' })
    expect(isStepUpCancelled(cancelled)).toBe(isReverificationCancelledError(cancelled))
    expect(isStepUpCancelled(cancelled)).toBe(true)
    expect(isStepUpCancelled(other)).toBe(false)
    expect(isStepUpCancelled(new Error('x'))).toBe(false)
  })
})
