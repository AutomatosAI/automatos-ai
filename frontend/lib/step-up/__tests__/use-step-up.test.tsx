/**
 * PRD-261 A-07c/A-07d — useStepUp (lib/step-up/use-step-up.ts).
 *
 * Clerk's real useReverification needs a mounted ClerkProvider, which no unit test has. So
 * `@clerk/nextjs` is mocked with a double that does what Clerk's `createReverificationHandler`
 * does (`@clerk/shared` react/index.mjs:1529-1576): call the fetcher; when the result passes
 * Clerk's own `isReverificationHint`, call `onNeedsReverification({ cancel, complete, level })`,
 * wait, and call the fetcher ONE more time; a cancel rejects with Clerk's ClerkRuntimeError.
 */
import { renderHook } from '@testing-library/react'
import { isReverificationHint } from '@clerk/shared/authorization-errors'
import { ClerkRuntimeError } from '@clerk/shared/error'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const clerk = vi.hoisted(() => ({ useReverification: vi.fn() }))
vi.mock('@clerk/nextjs', () => ({ useReverification: clerk.useReverification }))

type Fetcher = (...args: unknown[]) => Promise<unknown>
type NeedsReverification = (params: { cancel: () => void; complete: () => void; level: string | undefined }) => void

function levelOf(hint: unknown): string | undefined {
  return (hint as { clerk_error: { metadata?: { reverification?: { level?: string } } } }).clerk_error.metadata
    ?.reverification?.level
}

/** Clerk's createReverificationHandler, minus its modal: onNeedsReverification is required here. */
function clerkReverification(fetcher: Fetcher, options?: { onNeedsReverification?: NeedsReverification }) {
  return async (...args: unknown[]) => {
    let result = await fetcher(...args)
    if (isReverificationHint(result)) {
      await new Promise((resolve, reject) => {
        const cancel = () =>
          reject(new ClerkRuntimeError('User cancelled attempted verification', { code: 'reverification_cancelled' }))
        options?.onNeedsReverification?.({ cancel, complete: () => resolve(true), level: levelOf(result) })
      })
      result = await fetcher(...args)
    }
    return result
  }
}

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

function stepUpError(): Error {
  return Object.assign(new Error(STEP_UP_BODY.detail), { status: 403, body: STEP_UP_BODY })
}

async function loadStepUp(edition: 'saas' | 'local') {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  vi.resetModules()
  const hook = await import('@/lib/step-up/use-step-up')
  const hint = await import('@/lib/step-up/hint')
  return { ...hook, ...hint }
}

beforeEach(() => {
  clerk.useReverification.mockReset()
  clerk.useReverification.mockImplementation(clerkReverification)
})

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('useStepUp (saas)', () => {
  it('returns the answer and never prompts when the action passes', async () => {
    const { useStepUp } = await loadStepUp('saas')
    const action = vi.fn(async (id: string) => ({ id }))
    const onNeedsReverification = vi.fn()
    const { result } = renderHook(() => useStepUp(action, { onNeedsReverification }))

    await expect(result.current('install-1')).resolves.toEqual({ id: 'install-1' })
    expect(action).toHaveBeenCalledTimes(1)
    expect(onNeedsReverification).not.toHaveBeenCalled()
  })

  it('shows the reverification on a step-up 403 and retries the action once', async () => {
    const { useStepUp } = await loadStepUp('saas')
    const action = vi.fn().mockRejectedValueOnce(stepUpError()).mockResolvedValueOnce('saved')
    const onNeedsReverification = vi.fn<NeedsReverification>((params) => params.complete())
    const { result } = renderHook(() => useStepUp(action, { onNeedsReverification }))

    await expect(result.current('changes')).resolves.toBe('saved')
    expect(onNeedsReverification).toHaveBeenCalledTimes(1)
    expect(onNeedsReverification.mock.calls[0][0].level).toBe('first_factor')
    expect(action).toHaveBeenCalledTimes(2)
    expect(action).toHaveBeenNthCalledWith(2, 'changes')
  })

  it('throws StepUpFailedError when the retry is refused again (no second prompt)', async () => {
    const { useStepUp, StepUpFailedError, STEP_UP_FAILED_MESSAGE } = await loadStepUp('saas')
    const action = vi.fn().mockRejectedValue(stepUpError())
    const onNeedsReverification = vi.fn<NeedsReverification>((params) => params.complete())
    const { result } = renderHook(() => useStepUp(action, { onNeedsReverification }))

    const error = await result.current().catch((e: unknown) => e)
    expect(error).toBeInstanceOf(StepUpFailedError)
    expect((error as Error).message).toBe(STEP_UP_FAILED_MESSAGE)
    expect(action).toHaveBeenCalledTimes(2)
    expect(onNeedsReverification).toHaveBeenCalledTimes(1)
  })

  it('rejects with Clerk cancel error when the person closes the prompt', async () => {
    const { useStepUp, isStepUpCancelled } = await loadStepUp('saas')
    const action = vi.fn().mockRejectedValue(stepUpError())
    const onNeedsReverification = vi.fn<NeedsReverification>((params) => params.cancel())
    const { result } = renderHook(() => useStepUp(action, { onNeedsReverification }))

    const error = await result.current().catch((e: unknown) => e)
    expect(isStepUpCancelled(error)).toBe(true)
    expect(action).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['a conflict', Object.assign(new Error('pinned'), { status: 409, body: { detail: 'pinned' } })],
    [
      'step_up_unavailable (a key cannot reverify)',
      Object.assign(new Error('keys'), { status: 403, body: { detail: { code: 'step_up_unavailable', message: 'keys' } } }),
    ],
    ['a network failure', new Error('Failed to fetch')],
  ])('rethrows %s unchanged without prompting', async (_name, failure) => {
    const { useStepUp } = await loadStepUp('saas')
    const action = vi.fn().mockRejectedValue(failure)
    const onNeedsReverification = vi.fn()
    const { result } = renderHook(() => useStepUp(action, { onNeedsReverification }))

    await expect(result.current()).rejects.toBe(failure)
    expect(onNeedsReverification).not.toHaveBeenCalled()
    expect(action).toHaveBeenCalledTimes(1)
  })

  it('a step-up 403 from a real apiClient call reaches onNeedsReverification, and the retry succeeds', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { useStepUp } = await loadStepUp('saas')
    const { apiClient } = await import('@/lib/api-client')
    const settings = { membersMayChoose: false }
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(STEP_UP_BODY), { status: 403 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(settings), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const onNeedsReverification = vi.fn<NeedsReverification>((params) => params.complete())
    const { result } = renderHook(() =>
      useStepUp(apiClient.putThemeSettings.bind(apiClient), { onNeedsReverification }),
    )

    await expect(result.current({ membersMayChoose: false })).resolves.toEqual(settings)
    expect(onNeedsReverification).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})

describe('useStepUp (local edition)', () => {
  it('is a no-op: the action itself, and Clerk is never asked', async () => {
    const { useStepUp } = await loadStepUp('local')
    const action = vi.fn().mockRejectedValue(stepUpError())
    const { result } = renderHook(() => useStepUp(action))

    expect(result.current).toBe(action)
    await expect(result.current()).rejects.toThrow(STEP_UP_BODY.detail)
    expect(clerk.useReverification).not.toHaveBeenCalled()
  })
})
