/**
 * PRD-261 A-07f (#1157) — creating an SDK key under step-up (Settings › Widget SDK › API keys).
 *
 * In SaaS the server refuses `POST /api/api-keys` without a sign-in from the last few minutes
 * (orchestrator/api/api_keys.py `STEP_UP`) with HTTP 403 and Clerk's reverification hint at the
 * top level. ApiKeyManager calls it through `useStepUp` (lib/theme-host/api.ts), so the person
 * sees Clerk's "confirm it's you" and the key is requested once more, instead of a failure.
 *
 * The real `apiClient` runs over a stubbed `fetch`, so the hint travels the real error path.
 * `@clerk/nextjs` is a double of Clerk's `createReverificationHandler` (`@clerk/shared`
 * react/index.mjs:1529-1576): call the fetcher; on Clerk's own `isReverificationHint`, prompt
 * (the person confirms, or closes it), then call the fetcher ONE more time. The two Radix
 * `<Select>`s are native `<select>`s: Radix's listbox needs pointer capture jsdom lacks, and the
 * form's selects are not under test (the add-api-key-dialog.test.tsx precedent).
 */
import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { isReverificationHint } from '@clerk/shared/authorization-errors'
import { ClerkRuntimeError } from '@clerk/shared/error'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const clerk = vi.hoisted(() => ({
  useReverification: vi.fn(),
  answer: 'confirm' as 'confirm' | 'close',
  prompts: 0,
}))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))

vi.mock('@clerk/nextjs', () => ({ useReverification: clerk.useReverification }))
vi.mock('sonner', () => ({ toast }))

type WrapperProps = { children?: React.ReactNode }
vi.mock('@/components/ui/select', () => ({
  Select: ({ value, onValueChange, children }: WrapperProps & { value: string; onValueChange: (v: string) => void }) => (
    <select value={value} onChange={(e) => onValueChange(e.target.value)}>
      {children}
    </select>
  ),
  SelectTrigger: ({ children }: WrapperProps) => <>{children}</>,
  SelectValue: () => null,
  SelectContent: ({ children }: WrapperProps) => <>{children}</>,
  SelectItem: ({ value, children }: WrapperProps & { value: string }) => <option value={value}>{children}</option>,
}))

type Fetcher = (...args: unknown[]) => Promise<unknown>

/** Clerk's createReverificationHandler, with the person's answer to the prompt scripted. */
function clerkReverification(fetcher: Fetcher) {
  return async (...args: unknown[]) => {
    const result = await fetcher(...args)
    if (!isReverificationHint(result)) return result
    clerk.prompts += 1
    if (clerk.answer === 'close') {
      throw new ClerkRuntimeError('User cancelled attempted verification', { code: 'reverification_cancelled' })
    }
    return fetcher(...args)
  }
}

// orchestrator/core/auth/step_up.py step_up_body(10).
const STEP_UP_BODY = {
  clerk_error: {
    type: 'forbidden',
    reason: 'reverification-error',
    metadata: { reverification: { level: 'first_factor', afterMinutes: 10 } },
  },
  code: 'step_up_required',
  detail: "Confirm it's you to continue: this action needs a sign-in from the last 10 minutes.",
}

const CREATED = { id: 'key-1', name: 'CI', key: 'minted-in-the-test', key_type: 'server', permissions: ['chat'] }

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status })
}

/** `fetch` for /api/api-keys: GET lists none; each POST takes the next answer (the last repeats). */
function stubKeys(postAnswers: Response[]) {
  const posts: unknown[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (!String(url).endsWith('/api/api-keys') || (init?.method || 'GET') !== 'POST') return json([])
    posts.push(JSON.parse(String(init?.body)))
    const answer = postAnswers.length > 1 ? postAnswers.shift() : postAnswers[0]
    return (answer as Response).clone()
  }))
  return posts
}

async function createKey(edition: 'saas' | 'local', postAnswers: Response[]) {
  const posts = stubKeys(postAnswers)
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  const { ApiKeyManager } = await import('@/components/settings/ApiKeyManager')
  const messages = await import('@/lib/theme-host/api')
  render(<ApiKeyManager />)
  await screen.findByText('No API keys yet')
  fireEvent.click(screen.getAllByRole('button', { name: /Create API Key/ })[0])
  fireEvent.change(await screen.findByPlaceholderText('e.g. Production Backend'), { target: { value: 'CI' } })
  fireEvent.click(screen.getByRole('button', { name: 'Select All' }))
  fireEvent.click(screen.getByRole('button', { name: 'Create Key' }))
  return { posts, messages }
}

beforeEach(() => {
  clerk.useReverification.mockReset()
  clerk.useReverification.mockImplementation(clerkReverification)
  clerk.answer = 'confirm'
  clerk.prompts = 0
  toast.success.mockReset()
  toast.error.mockReset()
  vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.spyOn(console, 'log').mockImplementation(() => {})
})

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.resetModules()
})

describe('ApiKeyManager key creation (saas)', () => {
  it('a step-up 403 shows Clerk\'s prompt, the key is requested once more and shown once', async () => {
    const { posts } = await createKey('saas', [json(STEP_UP_BODY, 403), json(CREATED, 201)])

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('API key created'))
    expect(posts).toHaveLength(2)
    expect(posts[1]).toEqual(posts[0])
    expect(clerk.prompts).toBe(1)
    expect(screen.getByDisplayValue('minted-in-the-test')).toBeInTheDocument()
    expect(toast.error).not.toHaveBeenCalled()
  })

  it('a closed prompt mints nothing and says so', async () => {
    clerk.answer = 'close'
    const { posts, messages } = await createKey('saas', [json(STEP_UP_BODY, 403)])

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(messages.STEP_UP_CANCELLED_MESSAGE))
    expect(posts).toHaveLength(1)
    expect(toast.success).not.toHaveBeenCalled()
  })

  it('a refusal after the one retry says the confirmation failed (no second prompt)', async () => {
    const { posts, messages } = await createKey('saas', [json(STEP_UP_BODY, 403)])

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(messages.STEP_UP_FAILED_MESSAGE))
    expect(posts).toHaveLength(2)
    expect(clerk.prompts).toBe(1)
  })

  it('a key caller (step_up_unavailable) is told why, with no prompt', async () => {
    const unavailable = { detail: { code: 'step_up_unavailable', message: 'keys cannot confirm it' } }
    const { posts } = await createKey('saas', [json(unavailable, 403)])

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith('keys cannot confirm it'))
    expect(posts).toHaveLength(1)
    expect(clerk.prompts).toBe(0)
  })
})

describe('ApiKeyManager key creation (local edition)', () => {
  it('requests the key once and never asks Clerk', async () => {
    const { posts } = await createKey('local', [json(CREATED, 201)])

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('API key created'))
    expect(posts).toHaveLength(1)
    expect(clerk.useReverification).not.toHaveBeenCalled()
  })
})
