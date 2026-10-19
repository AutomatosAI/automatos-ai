/**
 * PRD-261 A-07f (Gerard D1-d, #1157) — the team page's role change under step-up.
 *
 * In SaaS the server refuses a role change without a sign-in from the last few minutes
 * (orchestrator/api/team.py `STEP_UP`) with HTTP 403 and Clerk's reverification hint at the top
 * level. The role change goes through `useStepUp` (lib/theme-host/api.ts), so the person sees
 * Clerk's "confirm it's you" and the request is sent once more, instead of a failure. D1-d scopes
 * step-up to the role change only: the invite modal is unchanged.
 *
 * The real `apiClient` runs over a stubbed `fetch`, so the hint travels the real error path
 * (`httpErrorOf` attaches the body). Clerk's real `useReverification` needs a mounted
 * ClerkProvider, so `@clerk/nextjs` is a double that does what Clerk's
 * `createReverificationHandler` does (`@clerk/shared` react/index.mjs:1529-1576): call the
 * fetcher; on Clerk's own `isReverificationHint`, prompt (here: the person confirms, or closes the
 * prompt), then call the fetcher ONE more time.
 */
import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { isReverificationHint } from '@clerk/shared/authorization-errors'
import { ClerkRuntimeError } from '@clerk/shared/error'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const WS = '00000000-0000-0000-0000-00000000a07f'

const clerk = vi.hoisted(() => ({
  useReverification: vi.fn(),
  answer: 'confirm' as 'confirm' | 'close',
  prompts: 0,
}))
vi.mock('@clerk/nextjs', () => ({ useReverification: clerk.useReverification }))
vi.mock('@/hooks/use-workspace', () => ({ useWorkspace: () => ({ workspaceId: WS, loading: false }) }))
vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ orgId: null }) }))
vi.mock('@/components/shared', () => ({
  PageHeader: ({ title, actions }: { title: string; actions?: React.ReactNode }) => (
    <header>
      <h1>{title}</h1>
      {actions}
    </header>
  ),
  SearchInput: () => null,
}))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
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

const MEMBER = {
  id: 5, user_id: '9', email: 'sam@example.com', name: 'Sam', role: 'editor', joined_at: '2026-10-01T00:00:00Z',
  last_active_at: null, tool_runs_30d: null, chats_30d: null, invited_by_email: null,
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status })
}

/** `fetch` that answers each "METHOD path-suffix" from its own queue (the last answer repeats). */
function stubBackend(answers: Record<string, Response[]>) {
  const calls: string[] = []
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const key = Object.keys(answers).find((k) => {
      const [method, suffix] = k.split(' ')
      return (init?.method || 'GET') === method && String(url).endsWith(suffix)
    })
    if (!key) return json([], 200)
    calls.push(key)
    const queue = answers[key]
    const answer = queue.length > 1 ? queue.shift() : queue[0]
    return (answer as Response).clone()
  })
  vi.stubGlobal('fetch', fetchMock)
  return calls
}

async function load(edition: 'saas' | 'local') {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  return {
    ...(await import('@/components/team/team-management')),
    ...(await import('@/lib/theme-host/api')),
  }
}

beforeEach(() => {
  clerk.useReverification.mockReset()
  clerk.useReverification.mockImplementation(clerkReverification)
  clerk.answer = 'confirm'
  clerk.prompts = 0
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

const ROLE = `PATCH /api/workspaces/${WS}/team/members/5/role`
const MEMBERS = `GET /api/workspaces/${WS}/team/members`

async function changeSamToAdmin(edition: 'saas' | 'local', roleAnswers: Response[]) {
  const calls = stubBackend({ [MEMBERS]: [json([MEMBER])], [ROLE]: roleAnswers })
  const alert = vi.fn()
  vi.stubGlobal('alert', alert)
  const { TeamManagement } = await load(edition)
  render(<TeamManagement />)
  fireEvent.change(await screen.findByDisplayValue('Editor'), { target: { value: 'admin' } })
  return { calls, alert }
}

describe('TeamManagement role change (saas)', () => {
  it('a step-up 403 shows Clerk\'s prompt and the change is sent once more', async () => {
    const { calls, alert } = await changeSamToAdmin('saas', [json(STEP_UP_BODY, 403), json({ status: 'success' })])

    await waitFor(() => expect(calls.filter((c) => c === ROLE)).toHaveLength(2))
    expect(clerk.prompts).toBe(1)
    expect(alert).not.toHaveBeenCalled()
  })

  it('a closed prompt changes nothing and says so', async () => {
    clerk.answer = 'close'
    const { calls, alert } = await changeSamToAdmin('saas', [json(STEP_UP_BODY, 403)])
    const { STEP_UP_CANCELLED_MESSAGE } = await import('@/lib/theme-host/api')

    await waitFor(() => expect(alert).toHaveBeenCalledWith(STEP_UP_CANCELLED_MESSAGE))
    expect(calls.filter((c) => c === ROLE)).toHaveLength(1)
  })

  it('a refusal after the one retry says the confirmation failed (no second prompt)', async () => {
    const { calls, alert } = await changeSamToAdmin('saas', [json(STEP_UP_BODY, 403)])
    const { STEP_UP_FAILED_MESSAGE } = await import('@/lib/theme-host/api')

    await waitFor(() => expect(alert).toHaveBeenCalledWith(STEP_UP_FAILED_MESSAGE))
    expect(calls.filter((c) => c === ROLE)).toHaveLength(2)
    expect(clerk.prompts).toBe(1)
  })
})

describe('TeamManagement role change (local edition)', () => {
  it('sends the change once and never asks Clerk', async () => {
    const { calls, alert } = await changeSamToAdmin('local', [json({ status: 'success' })])

    await waitFor(() => expect(calls.filter((c) => c === ROLE)).toHaveLength(1))
    expect(clerk.useReverification).not.toHaveBeenCalled()
    expect(alert).not.toHaveBeenCalled()
  })
})
