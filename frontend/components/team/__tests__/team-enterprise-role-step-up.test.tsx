/**
 * PRD-277 US-504 — a role change on the Team page goes through step-up in enterprise.
 *
 * The API refuses a role change from a sign-in older than its window with 403
 * `step_up_required` and the API's own start address (orchestrator/core/auth/step_up.py). The page
 * calls `useStepUp` (lib/theme-host/api.ts), which in enterprise is `useEnterpriseStepUp`: the
 * browser goes to sign in again at that start address, bringing the person back to this page, and
 * the change is not sent twice or shown as a failure. Clerk's reverification never runs.
 *
 * The real apiClient runs over a stubbed `fetch`, so the refusal travels the real error path. Where
 * the browser is sent is read from `signInStartUrl`, whose answer here is a hash, so jsdom can follow it.
 */
import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const WS = '00000000-0000-0000-0000-00000000e504'
const START = '/api/v1/auth/oidc/start?step_up=1'

const clerk = vi.hoisted(() => ({ useReverification: vi.fn() }))
const signIn = vi.hoisted(() => ({ startUrl: vi.fn((_returnTo: string | null | undefined, _start?: string) => '#signing-in-again') }))

vi.mock('@clerk/nextjs', () => ({ useReverification: clerk.useReverification }))
vi.mock('@/lib/enterprise/auth-api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/enterprise/auth-api')>()),
  signInStartUrl: signIn.startUrl,
}))
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

// orchestrator/core/auth/step_up.py sign_in_again_body(10).
const STEP_UP_BODY = {
  code: 'step_up_required',
  detail: "Confirm it's you to continue: this action needs a sign-in from the last 10 minutes.",
  step_up: { start: START, max_age_minutes: 10 },
}

const MEMBER = {
  id: 5, user_id: '9', email: 'sam@example.test', name: 'Sam', role: 'editor', joined_at: '2026-10-01T00:00:00Z',
  last_active_at: null, tool_runs_30d: null, chats_30d: null, invited_by_email: null,
}

const ROLE = `PATCH /api/workspaces/${WS}/team/members/5/role`
const MEMBERS = `GET /api/workspaces/${WS}/team/members`

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status })
}

/** `fetch` that answers each "METHOD path-suffix"; every call is recorded. */
function stubBackend(answers: Record<string, Response>) {
  const calls: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const key = Object.keys(answers).find((candidate) => {
      const [method, suffix] = candidate.split(' ')
      return (init?.method || 'GET') === method && String(url).endsWith(suffix)
    })
    if (!key) return json([], 200)
    calls.push(key)
    return answers[key].clone()
  }))
  return calls
}

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.resetModules()
  window.history.replaceState(null, '', '/')
})

describe('TeamManagement role change (enterprise)', () => {
  it('a step-up 403 sends the browser to sign in again at the API start address, back to the Team page', async () => {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'enterprise')
    window.history.replaceState(null, '', '/team?tab=members')
    const calls = stubBackend({ [MEMBERS]: json([MEMBER]), [ROLE]: json(STEP_UP_BODY, 403) })
    const alert = vi.fn()
    vi.stubGlobal('alert', alert)
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { TeamManagement } = await import('@/components/team/team-management')

    render(<TeamManagement />)
    fireEvent.change(await screen.findByDisplayValue('Editor'), { target: { value: 'admin' } })

    await waitFor(() => expect(signIn.startUrl).toHaveBeenCalledWith('/team?tab=members', START))
    await waitFor(() => expect(window.location.hash).toBe('#signing-in-again'))
    expect(calls.filter((call) => call === ROLE)).toHaveLength(1)
    expect(alert).not.toHaveBeenCalled()
    expect(clerk.useReverification).not.toHaveBeenCalled()
  })
})
