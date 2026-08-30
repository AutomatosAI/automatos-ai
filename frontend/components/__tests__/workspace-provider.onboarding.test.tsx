/**
 * PRD-222 W1S2/US-002 — the current-workspace onboarding snapshot reaches the
 * WorkspaceProvider. The backend now returns `onboarding: {stage, trial}` on
 * GET /api/workspaces/current; this asserts the provider maps it onto
 * `workspace.onboarding` and stays null-safe when the field is absent.
 * Pure/mocked — Clerk auth, navigation, and fetch are stubbed; no server.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'

const { getTokenMock } = vi.hoisted(() => ({
  getTokenMock: vi.fn(async () => 'test-token'),
}))

vi.mock('@clerk/nextjs', () => ({
  useAuth: () => ({ isSignedIn: true, getToken: getTokenMock }),
  useOrganization: () => ({ organization: null }),
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: null }),
  useSession: () => ({ isLoaded: true, isSignedIn: true, session: null }),
  useClerk: () => ({ signOut: async () => {} }),
}))

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
}))

import { WorkspaceProvider, useWorkspace } from '../workspace-provider'

function Probe() {
  const { workspace, isLoading } = useWorkspace()
  if (isLoading) return <div>loading</div>
  return (
    <div>
      <span data-testid="stage">{workspace?.onboarding?.stage ?? 'NONE'}</span>
      <span data-testid="granted">
        {workspace?.onboarding?.trial?.granted_usd ?? 'NONE'}
      </span>
      <span data-testid="spent">
        {workspace?.onboarding?.trial?.spent_usd ?? 'NONE'}
      </span>
      <span data-testid="state">
        {workspace?.onboarding?.trial?.state ?? 'NONE'}
      </span>
    </div>
  )
}

const BASE_RESPONSE = {
  id: 'ws-1',
  name: 'Acme',
  slug: 'acme',
  plan: 'starter',
  role: 'owner',
  plan_limits: {},
  webhook_url: null,
  webhook_key: null,
  settings: {},
}

function mockFetch(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => body,
    })),
  )
}

describe('WorkspaceProvider — onboarding snapshot (US-002)', () => {
  beforeEach(() => {
    getTokenMock.mockClear()
    process.env.NEXT_PUBLIC_API_URL = 'http://api.test'
  })

  it('exposes workspace.onboarding.stage and trial from the API', async () => {
    mockFetch({
      ...BASE_RESPONSE,
      onboarding: {
        stage: 'powerup',
        trial: { granted_usd: 5, spent_usd: 1.6, state: 'active' },
      },
    })

    render(
      <WorkspaceProvider>
        <Probe />
      </WorkspaceProvider>,
    )

    await waitFor(() =>
      expect(screen.getByTestId('stage')).toHaveTextContent('powerup'),
    )
    expect(screen.getByTestId('granted')).toHaveTextContent('5')
    expect(screen.getByTestId('spent')).toHaveTextContent('1.6')
    expect(screen.getByTestId('state')).toHaveTextContent('active')
  })

  it('is null-safe when the response omits onboarding', async () => {
    mockFetch({ ...BASE_RESPONSE })

    render(
      <WorkspaceProvider>
        <Probe />
      </WorkspaceProvider>,
    )

    await waitFor(() =>
      expect(screen.getByTestId('stage')).toHaveTextContent('NONE'),
    )
    expect(screen.getByTestId('granted')).toHaveTextContent('NONE')
    expect(screen.getByTestId('state')).toHaveTextContent('NONE')
  })
})
