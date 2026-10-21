/**
 * PRD-222 W1S2/US-002 — the current-workspace onboarding snapshot reaches the
 * WorkspaceProvider. The backend returns `onboarding: {stage}` on
 * GET /api/workspaces/current (PRD-265: the trial is credits in the ledger now,
 * served by GET /api/v1/billing/summary); this asserts the provider maps it onto
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

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { WorkspaceProvider, useWorkspace } from '../workspace-provider'

function Probe() {
  const { workspace, isLoading } = useWorkspace()
  if (isLoading) return <div>loading</div>
  return (
    <div>
      <span data-testid="stage">{workspace?.onboarding?.stage ?? 'NONE'}</span>
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

  it('exposes workspace.onboarding.stage from the API', async () => {
    mockFetch({
      ...BASE_RESPONSE,
      onboarding: { stage: 'powerup' },
    })

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorkspaceProvider>
          <Probe />
        </WorkspaceProvider>
      </QueryClientProvider>,
    )

    await waitFor(() =>
      expect(screen.getByTestId('stage')).toHaveTextContent('powerup'),
    )
  })

  it('is null-safe when the response omits onboarding', async () => {
    mockFetch({ ...BASE_RESPONSE })

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorkspaceProvider>
          <Probe />
        </WorkspaceProvider>
      </QueryClientProvider>,
    )

    await waitFor(() =>
      expect(screen.getByTestId('stage')).toHaveTextContent('NONE'),
    )
  })
})
