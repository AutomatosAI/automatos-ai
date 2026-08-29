/**
 * PRD-222 W2·S1b (US-024) — the exposure profile reaches the WorkspaceProvider.
 * The backend returns `exposure: {plan, families, marketplace_depth, nav}` on
 * GET /api/workspaces/current; this asserts the provider maps it onto
 * `workspace.exposure` and stays null-safe when the field is absent.
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
      <span data-testid="plan">{workspace?.plan ?? 'NONE'}</span>
      <span data-testid="nav-analytics">{String(workspace?.exposure?.nav?.analytics)}</span>
      <span data-testid="nav-team">{String(workspace?.exposure?.nav?.team)}</span>
      <span data-testid="depth">{workspace?.exposure?.marketplace_depth ?? 'NONE'}</span>
    </div>
  )
}

const BASE_RESPONSE = {
  id: 'ws-1',
  name: 'Acme',
  slug: 'acme',
  plan: 'basic',
  role: 'owner',
  plan_limits: {},
  webhook_url: null,
  webhook_key: null,
  settings: {},
}

function mockFetch(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => body })),
  )
}

describe('WorkspaceProvider — exposure snapshot (US-024)', () => {
  beforeEach(() => {
    getTokenMock.mockClear()
    process.env.NEXT_PUBLIC_API_URL = 'http://api.test'
  })

  it('exposes workspace.exposure (nav + depth) from the API', async () => {
    mockFetch({
      ...BASE_RESPONSE,
      exposure: {
        plan: 'basic',
        families: { codegraph: false, nl2sql: false, team: false, voice: false },
        marketplace_depth: 1,
        nav: { analytics: false, team: false },
      },
    })

    render(
      <WorkspaceProvider>
        <Probe />
      </WorkspaceProvider>,
    )

    await waitFor(() => expect(screen.getByTestId('plan')).toHaveTextContent('basic'))
    expect(screen.getByTestId('nav-analytics')).toHaveTextContent('false')
    expect(screen.getByTestId('nav-team')).toHaveTextContent('false')
    expect(screen.getByTestId('depth')).toHaveTextContent('1')
  })

  it('is null-safe when the response omits exposure', async () => {
    mockFetch({ ...BASE_RESPONSE })

    render(
      <WorkspaceProvider>
        <Probe />
      </WorkspaceProvider>,
    )

    await waitFor(() => expect(screen.getByTestId('plan')).toHaveTextContent('basic'))
    expect(screen.getByTestId('nav-analytics')).toHaveTextContent('undefined')
    expect(screen.getByTestId('depth')).toHaveTextContent('NONE')
  })
})
