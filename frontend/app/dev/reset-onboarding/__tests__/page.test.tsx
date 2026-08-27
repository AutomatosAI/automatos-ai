/**
 * PRD-222 US-016 (W1·S10 / D9) — the dev onboarding-reset console.
 *
 * Pure/mocked: Clerk, navigation, the workspace provider, the tour-storage
 * helpers, and fetch are all stubbed — no server, no real localStorage writes.
 * Covers: renders stage + trial; the wipe_credentials-without-reset_trial
 * warning; the plain "disabled" state on a 404; and that a successful reset
 * clears the tour/onboarding localStorage via the imported shepherd helpers and
 * redirects to /chat.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'

const { useWorkspaceMock, pushMock, resetAllToursMock, resetOnboardingMock, getAuthHeadersMock, refreshWorkspaceMock } =
  vi.hoisted(() => ({
    useWorkspaceMock: vi.fn(),
    pushMock: vi.fn(),
    resetAllToursMock: vi.fn(),
    resetOnboardingMock: vi.fn(),
    getAuthHeadersMock: vi.fn(async () => ({})),
    refreshWorkspaceMock: vi.fn(async () => {}),
  }))

vi.mock('@clerk/nextjs', () => ({
  useUser: () => ({ user: { id: 'user_dev_1' } }),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: pushMock }),
}))

vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => useWorkspaceMock(),
}))

vi.mock('@/lib/shepherd/tour-storage', () => ({
  resetAllTours: resetAllToursMock,
  resetOnboarding: resetOnboardingMock,
}))

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    getBaseUrl: () => 'https://api.test',
    getAuthHeaders: getAuthHeadersMock,
  },
}))

import ResetOnboardingPage from '../page'

function mockWorkspace(onboarding: any) {
  useWorkspaceMock.mockReturnValue({
    workspace: { onboarding },
    refreshWorkspace: refreshWorkspaceMock,
  })
}

describe('ResetOnboardingPage (US-016)', () => {
  beforeEach(() => {
    useWorkspaceMock.mockReset()
    pushMock.mockReset()
    resetAllToursMock.mockReset()
    resetOnboardingMock.mockReset()
    refreshWorkspaceMock.mockReset()
    getAuthHeadersMock.mockClear()
    vi.restoreAllMocks()
  })

  it('renders the current stage and trial state from the workspace provider', () => {
    mockWorkspace({ stage: 'boom', trial: { state: 'warned', spent_usd: 4.5, granted_usd: 5 } })
    render(<ResetOnboardingPage />)

    expect(screen.getByTestId('reset-stage')).toHaveTextContent('boom')
    expect(screen.getByTestId('reset-trial')).toHaveTextContent('warned')
    expect(screen.getByTestId('reset-trial')).toHaveTextContent('$4.50 of $5.00')
  })

  it('warns when wipe_credentials is on while reset_trial is off', () => {
    mockWorkspace({ stage: 'completed', trial: null })
    render(<ResetOnboardingPage />)

    // Not shown until credentials wipe is toggled on.
    expect(screen.queryByTestId('reset-credentials-warning')).toBeNull()

    fireEvent.click(screen.getByTestId('toggle-wipe-credentials'))
    expect(screen.getByTestId('reset-credentials-warning')).toBeInTheDocument()

    // Turning reset_trial on too clears the warning.
    fireEvent.click(screen.getByTestId('toggle-reset-trial'))
    expect(screen.queryByTestId('reset-credentials-warning')).toBeNull()
  })

  it('renders the plain disabled state when the endpoint 404s', async () => {
    mockWorkspace({ stage: 'questions', trial: null })
    vi.stubGlobal('fetch', vi.fn(async () => ({ status: 404, ok: false })) as any)
    render(<ResetOnboardingPage />)

    fireEvent.click(screen.getByTestId('reset-submit'))

    await waitFor(() => expect(screen.getByTestId('reset-disabled')).toBeInTheDocument())
    expect(pushMock).not.toHaveBeenCalled()
    expect(resetAllToursMock).not.toHaveBeenCalled()
  })

  it('on success clears tour/onboarding localStorage via shepherd helpers and redirects to /chat', async () => {
    mockWorkspace({ stage: 'boom', trial: { state: 'exhausted', spent_usd: 5, granted_usd: 5 } })
    const fetchMock = vi.fn(async () => ({
      status: 200,
      ok: true,
      json: async () => ({ stage: 'not_started', resets: 1 }),
    }))
    vi.stubGlobal('fetch', fetchMock as any)
    render(<ResetOnboardingPage />)

    fireEvent.click(screen.getByTestId('reset-submit'))

    await waitFor(() => expect(pushMock).toHaveBeenCalledWith('/chat'))
    expect(resetAllToursMock).toHaveBeenCalledWith('user_dev_1')
    expect(resetOnboardingMock).toHaveBeenCalledWith('user_dev_1')
    expect(refreshWorkspaceMock).toHaveBeenCalled()
    // The reset endpoint was called with the three flags.
    const body = JSON.parse((fetchMock.mock.calls[0][1] as any).body)
    expect(body).toEqual({ reset_trial: false, wipe_built: false, wipe_credentials: false })
  })
})
