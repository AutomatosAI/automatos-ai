/**
 * PRD-222 US-016 (W1·S10 / D9) — the dev onboarding-reset console.
 *
 * Pure/mocked: Clerk, navigation, the workspace provider, the billing summary and
 * fetch are stubbed. Covers: renders the stage and the trial's credits (PRD-265:
 * the trial is credits, read off the billing summary); the
 * wipe_credentials-without-reset_trial warning; the plain "disabled" state on a
 * 404; and that a successful reset sweeps residual legacy tour/onboarding browser
 * flags (real jsdom localStorage) and redirects to /chat.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'

const { useWorkspaceMock, pushMock, getAuthHeadersMock, refreshWorkspaceMock, billing } =
  vi.hoisted(() => ({
    useWorkspaceMock: vi.fn(),
    pushMock: vi.fn(),
    getAuthHeadersMock: vi.fn(async () => ({})),
    refreshWorkspaceMock: vi.fn(async () => {}),
    billing: { data: undefined as unknown },
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

vi.mock('@/hooks/use-billing-summary', () => ({
  useBillingSummary: () => billing,
}))

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    getBaseUrl: () => 'https://api.test',
    getAuthHeaders: getAuthHeadersMock,
  },
}))

import ResetOnboardingPage from '../page'

// A legacy tour/onboarding flag the reset sweep should clear (scoped by user).
const LEGACY_KEY = 'automatos-tour:welcome:user_dev_1'

function mockWorkspace(onboarding: any) {
  useWorkspaceMock.mockReturnValue({
    workspace: { onboarding },
    refreshWorkspace: refreshWorkspaceMock,
  })
}

/** The billing summary's trial: `trial` of 500 credits left. */
function mockTrialCredits(trial: number, paused = false) {
  billing.data = {
    metered: true,
    paused,
    trial_credits: 500,
    balance: { trial, monthly: 0, topup: 0, spendable: trial },
  }
}

describe('ResetOnboardingPage (US-016)', () => {
  beforeEach(() => {
    useWorkspaceMock.mockReset()
    pushMock.mockReset()
    refreshWorkspaceMock.mockReset()
    getAuthHeadersMock.mockClear()
    localStorage.clear()
    vi.restoreAllMocks()
    billing.data = undefined
  })

  it("renders the current stage from the workspace provider and the trial's credits from the billing summary", () => {
    mockWorkspace({ stage: 'boom' })
    mockTrialCredits(50)
    render(<ResetOnboardingPage />)

    expect(screen.getByTestId('reset-stage')).toHaveTextContent('boom')
    expect(screen.getByTestId('reset-trial')).toHaveTextContent('50 of 500 credits')
    expect(screen.getByTestId('reset-trial')).not.toHaveTextContent('paused')
  })

  it('says when the trial is spent and work is paused, and "none" when nothing is metered', () => {
    mockWorkspace({ stage: 'boom' })
    mockTrialCredits(0, true)
    const view = render(<ResetOnboardingPage />)
    expect(screen.getByTestId('reset-trial')).toHaveTextContent('0 of 500 credits · paused')
    view.unmount()

    billing.data = undefined
    render(<ResetOnboardingPage />)
    expect(screen.getByTestId('reset-trial')).toHaveTextContent('none')
  })

  it('warns when wipe_credentials is on while reset_trial is off', () => {
    mockWorkspace({ stage: 'completed' })
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
    mockWorkspace({ stage: 'questions' })
    localStorage.setItem(LEGACY_KEY, 'completed')
    vi.stubGlobal('fetch', vi.fn(async () => ({ status: 404, ok: false })) as any)
    render(<ResetOnboardingPage />)

    fireEvent.click(screen.getByTestId('reset-submit'))

    await waitFor(() => expect(screen.getByTestId('reset-disabled')).toBeInTheDocument())
    expect(pushMock).not.toHaveBeenCalled()
    // Disabled → no reset ran → the legacy flag is left untouched.
    expect(localStorage.getItem(LEGACY_KEY)).toBe('completed')
  })

  it('on success sweeps residual legacy tour/onboarding flags and redirects to /chat', async () => {
    mockWorkspace({ stage: 'boom' })
    mockTrialCredits(0, true)
    localStorage.setItem(LEGACY_KEY, 'completed')
    localStorage.setItem('automatos-onboarding-completed:user_dev_1', 'true')
    // A key for a different user must survive (scoping).
    localStorage.setItem('automatos-tour:welcome:user_other', 'completed')
    const fetchMock = vi.fn(async () => ({
      status: 200,
      ok: true,
      json: async () => ({ stage: 'not_started', resets: 1 }),
    }))
    vi.stubGlobal('fetch', fetchMock as any)
    render(<ResetOnboardingPage />)

    fireEvent.click(screen.getByTestId('reset-submit'))

    await waitFor(() => expect(pushMock).toHaveBeenCalledWith('/chat'))
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull()
    expect(localStorage.getItem('automatos-onboarding-completed:user_dev_1')).toBeNull()
    expect(localStorage.getItem('automatos-tour:welcome:user_other')).toBe('completed')
    expect(refreshWorkspaceMock).toHaveBeenCalled()
    // The reset endpoint was called with the three flags.
    const body = JSON.parse((fetchMock.mock.calls[0][1] as any).body)
    expect(body).toEqual({ reset_trial: false, wipe_built: false, wipe_credentials: false })
  })
})
