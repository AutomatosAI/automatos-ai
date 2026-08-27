/**
 * PRD-222 US-012 — Auto's opener.
 *
 * The opener is Auto's first move on a brand-new workspace: it renders in the
 * empty chat ONLY when onboarding.stage === 'not_started', asks the first
 * question, and invites the user to just start typing. Once onboarding advances
 * (or there is no onboarding snapshot) it renders nothing. A new-workspace
 * render shows the opener and no welcome modal — the modal mount was retired
 * from providers.tsx, so nothing on the opener path can surface it.
 *
 * Pure/mocked — the workspace provider is stubbed; no server, no Clerk.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const { useWorkspaceMock } = vi.hoisted(() => ({ useWorkspaceMock: vi.fn() }))

vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => useWorkspaceMock(),
}))

import { OnboardingOpener } from '../onboarding-opener'

function withStage(stage: string | undefined) {
  useWorkspaceMock.mockReturnValue({
    workspace: stage === undefined ? {} : { onboarding: { stage } },
  })
}

describe('OnboardingOpener (US-012)', () => {
  beforeEach(() => useWorkspaceMock.mockReset())

  it('renders Auto\'s opener in the empty chat when stage is not_started', () => {
    withStage('not_started')
    render(<OnboardingOpener />)

    expect(screen.getByTestId('onboarding-opener')).toBeInTheDocument()
    // asks the first onboarding question…
    expect(screen.getByText(/what.?s your business/i)).toBeInTheDocument()
    // …and invites "or just start typing"
    expect(screen.getByText(/just start typing/i)).toBeInTheDocument()
  })

  it('shows the opener and NO welcome modal for a new workspace', () => {
    withStage('not_started')
    render(<OnboardingOpener />)

    expect(screen.getByTestId('onboarding-opener')).toBeInTheDocument()
    // the retired WelcomeModal's CTA must not appear on the opener path
    expect(screen.queryByTestId('business-intake-cta')).toBeNull()
    expect(screen.queryByTestId('business-intake-start')).toBeNull()
  })

  it('renders nothing once onboarding has moved past not_started', () => {
    withStage('questions')
    const { container } = render(<OnboardingOpener />)

    expect(screen.queryByTestId('onboarding-opener')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing when there is no onboarding snapshot', () => {
    withStage(undefined)
    const { container } = render(<OnboardingOpener />)

    expect(screen.queryByTestId('onboarding-opener')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })
})
