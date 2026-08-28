/**
 * PRD-222 US-019 — the chat-surface mount for inline connect cards.
 *
 * Renders a ConnectAppCard during the onboarding build stage for every app the
 * workspace has been asked to connect but hasn't finished (a non-`active`
 * Composio connection). Off the build stage, or with nothing mid-connect, it
 * renders nothing. ConnectAppCard is stubbed — the card's own OAuth/callback
 * behaviour is covered by connect-app-card.test.tsx.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const { useWorkspaceMock, useConnectedAppsMock } = vi.hoisted(() => ({
  useWorkspaceMock: vi.fn(),
  useConnectedAppsMock: vi.fn(),
}))

vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => useWorkspaceMock() }))
vi.mock('@/hooks/use-composio-api', () => ({ useConnectedApps: () => useConnectedAppsMock() }))
vi.mock('../connect-app-card', () => ({
  ConnectAppCard: ({ appName }: { appName: string }) => (
    <div data-testid="stub-connect-card" data-app={appName.toUpperCase()} />
  ),
}))

import { OnboardingConnectCards } from '../onboarding-connect-cards'

function setup(stage: string | undefined, connections: Array<{ app_name: string; status: string }>) {
  useWorkspaceMock.mockReturnValue({ workspace: stage ? { onboarding: { stage } } : {} })
  useConnectedAppsMock.mockReturnValue({ data: connections })
}

describe('OnboardingConnectCards (US-019)', () => {
  beforeEach(() => {
    useWorkspaceMock.mockReset()
    useConnectedAppsMock.mockReset()
  })

  it('renders nothing outside the build stage', () => {
    setup('powerup', [{ app_name: 'GMAIL', status: 'pending' }])
    const { container } = render(<OnboardingConnectCards />)
    expect(container).toBeEmptyDOMElement()
  })

  it('surfaces a connect card for each app mid-connect during building', () => {
    setup('building', [
      { app_name: 'GMAIL', status: 'pending' },
      { app_name: 'SLACK', status: 'added' },
      { app_name: 'NOTION', status: 'active' }, // already connected — not offered
    ])
    render(<OnboardingConnectCards />)
    const cards = screen.getAllByTestId('stub-connect-card')
    expect(cards.map((c) => c.getAttribute('data-app'))).toEqual(['GMAIL', 'SLACK'])
  })

  it('renders nothing when every connection is already active', () => {
    setup('building', [{ app_name: 'GMAIL', status: 'active' }])
    const { container } = render(<OnboardingConnectCards />)
    expect(container).toBeEmptyDOMElement()
  })

  it('applies the caller-supplied positioning className', () => {
    setup('building', [{ app_name: 'GMAIL', status: 'pending' }])
    render(<OnboardingConnectCards className="fixed bottom-28" />)
    expect(screen.getByTestId('onboarding-connect-cards').className).toContain('fixed bottom-28')
  })
})
