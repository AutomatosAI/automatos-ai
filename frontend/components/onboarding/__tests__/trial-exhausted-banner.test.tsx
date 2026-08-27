/**
 * PRD-222 US-014 — the deterministic exhausted state.
 *
 * The banner renders on either trigger — the exhausted trial snapshot OR the
 * typed `trial_exhausted` error a live request returns — and embeds the US-013
 * power-up card so a broke workspace can still take a key. It carries ZERO
 * LLM/network dependency: rendering it fires no fetch and no apiClient call; the
 * only network hop reachable from the surface is the credentials POST the user
 * triggers inside the embedded card.
 *
 * Pure/mocked — apiClient, the workspace provider, and next/link are stubbed
 * (the embedded PowerUpCard consumes them).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const { postMock, useWorkspaceMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
  useWorkspaceMock: vi.fn(),
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { post: postMock } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => useWorkspaceMock(),
}))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: Record<string, unknown>) => (
    <a href={href as string} {...(rest as Record<string, string>)}>
      {children as React.ReactNode}
    </a>
  ),
}))

import { TrialExhaustedBanner } from '../trial-exhausted-banner'

function withState(state: string | undefined) {
  useWorkspaceMock.mockReturnValue({
    workspace: {
      onboarding: {
        stage: 'questions',
        trial:
          state === undefined
            ? null
            : { granted_usd: 5, spent_usd: 5, state },
      },
    },
    refreshWorkspace: vi.fn(),
  })
}

describe('TrialExhaustedBanner (US-014)', () => {
  beforeEach(() => {
    postMock.mockReset()
    useWorkspaceMock.mockReset()
  })
  afterEach(() => vi.restoreAllMocks())

  it('renders on the exhausted trial snapshot, embedding the power-up card', () => {
    withState('exhausted')
    render(<TrialExhaustedBanner />)

    expect(screen.getByTestId('trial-exhausted-banner')).toBeInTheDocument()
    expect(screen.getByText(/your trial credit is used up/i)).toBeInTheDocument()
    // the US-013 power-up card is embedded (its key input is the way out)
    expect(screen.getByTestId('power-up-card')).toBeInTheDocument()
    expect(screen.getByTestId('power-up-key-input')).toBeInTheDocument()
  })

  it('renders on the typed trial_exhausted error even before the snapshot flips', () => {
    withState('active') // snapshot not yet exhausted…
    render(<TrialExhaustedBanner errorCode="trial_exhausted" />) // …but the live request was blocked

    expect(screen.getByTestId('trial-exhausted-banner')).toBeInTheDocument()
    expect(screen.getByTestId('power-up-card')).toBeInTheDocument()
  })

  it('does not render while the trial is still active with no error', () => {
    withState('active')
    const { container } = render(<TrialExhaustedBanner />)
    expect(screen.queryByTestId('trial-exhausted-banner')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('does not render for a converted workspace', () => {
    withState('converted')
    const { container } = render(<TrialExhaustedBanner />)
    expect(screen.queryByTestId('trial-exhausted-banner')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('makes no LLM/network call on render — only the credentials POST is reachable', () => {
    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue(new Response('{}'))
    withState('exhausted')
    render(<TrialExhaustedBanner />)

    // rendering the exhausted state is fully deterministic
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(postMock).not.toHaveBeenCalled()
    // the credentials endpoint is the only wired call, and only on user submit
    expect(screen.getByTestId('power-up-connect')).toBeInTheDocument()
  })
})
