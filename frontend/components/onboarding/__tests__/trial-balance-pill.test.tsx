/**
 * PRD-222 US-014 — the trial balance pill.
 *
 * Reads the trial straight off the US-002 workspace snapshot: renders the
 * remaining/granted balance, goes amber once warned (and stays visible while
 * exhausted), and hides entirely once the workspace converts or never had a
 * trial. Pure/mocked — only the workspace provider is stubbed.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const { useWorkspaceMock } = vi.hoisted(() => ({ useWorkspaceMock: vi.fn() }))

vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => useWorkspaceMock(),
}))

import { TrialBalancePill } from '../trial-balance-pill'

function withTrial(
  trial: { granted_usd: number; spent_usd: number; state: string } | null,
) {
  useWorkspaceMock.mockReturnValue({ workspace: { onboarding: { trial } } })
}

describe('TrialBalancePill (US-014)', () => {
  beforeEach(() => useWorkspaceMock.mockReset())

  it('renders the balance from workspace.onboarding.trial', () => {
    withTrial({ granted_usd: 5, spent_usd: 1.6, state: 'active' })
    render(<TrialBalancePill />)

    const pill = screen.getByTestId('trial-balance-pill')
    expect(pill).toHaveTextContent('$3.40 of $5.00 trial')
    expect(pill).toHaveAttribute('data-state', 'active')
  })

  it('goes amber once the trial is warned', () => {
    withTrial({ granted_usd: 5, spent_usd: 4.5, state: 'warned' })
    render(<TrialBalancePill />)

    const pill = screen.getByTestId('trial-balance-pill')
    expect(pill).toHaveAttribute('data-state', 'warned')
    expect(pill.className).toContain('amber')
  })

  it('stays visible (amber) when exhausted', () => {
    withTrial({ granted_usd: 5, spent_usd: 5, state: 'exhausted' })
    render(<TrialBalancePill />)

    const pill = screen.getByTestId('trial-balance-pill')
    expect(pill).toHaveTextContent('$0.00 of $5.00 trial')
    expect(pill.className).toContain('amber')
  })

  it('is absent once the workspace has converted', () => {
    withTrial({ granted_usd: 5, spent_usd: 2, state: 'converted' })
    const { container } = render(<TrialBalancePill />)
    expect(screen.queryByTestId('trial-balance-pill')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('is absent when there is no trial', () => {
    withTrial(null)
    const { container } = render(<TrialBalancePill />)
    expect(screen.queryByTestId('trial-balance-pill')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('applies the caller-supplied positioning className', () => {
    withTrial({ granted_usd: 5, spent_usd: 0, state: 'active' })
    render(<TrialBalancePill className="absolute top-3 right-4 z-30" />)
    expect(screen.getByTestId('trial-balance-pill').className).toContain(
      'absolute top-3 right-4 z-30',
    )
  })
})
