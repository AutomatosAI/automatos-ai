/**
 * PRD-181 S5 — EU-AI-Act Art.14 human oversight on the approval card.
 *
 * The backend `mission_approval` payload now carries `risk_tier`,
 * `risk_class`, and `oversight_rationale`. This pins the frontend half: the card
 * renders the risk tier + the rationale (why a human is in the loop), and when no
 * oversight fields are present it renders no oversight banner.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@/hooks/use-missions-api', () => ({
  useApproveMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useRejectMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useUpdateMissionPlan: () => ({ isLoading: false, mutateAsync: vi.fn() }),
}))

import { MissionApprovalWidget, oversightTierLabel } from '../MissionApprovalWidget'
import type { MissionApprovalWidgetData, WidgetMetadata } from '../types'

const metadata: WidgetMetadata = {
  source: { type: 'tool', name: 'platform_create_mission' },
  createdAt: new Date(),
}

function data(overrides?: Partial<MissionApprovalWidgetData>): MissionApprovalWidgetData {
  return {
    mission_id: 'm-1',
    goal: 'Refund the customer and email them',
    task_count: 2,
    tasks: [
      { title: 'Issue refund', agent_role: 'ops', sequence: 1 },
      { title: 'Send email', agent_role: 'comms', sequence: 2 },
    ],
    ...overrides,
  }
}

describe('MissionApprovalWidget — AI-Act oversight (PRD-181 S5)', () => {
  it('shows the risk tier and oversight rationale', () => {
    render(
      <MissionApprovalWidget
        id="w1"
        title="Mission plan"
        data={data({
          risk_tier: 'human_in_the_loop',
          risk_class: 'external_side_effect',
          oversight_rationale:
            'External side-effect: acts on a third-party system. A human must approve before it runs.',
        })}
        metadata={metadata}
      />,
    )
    // the tier label + the (humanised) risk class
    expect(screen.getByText(/Human approval required/)).toBeInTheDocument()
    expect(screen.getByText(/external side effect/)).toBeInTheDocument()
    // the rationale
    expect(screen.getByText(/acts on a third-party system/)).toBeInTheDocument()
    // it is flagged as an oversight note for a11y
    expect(screen.getByRole('note', { name: /human oversight/i })).toBeInTheDocument()
  })

  it('renders no oversight banner when no risk fields are present', () => {
    render(<MissionApprovalWidget id="w2" title="Mission plan" data={data()} metadata={metadata} />)
    expect(screen.queryByRole('note', { name: /human oversight/i })).not.toBeInTheDocument()
  })

  it('maps oversight tiers to human-readable labels', () => {
    expect(oversightTierLabel('monitor')).toBe('Monitored')
    expect(oversightTierLabel('human_on_the_loop')).toBe('Human on the loop')
    expect(oversightTierLabel('human_in_the_loop')).toBe('Human approval required')
    // unknown / absent falls safe to the strongest label
    expect(oversightTierLabel(undefined)).toBe('Human approval required')
  })
})
