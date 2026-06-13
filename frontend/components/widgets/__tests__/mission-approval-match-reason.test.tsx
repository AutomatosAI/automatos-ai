/**
 * PRD-164 S2 — the approval card renders the agent-match preview.
 *
 * The backend persists per-task match reasons (plan snapshot →
 * `mission_approval` payload). This pins the frontend half: a task carrying
 * `match_agent`/`match_reason` shows them on the card, and a task without a
 * preview renders no reason line.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@/hooks/use-missions-api', () => ({
  useApproveMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useRejectMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useUpdateMissionPlan: () => ({ isLoading: false, mutateAsync: vi.fn() }),
}))

import { MissionApprovalWidget } from '../MissionApprovalWidget'
import type { MissionApprovalWidgetData, WidgetMetadata } from '../types'

const metadata: WidgetMetadata = {
  source: { type: 'tool', name: 'platform_create_mission' },
  createdAt: new Date(),
}

function data(overrides?: Partial<MissionApprovalWidgetData>): MissionApprovalWidgetData {
  return {
    mission_id: 'm-1',
    goal: 'Research competitor pricing',
    task_count: 2,
    tasks: [
      {
        title: 'Research pricing',
        agent_role: 'research',
        sequence: 1,
        match_agent: 'SCOUT',
        match_reason: "Strong role match for 'research'; capability profile closely matches the task (similarity 0.84)",
      },
      { title: 'Write summary', agent_role: 'writer', sequence: 2 },
    ],
    ...overrides,
  }
}

describe('MissionApprovalWidget — match preview (PRD-164 S2)', () => {
  it('shows the matched agent and reason for an annotated task', () => {
    render(
      <MissionApprovalWidget id="w1" title="Mission plan" data={data()} metadata={metadata} />,
    )
    expect(screen.getByText(/→ SCOUT/)).toBeInTheDocument()
    expect(screen.getByText(/Strong role match for 'research'/)).toBeInTheDocument()
    // The un-annotated task renders no reason line.
    expect(screen.queryByText(/→ SCRIBE/)).not.toBeInTheDocument()
  })

  it('labels an explicit override as the user pick', () => {
    const d = data()
    const overridden = {
      ...d.tasks[0],
      match_agent: 'VECTOR',
      match_is_override: true,
      match_reason: "Explicitly assigned: the plan names agent 'VECTOR' for this task",
    }
    render(
      <MissionApprovalWidget
        id="w2"
        title="Mission plan"
        data={{ ...d, tasks: [overridden, d.tasks[1]] }}
        metadata={metadata}
      />,
    )
    expect(screen.getByText(/→ VECTOR \(your pick\)/)).toBeInTheDocument()
  })
})
