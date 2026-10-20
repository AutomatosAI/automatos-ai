/**
 * The Mission copy names no fixed team size (facts sheet, 19 Oct 2026).
 *
 * The Studio entry tile and the Mission card said "6–9 agents". The mission
 * planner sets no agent count: a plan has 1 to 20 tasks (MIN_TASKS / MAX_TASKS
 * in orchestrator/modules/coordination/planner.py), each task goes to one
 * best-fit agent, and several tasks may share one. Both now say "a team of
 * agents".
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@/hooks/use-missions-api', () => ({ useMissions: () => ({ data: { missions: [] } }) }))

import { MissionConstellationCard } from '@/components/assignments/mission-card-constellation'
import { ENTRY_TYPES } from '@/components/assignments/studio/entry-grid'

/** "6–9 agents", "6-9 agents", "3 – 5 agents": a fixed range of agents. */
const FIXED_TEAM_SIZE = /\d+\s*[–-]\s*\d+\s+agents/

describe('Mission copy', () => {
  it('the Studio entry tile says a team of agents, not a fixed count', () => {
    const mission = ENTRY_TYPES.find((entry) => entry.id === 'mission')

    expect(mission?.desc).toContain('A team of agents')
    expect(mission?.desc).not.toMatch(FIXED_TEAM_SIZE)
  })

  it('the Mission card says a team of agents, not a fixed count', () => {
    render(<MissionConstellationCard onStart={() => {}} />)

    expect(screen.getByText(/A team of agents/)).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(FIXED_TEAM_SIZE)
  })
})
