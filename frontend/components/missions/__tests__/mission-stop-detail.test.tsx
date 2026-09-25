/**
 * F153 — the mission page says why a mission paused.
 *
 * A budget pause records its reason and numbers on the run (stop_detail) and
 * on its run_paused event. The budget bar said "Mission paused — budget
 * exceeded" only when tokens passed half the estimate (a budget is spend in
 * dollars now, and a Claude Code session's tokens cost nothing), and the
 * activity feed said only "Mission paused". Both now show the detail.
 */
import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'

import { MissionBudgetBar } from '../mission-budget-bar'
import { MissionActivityFeed } from '../mission-activity-feed'

beforeAll(() => {
  // jsdom has no ResizeObserver; the feed's Radix scroll area observes its viewport.
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
})

const DETAIL = "Paused: spent $3.10 of the $2.50 budget (the plan's 175,000-token estimate); " +
  '730,153 tokens ran in Claude Code sessions at no cost — raise the budget or resume'

describe('MissionBudgetBar (F153)', () => {
  it('says why a paused mission paused, whatever its token percentage', () => {
    render(<MissionBudgetBar tokensUsed={20_000} tokenBudgetEstimate={175_000} missionState="paused"
                             stopDetail={DETAIL} onResume={vi.fn()} />)
    expect(screen.getByText(DETAIL)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Resume/ })).toBeInTheDocument()
  })

  it('does not call a pause a budget pause when it does not say so', () => {
    render(<MissionBudgetBar tokensUsed={120_000} tokenBudgetEstimate={175_000} missionState="paused" />)
    expect(screen.getByText('Mission paused')).toBeInTheDocument()
    expect(screen.queryByText(/budget exceeded/)).not.toBeInTheDocument()
  })
})

describe('MissionActivityFeed (F153)', () => {
  it('shows why the run paused under the pause', () => {
    render(<MissionActivityFeed events={[{
      id: 'e-1', event_type: 'run_paused', actor_type: 'coordinator', actor_id: 'dispatcher',
      old_state: 'running', new_state: 'paused', task_id: null, stop_detail: DETAIL,
      created_at: '2026-09-25T12:00:00Z',
    }]} />)
    expect(screen.getByText('Mission paused')).toBeInTheDocument()
    expect(screen.getByText(DETAIL)).toBeInTheDocument()
  })
})
