/**
 * PRD-266 P266W3-RVW-1 (D27): the schedule feed now carries a paused scheduled task
 * (`paused: true`, US-018a) so the calendar can grey it. The Calendar tab's badge counts
 * only the work that will run, so pausing a task lowers it, as it did when the feed
 * dropped paused tasks.
 *
 * The shell's data hooks and child tabs are stubbed (the command-center-shell-questions
 * pattern); the focus is the Calendar badge.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

const schedule = vi.hoisted(() => ({ items: [] as unknown[] }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), refresh: vi.fn() }),
  usePathname: () => '/command-center',
  useSearchParams: () => new URLSearchParams('tab=summary'),
}))
vi.mock('@/hooks/use-activity-api', () => ({
  useActivityStats: () => ({ data: { working_now: 0, needs_attention: 0 } }),
  useActivityFeed: () => ({ data: { total: 0, items: [] } }),
  useActivitySchedule: () => ({ data: { scheduled: schedule.items } }),
}))
vi.mock('@/hooks/use-board-event-stream', () => ({ useBoardEventStream: () => undefined }))
vi.mock('@/hooks/use-needs-you', () => ({ useNeedsYou: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-watches-api', () => ({ useWatches: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-approval-grants', () => ({ useQuestions: () => ({ data: { grants: [] } }) }))

// Stub the child tabs/strips so the shell renders without their dependency trees.
vi.mock('../stats-strip', () => ({ StatsStrip: () => <div /> }))
vi.mock('../is-it-working-strip', () => ({ IsItWorkingStrip: () => <div /> }))
vi.mock('../summary-tab', () => ({ SummaryTab: () => <div>summary</div> }))
vi.mock('../board-tab', () => ({ BoardTab: () => <div>board</div> }))
vi.mock('../calendar-tab', () => ({ CalendarTab: () => <div>calendar</div> }))
vi.mock('../activity-tab', () => ({ ActivityTab: () => <div>activity</div> }))
vi.mock('../watchlist-tab', () => ({ WatchlistTab: () => <div>watchlist</div> }))
vi.mock('../governance-tab', () => ({ GovernanceTab: () => <div>governance</div> }))
vi.mock('../questions-tab', () => ({ QuestionsTab: () => <div>questions</div> }))
vi.mock('@/components/onboarding/trial-balance-pill', () => ({ TrialBalancePill: () => <div /> }))
vi.mock('@/components/onboarding/setup-checklist-card', () => ({ SetupChecklistCard: () => <div /> }))

import { CommandCenterShell } from '../command-center-shell'

const ACTIVE_TASK = { id: 'task-1', scheduled_task_id: 1, name: 'Morning briefing', type: 'task' }
const PAUSED_TASK = { id: 'task-2', scheduled_task_id: 2, name: 'Weekly digest', type: 'task', paused: true }

afterEach(() => cleanup())

describe('the Calendar tab badge with a paused task (PRD-266 D27)', () => {
  it('counts 1 for one active and one paused task', () => {
    schedule.items = [ACTIVE_TASK, PAUSED_TASK]
    render(<CommandCenterShell />)
    expect(screen.getByRole('button', { name: /Calendar/ }).textContent).toBe('Calendar1')
  })

  it('shows no badge when the only task is paused', () => {
    schedule.items = [PAUSED_TASK]
    render(<CommandCenterShell />)
    expect(screen.getByRole('button', { name: /Calendar/ }).textContent).not.toMatch(/\d/)
  })
})
