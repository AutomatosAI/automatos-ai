/**
 * PRD-225 — the Questions tab is wired into the Command Center shell with a
 * live count badge (open question count) and renders when selected.
 *
 * The shell's data hooks and child tabs are stubbed; the focus is the tab strip
 * badge = open-question count and the ?tab=questions render path.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

const questionsData = vi.hoisted(() => ({ current: { grants: [] as unknown[] } }))
const activeTab = vi.hoisted(() => ({ current: 'questions' }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), refresh: vi.fn() }),
  usePathname: () => '/command-center',
  useSearchParams: () => new URLSearchParams(`tab=${activeTab.current}`),
}))

vi.mock('@/hooks/use-activity-api', () => ({
  useActivityStats: () => ({ data: { working_now: 0, needs_attention: 0 } }),
  useActivityFeed: () => ({ data: { total: 0, items: [] } }),
  useActivitySchedule: () => ({ data: { scheduled: [] } }),
}))
vi.mock('@/hooks/use-board-tasks', () => ({ useBoardTasks: () => ({ columns: [] }) }))
vi.mock('@/hooks/use-board-event-stream', () => ({ useBoardEventStream: () => undefined }))
vi.mock('@/hooks/use-kpi-api', () => ({ useDecisionsNeeded: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-watches-api', () => ({ useWatches: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-approval-grants', () => ({
  useQuestions: () => ({ data: questionsData.current }),
}))

// Stub the child tabs/strips so the shell renders without their dependency trees.
vi.mock('../stats-strip', () => ({ StatsStrip: () => <div /> }))
vi.mock('../is-it-working-strip', () => ({ IsItWorkingStrip: () => <div /> }))
vi.mock('../summary-tab', () => ({ SummaryTab: () => <div>summary</div> }))
vi.mock('../board-tab', () => ({ BoardTab: () => <div>board</div> }))
vi.mock('../calendar-tab', () => ({ CalendarTab: () => <div>calendar</div> }))
vi.mock('../activity-tab', () => ({ ActivityTab: () => <div>activity</div> }))
vi.mock('../watchlist-tab', () => ({ WatchlistTab: () => <div>watchlist</div> }))
vi.mock('../governance-tab', () => ({ GovernanceTab: () => <div>governance</div> }))
vi.mock('../questions-tab', () => ({ QuestionsTab: () => <div>questions-body</div> }))
// PRD-222's onboarding widgets landed in the shell after this test was written
// and both call useWorkspace, so rendering the shell without a WorkspaceProvider
// throws. They are irrelevant to the Questions-tab wiring under test — stub them
// like every other shell child.
vi.mock('@/components/onboarding/trial-balance-pill', () => ({ TrialBalancePill: () => <div /> }))
vi.mock('@/components/onboarding/setup-checklist-card', () => ({ SetupChecklistCard: () => <div /> }))

import { CommandCenterShell } from '../command-center-shell'

describe('CommandCenterShell — PRD-225 Questions tab', () => {
  it('badges the Questions tab with the open-question count', () => {
    questionsData.current = { grants: [{ id: 1 }, { id: 2 }, { id: 3 }] }
    render(<CommandCenterShell />)
    const tab = screen.getByRole('button', { name: /Questions/ })
    expect(tab.textContent).toContain('3')
  })

  it('renders the Questions tab body when ?tab=questions', () => {
    questionsData.current = { grants: [] }
    activeTab.current = 'questions'
    render(<CommandCenterShell />)
    expect(screen.getByText('questions-body')).toBeInTheDocument()
    // No badge when the count is zero.
    const tab = screen.getByRole('button', { name: /Questions/ })
    expect(tab.textContent).not.toMatch(/\d/)
  })
})
