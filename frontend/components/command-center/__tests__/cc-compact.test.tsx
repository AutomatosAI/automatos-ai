/**
 * PRD-246 US-002 — the Command Centre's compact form, the parts that are
 * behaviour rather than CSS.
 *
 * Width is mocked through `@/hooks/use-mobile`, which is the only place the
 * app reads a breakpoint. The CSS half of the story (stats 2-up, the board's
 * snap, the gutter) is asserted against the compact region in
 * components/__tests__/studio-mobile-scope.test.ts.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const width = vi.hoisted(() => ({ phone: false, compact: false }))

/** One real event, so the stream renders a view rather than its empty state. */
const FEED = vi.hoisted(() => ({
  total: 1,
  items: [
    {
      id: 'evt-1',
      type: 'routine',
      name: 'Morning digest',
      status: 'completed',
      started_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
      duration_seconds: 4,
      agent: { id: 'a1', name: 'Scout' },
      agents: [],
      summary: 'Sent the digest.',
      source_id: null,
      source_url: null,
      trigger: 'scheduled',
      error_message: null,
    },
  ],
}))

vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => width.phone,
  useIsTabletOrBelow: () => width.compact,
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), refresh: vi.fn() }),
  usePathname: () => '/command-center',
  useSearchParams: () => new URLSearchParams('tab=governance'),
}))

vi.mock('@/hooks/use-activity-api', () => ({
  activityQueryKeys: { all: ['activity'] },
  useActivityStats: () => ({ data: { working_now: 0, needs_attention: 0 } }),
  useActivityFeed: () => ({ data: FEED, isLoading: false }),
  useActivitySchedule: () => ({
    data: { scheduled: [] },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  }),
  useSchedulerHealth: () => ({ data: { healthy: null, last_fired_at: null } }),
}))
vi.mock('@/hooks/use-board-tasks', () => ({ useBoardTasks: () => ({ columns: [] }) }))
vi.mock('@/hooks/use-board-event-stream', () => ({ useBoardEventStream: () => undefined }))
vi.mock('@/hooks/use-kpi-api', () => ({ useDecisionsNeeded: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-watches-api', () => ({ useWatches: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-approval-grants', () => ({ useQuestions: () => ({ data: { grants: [] } }) }))
vi.mock('@/hooks/use-heartbeats-api', () => ({ useToggleHeartbeat: () => ({ mutate: vi.fn() }) }))
vi.mock('@/hooks/use-scheduled-tasks-api', () => ({
  useUpdateScheduledTaskStatus: () => ({ mutate: vi.fn() }),
}))

vi.mock('../stats-strip', () => ({ StatsStrip: () => <div /> }))
vi.mock('../is-it-working-strip', () => ({ IsItWorkingStrip: () => <div /> }))
vi.mock('../summary-tab', () => ({ SummaryTab: () => <div>summary</div> }))
vi.mock('../board-tab', () => ({ BoardTab: () => <div>board</div> }))
// ActivityTab and CalendarTab are NOT stubbed — they are under test here, and
// the shell only mounts the tab named by `?tab=` (governance).
vi.mock('../watchlist-tab', () => ({ WatchlistTab: () => <div>watchlist</div> }))
vi.mock('../governance-tab', () => ({ GovernanceTab: () => <div>governance</div> }))
vi.mock('../questions-tab', () => ({ QuestionsTab: () => <div>questions</div> }))
vi.mock('@/components/onboarding/trial-balance-pill', () => ({ TrialBalancePill: () => <div /> }))
vi.mock('@/components/onboarding/setup-checklist-card', () => ({ SetupChecklistCard: () => <div /> }))

import { CommandCenterShell } from '../command-center-shell'
import { ActivityTab } from '../activity-tab'
import { CalendarTab } from '../calendar-tab'

/** jsdom implements no layout, so scrollIntoView has to be provided. */
const intoView = vi.fn()
beforeEach(() => {
  width.phone = false
  width.compact = false
  intoView.mockClear()
  Element.prototype.scrollIntoView = intoView
})
afterEach(cleanup)

describe('the tab strip keeps its active tab in view', () => {
  it('scrolls the active tab into view on a compact viewport', () => {
    width.compact = true
    const { container } = render(<CommandCenterShell />)
    // Governance is the seventh of seven tabs — off-screen on a phone.
    expect(container.querySelector('.cc-tab.active')).toHaveTextContent('Governance')
    expect(intoView).toHaveBeenCalledWith({ block: 'nearest', inline: 'center' })
  })

  it('leaves the scroll alone on a desktop, where the whole strip is visible', () => {
    render(<CommandCenterShell />)
    expect(intoView).not.toHaveBeenCalled()
  })
})

describe('Activity: the table is desktop-only', () => {
  it('a phone gets the cards and is not offered the density toggle', () => {
    width.phone = true
    const { container } = render(<ActivityTab />)
    expect(container.querySelector('.cc-act-cards')).not.toBeNull()
    expect(container.querySelector('.cc-act-table')).toBeNull()
    expect(screen.queryByRole('group', { name: 'Density' })).toBeNull()
  })

  it('a desktop still chooses, and still starts on cards', () => {
    const { container } = render(<ActivityTab />)
    expect(screen.getByRole('group', { name: 'Density' })).toBeInTheDocument()
    expect(container.querySelector('.cc-act-cards')).not.toBeNull()
    expect(screen.getByRole('button', { name: /Table/ })).toBeInTheDocument()
  })
})

describe('Calendar: the week grid is desktop-only', () => {
  it('a phone is offered Day and Month, and lands on Day', () => {
    width.phone = true
    const { container } = render(<CalendarTab />)
    expect(screen.queryByRole('button', { name: 'Week' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Day' })).toHaveClass('on')
    expect(screen.getByRole('button', { name: 'Month' })).toBeInTheDocument()
    // One day column, not seven at 47px each.
    expect(container.querySelector<HTMLElement>('.cc-cal-grid')!.style.gridTemplateColumns).toBe(
      '60px 1fr',
    )
  })

  it('a desktop keeps the week', () => {
    const { container } = render(<CalendarTab />)
    expect(screen.getByRole('button', { name: 'Week' })).toHaveClass('on')
    expect(container.querySelector<HTMLElement>('.cc-cal-grid')!.style.gridTemplateColumns).toBe(
      '60px repeat(7, 1fr)',
    )
  })
})
