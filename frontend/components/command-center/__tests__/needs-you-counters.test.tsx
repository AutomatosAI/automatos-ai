/**
 * PRD-252 R5 — one "Needs you" number. The Board tab badge, the lede's "need
 * your eyes" and the stats strip's ATTENTION all show useNeedsYou's total for
 * the shell's period, and the Summary tab carries no second count. (Auto's pill:
 * auto-now-rail.test.tsx. The widget's rows: needs-you-widget.test.tsx.)
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'

const needs = vi.hoisted(() => ({ periods: [] as string[] }))

vi.mock('@/hooks/use-needs-you', () => ({
  useNeedsYou: (period: string) => {
    needs.periods.push(period)
    return { data: { period, total: 5, counts: { review: 2, question: 1, approval: 1, failed: 1 } } }
  },
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), refresh: vi.fn() }),
  usePathname: () => '/command-center',
  useSearchParams: () => new URLSearchParams('tab=summary'),
}))
// The stats' own needs_attention is not what any counter shows any more.
vi.mock('@/hooks/use-activity-api', () => ({
  useActivityStats: () => ({ data: { working_now: 1, agents_active: 1, tasks_in_queue: 0, needs_attention: 99 } }),
  useActivityFeed: () => ({ data: { total: 0, items: [] } }),
  useActivitySchedule: () => ({ data: { scheduled: [] } }),
}))
vi.mock('@/hooks/use-board-event-stream', () => ({ useBoardEventStream: () => undefined }))
vi.mock('@/hooks/use-watches-api', () => ({ useWatches: () => ({ data: { total: 0 } }) }))
vi.mock('@/hooks/use-approval-grants', () => ({ useQuestions: () => ({ data: { grants: [] } }) }))
vi.mock('@/hooks/use-heartbeats-api', () => ({ useHeartbeats: () => ({ data: { heartbeats: [] } }) }))
vi.mock('@/hooks/use-unified-analytics', () => ({ useCostAnalyticsUnified: () => ({ data: undefined }) }))
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

describe('the one Needs-you number in the Command Centre', () => {
  it('is the Board badge, the lede and ATTENTION, for the same period', () => {
    render(<CommandCenterShell />)

    expect(screen.getByRole('button', { name: /Board/ }).textContent).toContain('5')
    expect(screen.getByRole('button', { name: /Summary/ }).textContent).not.toMatch(/\d/)
    expect(screen.getByText(/need your eyes/).textContent).toContain('5')
    const attention = screen.getByText('ATTENTION').closest('.cell') as HTMLElement
    expect(within(attention).getByText('5')).toBeInTheDocument()
    expect(new Set(needs.periods)).toEqual(new Set(['1d']))
  })
})
