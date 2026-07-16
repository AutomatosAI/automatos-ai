/**
 * PRD-204 S11 -- the Watchlist panel.
 *
 * Empty state is honest ("Nothing being watched"); a populated list renders
 * title/type/status/last-check/score columns off the watch registry shape
 * (score is the backend's x10 display string); rows link to the watched
 * mission / playbook execution; cancel goes through the house confirm
 * primitive (PRD-169 -- no window.confirm) and calls the cancel mutation;
 * closed watches offer no cancel.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import type { WatchRow } from '@/lib/api-client'

const { cancelMutate } = vi.hoisted(() => ({
  cancelMutate: vi.fn().mockResolvedValue({}),
}))

vi.mock('@/hooks/use-watches-api', () => ({
  useWatches: vi.fn(),
  useWatchDetail: vi.fn(() => ({ data: undefined, isLoading: false })),
  useCancelWatch: () => ({ isLoading: false, mutateAsync: cancelMutate }),
}))

vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: any) => (
    <a href={typeof href === 'string' ? href : String(href)} {...rest}>
      {children}
    </a>
  ),
}))

vi.mock('sonner', () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

import { WatchlistTab } from '../watchlist-tab'
import { useWatches } from '@/hooks/use-watches-api'

const mockUseWatches = vi.mocked(useWatches)

function watch(overrides?: Partial<WatchRow>): WatchRow {
  return {
    id: 'w-1',
    title: 'Watch: Draft the Q4 memo',
    watch_type: 'mission',
    target_type: 'mission',
    target_id: 'run-42',
    status: 'watching',
    policy: 'run_and_report',
    success_criteria: 'The memo ships.',
    quality_threshold: 0.8,
    final_score: null,
    final_score_display: 'unscored',
    final_verdict: null,
    actions_taken: 0,
    action_budget: 2,
    last_checked_at: new Date(Date.now() - 5 * 60_000).toISOString(),
    next_check_at: null,
    deadline_at: null,
    created_at: new Date().toISOString(),
    closed_at: null,
    ...overrides,
  }
}

function setWatches(watches: WatchRow[]) {
  mockUseWatches.mockReturnValue({
    data: { watches, total: watches.length },
    isLoading: false,
    isError: false,
  } as any)
}

describe('WatchlistTab -- PRD-204 S11', () => {
  beforeEach(() => {
    cancelMutate.mockClear()
  })

  it('shows the simple empty state when nothing is being watched', () => {
    setWatches([])
    render(<WatchlistTab />)
    expect(screen.getByText(/Nothing being watched/)).toBeInTheDocument()
  })

  it('renders a live watch row with type chip, status chip, relative check, and score', () => {
    setWatches([watch()])
    render(<WatchlistTab />)
    expect(screen.getByText('Watch: Draft the Q4 memo')).toBeInTheDocument()
    expect(screen.getByText('mission')).toBeInTheDocument()
    expect(screen.getByText('watching')).toBeInTheDocument()
    expect(screen.getByText(/minutes ago/)).toBeInTheDocument()
    expect(screen.getByText('unscored')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Cancel/ })).toBeInTheDocument()
  })

  it('shows the x10 score display for a scored watch and no cancel when closed', () => {
    setWatches([
      watch({
        id: 'w-2',
        status: 'passed',
        final_score: 0.83,
        final_score_display: '8.3/10',
        final_verdict: 'Delivered what was asked.',
        closed_at: new Date().toISOString(),
      }),
    ])
    render(<WatchlistTab />)
    expect(screen.getByText('8.3/10')).toBeInTheDocument()
    expect(screen.getByText('Delivered what was asked.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Cancel/ })).not.toBeInTheDocument()
  })

  it('links the row to the watched mission / playbook execution', () => {
    setWatches([
      watch(),
      watch({
        id: 'w-3',
        title: 'Watch: Weekly digest',
        target_type: 'playbook_execution',
        target_id: 'exec-9',
      }),
    ])
    render(<WatchlistTab />)
    expect(
      screen.getByRole('link', { name: 'Watch: Draft the Q4 memo' }),
    ).toHaveAttribute('href', '/assignments?tab=missions&mission=run-42')
    expect(
      screen.getByRole('link', { name: 'Watch: Weekly digest' }),
    ).toHaveAttribute('href', '/assignments?tab=playbooks&execution=exec-9')
  })

  it('cancel rides the house confirm primitive and calls the mutation', async () => {
    setWatches([watch({ id: 'w-7' })])
    render(<WatchlistTab />)

    // No mutation before the confirm dialog approves it.
    fireEvent.click(screen.getByRole('button', { name: /Cancel/ }))
    expect(cancelMutate).not.toHaveBeenCalled()
    expect(screen.getByText(/Stop watching this work/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /^Stop watching$/ }))
    await waitFor(() => expect(cancelMutate).toHaveBeenCalledWith('w-7'))
  })
})
