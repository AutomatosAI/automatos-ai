/**
 * PRD-246 US-004 — the Assignments hub's compact form, the parts that are
 * behaviour rather than CSS.
 *
 * The 1-up grids, the stacked card meta, the wrapping status head and the
 * grouped row's two-line form are CSS in the compact region and are asserted
 * in components/__tests__/studio-mobile-scope.test.ts. Here: the hub's tab
 * strip uses the SAME scroll-into-view hook as the Command Centre, and the
 * two ledger views (a 7-column row, a 7-column table) are desktop-only.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const width = vi.hoisted(() => ({ phone: false, compact: false }))
const missions = vi.hoisted(() => ({
  current: [
    {
      id: 'm-1111-2222',
      goal: 'Close the quarterly books',
      state: 'running',
      state_type: 'running',
      tokens_used: 1200,
      max_concurrent: 2,
      complexity_tier: 'standard',
      parallel_groups: [{}],
      tasks: [],
      created_by: 'gerard',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
  ] as unknown[],
}))

vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => width.phone,
  useIsTabletOrBelow: () => width.compact,
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => '/assignments',
  useSearchParams: () => new URLSearchParams('tab=missions'),
}))
vi.mock('@/lib/auth-hooks', () => ({ useUser: () => ({ user: { id: 'u-1' } }) }))
vi.mock('@/hooks/use-missions-api', () => ({
  useMissions: () => ({ data: { missions: missions.current }, isLoading: false }),
}))
vi.mock('@/hooks/use-playbook-api', () => ({
  useWorkflowPlaybooks: () => ({
    data: {
      items: [
        {
          id: 7,
          name: 'Weekly digest',
          description: 'Summarise the week.',
          trigger_type: 'cron',
          schedule_config: { cron: '0 9 * * 1' },
          use_count: 4,
          success_rate: 0.9,
          steps: [{}, {}],
        },
      ],
      total: 1,
    },
    isLoading: false,
  }),
  useExecutePlaybook: () => ({ mutate: vi.fn(), isPending: false }),
}))
// The two bodies are NOT stubbed — they are under test here, and `vi.mock`
// would replace them for the hub and for this file alike. The hub mounts the
// real MissionsBody for `?tab=missions`, on the mocked hooks above.
import { StudioAssignmentsHub } from '../studio/assignments-hub'
import { MissionsBody } from '../studio/missions-body'
import { PlaybooksBody } from '../studio/playbooks-body'

const intoView = vi.fn()
beforeEach(() => {
  width.phone = false
  width.compact = false
  intoView.mockClear()
  Element.prototype.scrollIntoView = intoView
})
afterEach(cleanup)

describe('the hub’s flip tabs behave like the Command Centre’s', () => {
  it('scrolls the active tab into view on a compact viewport', () => {
    width.compact = true
    const { container } = render(<StudioAssignmentsHub />)
    expect(container.querySelector('.cc-tab.active')).toHaveTextContent('Missions')
    expect(intoView).toHaveBeenCalledWith({ block: 'nearest', inline: 'center' })
  })

  it('leaves a desktop strip alone', () => {
    render(<StudioAssignmentsHub />)
    expect(intoView).not.toHaveBeenCalled()
  })
})

describe('the card grids are classes, so the compact region can answer them', () => {
  it('missions: the cards view is .mis-cards, not an inline grid', () => {
    const { container } = render(<MissionsBody />)
    fireEvent.click(screen.getByRole('button', { name: /Cards/ }))
    expect(container.querySelector('.mis-cards')).not.toBeNull()
    expect(container.querySelector('[style*="grid-template-columns"]')).toBeNull()
  })

  it('playbooks: the grid view is .pb-grid', () => {
    const { container } = render(<PlaybooksBody />)
    expect(container.querySelector('.pb-grid')).not.toBeNull()
  })
})

describe('the ledger views are desktop-only', () => {
  it('missions: a phone is not offered Table, and lands on Grouped', () => {
    width.phone = true
    const { container } = render(<MissionsBody />)
    expect(screen.queryByRole('button', { name: /Table/ })).toBeNull()
    expect(screen.getByRole('button', { name: /Grouped/ })).toHaveClass('on')
    expect(container.querySelector('.mis-row')).not.toBeNull()
  })

  it('missions: a desktop keeps all three views', () => {
    render(<MissionsBody />)
    for (const v of [/Grouped/, /Cards/, /Table/]) {
      expect(screen.getByRole('button', { name: v })).toBeInTheDocument()
    }
  })

  it('playbooks: a phone is not offered List, and keeps the card grid', () => {
    width.phone = true
    const { container } = render(<PlaybooksBody />)
    expect(screen.queryByRole('button', { name: /List/ })).toBeNull()
    expect(screen.getByRole('button', { name: /Grid/ })).toHaveClass('on')
    expect(container.querySelector('table')).toBeNull()
  })
})
