/** PRD-244 review batch 2 — "Activity": running first, then recent; rows open the thing itself. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const push = vi.hoisted(() => vi.fn())
vi.mock('next/navigation', () => ({ useRouter: () => ({ push }) }))
vi.mock('@/hooks/use-activity-api', () => ({
  useActivityFeed: (f: { status?: string }) => ({
    isLoading: false,
    data: {
      items:
        f.status === 'working'
          ? [{ id: 'recipe-1', type: 'recipe', name: 'Weekly digest', status: 'running', started_at: new Date().toISOString(), duration_seconds: null, step_progress: { current: 2, total: 4 }, source_id: 'r1' }]
          : f.status === 'done'
            ? [{ id: 'task-2', type: 'task', name: 'Draft the vendor email', status: 'completed', started_at: new Date().toISOString(), duration_seconds: 95, source_id: '121' }]
            : [],
    },
  }),
}))
vi.mock('@/components/command-center/activity-tab', () => ({ rowHref: (i: { source_id: string }) => `/x/${i.source_id}` }))

import { ActivityWidget } from '@/components/activity/widgets/activity-widget'

afterEach(() => { cleanup(); push.mockClear() })

describe('ActivityWidget', () => {
  it('shows what is running with its step, then what finished, and the running badge', () => {
    render(<ActivityWidget period="1d" />)
    expect(screen.getByText('1 running')).toBeInTheDocument()
    expect(screen.getByText('Weekly digest')).toBeInTheDocument()
    expect(screen.getByText('Step 2/4')).toBeInTheDocument()
    expect(screen.getByText('Draft the vendor email')).toBeInTheDocument()
    expect(screen.getByText('1m 35s')).toBeInTheDocument()
  })

  it('opens a row through the Activity tab link rules', () => {
    render(<ActivityWidget period="1d" />)
    fireEvent.click(screen.getByRole('button', { name: 'Open Draft the vendor email' }))
    expect(push).toHaveBeenCalledWith('/x/121')
  })
})
