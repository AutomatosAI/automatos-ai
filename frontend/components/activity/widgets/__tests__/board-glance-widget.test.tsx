/** PRD-244 review batch 2 — "Board at a glance": four facets of one read. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import type { ReactNode } from 'react'

const stats = vi.hoisted(() => ({ data: undefined as any, isLoading: false }))
vi.mock('@/hooks/use-activity-api', () => ({ useBoardStats: () => stats }))
vi.mock('recharts', () => ({ ResponsiveContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>, PieChart: ({ children }: { children: ReactNode }) => <div>{children}</div>, Pie: () => null, Cell: () => null, Tooltip: () => null }))
vi.mock('@/components/shared', () => ({ PremiumIcon: () => null }))

import { BoardGlanceWidget } from '@/components/activity/widgets/board-glance-widget'

afterEach(() => { cleanup(); stats.data = undefined })

describe('BoardGlanceWidget', () => {
  it('with no tasks, says so', () => {
    stats.data = { columns: [], total_tasks: 0, priorities: [], types: [], workload: [] }
    render(<BoardGlanceWidget period="1d" />)
    expect(screen.getByText('No tasks yet')).toBeInTheDocument()
  })
  it('renders the four facets from one read', () => {
    stats.data = {
      total_tasks: 6,
      columns: [{ status: 'review', count: 6 }, { status: 'done', count: 0 }],
      priorities: [{ priority: 'high', count: 4 }, { priority: 'low', count: 2 }],
      types: [{ type: 'recipe', count: 6, percentage: 100 }],
      workload: [{ agent_id: 1, agent_name: 'DRG-DEV', agent_icon: null, task_count: 6 }],
    }
    render(<BoardGlanceWidget period="1d" />)
    for (const f of ['Status', 'Priority', 'Type of work', 'Workload']) expect(screen.getByLabelText(f)).toBeInTheDocument()
    expect(screen.getByText('6 tasks')).toBeInTheDocument()
    expect(screen.getByText('In Review')).toBeInTheDocument()
    expect(screen.getByText('High')).toBeInTheDocument()
    expect(screen.getByText('Task')).toBeInTheDocument()
    expect(screen.getByText('DRG-DEV')).toBeInTheDocument()
  })
})
