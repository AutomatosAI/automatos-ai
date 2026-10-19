/**
 * PRD-263 US-022 — the Self-Learning tile shows the fields the backend sends,
 * and "—" where there is no value. It read `harness.iteration` (the backend sends
 * `iteration_count`), counted `data.prescriptions` (gone with PRD-187) and showed
 * the recorder's process-global signal counter, so it showed fake zeros.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import type { SelfLearningHealthData } from '@/hooks/use-learning-api'

const health = vi.hoisted(() => ({
  data: undefined as SelfLearningHealthData | undefined,
  isLoading: false,
  isError: false,
}))
vi.mock('@/hooks/use-learning-api', () => ({ useSelfLearningHealth: () => health }))

import { SelfLearningHealthWidget } from '@/components/activity/widgets/self-learning-health-widget'

function payload(overrides: Partial<SelfLearningHealthData> = {}): SelfLearningHealthData {
  return {
    workspace_id: 'ws-1',
    status: 'completed',
    iteration_count: 3,
    coverage: 0.5,
    insufficient_data: 2,
    watch: 1,
    regression: 0,
    mode: 'report_only',
    signals_7d: 12,
    unavailable: [],
    ...overrides,
  }
}

afterEach(() => {
  cleanup()
  health.data = undefined
})

describe('SelfLearningHealthWidget', () => {
  it('renders an absent iteration as "—", never 0', () => {
    health.data = payload({ status: 'scheduled_not_run_yet', iteration_count: null, coverage: null })
    render(<SelfLearningHealthWidget />)
    expect(screen.getByText('Iteration —')).toBeInTheDocument()
    expect(screen.queryByText('Iteration 0')).not.toBeInTheDocument()
  })

  it('renders scheduled_not_run_yet as words, without underscores', () => {
    health.data = payload({ status: 'scheduled_not_run_yet', iteration_count: 0 })
    render(<SelfLearningHealthWidget />)
    expect(screen.getByText('scheduled not run yet')).toBeInTheDocument()
    expect(screen.queryByText(/scheduled_not_run_yet/)).not.toBeInTheDocument()
  })

  it('renders coverage as a percentage', () => {
    health.data = payload({ coverage: 0.6667 })
    render(<SelfLearningHealthWidget />)
    expect(screen.getByText('67%')).toBeInTheDocument()
    expect(screen.getByText('Coverage')).toBeInTheDocument()
  })

  it('shows the verdict counts, the 7-day signals and the mode the backend sends', () => {
    health.data = payload()
    render(<SelfLearningHealthWidget />)
    expect(screen.getByText('12')).toBeInTheDocument()
    expect(screen.getByText('Signals (7 days)')).toBeInTheDocument()
    expect(screen.getByText('Agents with not enough data: 2')).toBeInTheDocument()
    expect(screen.getByText('· Mode Report only')).toBeInTheDocument()
    expect(screen.queryByText('Applied')).not.toBeInTheDocument()
    expect(screen.queryByText('Queued')).not.toBeInTheDocument()
  })

  it('shows the mode as the owner reads it, Propose included', () => {
    health.data = payload({ mode: 'propose' })
    render(<SelfLearningHealthWidget />)
    expect(screen.getByText('· Mode Propose')).toBeInTheDocument()
  })

  it('a failed run reads in the destructive tone', () => {
    health.data = payload({ status: 'failed' })
    render(<SelfLearningHealthWidget />)
    expect(screen.getByText('failed')).toHaveClass('text-destructive')
  })

  it('a section the backend could not read shows "—" and says it is unavailable', () => {
    health.data = payload({
      status: null, iteration_count: null, coverage: null, insufficient_data: null,
      watch: null, regression: null, mode: null, signals_7d: null, unavailable: ['harness', 'signals'],
    })
    render(<SelfLearningHealthWidget />)
    expect(screen.getByText('degraded')).toBeInTheDocument()
    expect(screen.getByText('HARNESS section unavailable')).toBeInTheDocument()
    expect(screen.getAllByText('—')).toHaveLength(4)
  })
})
