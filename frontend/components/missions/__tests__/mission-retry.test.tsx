/**
 * F247 — a failed mission is retried from its page.
 *
 * #0176 failed while the AI credit was out. Resume answered only for a paused
 * mission and both re-plans were spent, so nothing on the page could run it
 * again. The budget bar now offers Retry for a failed mission, through the same
 * Resume: its failed steps run again, with the same plan.
 */
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'

import { MissionBudgetBar } from '../mission-budget-bar'

const HALT = 'Joiner halt: no forward progress across 3 ledger checks.'

describe('MissionBudgetBar (F247)', () => {
  it('offers Retry for a failed mission, and says what it does', () => {
    const onResume = vi.fn()
    render(<MissionBudgetBar tokensUsed={20_000} tokenBudgetEstimate={175_000} missionState="failed"
                             stopDetail={HALT} onResume={onResume} />)

    expect(screen.getByText('Joiner halt: no forward progress across 3 ledger checks. '
      + 'Retry runs its failed steps again, with the same plan.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Retry/ }))
    expect(onResume).toHaveBeenCalledTimes(1)
  })

  it('keeps Resume for a paused mission', () => {
    render(<MissionBudgetBar tokensUsed={20_000} tokenBudgetEstimate={175_000} missionState="paused"
                             onResume={vi.fn()} />)
    expect(screen.getByRole('button', { name: /Resume/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Retry/ })).not.toBeInTheDocument()
  })

  it('offers neither to a running mission under half its budget', () => {
    render(<MissionBudgetBar tokensUsed={20_000} tokenBudgetEstimate={175_000} missionState="running"
                             onResume={vi.fn()} />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})
