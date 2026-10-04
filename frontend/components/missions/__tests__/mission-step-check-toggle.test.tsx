/**
 * F282 (night 8) — the mission page's switch mirrors the mission's own
 * check_each_step (whatever spelling Auto wrote it under, normalised server
 * side), and is disabled once the mission is done.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

const useMission = vi.hoisted(() => vi.fn())
const mutate = vi.hoisted(() => vi.fn())
const toastError = vi.hoisted(() => vi.fn())

vi.mock('@/hooks/use-missions-api', () => ({
  useMission: (...args: unknown[]) => useMission(...args),
  useUpdateMissionSettings: () => ({ mutate, isLoading: false }),
}))
vi.mock('sonner', () => ({ toast: { error: toastError, success: vi.fn() } }))

import { MissionStepCheckToggle } from '../mission-step-check-toggle'

function mission(overrides: Partial<{ check_each_step: boolean; state: string }> = {}) {
  return { check_each_step: false, state: 'running', ...overrides }
}

beforeEach(() => {
  mutate.mockClear()
  toastError.mockClear()
})

describe('MissionStepCheckToggle', () => {
  it('renders nothing while the mission is still loading', () => {
    useMission.mockReturnValue({ data: undefined, isLoading: true })
    render(<MissionStepCheckToggle missionId="0324" />)
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
  })

  it('is off when the mission does not check each step, and turns it on', () => {
    useMission.mockReturnValue({ data: mission({ check_each_step: false }), isLoading: false })
    render(<MissionStepCheckToggle missionId="0324" />)

    const toggle = screen.getByRole('switch', { name: /check each step/i })
    expect(toggle).toHaveAttribute('aria-checked', 'false')

    fireEvent.click(toggle)
    expect(mutate).toHaveBeenCalledWith(
      { id: '0324', body: { check_each_step: true } },
      expect.objectContaining({ onError: expect.any(Function) }),
    )
  })

  it('is on when the mission already checks each step, and turns it off', () => {
    useMission.mockReturnValue({ data: mission({ check_each_step: true }), isLoading: false })
    render(<MissionStepCheckToggle missionId="0324" />)

    const toggle = screen.getByRole('switch', { name: /check each step/i })
    expect(toggle).toHaveAttribute('aria-checked', 'true')

    fireEvent.click(toggle)
    expect(mutate).toHaveBeenCalledWith(
      { id: '0324', body: { check_each_step: false } },
      expect.objectContaining({ onError: expect.any(Function) }),
    )
  })

  it('is disabled once the mission has finished', () => {
    useMission.mockReturnValue({ data: mission({ state: 'completed' }), isLoading: false })
    render(<MissionStepCheckToggle missionId="0324" />)

    expect(screen.getByRole('switch', { name: /check each step/i })).toBeDisabled()
  })
})
