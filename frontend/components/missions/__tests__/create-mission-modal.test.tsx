/**
 * F282 (night 8) — the New mission form's "Check each step with me" switch.
 * Off by default (nothing sent); on sends config.check_each_step so the
 * mission starts with the setting under its one name, not one of Auto's
 * eight spellings (modules/coordination/owner_checks.py).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

const push = vi.hoisted(() => vi.fn())
const mutate = vi.hoisted(() => vi.fn())
const requestMock = vi.hoisted(() => vi.fn().mockResolvedValue({}))

vi.mock('next/navigation', () => ({ useRouter: () => ({ push }) }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({
  apiClient: { request: requestMock, uploadAttachment: vi.fn() },
}))
vi.mock('@/hooks/use-missions-api', () => ({
  useCreateMission: () => ({ mutate, isLoading: false }),
}))

import { CreateMissionModal } from '../create-mission-modal'

beforeEach(() => {
  mutate.mockClear()
})

function submitWithName() {
  fireEvent.change(screen.getByLabelText('Mission Name'), { target: { value: 'Ship the newsletter' } })
  fireEvent.click(screen.getByRole('button', { name: /create mission/i }))
}

describe('CreateMissionModal — check each step (F282)', () => {
  it('is off by default, so nothing is sent for it', () => {
    render(<CreateMissionModal open onOpenChange={() => {}} />)

    expect(screen.getByRole('switch', { name: /check each step/i })).toHaveAttribute('aria-checked', 'false')

    submitWithName()

    expect(mutate).toHaveBeenCalledTimes(1)
    const [body] = mutate.mock.calls[0]
    expect((body.config as Record<string, unknown>).check_each_step).toBeUndefined()
  })

  it('sends check_each_step true once switched on', () => {
    render(<CreateMissionModal open onOpenChange={() => {}} />)

    fireEvent.click(screen.getByRole('switch', { name: /check each step/i }))
    submitWithName()

    expect(mutate).toHaveBeenCalledTimes(1)
    const [body] = mutate.mock.calls[0]
    expect((body.config as Record<string, unknown>).check_each_step).toBe(true)
  })
})
