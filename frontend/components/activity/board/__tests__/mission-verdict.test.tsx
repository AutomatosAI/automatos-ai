/**
 * PRD-252 D6 — a mission's card in Review is decided with the mission's own
 * buttons: Approve starts it, Reject (with a reason) cancels it. A step is its
 * mission's to check.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'
import type { BoardTask } from '@/types/board'

const mission = vi.hoisted(() => ({
  approve: { mutate: vi.fn(), isLoading: false },
  reject: { mutate: vi.fn(), isLoading: false },
}))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))

vi.mock('@/hooks/use-missions-api', () => ({ useApproveMission: () => mission.approve, useRejectMission: () => mission.reject }))
vi.mock('next/link', () => ({ default: ({ href, children, ...rest }: any) => <a href={String(href)} {...rest}>{children}</a> }))
vi.mock('sonner', () => ({ toast }))

import { MissionVerdict } from '../mission-verdict'

function card(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '51', type: 'mission', name: 'Order the green coffee', status: 'review', priority: 'medium', tags: [],
    review_mode: 'human', source_id: '51', mission_id: 'run-9', review_reason: 'mission_plan', ...over }
}

function show(task: BoardTask, onDecided = vi.fn()) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <MissionVerdict task={task} onDecided={onDecided} />
    </QueryClientProvider>,
  )
  return onDecided
}

beforeEach(() => {
  mission.approve.mutate.mockReset()
  mission.reject.mutate.mockReset()
  toast.success.mockReset()
})
afterEach(cleanup)

describe('MissionVerdict', () => {
  it("approves the plan through the mission's endpoint, not the ticket's", () => {
    mission.approve.mutate.mockImplementation((_vars, opts) => opts.onSuccess())
    const onDecided = show(card())
    fireEvent.click(screen.getByRole('button', { name: /Approve the plan/ }))
    expect(mission.approve.mutate).toHaveBeenCalledWith({ id: 'run-9', body: {} }, expect.anything())
    expect(toast.success).toHaveBeenCalledWith('Plan approved. The mission is starting.')
    expect(onDecided).toHaveBeenCalled()
  })

  it("sends the owner's note with the approval, for every step (F291)", () => {
    show(card())
    fireEvent.change(screen.getByRole('textbox', { name: /A note for every step/ }),
      { target: { value: ' Use our real Thursday delivery day. ' } })
    fireEvent.click(screen.getByRole('button', { name: /Approve the plan/ }))
    expect(mission.approve.mutate).toHaveBeenCalledWith(
      { id: 'run-9', body: { note: 'Use our real Thursday delivery day.' } }, expect.anything())
  })

  it('rejects the plan only with a reason, and says the mission is cancelled', () => {
    show(card())
    fireEvent.click(screen.getByRole('button', { name: /Reject the plan/ }))
    const confirm = screen.getByRole('button', { name: /Reject and cancel the mission/ })
    expect(confirm).toBeDisabled()
    fireEvent.change(screen.getByRole('textbox', { name: /Why reject the plan\?/ }), { target: { value: ' Wrong supplier. ' } })
    fireEvent.click(confirm)
    expect(mission.reject.mutate).toHaveBeenCalledWith({ id: 'run-9', body: { reason: 'Wrong supplier.' } }, expect.anything())
  })

  it("leaves a step to its mission: only the way to it", () => {
    show(card({ review_reason: 'awaiting_review' }))
    expect(screen.getByTestId('review-on-mission')).toHaveAttribute('href', '/missions/run-9')
    expect(screen.queryByRole('button')).toBeNull()
  })
})
