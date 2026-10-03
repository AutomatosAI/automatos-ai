/**
 * PRD-252 R2 — a review is Reject (with the owner's words, required) or Approve
 * (named for what it does, with an optional note); both say what happened, and
 * why when they fail.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import type { BoardTask } from '@/types/board'

const verdict = vi.hoisted(() => ({
  approve: { mutate: vi.fn(), isLoading: false },
  reject: { mutate: vi.fn(), isLoading: false },
}))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))

vi.mock('@/hooks/use-board-tasks', () => ({
  useApproveTask: () => verdict.approve,
  useRejectTask: () => verdict.reject,
}))
vi.mock('sonner', () => ({ toast }))

import { ReviewVerdict } from '../review-verdict'
import { approveEffect, approveLabel, approvedMessage } from '../approval-effect'

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return {
    id: '5', type: 'task', name: 'Welcome email', status: 'review', priority: 'medium', tags: [],
    review_mode: 'human', source_id: '5', assignee: { agent_id: 3, agent_name: 'Words' }, ...over,
  }
}

beforeEach(() => {
  verdict.approve.mutate.mockReset()
  verdict.reject.mutate.mockReset()
  toast.success.mockReset()
  toast.error.mockReset()
})
afterEach(cleanup)

describe('Reject', () => {
  it('needs the owner to say what is wrong, and sends those words', () => {
    render(<ReviewVerdict task={ticket()} onDecided={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: /Reject/ }))

    const sendBack = screen.getByRole('button', { name: /Send back/ })
    expect(sendBack).toBeDisabled()                     // night 1: Reject sent no note at all
    fireEvent.change(screen.getByRole('textbox', { name: /What's wrong\?/ }), { target: { value: '  Name the café.  ' } })
    expect(sendBack).toBeEnabled()
    fireEvent.click(sendBack)

    expect(verdict.reject.mutate).toHaveBeenCalledWith({ taskId: '5', feedback: 'Name the café.' }, expect.anything())
    expect(verdict.approve.mutate).not.toHaveBeenCalled()
  })

  it('says why when sending back fails', () => {
    verdict.reject.mutate.mockImplementation((_vars, opts) => opts.onError(new Error('Ticket #5 was already decided')))
    render(<ReviewVerdict task={ticket()} onDecided={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: /Reject/ }))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Shorter.' } })
    fireEvent.click(screen.getByRole('button', { name: /Send back/ }))
    expect(toast.error).toHaveBeenCalledWith('Ticket #5 was already decided')
  })
})

describe('Approve', () => {
  it('names its effect from the approval action, before anyone clicks', () => {
    const publish = { type: 'publish_blog', post_id: 'p1' }
    render(<ReviewVerdict task={ticket({ planning_data: { approval_action: publish } })} onDecided={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Approve and publish' })).toBeInTheDocument()
    expect(screen.getByTestId('approve-effect')).toHaveTextContent('publishes the blog post to the live site')
    expect(approveLabel(null)).toBe('Approve and mark done')
    expect(approveEffect(null)).toBe('Approving marks the ticket done. Nothing else runs.')
    expect(approveLabel({ type: 'create_blog', topic: 'Oat milk' })).toBe('Approve and start the post')
  })

  it('is one click without a note, and confirms what happened', () => {
    const onDecided = vi.fn()
    verdict.approve.mutate.mockImplementation((_vars, opts) => opts.onSuccess({ action_result: null }))
    render(<ReviewVerdict task={ticket()} onDecided={onDecided} />)
    fireEvent.click(screen.getByRole('button', { name: 'Approve and mark done' }))
    expect(verdict.approve.mutate).toHaveBeenCalledWith({ taskId: '5', note: undefined }, expect.anything())
    expect(toast.success).toHaveBeenCalledWith('Approved. Ticket #5 is done.')
    expect(onDecided).toHaveBeenCalled()
  })

  it('carries an optional note to keep on the ticket (F038)', () => {
    render(<ReviewVerdict task={ticket()} onDecided={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: /Add a note to the approval/ }))
    fireEvent.change(screen.getByRole('textbox', { name: /Note on the approval/ }), { target: { value: 'Send it Monday.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Approve and mark done' }))
    expect(verdict.approve.mutate).toHaveBeenCalledWith({ taskId: '5', note: 'Send it Monday.' }, expect.anything())
  })

  it('says why when the approval fails', () => {
    verdict.approve.mutate.mockImplementation((_vars, opts) => opts.onError(new Error('Approval action failed: planner unavailable')))
    render(<ReviewVerdict task={ticket()} onDecided={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Approve and mark done' }))
    expect(toast.error).toHaveBeenCalledWith('Approval action failed: planner unavailable')
  })

  it('confirms what an approval action did', () => {
    expect(approvedMessage({ action_result: { type: 'publish_blog', title: 'Oat milk' } }, '5')).toBe('Approved. "Oat milk" is published.')
    expect(approvedMessage({ action_result: { type: 'create_blog', topic: 'Oat milk' } }, '5')).toBe('Approved. The blog post on "Oat milk" has started.')
  })
})
