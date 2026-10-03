/** PRD-252 R1 — "View on Board" after creating a task opens that ticket, not the whole board. */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

const push = vi.hoisted(() => vi.fn())
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
const create = vi.hoisted(() => ({ mutateAsync: vi.fn(), isLoading: false }))
const schedule = vi.hoisted(() => ({ mutateAsync: vi.fn(), isLoading: false }))

vi.mock('next/navigation', () => ({ useRouter: () => ({ push }) }))
vi.mock('sonner', () => ({ toast }))
vi.mock('@/hooks/use-board-tasks-api', () => ({ useCreateTask: () => create }))
vi.mock('@/hooks/use-scheduled-tasks-api', () => ({ useCreateScheduledBoardTask: () => schedule }))

import { filedTicketHref, useCreateTaskSubmit } from '../create-task-submit'

const payload = { title: 'Welcome email', description: 'For the new café', priority: 'medium' as const }

beforeEach(() => {
  push.mockClear(); toast.success.mockClear(); toast.error.mockClear(); create.mutateAsync.mockReset()
})

describe('filing a task now', () => {
  it('"View on Board" opens the ticket just filed', async () => {
    create.mutateAsync.mockResolvedValue({ id: 1206, title: 'Welcome email' })
    const onDone = vi.fn()
    const { result } = renderHook(() => useCreateTaskSubmit({
      buildPayload: () => payload, scheduleMode: 'now', scheduleAt: '', attachmentCount: 0, agentName: 'Words', onDone,
    }))

    await act(() => result.current.submit())

    expect(toast.success).toHaveBeenCalledWith('Task assigned to Words.', expect.anything())
    toast.success.mock.calls[0][1].action.onClick()
    expect(push).toHaveBeenCalledWith('/command-center?tab=board&task_id=1206')   // it was ?tab=board
    expect(onDone).toHaveBeenCalled()
  })

  it('opens the board when the create returned no id', () => {
    expect(filedTicketHref(undefined)).toBe('/command-center?tab=board')
    expect(filedTicketHref({ id: '7' })).toBe('/command-center?tab=board&task_id=7')
  })

  it('refuses an untitled task without filing it', async () => {
    const { result } = renderHook(() => useCreateTaskSubmit({
      buildPayload: () => ({ ...payload, title: '  ' }), scheduleMode: 'now', scheduleAt: '', attachmentCount: 0, agentName: null, onDone: vi.fn(),
    }))
    await act(() => result.current.submit())
    expect(toast.error).toHaveBeenCalledWith('Title is required')
    expect(create.mutateAsync).not.toHaveBeenCalled()
  })
})
