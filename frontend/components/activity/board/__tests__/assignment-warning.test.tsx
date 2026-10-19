/**
 * PRD-256 F430 — a ticket given to a CLI-session agent that cannot call what it names says so:
 * the create and the PATCH answer with `warning`, and the board shows it as a warning toast
 * beside the success. The assignment stands either way.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook } from '@testing-library/react'
import type { BoardTask } from '@/types/board'

const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn() }))
const answer = vi.hoisted(() => ({ current: {} as Record<string, unknown> }))

vi.mock('sonner', () => ({ toast }))
vi.mock('@/hooks/use-agent-api', () => ({ useAssignableAgents: () => ({ data: [] }) }))
vi.mock('@/hooks/use-board-tasks-api', () => ({
  // the PATCH answers at once with what the test set
  useUpdateTask: () => ({ mutate: (_vars: unknown, opts: any) => opts.onSuccess(answer.current), isLoading: false }),
  useCancelTask: () => ({ mutate: vi.fn(), isLoading: false }),
}))
vi.mock('next/link', () => ({ default: ({ children }: any) => children }))

import { ASSIGNMENT_WARNING_DURATION_MS, assignmentWarning, toastAssignmentWarning } from '../assignment-warning'
import { useTicketActions } from '../ticket-actions-menu'

const WARNING = 'Social Media Director runs as a CLI session and cannot call `platform_create_social_post` '
  + '(its Socials tool group is off for this agent). Its Model tab turns the tool group on.'

function ticket(): BoardTask {
  return { id: '1139', type: 'task', name: 'Quote card', status: 'inbox', priority: 'medium', tags: [],
    review_mode: 'auto', source_id: '1139' } as BoardTask
}

beforeEach(() => { toast.success.mockClear(); toast.warning.mockClear(); answer.current = {} })

describe('the answer\'s warning', () => {
  it('is read only when it is a sentence', () => {
    expect(assignmentWarning({ warning: `  ${WARNING} ` })).toBe(WARNING)
    for (const none of [undefined, null, {}, { warning: '' }, { warning: '   ' }, { warning: 7 }, 'warning']) {
      expect(assignmentWarning(none)).toBeNull()
    }
  })

  it('shows as a warning toast long enough to read, and nothing shows without one', () => {
    toastAssignmentWarning({ warning: WARNING })
    expect(toast.warning).toHaveBeenCalledWith(WARNING, { duration: ASSIGNMENT_WARNING_DURATION_MS })
    toast.warning.mockClear()
    toastAssignmentWarning({ id: 1139 })
    expect(toast.warning).not.toHaveBeenCalled()
  })
})

describe('assigning from the board', () => {
  it('says the ticket is assigned, then what the agent cannot call', () => {
    answer.current = { id: 1139, cannot_call: ['platform_create_social_post'], warning: WARNING }
    const { result } = renderHook(() => useTicketActions(ticket()))
    result.current.assignTo({ id: 348, name: 'Social Media Director' })
    expect(toast.success).toHaveBeenCalledWith(expect.stringContaining('is assigned to Social Media Director.'))
    expect(toast.warning).toHaveBeenCalledWith(WARNING, expect.anything())
  })

  it('says only that it is assigned when the agent can do the work', () => {
    answer.current = { id: 1139 }
    const { result } = renderHook(() => useTicketActions(ticket()))
    result.current.assignTo({ id: 7, name: 'Words' })
    expect(toast.success).toHaveBeenCalled()
    expect(toast.warning).not.toHaveBeenCalled()
  })
})
