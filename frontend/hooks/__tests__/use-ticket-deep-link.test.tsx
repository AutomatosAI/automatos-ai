/**
 * PRD-252 R1 — the board's deep link opens the ticket: by id (so a filtered, old
 * or step ticket opens too), at a question when the link names one, and a
 * question link without a ticket finds its ticket among the open questions.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'

const nav = vi.hoisted(() => ({ query: '', replace: vi.fn() }))
const board = vi.hoisted(() => ({ tasks: {} as Record<string, { id: string; name: string }>, failing: new Set<string>() }))
const asks = vi.hoisted(() => ({ data: undefined as undefined | { grants: unknown[] } }))
const toastError = vi.hoisted(() => vi.fn())

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace }),
  usePathname: () => '/command-center',
  useSearchParams: () => new URLSearchParams(nav.query),
}))
vi.mock('@/hooks/use-board-tasks', () => ({
  useBoardTask: (id: string | null) => ({
    data: id ? board.tasks[id] : undefined,
    isError: id ? board.failing.has(id) : false,
  }),
}))
vi.mock('@/hooks/use-approval-grants', () => ({ useQuestions: () => ({ data: asks.data }) }))
vi.mock('sonner', () => ({ toast: { error: toastError } }))

import { useTicketDeepLink } from '@/hooks/use-ticket-deep-link'

const old = { id: '12', name: 'Done in July' }
const parked = { id: '612', name: 'Cafe questions' }

beforeEach(() => {
  board.tasks = { '12': old, '612': parked }
  board.failing = new Set()
  asks.data = undefined
  nav.replace.mockClear()
  toastError.mockClear()
})
afterEach(cleanup)

describe('useTicketDeepLink', () => {
  it('opens the ticket a link names, once', () => {
    nav.query = 'tab=board&task_id=12'
    const onOpen = vi.fn()
    const { rerender } = renderHook(() => useTicketDeepLink(onOpen))
    rerender()
    expect(onOpen).toHaveBeenCalledTimes(1)
    expect(onOpen).toHaveBeenCalledWith(old, null)
  })

  it('opens it at the question the link names', () => {
    nav.query = 'tab=board&task_id=612&question=41'
    const onOpen = vi.fn()
    renderHook(() => useTicketDeepLink(onOpen))
    expect(onOpen).toHaveBeenCalledWith(parked, 41)
  })

  it('finds the ticket of a question known only by its id (a notification)', () => {
    nav.query = 'tab=board&question=41'
    asks.data = { grants: [{ id: 41, subject_type: 'board_task', subject_id: '612' }] }
    const onOpen = vi.fn()
    renderHook(() => useTicketDeepLink(onOpen))
    expect(onOpen).toHaveBeenCalledWith(parked, 41)
  })

  it('sends a question that is no longer open to the Questions tab', () => {
    nav.query = 'tab=board&question=99'
    asks.data = { grants: [] }
    const onOpen = vi.fn()
    renderHook(() => useTicketDeepLink(onOpen))
    expect(onOpen).not.toHaveBeenCalled()
    expect(nav.replace).toHaveBeenCalledWith('/command-center?tab=questions')
  })

  it('says so when the ticket cannot be opened, instead of a silent board', () => {
    nav.query = 'tab=board&task_id=404'
    board.failing = new Set(['404'])
    renderHook(() => useTicketDeepLink(vi.fn()))
    expect(toastError).toHaveBeenCalledTimes(1)
    expect(toastError.mock.calls[0][0]).toContain('Ticket 404 could not be opened')   // PRD-252 R4: never '#404', a number
  })

  it('clear() drops the link, so a tab switch never reopens the ticket', () => {
    nav.query = 'tab=board&task_id=12&question=41'
    const { result } = renderHook(() => useTicketDeepLink(vi.fn()))
    result.current()
    expect(nav.replace).toHaveBeenCalledWith('/command-center?tab=board', { scroll: false })
  })
})
