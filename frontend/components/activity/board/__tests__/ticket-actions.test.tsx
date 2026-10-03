/**
 * PRD-252 R7 — Assign and Cancel on the board; Cancelled and Closed are one
 * stage; a cancelled ticket says who stopped it and when.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'
import type { BoardTask } from '@/types/board'

const hooks = vi.hoisted(() => ({
  update: { mutate: vi.fn(), isLoading: false },
  cancel: { mutate: vi.fn(), isLoading: false },
  agents: [{ id: 7, name: 'Words' }, { id: 9, name: 'Numbers' }],
}))
const requestMock = vi.hoisted(() => vi.fn())

vi.mock('@/hooks/use-board-tasks-api', () => ({ useUpdateTask: () => hooks.update, useCancelTask: () => hooks.cancel }))
vi.mock('@/hooks/use-agent-api', () => ({ useAssignableAgents: () => ({ data: hooks.agents }) }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: (...a: unknown[]) => requestMock(...a) } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('next/link', () => ({ default: ({ href, children, ...rest }: any) => <a href={String(href)} {...rest}>{children}</a> }))

import { canAssign, canCancel, whoStopped } from '../ticket-actions'
import { TicketActionsBar, CancelledBanner } from '../ticket-actions-bar'
import { useBoardTasks } from '@/hooks/use-board-tasks'
import { BOARD_COLUMNS } from '@/types/board'

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '1139', type: 'task', name: 'Price list for Tidewater', status: 'inbox', priority: 'medium', tags: [],
    review_mode: 'auto', source_id: '1139', ...over }
}

beforeEach(() => { hooks.update.mutate.mockReset(); hooks.cancel.mutate.mockReset(); requestMock.mockReset() })
afterEach(cleanup)

describe('what the board offers', () => {
  it('an Inbox ticket can be assigned and cancelled; a finished one neither', () => {
    expect([canAssign(ticket()), canCancel(ticket())]).toEqual([true, true])
    expect([canAssign(ticket({ status: 'done' })), canCancel(ticket({ status: 'done' }))]).toEqual([false, false])
    // F245: Cancel closes a failed ticket (night 7: the button refused one without a word).
    expect([canAssign(ticket({ status: 'failed' })), canCancel(ticket({ status: 'failed' }))]).toEqual([true, true])
    expect(canCancel(ticket({ status: 'cancelled' }))).toBe(false)
  })

  it("a mission's ticket is the mission's to run: no assign or cancel, a link to the mission", () => {
    const step = ticket({ type: 'mission', mission_id: 'run-1' })
    expect([canAssign(step), canCancel(step)]).toEqual([false, false])
    render(<TicketActionsBar task={step} />)
    expect(screen.getByText('Open the mission').closest('a')).toHaveAttribute('href', '/missions/run-1')
  })

  it('every ticket can be discussed with Auto, a finished one and a mission\'s too (PRD-252 R2, D4)', () => {
    for (const task of [ticket(), ticket({ status: 'done' }), ticket({ type: 'mission', mission_id: 'run-1' })]) {
      render(<TicketActionsBar task={task} />)
      expect(screen.getByTestId('discuss-ticket')).toHaveAttribute('href', '/chat?ticket=1139')
      cleanup()
    }
  })
})

describe('in the viewer', () => {
  it('assigns an Inbox ticket without leaving the board', () => {
    render(<TicketActionsBar task={ticket()} />)
    fireEvent.change(screen.getByRole('combobox', { name: 'Assign to an agent' }), { target: { value: '9' } })
    expect(hooks.update.mutate).toHaveBeenCalledWith({ taskId: '1139', payload: { assigned_agent_id: 9 } }, expect.anything())
  })

  it('cancels a ticket', () => {
    render(<TicketActionsBar task={ticket({ status: 'blocked' })} />)
    fireEvent.click(screen.getByRole('button', { name: /Cancel ticket/ }))
    expect(hooks.cancel.mutate).toHaveBeenCalledWith('1139', expect.anything())
  })

  it('a cancelled ticket says who stopped it, when and why', () => {
    const at = new Date(Date.now() - 2 * 3600_000).toISOString()
    const task = ticket({ status: 'cancelled', runtime_ref: { cancelled: { by: 'user:1', reason: 'cancelled on the board', at } } })
    expect(whoStopped(task)).toMatchObject({ reason: 'cancelled on the board', at, closed: false })
    render(<CancelledBanner task={task} />)
    expect(screen.getByTestId('cancelled-banner')).toHaveTextContent(/Cancelled by .+ about 2 hours ago/)
  })
})

describe('one stage', () => {
  it('a closed ticket sits in the Cancelled column; there is no Closed column', async () => {
    expect(BOARD_COLUMNS.map((c) => c.status)).not.toContain('closed')
    requestMock.mockResolvedValue({ tasks: [
      { id: 1, title: 'Superseded', status: 'closed' },
      { id: 2, title: 'No longer wanted', status: 'cancelled' },
    ], total: 2 })
    const client = new QueryClient()
    const wrapper = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    const { result } = renderHook(() => useBoardTasks(), { wrapper })
    await waitFor(() => expect(result.current.columns.find((c) => c.status === 'cancelled')?.tasks).toHaveLength(2))
  })
})
