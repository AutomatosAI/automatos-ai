/**
 * P256-FIX-N13-7 (F414) — one ticket number everywhere.
 *
 * Night 13, Auto said #1394 (the ticket's workspace_seq) while a surface showed its row id,
 * 2680, and "did that go through?" got yes, then no. The board's card label, the chat's
 * ticket card and the board's toasts name a ticket by its board number, never its row id.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import React from 'react'
import { renderHook, waitFor, cleanup } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const api = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: api.request } }))

import { ticketCalled, ticketNumber } from '@/lib/ticket-number'
import { useBoardTask } from '@/hooks/use-board-tasks'

const read = (rel: string) => readFileSync(path.resolve(__dirname, '..', '..', rel), 'utf8')

afterEach(cleanup)

describe('ticketNumber', () => {
  it('writes workspace_seq 1394 as the board number #1394, never the row id', () => {
    expect(ticketNumber({ workspace_seq: 1394 })).toBe('#1394')
    expect(ticketNumber({ workspace_seq: 42 })).toBe('#0042')
    expect(ticketNumber({ number: '#0051.3', workspace_seq: 99 })).toBe('#0051.3')   // the API's number wins
    expect(ticketNumber({ number: null })).toBeNull()
    expect(ticketNumber({ workspace_seq: 7, source_type: 'orchestration_task' })).toBeNull()   // a step: its card's
  })

  it('calls a ticket by its number, and "this ticket" without one', () => {
    expect(ticketCalled({ workspace_seq: 1394 })).toBe('ticket #1394')
    expect(ticketCalled({ number: '#1394' }, true)).toBe('Ticket #1394')
    expect(ticketCalled({})).toBe('this ticket')
  })
})

describe('the board card label', () => {
  it('shows #1394 for a ticket with workspace_seq 1394 and row id 2680', async () => {
    api.request.mockResolvedValueOnce({ id: 2680, workspace_seq: 1394, number: null, title: 'Price list', status: 'review' })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
    const { result } = renderHook(() => useBoardTask('2680'), { wrapper })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.number).toBe('#1394')
    expect(result.current.data?.id).toBe('2680')   // the id stays for the API, never on the label
  })

  it('every label renders the number the board mapped, and no id fallback is left', () => {
    expect(read('components/activity/board/board-card.tsx')).toContain('{task.number && ')
    expect(read('components/command-center/board-tab.tsx')).toContain('{task.number && <span className="num">{task.number}</span>}')
    expect(read('components/chatbot/task-card.tsx')).toContain('{ticketCalled(card, true)}')
    for (const file of ['components/chatbot/task-card.tsx', 'components/activity/board/ticket-actions-menu.tsx',
      'components/activity/board/review-verdict.tsx']) {
      expect(read(file), file).not.toMatch(/number \?\? (task|card)\.id/)
    }
  })
})
