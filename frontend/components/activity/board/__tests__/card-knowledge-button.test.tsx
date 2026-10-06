/**
 * PRE-11 (Gerard, 7 Oct): an approved ticket's view offers Add to Knowledge for its
 * answer, then "Added to Knowledge" with Remove (the ticket stays). The board's answer
 * carries knowledge_document_id, which the ticket keeps.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'
import type { BoardTask } from '@/types/board'

const request = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({ apiClient: { request: (...args: unknown[]) => request(...args) } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/hooks/use-board-tasks-api', () => ({
  useUpdateTask: () => ({ mutate: vi.fn(), isLoading: false }),
  useCancelTask: () => ({ mutate: vi.fn(), isLoading: false }),
}))
vi.mock('@/hooks/use-agent-api', () => ({ useAssignableAgents: () => ({ data: [] }) }))
vi.mock('next/link', () => ({ default: ({ href, children, ...rest }: any) => <a href={String(href)} {...rest}>{children}</a> }))

import { CardKnowledgeButton, offersKnowledge } from '../card-knowledge-button'
import { TicketActionsBar } from '../ticket-actions-bar'
import { useBoardTask } from '@/hooks/use-board-tasks'

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '1139', type: 'task', name: 'Payment terms for cafés', status: 'done', priority: 'medium', tags: [],
    review_mode: 'human', source_id: '1139', result: 'Cafés pay on 30-day terms.', ...over }
}

function client() {
  return new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
}

function withClient(node: React.ReactNode) {
  return render(<QueryClientProvider client={client()}>{node}</QueryClientProvider>)
}

beforeEach(() => {
  request.mockReset()
  request.mockResolvedValue({ success: true, task_id: 1139, document_id: 42, already_added: false })
})
afterEach(() => cleanup())

describe('which tickets offer Add to Knowledge', () => {
  it('an approved ticket with an answer, or one already added; nothing else', () => {
    expect(offersKnowledge(ticket())).toBe(true)
    expect(offersKnowledge(ticket({ status: 'review' }))).toBe(false)
    expect(offersKnowledge(ticket({ result: '  ' }))).toBe(false)
    expect(offersKnowledge(ticket({ result: undefined }))).toBe(false)
    expect(offersKnowledge(ticket({ status: 'in_progress', knowledge_document_id: 42 }))).toBe(true)
  })
})

describe('CardKnowledgeButton', () => {
  it("adds the ticket's answer", async () => {
    withClient(<CardKnowledgeButton task={ticket()} />)

    fireEvent.click(screen.getByRole('button', { name: /Add to Knowledge/ }))

    await waitFor(() => expect(request).toHaveBeenCalledWith('/api/v1/tasks/1139/add-to-knowledge', { method: 'POST' }))
  })

  it('says it was added and removes the copy', async () => {
    withClient(<CardKnowledgeButton task={ticket({ knowledge_document_id: 42 })} />)

    expect(screen.getByText('Added to Knowledge')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Remove from Knowledge' }))

    await waitFor(() => expect(request).toHaveBeenCalledWith('/api/v1/tasks/1139/add-to-knowledge', { method: 'DELETE' }))
  })
})

describe("the ticket's view", () => {
  it('offers it on an approved ticket, not on one in review', () => {
    withClient(<TicketActionsBar task={ticket()} />)
    expect(screen.getByRole('button', { name: /Add to Knowledge/ })).toBeInTheDocument()
    cleanup()

    withClient(<TicketActionsBar task={ticket({ status: 'review' })} />)
    expect(screen.queryByRole('button', { name: /Add to Knowledge/ })).toBeNull()
  })

  it("keeps the board's knowledge_document_id on the ticket", async () => {
    request.mockResolvedValue({ id: 1139, title: 'Payment terms for cafés', status: 'done', knowledge_document_id: 42 })
    const queryClient = client()
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )
    const { result } = renderHook(() => useBoardTask('1139'), { wrapper })

    await waitFor(() => expect(result.current.data?.knowledge_document_id).toBe(42))
  })
})
