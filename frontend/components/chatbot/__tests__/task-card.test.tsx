/**
 * PRD-238 S6 — the ticket card: renders the snapshot, follows the board's
 * events for its own ticket, stops listening once terminal.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, act, waitFor } from '@testing-library/react'
import React from 'react'

const push = vi.fn()
vi.mock('next/navigation', () => ({ useRouter: () => ({ push, replace: vi.fn() }) }))
const api = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))

import { TaskCard } from '../task-card'
import type { TaskCardData } from '@/types'

const card: TaskCardData = {
  id: 92,
  title: 'Write basic webpage for AI workflow business',
  status: 'in_progress',
  assigned_agent: 'Bob',
  last_tool: 'Bash',
  files_touched: 2,
  started_at: new Date(Date.now() - 65_000).toISOString(),
}

beforeEach(() => {
  api.request.mockReset()
  push.mockReset()
})
afterEach(cleanup)

describe('TaskCard', () => {
  it('shows the ticket, its status, who has it and what it touched', () => {
    render(<TaskCard card={card} />)
    expect(screen.getByText('Ticket #92')).toBeInTheDocument()
    expect(screen.getByText(card.title)).toBeInTheDocument()
    expect(screen.getByTestId('task-card-status')).toHaveTextContent('In Progress')
    expect(screen.getByText(/Bob · 1 min so far · last tool: Bash · 2 files touched/)).toBeInTheDocument()
  })

  it('refreshes from the API when the board reports its ticket changed', async () => {
    api.request.mockResolvedValue({ id: 92, title: card.title, status: 'done', started_at: card.started_at, completed_at: new Date().toISOString(), runtime_ref: { exit_reason: 'completed', files_touched: ['a', 'b', 'c'] } })
    render(<TaskCard card={card} />)
    act(() => {
      window.dispatchEvent(new CustomEvent('automatos:board-changed', { detail: { task_id: 91 } }))
    })
    expect(api.request).not.toHaveBeenCalled()
    act(() => {
      window.dispatchEvent(new CustomEvent('automatos:board-changed', { detail: { task_id: 92 } }))
    })
    await waitFor(() => expect(screen.getByTestId('task-card-status')).toHaveTextContent('Done'))
    expect(api.request).toHaveBeenCalledWith('/api/v1/tasks/92')
    expect(screen.getByText(/3 files touched/)).toBeInTheDocument()
  })

  it('opens the ticket on the board', () => {
    render(<TaskCard card={card} />)
    screen.getByRole('button', { name: /Open on the board/ }).click()
    expect(push).toHaveBeenCalledWith('/command-center?tab=board&task_id=92')
  })
})
