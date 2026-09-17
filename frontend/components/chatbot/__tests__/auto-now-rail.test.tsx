/**
 * PRD-244 D5 — "Auto now" on fake hook data: with one open question, one due
 * watch and one running agent the rail shows three honest rows, each landing
 * on the right Command Centre tab; with nothing, it says so; answering sends
 * the grant id and the option or text.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'

const data = vi.hoisted(() => ({
  stats: { data: undefined as any, isLoading: false },
  fleet: { data: undefined as any, isLoading: false },
  questions: { data: undefined as any, isLoading: false },
  watches: { data: undefined as any, isLoading: false },
  decisions: { data: undefined as any, isLoading: false },
  answer: { mutateAsync: vi.fn(async () => ({})), isLoading: false },
}))
vi.mock('@/hooks/use-activity-api', () => ({ useActivityStats: () => data.stats }))
vi.mock('@/hooks/use-agent-api', () => ({ useFleetState: () => data.fleet }))
vi.mock('@/hooks/use-approval-grants', () => ({ useQuestions: () => data.questions, useAnswerQuestion: () => data.answer }))
vi.mock('@/hooks/use-watches-api', () => ({ useWatches: () => data.watches }))
vi.mock('@/hooks/use-kpi-api', () => ({ useDecisionsNeeded: () => data.decisions }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { AutoNowRail, questionPreview } from '@/components/chatbot/auto-now-rail'
import { AutoNowPill } from '@/components/chatbot/auto-now-pill'

const question = (over: Partial<Record<string, unknown>> = {}) => ({
  id: 41, kind: 'question', status: 'pending', subject_type: 'board_task', subject_id: '7',
  question_md: '## Which vendor?\nPick one.', options: null, asked_by_agent_id: 3, requested_at: '2026-09-17T10:00:00Z', ...over,
})

beforeEach(() => {
  data.stats.data = undefined; data.fleet.data = undefined; data.questions.data = undefined; data.watches.data = undefined; data.decisions.data = undefined
  data.answer.mutateAsync.mockClear()
})
afterEach(cleanup)

describe('AutoNowRail', () => {
  it('with nothing on the floor, every section says so and fabricates no count', () => {
    render(<AutoNowRail />)
    expect(screen.getByText('No one is working right now.')).toBeInTheDocument()
    expect(screen.getByText('Nothing waiting on you.')).toBeInTheDocument()
    expect(screen.getByText('Nothing being watched.')).toBeInTheDocument()
    expect(screen.getByText('No decisions waiting.')).toBeInTheDocument()
    expect(screen.getAllByText('—')).toHaveLength(4)
  })

  it('with one running agent, one open question, one due watch and one decision, shows honest rows that land on the right tabs', () => {
    data.stats.data = { working_now: 1, agents_active: 2, tasks_in_queue: 3, needs_attention: 1 }
    data.fleet.data = { agents: [
      { agent_id: 9, name: 'OPS', current: { kind: 'board_task', id: 1, title: 'Draft the Q3 vendor email', since: '2026-09-17T09:00:00Z' } },
      { agent_id: 4, name: 'IDLE', current: null },
    ] }
    data.questions.data = { grants: [question()] }
    data.watches.data = { watches: [{ id: 'w1', title: 'Invoice run', next_check_at: '2026-09-17T16:00:00Z' }] }
    data.decisions.data = { total: 1, items: [] }
    render(<AutoNowRail />)
    expect(screen.getByText('OPS').closest('a')).toHaveAttribute('href', '/command-center?tab=board')
    expect(screen.getByText('Draft the Q3 vendor email')).toBeInTheDocument()
    expect(screen.queryByText('IDLE')).toBeNull()
    expect(screen.getByText('Which vendor?').closest('a')).toHaveAttribute('href', '/command-center?tab=questions')
    expect(screen.getByText('Invoice run').closest('a')).toHaveAttribute('href', '/command-center?tab=watchlist')
    expect(screen.getByText('1 waiting for a decision').closest('a')).toHaveAttribute('href', '/command-center?tab=governance')
    expect(screen.getByText('3')).toBeInTheDocument() // queue stat
  })

  it('answers inline — an option sends the option, free text sends answer_text, with the grant id', async () => {
    data.questions.data = { grants: [question({ options: ['Acme', 'Globex'] }), question({ id: 42, options: null, question_md: 'Budget?' })] }
    render(<AutoNowRail />)
    fireEvent.click(screen.getByRole('button', { name: 'Globex' }))
    await waitFor(() => expect(data.answer.mutateAsync).toHaveBeenCalledWith({ grantId: 41, option: 'Globex' }))
    fireEvent.change(screen.getByLabelText('Answer'), { target: { value: '5k' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(data.answer.mutateAsync).toHaveBeenCalledWith({ grantId: 42, answer_text: '5k' }))
  })

  it('strips markdown for the one-line preview', () => {
    expect(questionPreview('## **Which** _vendor_?\nmore')).toBe('Which vendor?')
    expect(questionPreview(null)).toBe('')
  })
})

describe('AutoNowPill', () => {
  it('carries the two counts that need a human and toggles the rail', () => {
    data.questions.data = { grants: [question(), question({ id: 2 })] }
    data.decisions.data = { total: 1, items: [] }
    const onToggle = vi.fn()
    render(<AutoNowPill open={false} onToggle={onToggle} />)
    const btn = screen.getByRole('button', { name: 'Show Auto now rail' })
    expect(btn).toHaveAttribute('title', 'Auto now · 2 questions · 1 decision')
    expect(screen.getByText('3')).toBeInTheDocument()
    fireEvent.click(btn)
    expect(onToggle).toHaveBeenCalled()
  })
})
