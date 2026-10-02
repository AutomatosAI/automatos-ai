/** PRD-252 R1 — a question link opens the question inside its ticket. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const asks = vi.hoisted(() => ({ grants: [] as unknown[] }))

vi.mock('@/hooks/use-approval-grants', () => ({ useQuestions: () => ({ data: { grants: asks.grants } }) }))
vi.mock('@/components/command-center/question-card', () => ({
  QuestionCard: ({ q, inTicket }: { q: { question_md: string }; inTicket?: boolean }) => (
    <div data-in-ticket={String(Boolean(inTicket))}>{q.question_md}</div>
  ),
}))

import { TicketQuestions } from '../ticket-questions'

afterEach(cleanup)

describe('TicketQuestions', () => {
  it("shows this ticket's open questions, the linked one marked, and no other ticket's", () => {
    asks.grants = [
      { id: 41, question_md: 'Which café?', subject_type: 'board_task', subject_id: '612' },
      { id: 42, question_md: 'Which roast?', subject_type: 'board_task', subject_id: '612' },
      { id: 43, question_md: 'Another ticket', subject_type: 'board_task', subject_id: '7' },
    ]
    render(<TicketQuestions taskId="612" focusQuestionId={42} />)

    expect(screen.getByText('Waiting for 2 answers')).toBeInTheDocument()
    expect(screen.getByText('Which café?')).toBeInTheDocument()
    expect(screen.queryByText('Another ticket')).toBeNull()
    expect(screen.getByText('Which roast?').parentElement).toHaveAttribute('data-focused', 'true')
    expect(screen.getByText('Which café?').parentElement).not.toHaveAttribute('data-focused')
    // Inside the ticket, the card does not link back to the ticket it is in.
    expect(screen.getByText('Which café?')).toHaveAttribute('data-in-ticket', 'true')
  })

  it('renders nothing for a ticket with no open question', () => {
    asks.grants = [{ id: 43, question_md: 'Another ticket', subject_type: 'board_task', subject_id: '7' }]
    const { container } = render(<TicketQuestions taskId="612" />)
    expect(container).toBeEmptyDOMElement()
  })
})
