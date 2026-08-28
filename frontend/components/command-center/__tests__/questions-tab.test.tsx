/**
 * PRD-225 (G2) — the Questions tab.
 *
 * Open asks render newest-first with markdown + the blocked cascade (capped
 * with "+N more"); Answer posts to the answer endpoint and the card leaves the
 * queue; Dismiss keeps the trail visible (the subject stays blocked); an option
 * button answers with that choice.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import type { ApprovalGrant } from '@/lib/api-client'

const { answerMutate, dismissMutate } = vi.hoisted(() => ({
  answerMutate: vi.fn().mockResolvedValue({}),
  dismissMutate: vi.fn().mockResolvedValue({}),
}))

const useQuestionsMock = vi.hoisted(() => vi.fn())

vi.mock('@/hooks/use-approval-grants', () => ({
  useQuestions: useQuestionsMock,
  useAnswerQuestion: () => ({ isLoading: false, mutateAsync: answerMutate }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: dismissMutate }),
}))

vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: any) => (
    <a href={typeof href === 'string' ? href : String(href)} {...rest}>
      {children}
    </a>
  ),
}))

vi.mock('sonner', () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

import { QuestionsTab } from '../questions-tab'

function question(overrides?: Partial<ApprovalGrant>): ApprovalGrant {
  return {
    id: 1,
    workspace_id: 'ws',
    subject_type: 'board_task',
    subject_id: '42',
    tool_name: null,
    risk_tier: null,
    agent_id: 9,
    status: 'pending',
    reason: 'Awaiting human answer',
    estimated_cost_usd: null,
    requested_at: '2026-08-27T10:00:00Z',
    expires_at: null,
    granted_at: null,
    granted_by: null,
    revoked_at: null,
    revoked_by: null,
    kind: 'question',
    question_md: 'Ship **A** or **B**?',
    options: null,
    answer_text: null,
    answered_by: null,
    answered_at: null,
    asked_by_agent_id: 9,
    ...overrides,
  }
}

function setQuestions(grants: ApprovalGrant[]) {
  useQuestionsMock.mockReturnValue({ data: { grants }, isLoading: false, isError: false })
}

describe('QuestionsTab — PRD-225', () => {
  beforeEach(() => {
    answerMutate.mockClear().mockResolvedValue({})
    dismissMutate.mockClear().mockResolvedValue({})
  })

  it('renders open asks newest-first with markdown + asker + subject link', () => {
    setQuestions([
      question({ id: 1, question_md: 'First **ask**', asked_by_agent_id: 3 }),
      question({ id: 2, question_md: 'Second ask', asked_by_agent_id: 4 }),
    ])
    render(<QuestionsTab />)
    // Markdown renders — the bold segment is a distinct node.
    expect(screen.getByText('ask')).toBeInTheDocument()
    // Asker badge + subject link.
    expect(screen.getByText('Agent #3')).toBeInTheDocument()
    const link = screen.getAllByRole('link', { name: /board_task:42/ })[0]
    expect(link).toHaveAttribute('href', '/command-center?tab=board')
    // Order preserved (newest-first is the backend's job; the tab keeps order).
    const asked = screen.getAllByText(/Agent #/).map((n) => n.textContent)
    expect(asked).toEqual(['Agent #3', 'Agent #4'])
  })

  it('renders the blocked cascade capped at 6 with "+N more"', () => {
    const tasks = Array.from({ length: 6 }, (_, i) => ({
      id: i + 100,
      title: `Downstream ${i + 1}`,
      status: 'blocked',
    }))
    setQuestions([question({ cascade: { total: 8, tasks } })])
    render(<QuestionsTab />)
    expect(screen.getByText(/Blocking 8 downstream tasks/)).toBeInTheDocument()
    expect(screen.getByText('Downstream 1')).toBeInTheDocument()
    expect(screen.getByText('Downstream 6')).toBeInTheDocument()
    expect(screen.getByText('+2 more')).toBeInTheDocument()
  })

  it('a cascade fixture that would have cycled still renders a flat, deduped list', () => {
    // The backend de-cycles; the tab renders whatever flat list it receives.
    setQuestions([
      question({
        cascade: {
          total: 3,
          tasks: [
            { id: 1, title: 'A', status: 'blocked' },
            { id: 2, title: 'B', status: 'blocked' },
            { id: 3, title: 'C', status: 'assigned' },
          ],
        },
      }),
    ])
    render(<QuestionsTab />)
    expect(screen.getByText(/Blocking 3 downstream tasks/)).toBeInTheDocument()
    expect(screen.queryByText('+0 more')).not.toBeInTheDocument()
  })

  it('answering posts to the endpoint and the card leaves the list', async () => {
    setQuestions([question({ id: 7 })])
    render(<QuestionsTab />)
    fireEvent.change(screen.getByRole('textbox', { name: /Answer/ }), {
      target: { value: 'Ship A' },
    })
    fireEvent.click(screen.getByRole('button', { name: /Answer/ }))
    expect(answerMutate).toHaveBeenCalledWith({ grantId: 7, answer_text: 'Ship A' })
    await waitFor(() =>
      expect(screen.queryByText(/Ship/)).not.toBeInTheDocument(),
    )
  })

  it('⌘/Ctrl-Enter in the textarea submits the answer', () => {
    setQuestions([question({ id: 8 })])
    render(<QuestionsTab />)
    const box = screen.getByRole('textbox', { name: /Answer/ })
    fireEvent.change(box, { target: { value: 'use your judgment' } })
    fireEvent.keyDown(box, { key: 'Enter', metaKey: true })
    expect(answerMutate).toHaveBeenCalledWith({ grantId: 8, answer_text: 'use your judgment' })
  })

  it('option buttons answer with the chosen option', () => {
    setQuestions([question({ id: 9, options: ['A', 'B'] })])
    render(<QuestionsTab />)
    fireEvent.click(screen.getByRole('button', { name: 'B' }))
    expect(answerMutate).toHaveBeenCalledWith({ grantId: 9, option: 'B' })
  })

  it('dismiss keeps the trail visible (subject stays blocked, may re-ask)', async () => {
    setQuestions([question({ id: 11, question_md: 'Keep me visible' })])
    render(<QuestionsTab />)
    fireEvent.click(screen.getByRole('button', { name: /Dismiss/ }))
    expect(dismissMutate).toHaveBeenCalledWith(11)
    await waitFor(() =>
      expect(screen.getByRole('note')).toHaveTextContent(/may re-ask/i),
    )
    // The question text stays on screen — the trail is not removed.
    expect(screen.getByText('Keep me visible')).toBeInTheDocument()
  })

  it('shows an honest empty state when nothing is open', () => {
    setQuestions([])
    render(<QuestionsTab />)
    expect(screen.getByText(/No open questions/)).toBeInTheDocument()
  })

  it('a strict-held channel directive renders inert — no clickable link (P225-RVW-6)', () => {
    // The trust gate stores channel-sourced text fenced, so an attacker's link
    // syntax / bare URL is literal text, never a clickable anchor in the admin's
    // own trusted Questions tab.
    const fenced = [
      '**Inbound directive awaiting approval**',
      '',
      '```',
      '[route it now](https://attacker.example) then https://evil.example',
      '```',
      '',
      '_Answer "route it" to let it proceed, or dismiss to keep it held._',
    ].join('\n')
    setQuestions([
      question({ id: 21, subject_type: 'channel', subject_id: 'chan-1', question_md: fenced }),
    ])
    render(<QuestionsTab />)

    // The raw directive text stays readable by the operator...
    expect(document.body.textContent).toContain('[route it now](https://attacker.example)')
    // ...but nothing in the inbound text is a clickable anchor.
    expect(document.querySelector('a[href="https://attacker.example"]')).toBeNull()
    expect(document.querySelector('a[href="https://evil.example"]')).toBeNull()
    expect(screen.queryByRole('link', { name: /route it now/ })).not.toBeInTheDocument()
  })

  it('an agent-authored question still renders real markdown links (fix is scoped)', () => {
    setQuestions([
      question({ id: 22, question_md: 'See [the doc](https://trusted.example) first.' }),
    ])
    render(<QuestionsTab />)
    expect(screen.getByRole('link', { name: 'the doc' })).toHaveAttribute(
      'href',
      'https://trusted.example',
    )
  })
})
