/**
 * PRD-170 S4 — DiffCard render (vitest).
 *
 * A pending edit card renders the in-bundle Monaco diff + Approve/Deny buttons
 * that fire onDecide; an auto-accepted card renders the "auto-accepted" badge
 * and NO action buttons; a resolved card hides its buttons. @monaco-editor/react
 * and next/dynamic are stubbed so the card renders without a real editor.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

// Stub next/dynamic to synchronously return a lightweight component so the
// dynamically-imported DiffEditor renders deterministically in jsdom.
vi.mock('next/dynamic', () => ({
  default: () => (props: Record<string, unknown>) => (
    <div
      data-testid="monaco-diff"
      data-original={String(props.original ?? '')}
      data-modified={String(props.modified ?? '')}
      data-language={String(props.language ?? '')}
    />
  ),
}))

import { DiffCard } from './DiffCard'
import type { ApprovalCard } from './diffApproval'

const editCard = (overrides: Partial<ApprovalCard> = {}): ApprovalCard => ({
  requestId: 'r1',
  toolName: 'Write',
  diff: {
    path: 'src/app.py',
    oldContent: 'old',
    newContent: 'new',
    language: 'python',
  },
  status: 'pending',
  autoAccepted: false,
  ...overrides,
})

describe('DiffCard', () => {
  it('renders the tool name, path, and the in-bundle diff', () => {
    render(<DiffCard card={editCard()} onDecide={vi.fn()} />)
    expect(screen.getByText('Write')).toBeInTheDocument()
    expect(screen.getByText('src/app.py')).toBeInTheDocument()
    const diff = screen.getByTestId('monaco-diff')
    expect(diff).toHaveAttribute('data-original', 'old')
    expect(diff).toHaveAttribute('data-modified', 'new')
    expect(diff).toHaveAttribute('data-language', 'python')
  })

  it('fires onDecide("approve") when Approve is clicked', () => {
    const onDecide = vi.fn()
    render(<DiffCard card={editCard()} onDecide={onDecide} />)
    fireEvent.click(screen.getByTestId('diff-approve'))
    expect(onDecide).toHaveBeenCalledWith('r1', 'approve')
  })

  it('fires onDecide("deny") when Deny is clicked', () => {
    const onDecide = vi.fn()
    render(<DiffCard card={editCard()} onDecide={onDecide} />)
    fireEvent.click(screen.getByTestId('diff-deny'))
    expect(onDecide).toHaveBeenCalledWith('r1', 'deny')
  })

  it('an auto-accepted card shows the badge and NO action buttons', () => {
    render(
      <DiffCard
        card={editCard({ status: 'approved', autoAccepted: true })}
        onDecide={vi.fn()}
      />
    )
    expect(screen.getByTestId('auto-accepted-badge')).toBeInTheDocument()
    expect(screen.queryByTestId('diff-approve')).toBeNull()
    expect(screen.queryByTestId('diff-deny')).toBeNull()
  })

  it('a resolved (denied) card hides its action buttons', () => {
    render(
      <DiffCard card={editCard({ status: 'denied' })} onDecide={vi.fn()} />
    )
    expect(screen.queryByTestId('diff-approve')).toBeNull()
    expect(screen.getByText('denied')).toBeInTheDocument()
  })

  it('a bare permission card (no diff) renders without a diff view', () => {
    const card: ApprovalCard = {
      requestId: 'r-bash',
      toolName: 'Bash',
      diff: null,
      status: 'pending',
      autoAccepted: false,
    }
    render(<DiffCard card={card} onDecide={vi.fn()} />)
    expect(screen.getByText('Bash')).toBeInTheDocument()
    expect(screen.queryByTestId('monaco-diff')).toBeNull()
    // Still approvable/deniable.
    expect(screen.getByTestId('diff-approve')).toBeInTheDocument()
  })
})
