/** PRD-252 R1 — an approval row in Needs you opens the ticket at its approval, with Grant and Deny there. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const pending = vi.hoisted(() => ({ grants: [] as unknown[], asked: [] as unknown[] }))

vi.mock('@/hooks/use-approval-grants', () => ({
  useApprovalGrants: (status?: string, kind?: string) => {
    pending.asked.push([status, kind])
    return { data: { grants: pending.grants } }
  },
}))
vi.mock('@/components/command-center/governance/approvals-inbox', () => ({
  GrantCard: ({ grant }: { grant: { reason: string } }) => <div>{grant.reason}</div>,
}))

import { TicketApprovals } from '../ticket-approvals'

afterEach(cleanup)

describe('TicketApprovals', () => {
  it("shows this ticket's pending approvals, the linked one marked, and no other ticket's", () => {
    pending.grants = [
      { id: 11, reason: 'Send the price list', subject_type: 'board_task', subject_id: '760' },
      // A gated call made from the ticket: its ticket is on the grant's owner (F091-E1).
      { id: 12, reason: 'Post to Instagram', subject_type: 'tool_call', subject_id: 'c-1', owner: { ticket: { id: 760 } } },
      { id: 13, reason: 'Another ticket', subject_type: 'board_task', subject_id: '7' },
    ]
    render(<TicketApprovals taskId="760" focusGrantId={12} />)

    expect(pending.asked).toContainEqual(['pending', 'approval'])
    expect(screen.getByText('Waiting for 2 approvals')).toBeInTheDocument()
    expect(screen.queryByText('Another ticket')).toBeNull()
    expect(screen.getByText('Post to Instagram').parentElement).toHaveAttribute('data-focused', 'true')
    expect(screen.getByText('Send the price list').parentElement).not.toHaveAttribute('data-focused')
  })

  it("leaves a grant the blocked reason names to the Blocked panel's Approve button", () => {
    pending.grants = [
      { id: 11, reason: 'Send the price list', subject_type: 'board_task', subject_id: '760' },
      { id: 12, reason: 'Post to Instagram', subject_type: 'board_task', subject_id: '760' },
    ]
    render(<TicketApprovals taskId="760" blockedReason="Awaiting human approval (grant #11): board task requires approval" />)

    expect(screen.queryByText('Send the price list')).toBeNull()      // shown once, in the Blocked panel
    expect(screen.getByText('Post to Instagram')).toBeInTheDocument()
  })

  it('renders nothing for a ticket with no pending approval', () => {
    pending.grants = [{ id: 13, reason: 'Another ticket', subject_type: 'board_task', subject_id: '7' }]
    const { container } = render(<TicketApprovals taskId="760" />)
    expect(container).toBeEmptyDOMElement()
  })
})
