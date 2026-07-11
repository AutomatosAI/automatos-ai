/**
 * PRD-196 S1 (P2-15, governance I.1) — the Approvals inbox front door.
 *
 * A pending grant renders its Art.14 oversight tier + rationale + Grant/Deny
 * actions; Grant/Deny call the right mutation; a decided (granted) grant shows
 * Revoke, not Grant/Deny; a fully-decided (denied) grant is read-only.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import type { ApprovalGrant } from '@/lib/api-client'

const { grantMutate, denyMutate, revokeMutate } = vi.hoisted(() => ({
  grantMutate: vi.fn().mockResolvedValue({}),
  denyMutate: vi.fn().mockResolvedValue({}),
  revokeMutate: vi.fn().mockResolvedValue({}),
}))

vi.mock('@/hooks/use-approval-grants', () => ({
  useApprovalGrants: vi.fn(),
  useGrantApproval: () => ({ isLoading: false, mutateAsync: grantMutate }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: denyMutate }),
  useRevokeApproval: () => ({ isLoading: false, mutateAsync: revokeMutate }),
}))

vi.mock('sonner', () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

import { ApprovalsInbox } from '../governance/approvals-inbox'
import { useApprovalGrants } from '@/hooks/use-approval-grants'

const mockUse = vi.mocked(useApprovalGrants)

function grant(overrides?: Partial<ApprovalGrant>): ApprovalGrant {
  return {
    id: 1,
    workspace_id: 'ws',
    subject_type: 'board_task',
    subject_id: '42',
    tool_name: 'composio_gmail_send',
    risk_tier: 'external_side_effect',
    agent_id: null,
    status: 'pending',
    reason: 'external side-effect requires approval',
    estimated_cost_usd: '1.25',
    requested_at: '2026-07-11T10:00:00Z',
    expires_at: null,
    granted_at: null,
    granted_by: null,
    revoked_at: null,
    revoked_by: null,
    oversight: {
      risk_class: 'external_side_effect',
      tier: 'human_in_the_loop',
      rationale: 'External side-effect: acts on a third-party system. A human must approve before it runs.',
      requires_approval: true,
    },
    ...overrides,
  }
}

function setGrants(grants: ApprovalGrant[]) {
  mockUse.mockReturnValue({ data: { grants }, isLoading: false, isError: false } as any)
}

describe('ApprovalsInbox — PRD-196 S1', () => {
  beforeEach(() => {
    grantMutate.mockClear()
    denyMutate.mockClear()
    revokeMutate.mockClear()
  })

  it('renders a pending grant with oversight tier, rationale, and Grant/Deny actions', () => {
    setGrants([grant()])
    render(<ApprovalsInbox />)
    expect(screen.getByText(/Human approval required/)).toBeInTheDocument()
    expect(screen.getByText(/acts on a third-party system/)).toBeInTheDocument()
    expect(screen.getByRole('note', { name: /human oversight/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Grant/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Deny/ })).toBeInTheDocument()
  })

  it('Grant and Deny call the right mutation with the grant id', () => {
    setGrants([grant({ id: 7 })])
    render(<ApprovalsInbox />)
    fireEvent.click(screen.getByRole('button', { name: /Grant/ }))
    expect(grantMutate).toHaveBeenCalledWith(7)
    fireEvent.click(screen.getByRole('button', { name: /Deny/ }))
    expect(denyMutate).toHaveBeenCalledWith(7)
  })

  it('a granted grant offers Revoke, not Grant/Deny', () => {
    setGrants([grant({ id: 9, status: 'granted' })])
    render(<ApprovalsInbox />)
    expect(screen.getByRole('button', { name: /Revoke/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^Grant$/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^Deny$/ })).not.toBeInTheDocument()
  })

  it('a denied grant is read-only (no action buttons)', () => {
    setGrants([grant({ id: 11, status: 'denied' })])
    render(<ApprovalsInbox />)
    expect(screen.queryByRole('button', { name: /Grant/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Deny/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Revoke/ })).not.toBeInTheDocument()
  })

  it('shows an honest empty state when nothing is pending', () => {
    setGrants([])
    render(<ApprovalsInbox />)
    expect(screen.getByText(/No approvals pending/)).toBeInTheDocument()
  })
})
