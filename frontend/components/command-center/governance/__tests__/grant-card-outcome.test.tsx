/**
 * P256-FIX-N17-2 (F444, night 17): a click whose call failed says so on the card.
 *
 * The /grant route answers 200 whenever the yes is recorded; the resumed call's result
 * rides on `grant.details.executed_result`. Night 17's #2827 was granted, the board's
 * Reject refused the send-back ("Cannot reject a task with no assigned agent"), and the
 * Governance card (reused inside the ticket, ticket-approvals.tsx) toasted "granted".
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import type { ApprovalGrant } from '@/lib/api-client'

const { grantMutate, denyMutate, revokeMutate, toast } = vi.hoisted(() => ({
  grantMutate: vi.fn(),
  denyMutate: vi.fn(),
  revokeMutate: vi.fn(),
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

vi.mock('@/hooks/use-approval-grants', () => ({
  useApprovalGrants: vi.fn(),
  useGrantApproval: () => ({ isLoading: false, mutateAsync: grantMutate }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: denyMutate }),
  useRevokeApproval: () => ({ isLoading: false, mutateAsync: revokeMutate }),
}))
vi.mock('@/hooks/use-missions-api', () => ({
  useMissions: () => ({ data: { missions: [] }, isLoading: false, isError: false }),
  useApproveMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useRejectMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useUpdateMissionPlan: () => ({ isLoading: false, mutateAsync: vi.fn() }),
}))
vi.mock('sonner', () => ({ toast }))

import { GrantCard } from '../approvals-inbox'

const NO_AGENT = 'Cannot reject a task with no assigned agent'

function grant(overrides?: Partial<ApprovalGrant>): ApprovalGrant {
  return {
    id: 2827,
    workspace_id: 'ws',
    subject_type: 'tool_call',
    subject_id: 'platform_update_task_status:abc',
    tool_name: 'platform_update_task_status',
    risk_tier: null,
    agent_id: null,
    status: 'pending',
    reason: 'Waiting for the owner\'s click: send #1767 back to its agent with your note',
    estimated_cost_usd: null,
    requested_at: '2026-10-16T00:22:10Z',
    expires_at: null,
    granted_at: null,
    granted_by: null,
    revoked_at: null,
    revoked_by: null,
    ...overrides,
  }
}

function granted(executed_result: Record<string, unknown>) {
  return { grant: { ...grant({ status: 'granted' }), details: { executed_result } } }
}

describe('GrantCard — what the click ran', () => {
  beforeEach(() => {
    for (const mock of [grantMutate, denyMutate, revokeMutate, toast.success, toast.error]) mock.mockReset()
  })

  it("night 17's #2827: a granted click whose call was refused toasts the failure, not 'granted'", async () => {
    grantMutate.mockResolvedValue(granted({ success: false, error: NO_AGENT }))
    render(<GrantCard grant={grant()} />)

    fireEvent.click(screen.getByRole('button', { name: /Grant/ }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(`Approved, but execution failed: ${NO_AGENT}`))
    expect(toast.success).not.toHaveBeenCalled()
  })

  it('a granted click whose call ran toasts success', async () => {
    grantMutate.mockResolvedValue(granted({ success: true }))
    render(<GrantCard grant={grant()} />)

    fireEvent.click(screen.getByRole('button', { name: /Grant/ }))

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Grant 2827 granted'))
    expect(toast.error).not.toHaveBeenCalled()
  })

  it('a revoke is never read as a failed run, whatever the grant ran before', async () => {
    revokeMutate.mockResolvedValue(granted({ success: false, error: NO_AGENT }))
    render(<GrantCard grant={grant({ status: 'granted' })} />)

    fireEvent.click(screen.getByRole('button', { name: /Revoke/ }))

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Grant 2827 revoked'))
    expect(toast.error).not.toHaveBeenCalled()
  })
})
