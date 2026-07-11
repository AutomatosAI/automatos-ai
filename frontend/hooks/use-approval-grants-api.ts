/**
 * Approval-grant React Query hooks — PRD-193 S3 (P2-12).
 *
 * First frontend consumer of the PRD-181 approval-grant API. Shape mirrors
 * use-missions-api.ts (useApproveMission): a POST mutation per decision,
 * invalidating the grants list so the P2-15 Command Center approvals queue
 * stays fresh once it lands.
 *
 * No new backend routes — /api/v1/approval-grants shipped with PRD-181.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// ── Query Keys ────────────────────────────────────────────────

export const approvalGrantQueryKeys = {
  all: ['approval-grants'] as const,
}

// ── Types (PRD-181 ApprovalGrant.to_dict shape) ───────────────

export interface ApprovalGrantRecord {
  id: number
  workspace_id: string | null
  subject_type: string
  subject_id: string
  tool_name: string | null
  risk_tier: string | null
  agent_id: number | null
  status: 'pending' | 'granted' | 'revoked' | 'expired' | 'denied' | string
  reason: string | null
  requested_at: string | null
  expires_at: string | null
  granted_at: string | null
  granted_by: string | null
  revoked_at: string | null
  revoked_by: string | null
  /** PRD-193 S4: params snapshot + executed_result land here on resume. */
  details?: Record<string, unknown>
}

export interface ApprovalGrantResponse {
  grant: ApprovalGrantRecord
}

// ── Grant (approve) ───────────────────────────────────────────

export function useGrantApproval() {
  const queryClient = useQueryClient()

  return useMutation<ApprovalGrantResponse, Error, { id: number }>({
    mutationFn: ({ id }) =>
      apiClient.request<ApprovalGrantResponse>(`/api/v1/approval-grants/${id}/grant`, {
        method: 'POST',
        body: {} as unknown as BodyInit,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: approvalGrantQueryKeys.all })
    },
  })
}

// ── Deny ──────────────────────────────────────────────────────

export function useDenyApproval() {
  const queryClient = useQueryClient()

  return useMutation<ApprovalGrantResponse, Error, { id: number }>({
    mutationFn: ({ id }) =>
      apiClient.request<ApprovalGrantResponse>(`/api/v1/approval-grants/${id}/deny`, {
        method: 'POST',
        body: {} as unknown as BodyInit,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: approvalGrantQueryKeys.all })
    },
  })
}
