/**
 * Approval-grants React Query hooks — PRD-196 S1 (P2-15, governance I.1).
 *
 * The front door for the durable approval grants that gate board tasks,
 * playbook runs, and (future) scheduled/webhook agents. Reads
 * `GET /api/v1/approval-grants?status=` and drives the grant / deny / revoke
 * verbs — all ws-admin gated by PRD-196 S2. Same shape as use-missions-api.ts;
 * a decision invalidates the list so the inbox refreshes optimistically.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { ApprovalGrantsResponse } from '@/lib/api-client'

export const approvalGrantsQueryKeys = {
  all: ['approval-grants'] as const,
  list: (status?: string) => ['approval-grants', 'list', status ?? 'all'] as const,
}

export function useApprovalGrants(status?: string) {
  return useQuery<ApprovalGrantsResponse>({
    queryKey: approvalGrantsQueryKeys.list(status),
    queryFn: () => apiClient.listApprovalGrants(status),
    // Pending approvals are time-sensitive; keep them fresh without hammering.
    refetchInterval: 30_000,
  })
}

export function useGrantApproval() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (grantId: number) => apiClient.grantApproval(grantId),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: approvalGrantsQueryKeys.all }),
  })
}

export function useDenyApproval() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (grantId: number) => apiClient.denyApproval(grantId),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: approvalGrantsQueryKeys.all }),
  })
}

export function useRevokeApproval() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (grantId: number) => apiClient.revokeApproval(grantId),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: approvalGrantsQueryKeys.all }),
  })
}
