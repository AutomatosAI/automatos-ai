/**
 * Approval-grants React Query hooks — PRD-196 S1 (P2-15, governance I.1).
 *
 * The single front door for the durable approval grants that gate board tasks,
 * playbook runs, and (future) scheduled/webhook agents — AND, since PRD-225, the
 * question-kind asks. Reads `GET /api/v1/approval-grants?status=&kind=` and
 * drives the grant / deny / revoke / answer verbs — all ws-admin gated by
 * PRD-196 S2. A decision invalidates the list so the surface refreshes
 * optimistically.
 *
 * PRD-225 consolidation (house rule: no duplicate hooks): the former
 * `use-approval-grants-api.ts` (a second useGrantApproval/useDenyApproval pair)
 * was merged into this module and deleted; ToolApprovalWidget now imports here.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { ApprovalGrant, ApprovalGrantsResponse } from '@/lib/api-client'

// The single ApprovalGrant record + decision-response types (moved here from the
// deleted use-approval-grants-api.ts so its consumers keep one import surface).
export type ApprovalGrantRecord = ApprovalGrant
export interface ApprovalGrantResponse {
  grant: ApprovalGrant
}

export const approvalGrantsQueryKeys = {
  all: ['approval-grants'] as const,
  list: (status?: string, kind?: string) =>
    ['approval-grants', 'list', status ?? 'all', kind ?? 'all'] as const,
}

export function useApprovalGrants(status?: string, kind?: string) {
  return useQuery<ApprovalGrantsResponse>({
    queryKey: approvalGrantsQueryKeys.list(status, kind),
    queryFn: () => apiClient.listApprovalGrants(status, kind),
    // Pending approvals are time-sensitive; keep them fresh without hammering.
    refetchInterval: 30_000,
  })
}

// PRD-225: the Questions tab — open (pending) question-kind asks, newest first.
export function useQuestions() {
  return useApprovalGrants('pending', 'question')
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

// PRD-225: answer a question (free text and/or a chosen option) — resumes the
// parked subject server-side and refreshes the Questions list.
export function useAnswerQuestion() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (vars: { grantId: number; answer_text?: string; option?: string }) =>
      apiClient.answerQuestion(vars.grantId, {
        answer_text: vars.answer_text,
        option: vars.option,
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: approvalGrantsQueryKeys.all }),
  })
}
