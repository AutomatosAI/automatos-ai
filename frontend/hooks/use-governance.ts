/**
 * Governance React Query hooks — PRD-196 S3/S4 (P2-15).
 *
 * Reads the ws-admin-gated governance surface: the on/off status (policy plane,
 * grants, verdicts, retention) that feeds the is-it-working tile, and the
 * filterable audit-log view. Policy + budget hooks (S4) join this module.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type {
  GovernanceStatus,
  AuditLogResponse,
  AuditLogFilters,
  PolicyDocument,
  BudgetConfig,
} from '@/lib/api-client'

export const governanceQueryKeys = {
  status: ['governance', 'status'] as const,
  auditLog: (filters: AuditLogFilters) => ['governance', 'audit-log', filters] as const,
  policy: ['governance', 'policy'] as const,
  budget: ['governance', 'budget'] as const,
}

export function useGovernanceStatus() {
  return useQuery<GovernanceStatus>({
    queryKey: governanceQueryKeys.status,
    queryFn: () => apiClient.getGovernanceStatus(),
    // Fail quietly for non-admins (403) — the tile/pane handle the absence.
    retry: false,
    refetchInterval: 60_000,
  })
}

export function useGovernanceAuditLog(filters: AuditLogFilters = {}) {
  return useQuery<AuditLogResponse>({
    queryKey: governanceQueryKeys.auditLog(filters),
    queryFn: () => apiClient.getGovernanceAuditLog(filters),
    retry: false,
    keepPreviousData: true,
  })
}

export function usePolicy() {
  return useQuery<PolicyDocument>({
    queryKey: governanceQueryKeys.policy,
    queryFn: () => apiClient.getGovernancePolicy(),
    retry: false,
  })
}

export function useUpdatePolicy() {
  const queryClient = useQueryClient()
  return useMutation<PolicyDocument, Error, Partial<PolicyDocument>>({
    mutationFn: (body) => apiClient.putGovernancePolicy(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: governanceQueryKeys.policy })
      queryClient.invalidateQueries({ queryKey: governanceQueryKeys.status })
    },
  })
}

export function useBudget() {
  return useQuery<BudgetConfig>({
    queryKey: governanceQueryKeys.budget,
    queryFn: () => apiClient.getGovernanceBudget(),
    retry: false,
  })
}

export function useUpdateBudget() {
  const queryClient = useQueryClient()
  return useMutation<BudgetConfig, Error, BudgetConfig>({
    mutationFn: (body) => apiClient.putGovernanceBudget(body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: governanceQueryKeys.budget }),
  })
}
