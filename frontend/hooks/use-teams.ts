/**
 * PRD-158 — Teams hooks (React Query).
 *
 * One source for the workspace team list, team creation, per-team document
 * counts, and team-access edits. Replaces free-text team entry everywhere.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import apiClient from '../lib/api-client'

export interface Team {
  id: number
  name: string
  normalized_name: string
}

export const teamQueryKeys = {
  teams: ['teams'] as const,
  documentTeamCounts: ['documents', 'team-counts'] as const,
}

/** All teams in the workspace. */
export function useTeams() {
  return useQuery({
    queryKey: teamQueryKeys.teams,
    queryFn: () => apiClient.getTeams(),
    staleTime: 60000,
  })
}

/** Create a team (idempotent server-side by normalized name). */
export function useCreateTeam() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiClient.createTeam(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: teamQueryKeys.teams })
    },
    onError: () => toast.error('Failed to create team'),
  })
}

/** Per-team document counts, aggregated server-side. */
export function useDocumentTeamCounts() {
  return useQuery({
    queryKey: teamQueryKeys.documentTeamCounts,
    queryFn: () => apiClient.getDocumentTeamCounts(),
    staleTime: 30000,
  })
}

/** Edit a single document's team_access. */
export function useUpdateDocumentTeamAccess() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: { id: number; teamAccess: string[] }) =>
      apiClient.updateDocumentTeamAccess(data.id, data.teamAccess),
    onSuccess: () => {
      toast.success('Team access updated')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: teamQueryKeys.documentTeamCounts })
    },
    onError: () => toast.error('Failed to update team access'),
  })
}

/** Bulk-assign team_access across several documents. */
export function useBulkUpdateTeamAccess() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: { ids: number[]; teamAccess: string[] }) =>
      apiClient.bulkUpdateTeamAccess(data.ids, data.teamAccess),
    onSuccess: () => {
      toast.success('Team access updated for selected documents')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: teamQueryKeys.documentTeamCounts })
    },
    onError: () => toast.error('Failed to update team access'),
  })
}
