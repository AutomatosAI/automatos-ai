/**
 * Learning API hooks
 * Provides React Query integration for learning endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from "@/lib/api-client"

// Query keys for React Query
export const learningQueryKeys = {
  insights: ['learning', 'insights'] as const,
  effectivenessReport: ['learning', 'effectiveness-report'] as const,
  selfLearningHealth: ['learning', 'self-learning-health'] as const,
}

// PRD-263 US-021: the sections of GET /api/harness/self-learning; one that failed
// to load on the backend is named in `unavailable`.
export type SelfLearningSection = 'harness' | 'signals'

// PRD-263 US-021: shape of GET /api/harness/self-learning. Counts, coverage and
// mode only (any workspace member reads it; never grant text). A figure the
// backend does not have is null, never 0, and the tile shows it as "—".
export interface SelfLearningHealthData {
  workspace_id: string
  /** HARNESS status from the latest run: completed, running, failed, disabled, dormant_insufficient_agents or scheduled_not_run_yet. */
  status: string | null
  iteration_count: number | null
  /** Judged agents over active agents, 0..1, on a completed run. */
  coverage: number | null
  insufficient_data: number | null
  watch: number | null
  regression: number | null
  /** The workspace's HARNESS mode: report_only, propose or disabled. */
  mode: string | null
  /** This workspace's production tool calls over the last 7 days. */
  signals_7d: number | null
  unavailable: SelfLearningSection[]
}

// ============= QUERY HOOKS =============

// Get learning insights
export function useLearningInsights() {
  return useQuery({
    queryKey: learningQueryKeys.insights,
    queryFn: () => apiClient.getLearningInsights(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get learning effectiveness report
export function useLearningEffectivenessReport() {
  return useQuery({
    queryKey: learningQueryKeys.effectivenessReport,
    queryFn: () => apiClient.getLearningEffectivenessReport(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get self-learning health for the Command Center tile (the latest HARNESS run's
// status, coverage and verdict counts, the workspace's mode, and its 7-day
// production tool calls). Uses the generic request path — the endpoint is
// workspace-scoped server-side via the auth context.
export function useSelfLearningHealth() {
  return useQuery<SelfLearningHealthData>({
    queryKey: learningQueryKeys.selfLearningHealth,
    queryFn: () => apiClient.request<SelfLearningHealthData>('/api/harness/self-learning'),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// ============= MUTATION HOOKS =============

// Submit learning feedback
export function useSubmitLearningFeedback() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.submitLearningFeedback(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['learning'] })
      toast.success('Learning feedback submitted successfully!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to submit learning feedback: ${error.message}`)
      throw error
    },
  })
}

// Process learning feedback
export function useProcessLearningFeedback() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.processLearningFeedback(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['learning'] })
      toast.success('Learning feedback processed successfully!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to process learning feedback: ${error.message}`)
      throw error
    },
  })
}
