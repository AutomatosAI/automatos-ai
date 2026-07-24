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

// PRD-142 Wave 4 (W4-S16): shape of GET /api/harness/self-learning. Each section
// is best-effort on the backend — a section that failed to load arrives as
// `{ error }`, so every field is optional and an `error` key may be present.
export interface SelfLearningHealthData {
  workspace_id: string
  harness: { status?: string; iteration?: number; error?: string; [key: string]: unknown }
  tool_routing: {
    recorded?: number
    dropped?: number
    flushes?: number
    flush_errors?: number
    queue_depth?: number
    error?: string
    [key: string]: unknown
  }
  prescriptions: {
    applied?: number
    queued?: number
    proposed?: number
    rejected?: number
    rolled_back?: number
    error?: string
    [key: string]: number | string | undefined
  }
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

// Get self-learning health for the Command Center tile (HARNESS loop status,
// tool-routing recorder stats, prescription summary). Uses the generic request
// path — the endpoint is workspace-scoped server-side via the auth context.
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
