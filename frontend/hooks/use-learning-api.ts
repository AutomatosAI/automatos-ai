/**
 * Learning API hooks
 * Provides React Query integration for learning endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from "@/lib/api-client"

// Query keys for React Query
export const learningQueryKeys = {
  insights: ['learning', 'insights'] as const,
  effectivenessReport: ['learning', 'effectiveness-report'] as const,
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
