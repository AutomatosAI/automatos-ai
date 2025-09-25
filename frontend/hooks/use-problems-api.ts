/**
 * Problems API hooks
 * Provides React Query integration for problems endpoints
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const problemsQueryKeys = {
  problems: ['problems'] as const,
}

// ============= MUTATION HOOKS =============

// Submit problem
export function useSubmitProblem() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.submitProblem(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: problemsQueryKeys.problems })
      toast.success('Problem submitted successfully!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to submit problem: ${error.message}`)
      throw error
    },
  })
}

// Analyze problem
export function useAnalyzeProblem() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.analyzeProblem(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: problemsQueryKeys.problems })
      toast.success('Problem analysis completed!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to analyze problem: ${error.message}`)
      throw error
    },
  })
}
