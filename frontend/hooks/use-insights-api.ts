/**
 * Insights API hooks
 * Provides React Query integration for insights endpoints
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from "@/lib/api-client"

// Query keys for React Query
export const insightsQueryKeys = {
  insights: ['insights'] as const,
}

// ============= MUTATION HOOKS =============

// Extract insights
export function useExtractInsights() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.extractInsights(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: insightsQueryKeys.insights })
      toast.success('Insights extracted successfully!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to extract insights: ${error.message}`)
      throw error
    },
  })
}
