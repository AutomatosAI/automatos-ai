/**
 * Recommendations API hooks
 * Provides React Query integration for recommendations endpoints
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from "@/lib/api-client"

// Query keys for React Query
export const recommendationsQueryKeys = {
  recommendations: ['recommendations'] as const,
}

// ============= MUTATION HOOKS =============

// Generate recommendations
export function useGenerateRecommendations() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.generateRecommendations(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: recommendationsQueryKeys.recommendations })
      toast.success('Recommendations generated successfully!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to generate recommendations: ${error.message}`)
      throw error
    },
  })
}
