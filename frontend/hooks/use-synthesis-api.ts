/**
 * Synthesis API hooks
 * Provides React Query integration for synthesis endpoints
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from "@/lib/api-client'

// Query keys for React Query
export const synthesisQueryKeys = {
  synthesis: ['synthesis'] as const,
}

// ============= MUTATION HOOKS =============

// Synthesize comprehensive
export function useSynthesizeComprehensive() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.synthesizeComprehensive(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: synthesisQueryKeys.synthesis })
      toast.success('Comprehensive synthesis completed!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to synthesize: ${error.message}`)
      throw error
    },
  })
}
