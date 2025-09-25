/**
 * Knowledge API hooks
 * Provides React Query integration for knowledge endpoints
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const knowledgeQueryKeys = {
  knowledge: ['knowledge'] as const,
}

// ============= MUTATION HOOKS =============

// Share knowledge
export function useShareKnowledge() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.shareKnowledge(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: knowledgeQueryKeys.knowledge })
      toast.success('Knowledge shared successfully!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to share knowledge: ${error.message}`)
      throw error
    },
  })
}
