/**
 * Coordination API hooks
 * Provides React Query integration for coordination endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from "@/lib/api-client"

// Query keys for React Query
export const coordinationQueryKeys = {
  coordination: ['coordination'] as const,
  coordinationStatus: ['coordination', 'status'] as const,
}

// ============= QUERY HOOKS =============

// Get coordination status
export function useCoordinationStatus() {
  return useQuery({
    queryKey: coordinationQueryKeys.coordinationStatus,
    queryFn: () => apiClient.getAgentCoordination(),
    refetchInterval: 15000,
    staleTime: 10000,
  })
}

// ============= MUTATION HOOKS =============

// Coordinate agents
export function useCoordinateAgents() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.coordinateAgents(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: coordinationQueryKeys.coordinationStatus })
      toast.success('Agent coordination initiated!')
      return data
    },
    onError: (error) => {
      toast.error(`Coordination failed: ${error.message}`)
      throw error
    },
  })
}
