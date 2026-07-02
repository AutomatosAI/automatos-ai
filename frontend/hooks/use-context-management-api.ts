/**
 * Context Management API hooks
 * Provides React Query integration for context management endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const contextManagementQueryKeys = {
  contextStats: ['context', 'stats'] as const,
  contextPerformance: ['context', 'performance'] as const,
  contextSources: ['context', 'sources'] as const,
  recentQueries: ['context', 'queries', 'recent'] as const,
  contextPatterns: ['context', 'patterns'] as const,
  contextSystemHealth: ['context', 'system-health'] as const,
}

// ============= QUERY HOOKS =============

// Get context stats
export function useContextStats() {
  return useQuery({
    queryKey: contextManagementQueryKeys.contextStats,
    queryFn: () => apiClient.getContextStats(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get context performance
export function useContextPerformance(timeRange: string = '24h') {
  return useQuery({
    queryKey: [...contextManagementQueryKeys.contextPerformance, timeRange],
    queryFn: () => apiClient.getContextPerformance(timeRange),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get context sources
export function useContextSources() {
  return useQuery({
    queryKey: contextManagementQueryKeys.contextSources,
    queryFn: () => apiClient.getContextSources(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get recent context queries
export function useRecentContextQueries() {
  return useQuery({
    queryKey: contextManagementQueryKeys.recentQueries,
    queryFn: () => apiClient.getRecentContextQueries(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get context patterns
export function useContextPatterns() {
  return useQuery({
    queryKey: contextManagementQueryKeys.contextPatterns,
    queryFn: () => apiClient.getContextPatterns(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get context system health
export function useContextSystemHealth() {
  return useQuery({
    queryKey: contextManagementQueryKeys.contextSystemHealth,
    queryFn: () => apiClient.getContextSystemHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// ============= MUTATION HOOKS =============

// Initialize context
export function useInitializeContext() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.initializeContext(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['context'] })
      toast.success('Context initialized successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to initialize context: ${error.message}`)
      throw error
    },
  })
}

// Test context RAG
export function useTestContextRAG() {
  return useMutation({
    mutationFn: (configId: string) => apiClient.testContextRAG(configId),
    onSuccess: (data) => {
      toast.success('Context RAG test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Context RAG test failed: ${error.message}`)
      throw error
    },
  })
}

// Update context performance
export function useUpdateContextPerformance() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.updateSystemLearningState(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: contextManagementQueryKeys.contextPerformance })
      toast.success('Context performance updated!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update context performance: ${error.message}`)
      throw error
    },
  })
}

// Get optimization recommendations (query)
export function useOptimizationRecommendations() {
  return useQuery({
    queryKey: ['context', 'optimization'],
    queryFn: () => apiClient.getOptimizationRecommendations(),
    staleTime: 60000, // Cache for 1 minute
    refetchInterval: false, // Don't auto-refetch
  })
}

// Trigger optimization analysis (mutation for manual refresh)
export function useOptimizeContext() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: () => apiClient.getOptimizationRecommendations(),
    onSuccess: (data) => {
      queryClient.setQueryData(['context', 'optimization'], data)
      toast.success('Optimization analysis complete!')
      return data
    },
    onError: (error) => {
      toast.error(`Context optimization failed: ${error.message}`)
      throw error
    },
  })
}

// Clear context cache
export function useClearContextCache() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: () => Promise.resolve(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['context'] })
      toast.success('Context cache cleared!')
    },
    onError: (error) => {
      toast.error(`Failed to clear context cache: ${error.message}`)
      throw error
    },
  })
}

// Refresh context data
export function useRefreshContextData() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: () => Promise.resolve(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['context'] })
      toast.success('Context data refreshed!')
    },
    onError: (error) => {
      toast.error(`Failed to refresh context data: ${error.message}`)
      throw error
    },
  })
}
