/**
 * Performance Monitoring API hooks
 * Provides React Query integration for performance monitoring endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const performanceQueryKeys = {
  baseline: ['performance', 'baseline'] as const,
  agentPerformance: (agentId: string) => ['performance', 'agent', agentId] as const,
  evaluationMetrics: ['performance', 'evaluation-metrics'] as const,
  contextPerformance: ['performance', 'context'] as const,
  analytics: ['performance', 'analytics'] as const,
}

// ============= QUERY HOOKS =============

// Get performance baseline
export function usePerformanceBaseline() {
  return useQuery({
    queryKey: performanceQueryKeys.baseline,
    queryFn: () => apiClient.getSystemPerformanceBaseline(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get agent performance
export function useAgentPerformance(agentId: string | null) {
  return useQuery({
    queryKey: performanceQueryKeys.agentPerformance(agentId!),
    queryFn: () => apiClient.getAgentPerformance(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get evaluation performance metrics
export function useEvaluationPerformanceMetrics() {
  return useQuery({
    queryKey: performanceQueryKeys.evaluationMetrics,
    queryFn: () => apiClient.getEvaluationPerformanceMetrics(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get context performance
export function useContextPerformance() {
  return useQuery({
    queryKey: performanceQueryKeys.contextPerformance,
    queryFn: () => apiClient.getContextPerformance(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get performance analytics
export function usePerformanceAnalytics(timeRange: string = '24h') {
  return useQuery({
    queryKey: ['performance', 'analytics', timeRange],
    queryFn: () => apiClient.getPerformanceAnalytics(timeRange),
    refetchInterval: 60000,
  })
}

// ============= MUTATION HOOKS =============

// Run performance test
export function useRunPerformanceTest() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.runSystemPerformanceTest(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: performanceQueryKeys.baseline })
      queryClient.invalidateQueries({ queryKey: ['performance'] })
      toast.success('Performance test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Performance test failed: ${error.message}`)
      throw error
    },
  })
}

// Get performance comparison
export function usePerformanceComparison() {
  return useMutation({
    mutationFn: (data: any) => apiClient.getSystemPerformanceComparison(data),
    onSuccess: (data) => {
      toast.success('Performance comparison completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Performance comparison failed: ${error.message}`)
      throw error
    },
  })
}
