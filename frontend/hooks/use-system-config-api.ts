/**
 * System Configuration API hooks
 * Provides React Query integration for system configuration endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// Helper function to handle error messages consistently
function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// Query keys for React Query
export const systemConfigQueryKeys = {
  health: ['system', 'health'] as const,
  apiHealth: ['api', 'health'] as const,
  metrics: ['system', 'metrics'] as const,
  config: ['system', 'config'] as const,
  configKey: (key: string) => ['system', 'config', key] as const,
  rag: ['system', 'rag'] as const,
  ragConfig: (id: string) => ['system', 'rag', id] as const,
  agentTypes: ['system', 'agent-types'] as const,
  agentStatistics: ['system', 'agent-statistics'] as const,
  agentStatus: (agentId: string) => ['system', 'agent-status', agentId] as const,
  performanceBaseline: ['system', 'performance-baseline'] as const,
  activities: ['system', 'activities'] as const,
  agentLogs: (agentId: string) => ['system', 'agent-logs', agentId] as const,
  agentStats: (agentId: string) => ['system', 'agent-stats', agentId] as const,
}

// ============= QUERY HOOKS =============

// Get system health
export function useSystemHealth() {
  return useQuery({
    queryKey: systemConfigQueryKeys.health,
    queryFn: () => apiClient.getSystemHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get API endpoint health
export function useApiHealth() {
  return useQuery({
    queryKey: systemConfigQueryKeys.apiHealth,
    queryFn: () => apiClient.getApiHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get system metrics
export function useSystemMetrics() {
  return useQuery({
    queryKey: systemConfigQueryKeys.metrics,
    queryFn: () => apiClient.getSystemMetrics(),
    refetchInterval: 30000,
    staleTime: 15000,
    retry: (failureCount, error: any) => {
      // Only retry 3 times
      return failureCount < 3;
    },
    onError: (error: any) => {
      console.error('Failed to fetch system metrics:', error);
      // Toast notification for user is handled in API client
    }
  })
}

// Get system config
export function useSystemConfig() {
  return useQuery({
    queryKey: systemConfigQueryKeys.config,
    queryFn: () => apiClient.getSystemConfig(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get system config key
export function useSystemConfigKey(key: string | null) {
  return useQuery({
    queryKey: systemConfigQueryKeys.configKey(key!),
    queryFn: () => apiClient.getSystemConfigKey(key!),
    enabled: !!key,
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get system RAG
export function useSystemRAG() {
  return useQuery({
    queryKey: systemConfigQueryKeys.rag,
    queryFn: () => apiClient.getSystemRAG(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get system RAG config
export function useSystemRAGConfig(id: string | null) {
  return useQuery({
    queryKey: systemConfigQueryKeys.ragConfig(id!),
    queryFn: () => apiClient.getSystemRAGConfig(id!),
    enabled: !!id,
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get system agent types
export function useSystemAgentTypes() {
  return useQuery({
    queryKey: systemConfigQueryKeys.agentTypes,
    queryFn: () => apiClient.getSystemAgentTypes(),
    staleTime: 5 * 60 * 1000, // Agent types don't change often
  })
}

// Get system agent statistics
export function useSystemAgentStatistics() {
  return useQuery({
    queryKey: systemConfigQueryKeys.agentStatistics,
    queryFn: () => apiClient.getSystemAgentStatistics(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get system agent status
export function useSystemAgentStatus(agentId: string | null) {
  return useQuery({
    queryKey: systemConfigQueryKeys.agentStatus(agentId!),
    queryFn: () => apiClient.getSystemAgentStatus(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get system performance baseline
export function useSystemPerformanceBaseline() {
  return useQuery({
    queryKey: systemConfigQueryKeys.performanceBaseline,
    queryFn: () => apiClient.getSystemPerformanceBaseline(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get system activities
export function useSystemActivities(limit: number = 10) {
  return useQuery({
    queryKey: ['system', 'activities', limit],
    queryFn: () => apiClient.getSystemActivities(limit),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get system agent logs
export function useSystemAgentLogs(agentId: string | null) {
  return useQuery({
    queryKey: systemConfigQueryKeys.agentLogs(agentId!),
    queryFn: () => apiClient.getAgentLogs(agentId!),
    enabled: !!agentId,
    refetchInterval: 5000,
  })
}

// Get system agent stats
export function useSystemAgentStats(agentId: string | null) {
  return useQuery({
    queryKey: systemConfigQueryKeys.agentStats(agentId!),
    queryFn: () => apiClient.getAgentStats(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// ============= MUTATION HOOKS =============

// Update system config
export function useUpdateSystemConfig() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.updateSystemConfig(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.config })
      toast.success('System configuration updated!')
      return data
    },
    onError: (error: unknown) => {
      const errorMessage = error instanceof Error ? error.message : String(error);
      toast.error(`Failed to update system config: ${errorMessage}`)
      throw error
    },
  })
}

// Update system config key
export function useUpdateSystemConfigKey() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ key, value }: { key: string; value: any }) =>
      apiClient.updateSystemConfigKey(key, value),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.config })
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.configKey(variables.key) })
      toast.success('System configuration key updated!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update system config key: ${error.message}`)
      throw error
    },
  })
}

// Update system RAG
export function useUpdateSystemRAG() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.updateSystemRAG(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.rag })
      toast.success('System RAG configuration updated!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update system RAG: ${error.message}`)
      throw error
    },
  })
}

// Test system RAG
export function useTestSystemRAG() {
  return useMutation({
    mutationFn: (id: string) => apiClient.testSystemRAG(id),
    onSuccess: (data) => {
      toast.success('System RAG test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`System RAG test failed: ${error.message}`)
      throw error
    },
  })
}

// Test system route
export function useTestSystemRoute() {
  return useMutation({
    mutationFn: () => apiClient.testSystemRoute(),
    onSuccess: (data) => {
      toast.success('System route test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`System route test failed: ${error.message}`)
      throw error
    },
  })
}

// Execute system agent
export function useExecuteSystemAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: any }) =>
      apiClient.executeSystemAgent(agentId, data),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.agentStatus(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.agentStatistics })
      toast.success('System agent executed successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to execute system agent: ${error.message}`)
      throw error
    },
  })
}

// Update system learning state
export function useUpdateSystemLearningState() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.updateSystemLearningState(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['system'] })
      toast.success('System learning state updated!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update system learning state: ${error.message}`)
      throw error
    },
  })
}

// Run system performance test
export function useRunSystemPerformanceTest() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.runSystemPerformanceTest(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.performanceBaseline })
      toast.success('System performance test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`System performance test failed: ${error.message}`)
      throw error
    },
  })
}

// Get system performance comparison
export function useSystemPerformanceComparison() {
  return useMutation({
    mutationFn: (data: any) => apiClient.getSystemPerformanceComparison(data),
    onSuccess: (data) => {
      toast.success('System performance comparison completed!')
      return data
    },
    onError: (error) => {
      toast.error(`System performance comparison failed: ${error.message}`)
      throw error
    },
  })
}

// Save system config
export function useSaveSystemConfig() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ configKey, configValue, description }: { configKey: string; configValue: any; description?: string }) =>
      apiClient.saveSystemConfig(configKey, configValue, description),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.config })
      toast.success('System configuration saved!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to save system config: ${error.message}`)
      throw error
    },
  })
}

// Start system agent
export function useStartSystemAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (agentId: string) => apiClient.startAgent(agentId),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.agentStatus(agentId) })
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.agentStatistics })
      toast.success('System agent started!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to start system agent: ${error.message}`)
      throw error
    },
  })
}

// Stop system agent
export function useStopSystemAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (agentId: string) => apiClient.stopAgent(agentId),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.agentStatus(agentId) })
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.agentStatistics })
      toast.success('System agent stopped!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to stop system agent: ${error.message}`)
      throw error
    },
  })
}

// Pause system agent
export function usePauseSystemAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (agentId: string) => apiClient.pauseAgent(agentId),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.agentStatus(agentId) })
      queryClient.invalidateQueries({ queryKey: systemConfigQueryKeys.agentStatistics })
      toast.success('System agent paused!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to pause system agent: ${error.message}`)
      throw error
    },
  })
}
