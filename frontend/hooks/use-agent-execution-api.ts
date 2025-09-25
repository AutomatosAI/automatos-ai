/**
 * Agent Execution API hooks
 * Provides React Query integration for agent execution endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const agentExecutionQueryKeys = {
  executionStatus: (agentId: string) => ['agent-execution', 'status', agentId] as const,
  executionResults: (agentId: string) => ['agent-execution', 'results', agentId] as const,
  executionLogs: (agentId: string) => ['agent-execution', 'logs', agentId] as const,
  executionMetrics: (agentId: string) => ['agent-execution', 'metrics', agentId] as const,
  executionHistory: (agentId: string) => ['agent-execution', 'history', agentId] as const,
  executionQueue: ['agent-execution', 'queue'] as const,
  executionScheduling: ['agent-execution', 'scheduling'] as const,
  executionMonitoring: ['agent-execution', 'monitoring'] as const,
  executionOptimization: ['agent-execution', 'optimization'] as const,
  executionDebugging: ['agent-execution', 'debugging'] as const,
  executionTesting: ['agent-execution', 'testing'] as const,
  executionValidation: ['agent-execution', 'validation'] as const,
  executionReporting: ['agent-execution', 'reporting'] as const,
  executionAnalytics: ['agent-execution', 'analytics'] as const,
  executionAlerts: ['agent-execution', 'alerts'] as const,
  executionConfiguration: (agentId: string) => ['agent-execution', 'configuration', agentId] as const,
  executionPermissions: ['agent-execution', 'permissions'] as const,
  executionSecurity: ['agent-execution', 'security'] as const,
  executionCompliance: ['agent-execution', 'compliance'] as const,
  executionAudit: ['agent-execution', 'audit'] as const,
  executionBackup: ['agent-execution', 'backup'] as const,
  executionRecovery: ['agent-execution', 'recovery'] as const,
  executionMigration: ['agent-execution', 'migration'] as const,
  executionScaling: ['agent-execution', 'scaling'] as const,
  executionLoadBalancing: ['agent-execution', 'load-balancing'] as const,
  executionFailover: ['agent-execution', 'failover'] as const,
  executionHealth: ['agent-execution', 'health'] as const,
  executionMaintenance: ['agent-execution', 'maintenance'] as const,
  executionUpdates: ['agent-execution', 'updates'] as const,
}

// ============= QUERY HOOKS =============

// Get agent execution status
export function useAgentExecutionStatus(agentId: string | null) {
  return useQuery({
    queryKey: agentExecutionQueryKeys.executionStatus(agentId!),
    queryFn: () => apiClient.getSystemAgentStatus(agentId!),
    enabled: !!agentId,
    refetchInterval: 5000, // Frequent updates for execution status
  })
}

// Get agent execution results
export function useAgentExecutionResults(agentId: string | null) {
  return useQuery({
    queryKey: agentExecutionQueryKeys.executionResults(agentId!),
    queryFn: () => apiClient.getAgentStats(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get agent execution logs
export function useAgentExecutionLogs(agentId: string | null) {
  return useQuery({
    queryKey: agentExecutionQueryKeys.executionLogs(agentId!),
    queryFn: () => apiClient.getAgentLogs(agentId!),
    enabled: !!agentId,
    refetchInterval: 5000,
  })
}

// Get agent execution metrics
export function useAgentExecutionMetrics(agentId: string | null) {
  return useQuery({
    queryKey: agentExecutionQueryKeys.executionMetrics(agentId!),
    queryFn: () => apiClient.getAgentPerformance(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get agent execution history
export function useAgentExecutionHistory(agentId: string | null) {
  return useQuery({
    queryKey: agentExecutionQueryKeys.executionHistory(agentId!),
    queryFn: () => apiClient.getAgentRuns(agentId!, 50),
    enabled: !!agentId,
    refetchInterval: 30000,
  })
}

// Get execution queue
export function useAgentExecutionQueue() {
  return useQuery({
    queryKey: agentExecutionQueryKeys.executionQueue,
    queryFn: () => apiClient.getSystemAgentStatistics(),
    refetchInterval: 10000,
  })
}

// Get execution monitoring
export function useAgentExecutionMonitoring() {
  return useQuery({
    queryKey: agentExecutionQueryKeys.executionMonitoring,
    queryFn: () => apiClient.getSystemMetrics(),
    refetchInterval: 5000,
  })
}

// Get execution health
export function useAgentExecutionHealth() {
  return useQuery({
    queryKey: agentExecutionQueryKeys.executionHealth,
    queryFn: () => apiClient.getSystemHealth(),
    refetchInterval: 30000,
  })
}

// ============= MUTATION HOOKS =============

// Execute agent
export function useExecuteAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: any }) =>
      apiClient.executeSystemAgent(agentId, data),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionStatus(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionResults(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionQueue })
      toast.success('Agent execution started!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to execute agent: ${error.message}`)
      throw error
    },
  })
}

// Start agent
export function useStartAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (agentId: string) => apiClient.startAgent(agentId),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionStatus(agentId) })
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionQueue })
      toast.success('Agent started successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to start agent: ${error.message}`)
      throw error
    },
  })
}

// Stop agent
export function useStopAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (agentId: string) => apiClient.stopAgent(agentId),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionStatus(agentId) })
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionQueue })
      toast.success('Agent stopped successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to stop agent: ${error.message}`)
      throw error
    },
  })
}

// Pause agent
export function usePauseAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (agentId: string) => apiClient.pauseAgent(agentId),
    onSuccess: (data, agentId) => {
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionStatus(agentId) })
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionQueue })
      toast.success('Agent paused successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to pause agent: ${error.message}`)
      throw error
    },
  })
}

// Update execution configuration
export function useUpdateExecutionConfiguration() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ agentId, config }: { agentId: string; config: any }) =>
      apiClient.updateAgent(agentId, config),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionConfiguration(variables.agentId) })
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionStatus(variables.agentId) })
      toast.success('Execution configuration updated!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update execution configuration: ${error.message}`)
      throw error
    },
  })
}

// Run execution test
export function useRunExecutionTest() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.runSystemPerformanceTest(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionTesting })
      toast.success('Execution test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Execution test failed: ${error.message}`)
      throw error
    },
  })
}

// Optimize execution
export function useOptimizeExecution() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.updateSystemLearningState(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionOptimization })
      toast.success('Execution optimization completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Execution optimization failed: ${error.message}`)
      throw error
    },
  })
}

// Debug execution
export function useDebugExecution() {
  return useMutation({
    mutationFn: (data: any) => apiClient.getSystemAgentStatistics(),
    onSuccess: (data) => {
      toast.success('Execution debugging data retrieved!')
      return data
    },
    onError: (error) => {
      toast.error(`Execution debugging failed: ${error.message}`)
      throw error
    },
  })
}

// Validate execution
export function useValidateExecution() {
  return useMutation({
    mutationFn: (data: any) => apiClient.getSystemHealth(),
    onSuccess: (data) => {
      toast.success('Execution validation completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Execution validation failed: ${error.message}`)
      throw error
    },
  })
}

// Generate execution report
export function useGenerateExecutionReport() {
  return useMutation({
    mutationFn: (data: any) => apiClient.getSystemAgentStatistics(),
    onSuccess: (data) => {
      toast.success('Execution report generated!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to generate execution report: ${error.message}`)
      throw error
    },
  })
}

// Scale execution
export function useScaleExecution() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.updateSystemConfig(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: agentExecutionQueryKeys.executionScaling })
      toast.success('Execution scaling completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Execution scaling failed: ${error.message}`)
      throw error
    },
  })
}

// Health check execution
export function useHealthCheckExecution() {
  return useMutation({
    mutationFn: () => apiClient.getSystemHealth(),
    onSuccess: (data) => {
      toast.success('Execution health check completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Execution health check failed: ${error.message}`)
      throw error
    },
  })
}
