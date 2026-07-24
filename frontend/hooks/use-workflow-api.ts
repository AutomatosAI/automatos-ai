/**
 * Workflow API hooks
 * Provides React Query integration for all workflow-related endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const workflowQueryKeys = {
  all: ['workflows'] as const,
  detail: (id: string) => ['workflows', id] as const,
  active: ['workflows', 'active'] as const,
  stats: ['workflows', 'stats'] as const,
  statsDashboard: ['workflows', 'stats', 'dashboard'] as const,
  liveProgress: (id: string) => ['workflows', id, 'live-progress'] as const,
  executions: ['workflows', 'executions'] as const,
  execution: (id: string) => ['workflows', 'executions', id] as const,
  templates: ['workflows', 'templates'] as const,
  analytics: ['workflows', 'analytics'] as const,
}

// ============= QUERY HOOKS =============

// Get all workflows
export function useWorkflows() {
  return useQuery({
    queryKey: workflowQueryKeys.all,
    queryFn: () => apiClient.getWorkflows(),
    staleTime: 30000, // 30 seconds
  })
}

// Get single workflow
export function useWorkflow(workflowId: string | null) {
  return useQuery({
    queryKey: workflowQueryKeys.detail(workflowId!),
    queryFn: () => apiClient.getWorkflow(workflowId!),
    enabled: !!workflowId,
    staleTime: 30000,
  })
}

// Get active workflows - auto-refreshes every 5 seconds
export function useActiveWorkflows() {
  return useQuery({
    queryKey: workflowQueryKeys.active,
    queryFn: () => apiClient.getActiveWorkflows(),
    refetchInterval: 5000, // Auto-refresh every 5 seconds
    staleTime: 4000,
  })
}

// Get workflow stats
export function useWorkflowStats() {
  return useQuery({
    queryKey: workflowQueryKeys.stats,
    queryFn: () => apiClient.getWorkflowStats(),
    staleTime: 60000, // 1 minute
  })
}

// Get workflow stats dashboard
export function useWorkflowStatsDashboard() {
  return useQuery({
    queryKey: workflowQueryKeys.statsDashboard,
    queryFn: () => apiClient.getWorkflowStatsDashboard(),
    staleTime: 30000,
  })
}

// Get live progress for a workflow
export function useWorkflowLiveProgress(workflowId: string | null) {
  return useQuery({
    queryKey: workflowQueryKeys.liveProgress(workflowId!),
    queryFn: () => apiClient.getWorkflowLiveProgress(workflowId!),
    enabled: !!workflowId,
    refetchInterval: 2000, // Auto-refresh every 2 seconds for live data
    staleTime: 1000,
  })
}

// Get all workflow executions
export function useWorkflowExecutions() {
  return useQuery({
    queryKey: workflowQueryKeys.executions,
    queryFn: () => apiClient.getWorkflowExecutions(),
    staleTime: 30000,
  })
}

// Get single workflow execution
export function useWorkflowExecution(executionId: string | null) {
  return useQuery({
    queryKey: workflowQueryKeys.execution(executionId!),
    queryFn: () => apiClient.getWorkflowExecution(executionId!),
    enabled: !!executionId,
    staleTime: 30000,
  })
}

// Get workflow templates
export function useWorkflowTemplates() {
  return useQuery({
    queryKey: workflowQueryKeys.templates,
    queryFn: () => apiClient.getWorkflowTemplates(),
    staleTime: 300000, // 5 minutes - templates don't change often
  })
}

// Get workflow analytics
export function useWorkflowAnalytics() {
  return useQuery({
    queryKey: workflowQueryKeys.analytics,
    queryFn: () => apiClient.getWorkflowAnalytics(),
    staleTime: 60000,
  })
}

// ============= MUTATION HOOKS =============

// Create workflow
export function useCreateWorkflow() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.createWorkflow(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.all })
      toast.success('Workflow created successfully')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to create workflow')
    }
  })
}

// Update workflow
export function useUpdateWorkflow() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ id, data }: { id: string, data: any }) => 
      apiClient.updateWorkflow(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.detail(id) })
      toast.success('Workflow updated successfully')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to update workflow')
    }
  })
}

// Delete workflow
export function useDeleteWorkflow() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteWorkflow(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.all })
      toast.success('Workflow deleted successfully')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to delete workflow')
    }
  })
}

// Execute workflow
export function useExecuteWorkflow() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ id, data }: { id: string, data?: any }) => 
      apiClient.executeWorkflow(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.executions })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.liveProgress(id) })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.active })
      toast.success('Workflow execution started')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to execute workflow')
    }
  })
}

// Execute workflow with advanced options
export function useExecuteWorkflowAdvanced() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ workflowId, options }: { workflowId: string, options: any }) => 
      apiClient.executeWorkflowAdvanced(workflowId, options),
    onSuccess: (_, { workflowId }) => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.active })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.executions })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.liveProgress(workflowId) })
      toast.success('Advanced workflow execution started')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to execute workflow')
    }
  })
}

// Deploy workflow
export function useDeployWorkflow() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ id, version }: { id: string, version?: string }) => 
      apiClient.deployWorkflow(id, version),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.detail(id) })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.all })
      toast.success('Workflow deployed successfully')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to deploy workflow')
    }
  })
}

// Stop workflow execution
export function useStopWorkflowExecution() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (executionId: string) => apiClient.stopWorkflowExecution(executionId),
    onSuccess: (_, executionId) => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.execution(executionId) })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.active })
      toast.success('Workflow execution stopped')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to stop execution')
    }
  })
}

// Pause workflow execution
export function usePauseWorkflowExecution() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (executionId: string) => apiClient.pauseWorkflowExecution(executionId),
    onSuccess: (_, executionId) => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.execution(executionId) })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.active })
      toast.success('Workflow execution paused')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to pause execution')
    }
  })
}

// Resume workflow execution
export function useResumeWorkflowExecution() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (executionId: string) => apiClient.resumeWorkflowExecution(executionId),
    onSuccess: (_, executionId) => {
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.execution(executionId) })
      queryClient.invalidateQueries({ queryKey: workflowQueryKeys.active })
      toast.success('Workflow execution resumed')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to resume execution')
    }
  })
}
