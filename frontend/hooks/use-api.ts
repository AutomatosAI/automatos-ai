
/**
 * Enhanced API hooks for real-time data fetching
 * Provides React Query integration with automatic caching, retries, and real-time updates
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '../lib/api-client'
import { toast } from 'react-hot-toast'

// Query keys for React Query
export const queryKeys = {
  // System
  systemHealth: ['system', 'health'],
  systemMetrics: ['system', 'metrics'],
  systemActivities: ['system', 'activities'],
  
  // Agents
  agents: ['agents'],
  agent: (id: string) => ['agents', id],
  agentLogs: (id: string) => ['agents', id, 'logs'],
  agentStats: (id: string) => ['agents', id, 'stats'],
  
  // Documents
  documents: ['documents'],
  document: (id: string) => ['documents', id],
  documentAnalytics: ['documents', 'analytics'],
  documentProcessing: (id: string) => ['documents', id, 'processing'],
  
  // Workflows
  workflows: ['workflows'],
  workflow: (id: string) => ['workflows', id],
  workflowStatus: (id: string) => ['workflows', id, 'status'],
  workflowLogs: (id: string) => ['workflows', id, 'logs'],
  workflowTemplates: ['workflows', 'templates'],
  
  // Analytics
  performanceAnalytics: (timeRange: string) => ['analytics', 'performance', timeRange],
  usageAnalytics: (timeRange: string) => ['analytics', 'usage', timeRange],
  agentAnalytics: (timeRange: string) => ['analytics', 'agents', timeRange],
  documentAnalyticsTime: (timeRange: string) => ['analytics', 'documents', timeRange],
  workflowAnalytics: (timeRange: string) => ['analytics', 'workflows', timeRange],
  errorAnalytics: (timeRange: string) => ['analytics', 'errors', timeRange],
  
  // Settings
  settings: ['settings'],
  userPreferences: ['settings', 'user'],
  systemLogs: ['settings', 'logs'],
}

// System hooks
export function useSystemHealth() {
  return useQuery({
    queryKey: queryKeys.systemHealth,
    queryFn: () => apiClient.getSystemHealth(),
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // Consider data stale after 15 seconds
  })
}

export function useSystemMetrics() {
  return useQuery({
    queryKey: queryKeys.systemMetrics,
    queryFn: () => apiClient.getSystemMetrics(),
    refetchInterval: 10000, // Refetch every 10 seconds
    staleTime: 5000,
  })
}

export function useSystemActivities() {
  return useQuery({
    queryKey: queryKeys.systemActivities,
    queryFn: () => apiClient.getSystemActivities(),
    refetchInterval: 15000, // Refetch every 15 seconds
  })
}

// Agent hooks
export function useAgents() {
  return useQuery({
    queryKey: queryKeys.agents,
    queryFn: () => apiClient.getAgents(),
    staleTime: 30000,
  })
}

export function useAgent(id: string) {
  return useQuery({
    queryKey: queryKeys.agent(id),
    queryFn: () => apiClient.getAgent(id),
    enabled: !!id,
  })
}

export function useAgentLogs(id: string) {
  return useQuery({
    queryKey: queryKeys.agentLogs(id),
    queryFn: () => apiClient.getAgentLogs(id),
    enabled: !!id,
    refetchInterval: 5000, // Real-time logs
  })
}

export function useAgentStats(id: string) {
  return useQuery({
    queryKey: queryKeys.agentStats(id),
    queryFn: () => apiClient.getAgentStats(id),
    enabled: !!id,
    refetchInterval: 10000,
  })
}

export function useCreateAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.createAgent(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents })
      toast.success('Agent created successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to create agent: ${error.message}`)
    },
  })
}

export function useUpdateAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) => 
      apiClient.updateAgent(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents })
      queryClient.invalidateQueries({ queryKey: queryKeys.agent(id) })
      toast.success('Agent updated successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to update agent: ${error.message}`)
    },
  })
}

export function useDeleteAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteAgent(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents })
      toast.success('Agent deleted successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to delete agent: ${error.message}`)
    },
  })
}

export function useStartAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => apiClient.startAgent(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents })
      queryClient.invalidateQueries({ queryKey: queryKeys.agent(id) })
      toast.success('Agent started successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to start agent: ${error.message}`)
    },
  })
}

export function useStopAgent() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => apiClient.stopAgent(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents })
      queryClient.invalidateQueries({ queryKey: queryKeys.agent(id) })
      toast.success('Agent stopped successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to stop agent: ${error.message}`)
    },
  })
}

// Document hooks
export function useDocuments() {
  return useQuery({
    queryKey: queryKeys.documents,
    queryFn: () => apiClient.getDocuments(),
    staleTime: 30000,
  })
}

export function useDocument(id: string) {
  return useQuery({
    queryKey: queryKeys.document(id),
    queryFn: () => apiClient.getDocument(id),
    enabled: !!id,
  })
}

export function useUploadDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ file, metadata }: { file: File; metadata?: any }) => 
      apiClient.uploadDocument(file, metadata),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.documents })
      toast.success('Document uploaded successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to upload document: ${error.message}`)
    },
  })
}

export function useDeleteDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteDocument(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.documents })
      toast.success('Document deleted successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to delete document: ${error.message}`)
    },
  })
}

export function useDocumentAnalytics() {
  return useQuery({
    queryKey: queryKeys.documentAnalytics,
    queryFn: () => apiClient.getDocumentAnalytics(),
    refetchInterval: 60000, // Refetch every minute
  })
}

// Workflow hooks
export function useWorkflows() {
  return useQuery({
    queryKey: queryKeys.workflows,
    queryFn: () => apiClient.getWorkflows(),
    staleTime: 30000,
  })
}

export function useWorkflow(id: string) {
  return useQuery({
    queryKey: queryKeys.workflow(id),
    queryFn: () => apiClient.getWorkflow(id),
    enabled: !!id,
  })
}

export function useCreateWorkflow() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.createWorkflow(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.workflows })
      toast.success('Workflow created successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to create workflow: ${error.message}`)
    },
  })
}

export function useRunWorkflow() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ id, inputs }: { id: string; inputs?: any }) => 
      apiClient.runWorkflow(id, inputs),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.workflowStatus(id) })
      toast.success('Workflow started successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to run workflow: ${error.message}`)
    },
  })
}

export function useWorkflowTemplates() {
  return useQuery({
    queryKey: queryKeys.workflowTemplates,
    queryFn: () => apiClient.getWorkflowTemplates(),
    staleTime: 300000, // Templates don't change often
  })
}

// Analytics hooks
export function usePerformanceAnalytics(timeRange = '7d') {
  return useQuery({
    queryKey: queryKeys.performanceAnalytics(timeRange),
    queryFn: () => apiClient.getPerformanceAnalytics(timeRange),
    refetchInterval: 60000,
  })
}

export function useUsageAnalytics(timeRange = '7d') {
  return useQuery({
    queryKey: queryKeys.usageAnalytics(timeRange),
    queryFn: () => apiClient.getUsageAnalytics(timeRange),
    refetchInterval: 60000,
  })
}

export function useAgentAnalytics(timeRange = '7d') {
  return useQuery({
    queryKey: queryKeys.agentAnalytics(timeRange),
    queryFn: () => apiClient.getAgentAnalytics(timeRange),
    refetchInterval: 60000,
  })
}

// Settings hooks
export function useSettings() {
  return useQuery({
    queryKey: queryKeys.settings,
    queryFn: () => apiClient.getSettings(),
  })
}

export function useUpdateSettings() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.updateSettings(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.settings })
      toast.success('Settings updated successfully')
    },
    onError: (error: any) => {
      toast.error(`Failed to update settings: ${error.message}`)
    },
  })
}
