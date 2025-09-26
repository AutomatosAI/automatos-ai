/**
 * Health Checks API hooks
 * Provides React Query integration for health check endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from "@/lib/api-client'

// Query keys for React Query
export const healthQueryKeys = {
  systemHealth: ['health', 'system'] as const,
  systemMetrics: ['health', 'metrics'] as const,
  agentHealth: (agentId: string) => ['health', 'agent', agentId] as const,
  multiAgentHealth: ['health', 'multi-agent'] as const,
  toolsHealth: ['health', 'tools'] as const,
  permissionsHealth: ['health', 'permissions'] as const,
  credentialsHealth: ['health', 'credentials'] as const,
  fieldTheoryHealth: ['health', 'field-theory'] as const,
}

// ============= QUERY HOOKS =============

// Get system health
export function useSystemHealth() {
  return useQuery({
    queryKey: healthQueryKeys.systemHealth,
    queryFn: () => apiClient.getSystemHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get system metrics
export function useSystemMetrics() {
  return useQuery({
    queryKey: healthQueryKeys.systemMetrics,
    queryFn: () => apiClient.getSystemMetrics(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get agent health
export function useAgentHealth(agentId: string | null) {
  return useQuery({
    queryKey: healthQueryKeys.agentHealth(agentId!),
    queryFn: () => apiClient.getSystemAgentStatus(agentId!),
    enabled: !!agentId,
    refetchInterval: 10000,
  })
}

// Get multi-agent health
export function useMultiAgentHealth() {
  return useQuery({
    queryKey: healthQueryKeys.multiAgentHealth,
    queryFn: () => apiClient.getMultiAgentHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get tools health
export function useToolsHealth() {
  return useQuery({
    queryKey: healthQueryKeys.toolsHealth,
    queryFn: () => apiClient.getToolsHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get permissions health
export function usePermissionsHealth() {
  return useQuery({
    queryKey: healthQueryKeys.permissionsHealth,
    queryFn: () => apiClient.getPermissionsHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get credentials health
export function useCredentialsHealth() {
  return useQuery({
    queryKey: healthQueryKeys.credentialsHealth,
    queryFn: () => apiClient.getCredentialsHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get field theory health
export function useFieldTheoryHealth() {
  return useQuery({
    queryKey: healthQueryKeys.fieldTheoryHealth,
    queryFn: () => apiClient.getFieldTheoryHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// ============= MUTATION HOOKS =============

// Test system health
export function useTestSystemHealth() {
  return useMutation({
    mutationFn: () => apiClient.testSystemRoute(),
    onSuccess: (data) => {
      toast.success('System health test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`System health test failed: ${error.message}`)
      throw error
    },
  })
}
