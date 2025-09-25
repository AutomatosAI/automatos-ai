/**
 * Tools Management API hooks
 * Provides React Query integration for tools management endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const toolsManagementQueryKeys = {
  tools: ['tools'] as const,
  tool: (toolId: string) => ['tools', toolId] as const,
  toolStatus: (toolId: string) => ['tools', toolId, 'status'] as const,
  toolUsageStats: (toolId: string) => ['tools', toolId, 'usage-stats'] as const,
  toolsHealth: ['tools', 'health'] as const,
}

// ============= QUERY HOOKS =============

// Get all tools
export function useTools() {
  return useQuery({
    queryKey: toolsManagementQueryKeys.tools,
    queryFn: () => apiClient.getTools(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get specific tool
export function useTool(toolId: string | null) {
  return useQuery({
    queryKey: toolsManagementQueryKeys.tool(toolId!),
    queryFn: () => apiClient.getTool(toolId!),
    enabled: !!toolId,
    refetchInterval: 30000,
  })
}

// Get tool status
export function useToolStatus(toolId: string | null) {
  return useQuery({
    queryKey: toolsManagementQueryKeys.toolStatus(toolId!),
    queryFn: () => apiClient.getToolStatus(toolId!),
    enabled: !!toolId,
    refetchInterval: 30000,
  })
}

// Get tool usage stats
export function useToolUsageStats(toolId: string | null) {
  return useQuery({
    queryKey: toolsManagementQueryKeys.toolUsageStats(toolId!),
    queryFn: () => apiClient.getToolUsageStats(toolId!),
    enabled: !!toolId,
    refetchInterval: 60000,
  })
}

// Get tools health
export function useToolsHealth() {
  return useQuery({
    queryKey: toolsManagementQueryKeys.toolsHealth,
    queryFn: () => apiClient.getToolsHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// ============= MUTATION HOOKS =============

// Install tool
export function useInstallTool() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (toolId: string) => apiClient.installTool(toolId),
    onSuccess: (data, toolId) => {
      queryClient.invalidateQueries({ queryKey: toolsManagementQueryKeys.tools })
      queryClient.invalidateQueries({ queryKey: toolsManagementQueryKeys.tool(toolId) })
      queryClient.invalidateQueries({ queryKey: toolsManagementQueryKeys.toolStatus(toolId) })
      toast.success('Tool installed successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to install tool: ${error.message}`)
      throw error
    },
  })
}

// Configure tool
export function useConfigureTool() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ toolId, data }: { toolId: string; data: any }) =>
      apiClient.configureTool(toolId, data),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: toolsManagementQueryKeys.tool(variables.toolId) })
      queryClient.invalidateQueries({ queryKey: toolsManagementQueryKeys.toolStatus(variables.toolId) })
      toast.success('Tool configured successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to configure tool: ${error.message}`)
      throw error
    },
  })
}

// Uninstall tool
export function useUninstallTool() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (toolId: string) => apiClient.uninstallTool(toolId),
    onSuccess: (data, toolId) => {
      queryClient.invalidateQueries({ queryKey: toolsManagementQueryKeys.tools })
      queryClient.removeQueries({ queryKey: toolsManagementQueryKeys.tool(toolId) })
      toast.success('Tool uninstalled successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to uninstall tool: ${error.message}`)
      throw error
    },
  })
}
