/**
 * MCP Tools API Hooks (Phase 3: Skills & Tools Integration)
 * ===========================================================
 * 
 * Comprehensive React Query hooks for MCP (Model Context Protocol) tools management.
 * Provides CRUD operations, agent-tool assignments, and usage tracking.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useToast } from './use-toast'

// ===================================================================
// MCP TOOLS CRUD HOOKS
// ===================================================================

export function useMCPTools(params?: {
  status?: string
  category?: string
  provider?: string
  search?: string
  skip?: number
  limit?: number
}) {
  return useQuery({
    queryKey: ['mcp-tools', params],
    queryFn: () => apiClient.getMCPTools(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useMCPTool(id: number | null) {
  return useQuery({
    queryKey: ['mcp-tools', id],
    queryFn: () => apiClient.getMCPTool(id!),
    enabled: id !== null,
  })
}

export function useCreateMCPTool() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (data: any) => apiClient.createMCPTool(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mcp-tools'] })
      toast({
        title: 'Tool Created',
        description: 'MCP tool has been created successfully.',
      })
    },
    onError: (error: any) => {
      toast({
        title: 'Creation Failed',
        description: error.message || 'Failed to create MCP tool.',
        variant: 'destructive',
      })
    },
  })
}

export function useUpdateMCPTool() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: any }) =>
      apiClient.updateMCPTool(id, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['mcp-tools'] })
      queryClient.invalidateQueries({ queryKey: ['mcp-tools', variables.id] })
      toast({
        title: 'Tool Updated',
        description: 'MCP tool has been updated successfully.',
      })
    },
    onError: (error: any) => {
      toast({
        title: 'Update Failed',
        description: error.message || 'Failed to update MCP tool.',
        variant: 'destructive',
      })
    },
  })
}

export function useDeleteMCPTool() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (id: number) => apiClient.deleteMCPTool(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mcp-tools'] })
      toast({
        title: 'Tool Deleted',
        description: 'MCP tool has been deleted successfully.',
      })
    },
    onError: (error: any) => {
      toast({
        title: 'Deletion Failed',
        description: error.message || 'Failed to delete MCP tool.',
        variant: 'destructive',
      })
    },
  })
}

export function useTestMCPToolConnection() {
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({ id, params }: { id: number; params?: any }) =>
      apiClient.testMCPToolConnection(id, params),
    onSuccess: (data: any) => {
      toast({
        title: 'Connection Successful',
        description: data.message || 'MCP tool connection test passed.',
      })
    },
    onError: (error: any) => {
      toast({
        title: 'Connection Failed',
        description: error.message || 'MCP tool connection test failed.',
        variant: 'destructive',
      })
    },
  })
}

// ===================================================================
// MCP TOOLS DISCOVERY & STATS HOOKS
// ===================================================================

export function useMCPToolCategories() {
  return useQuery({
    queryKey: ['mcp-tools', 'categories'],
    queryFn: () => apiClient.getMCPToolCategories(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

export function useMCPToolsStats() {
  return useQuery({
    queryKey: ['mcp-tools', 'stats'],
    queryFn: () => apiClient.getMCPToolsStats(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })
}

// ===================================================================
// AGENT-TOOL ASSIGNMENT HOOKS
// ===================================================================

export function useMCPToolAssignments(enabledOnly: boolean = true) {
  return useQuery({
    queryKey: ['mcp-tool-assignments', { enabledOnly }],
    queryFn: () => apiClient.getMCPToolAssignments(enabledOnly),
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

export function useAgentTools(agentId: number | null, enabledOnly: boolean = true) {
  return useQuery({
    queryKey: ['agent-tools', agentId, { enabledOnly }],
    queryFn: () => apiClient.getAgentTools(agentId!, enabledOnly),
    enabled: agentId !== null,
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

export function useAssignToolToAgent() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({
      agentId,
      toolId,
      data,
    }: {
      agentId: number
      toolId: number
      data?: { enabled?: boolean; permissions?: any; configuration?: any }
    }) => apiClient.assignToolToAgent(agentId, toolId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['agent-tools', variables.agentId] })
      queryClient.invalidateQueries({ queryKey: ['agents', variables.agentId] })
      queryClient.invalidateQueries({ queryKey: ['mcp-tools', 'stats'] })
      toast({
        title: 'Tool Assigned',
        description: 'Tool has been assigned to the agent successfully.',
      })
    },
    onError: (error: any) => {
      toast({
        title: 'Assignment Failed',
        description: error.message || 'Failed to assign tool to agent.',
        variant: 'destructive',
      })
    },
  })
}

export function useRemoveToolFromAgent() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({ agentId, toolId }: { agentId: number; toolId: number }) =>
      apiClient.removeToolFromAgent(agentId, toolId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['agent-tools', variables.agentId] })
      queryClient.invalidateQueries({ queryKey: ['agents', variables.agentId] })
      queryClient.invalidateQueries({ queryKey: ['mcp-tools', 'stats'] })
      toast({
        title: 'Tool Removed',
        description: 'Tool has been removed from the agent successfully.',
      })
    },
    onError: (error: any) => {
      toast({
        title: 'Removal Failed',
        description: error.message || 'Failed to remove tool from agent.',
        variant: 'destructive',
      })
    },
  })
}

export function useUpdateToolPermissions() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({
      agentId,
      toolId,
      permissions,
    }: {
      agentId: number
      toolId: number
      permissions: any
    }) => apiClient.updateToolPermissions(agentId, toolId, permissions),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['agent-tools', variables.agentId] })
      queryClient.invalidateQueries({ queryKey: ['agents', variables.agentId] })
      toast({
        title: 'Permissions Updated',
        description: 'Tool permissions have been updated successfully.',
      })
    },
    onError: (error: any) => {
      toast({
        title: 'Update Failed',
        description: error.message || 'Failed to update tool permissions.',
        variant: 'destructive',
      })
    },
  })
}

// ===================================================================
// TOOL USAGE TRACKING HOOKS
// ===================================================================

export function useToolUsageLogs(params?: {
  tool_id?: number
  agent_id?: number
  success_only?: boolean
  skip?: number
  limit?: number
}) {
  return useQuery({
    queryKey: ['tool-usage-logs', params],
    queryFn: () => apiClient.getToolUsageLogs(params),
    staleTime: 1 * 60 * 1000, // 1 minute
  })
}

// ===================================================================
// HELPER HOOKS
// ===================================================================

/**
 * Get all tools available for a specific category
 */
export function useToolsByCategory(category: string) {
  return useMCPTools({ category, limit: 100 })
}

/**
 * Get all tools from a specific provider
 */
export function useToolsByProvider(provider: string) {
  return useMCPTools({ provider, limit: 100 })
}

/**
 * Search tools by name or description
 */
export function useSearchTools(search: string) {
  return useMCPTools({ search, limit: 50 })
}

/**
 * Get active tools only
 */
export function useActiveTools() {
  return useMCPTools({ status: 'active', limit: 100 })
}

