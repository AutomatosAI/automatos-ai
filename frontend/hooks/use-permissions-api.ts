/**
 * Permissions API hooks
 * Provides React Query integration for permissions endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from "@/lib/api-client"

// Query keys for React Query
export const permissionsQueryKeys = {
  permissionMatrix: ['permissions', 'matrix'] as const,
  permissionAssignments: ['permissions', 'assignments'] as const,
  permissionAuditLogs: ['permissions', 'audit-logs'] as const,
  permissionsHealth: ['permissions', 'health'] as const,
}

// ============= QUERY HOOKS =============

// Get permission matrix
export function usePermissionMatrix() {
  return useQuery({
    queryKey: permissionsQueryKeys.permissionMatrix,
    queryFn: () => apiClient.getPermissionMatrix(),
    refetchInterval: 300000, // 5 minutes - permissions don't change often
    staleTime: 300000,
  })
}

// Get permission assignments
export function usePermissionAssignments() {
  return useQuery({
    queryKey: permissionsQueryKeys.permissionAssignments,
    queryFn: () => apiClient.getPermissionAssignments(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get permission audit logs
export function usePermissionAuditLogs() {
  return useQuery({
    queryKey: permissionsQueryKeys.permissionAuditLogs,
    queryFn: () => apiClient.getPermissionAuditLogs(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get permissions health
export function usePermissionsHealth() {
  return useQuery({
    queryKey: permissionsQueryKeys.permissionsHealth,
    queryFn: () => apiClient.getPermissionsHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// ============= MUTATION HOOKS =============

// Assign permission
export function useAssignPermission() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.assignPermission(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: permissionsQueryKeys.permissionAssignments })
      queryClient.invalidateQueries({ queryKey: permissionsQueryKeys.permissionMatrix })
      toast.success('Permission assigned successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to assign permission: ${error.message}`)
      throw error
    },
  })
}

// Bulk assign permissions
export function useBulkAssignPermissions() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.bulkAssignPermissions(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: permissionsQueryKeys.permissionAssignments })
      queryClient.invalidateQueries({ queryKey: permissionsQueryKeys.permissionMatrix })
      toast.success('Permissions assigned in bulk successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to bulk assign permissions: ${error.message}`)
      throw error
    },
  })
}

// Revoke permission
export function useRevokePermission() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.revokePermission(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: permissionsQueryKeys.permissionAssignments })
      queryClient.invalidateQueries({ queryKey: permissionsQueryKeys.permissionMatrix })
      toast.success('Permission revoked successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to revoke permission: ${error.message}`)
      throw error
    },
  })
}
