/**
 * Credentials API hooks
 * Provides React Query integration for credentials endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from "@/lib/api-client"

// Query keys for React Query
export const credentialsQueryKeys = {
  credentials: ['credentials'] as const,
  credentialByTool: (toolId: string) => ['credentials', 'tool', toolId] as const,
  credentialAuditLogs: ['credentials', 'audit-logs'] as const,
  credentialsHealth: ['credentials', 'health'] as const,
}

// ============= QUERY HOOKS =============

// Get all credentials
export function useCredentials() {
  return useQuery({
    queryKey: credentialsQueryKeys.credentials,
    queryFn: () => apiClient.getCredentials(),
    refetchInterval: 300000, // 5 minutes - credentials don't change often
    staleTime: 300000,
  })
}

// Get credential by tool
export function useCredentialByTool(toolId: string | null) {
  return useQuery({
    queryKey: credentialsQueryKeys.credentialByTool(toolId!),
    queryFn: () => apiClient.getCredentialByTool(toolId!),
    enabled: !!toolId,
    refetchInterval: 300000,
    staleTime: 300000,
  })
}

// Get credential audit logs
export function useCredentialAuditLogs() {
  return useQuery({
    queryKey: credentialsQueryKeys.credentialAuditLogs,
    queryFn: () => apiClient.getCredentialAuditLogs(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get credentials health
export function useCredentialsHealth() {
  return useQuery({
    queryKey: credentialsQueryKeys.credentialsHealth,
    queryFn: () => apiClient.getCredentialsHealth(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// ============= MUTATION HOOKS =============

// Create credential
export function useCreateCredential() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.createCredential(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: credentialsQueryKeys.credentials })
      toast.success('Credential created successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to create credential: ${error.message}`)
      throw error
    },
  })
}

// Update credential
export function useUpdateCredential() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ credentialId, data }: { credentialId: string; data: any }) =>
      apiClient.updateCredential(credentialId, data),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: credentialsQueryKeys.credentials })
      queryClient.invalidateQueries({ queryKey: credentialsQueryKeys.credentialByTool(variables.credentialId) })
      toast.success('Credential updated successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update credential: ${error.message}`)
      throw error
    },
  })
}

// Delete credential
export function useDeleteCredential() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (credentialId: string) => apiClient.deleteCredential(credentialId),
    onSuccess: (data, credentialId) => {
      queryClient.invalidateQueries({ queryKey: credentialsQueryKeys.credentials })
      queryClient.removeQueries({ queryKey: credentialsQueryKeys.credentialByTool(credentialId) })
      toast.success('Credential deleted successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to delete credential: ${error.message}`)
      throw error
    },
  })
}

// Test credential connection
export function useTestCredentialConnection() {
  return useMutation({
    mutationFn: (data: any) => apiClient.testCredentialConnection(data),
    onSuccess: (data) => {
      toast.success('Credential connection test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Credential connection test failed: ${error.message}`)
      throw error
    },
  })
}
