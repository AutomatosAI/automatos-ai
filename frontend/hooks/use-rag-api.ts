/**
 * RAG System API hooks
 * Provides React Query integration for RAG system endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from "@/lib/api-client'

// Query keys for React Query
export const ragQueryKeys = {
  systemRAG: ['rag', 'system'] as const,
  systemRAGConfig: (id: string) => ['rag', 'system', id] as const,
  contextRAG: (configId: string) => ['rag', 'context', configId] as const,
}

// ============= QUERY HOOKS =============

// Get system RAG
export function useSystemRAG() {
  return useQuery({
    queryKey: ragQueryKeys.systemRAG,
    queryFn: () => apiClient.getSystemRAG(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// Get system RAG config
export function useSystemRAGConfig(id: string | null) {
  return useQuery({
    queryKey: ragQueryKeys.systemRAGConfig(id!),
    queryFn: () => apiClient.getSystemRAGConfig(id!),
    enabled: !!id,
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// ============= MUTATION HOOKS =============

// Update system RAG
export function useUpdateSystemRAG() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.updateSystemRAG(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ragQueryKeys.systemRAG })
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

// Test context RAG
export function useTestContextRAG() {
  return useMutation({
    mutationFn: (configId: string) => apiClient.testContextRAG(configId),
    onSuccess: (data) => {
      toast.success('Context RAG test completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Context RAG test failed: ${error.message}`)
      throw error
    },
  })
}
