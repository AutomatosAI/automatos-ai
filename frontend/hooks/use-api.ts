import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// Query key factories
export const systemQueryKeys = {
  health: ['system', 'health'] as const,
  metrics: ['system', 'metrics'] as const,
}

export const agentQueryKeys = {
  all: ['agents'] as const,
  detail: (id: string) => ['agents', id] as const,
}

export const documentQueryKeys = {
  all: ['documents'] as const,
  detail: (id: string) => ['documents', id] as const,
}

export const workflowQueryKeys = {
  all: ['workflows'] as const,
  detail: (id: string) => ['workflows', id] as const,
}

// System hooks
export function useSystemHealth() {
  return useQuery({
    queryKey: systemQueryKeys.health,
    queryFn: async () => {
      try {
        return await apiClient.getSystemHealth()
      } catch (error) {
        throw error
      }
    },
  })
}

export function useSystemMetrics() {
  return useQuery({
    queryKey: systemQueryKeys.metrics,
    queryFn: async () => {
      try {
        return await apiClient.getSystemMetrics()
      } catch (error) {
        throw error
      }
    },
  })
}

// Agent hooks
export function useAgents() {
  return useQuery({
    queryKey: agentQueryKeys.all,
    queryFn: async () => {
      try {
        return await apiClient.getAgents()
      } catch (error) {
        throw error
      }
    },
  })
}

// Document hooks
export function useDocuments() {
  return useQuery({
    queryKey: documentQueryKeys.all,
    queryFn: async () => {
      try {
        return await apiClient.getDocuments()
      } catch (error) {
        throw error
      }
    },
  })
}

// Workflow hooks
export function useWorkflows() {
  return useQuery({
    queryKey: workflowQueryKeys.all,
    queryFn: async () => {
      try {
        return await apiClient.getWorkflows()
      } catch (error) {
        throw error
      }
    },
  })
}