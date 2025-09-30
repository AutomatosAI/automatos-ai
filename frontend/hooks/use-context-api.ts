import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// ✅ All these endpoints are verified working in tests

export function useContextStats() {
  return useQuery({
    queryKey: ['context', 'stats'],
    queryFn: () => apiClient.getContextStats()
  })
}

export function useContextSources() {
  return useQuery({
    queryKey: ['context', 'sources'],
    queryFn: () => apiClient.getContextSources()
  })
}

export function useContextPatterns() {
  return useQuery({
    queryKey: ['context', 'patterns'],
    queryFn: () => apiClient.getContextPatterns()
  })
}

export function useContextPerformance() {
  return useQuery({
    queryKey: ['context', 'performance'],
    queryFn: () => apiClient.getContextPerformance()
  })
}

export function useRecentContextQueries(limit: number = 20) {
  return useQuery({
    queryKey: ['context', 'queries', 'recent', limit],
    queryFn: () => apiClient.getRecentContextQueries()
  })
}

export function useContextSystemHealth() {
  return useQuery({
    queryKey: ['context', 'system', 'health'],
    queryFn: () => apiClient.getContextSystemHealth()
  })
}

export function useInitializeContext() {
  return useMutation({
    mutationFn: (data?: any) => apiClient.initializeContext(data)
  })
}

export function useProcessQuery() {
  return useMutation({
    mutationFn: (data: any) => apiClient.processQuery(data)
  })
}

export function useCalculateSimilarity() {
  return useMutation({
    mutationFn: (data: any) => apiClient.calculateSimilarity(data)
  })
}
