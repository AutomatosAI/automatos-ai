import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

export function useContextStats() {
  return useQuery({
    queryKey: ['context', 'stats'],
    queryFn: () => apiClient.getContextStats()
  })
}

export function useContextPerformance() {
  return useQuery({
    queryKey: ['context', 'performance'],
    queryFn: () => apiClient.getContextPerformance()
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
