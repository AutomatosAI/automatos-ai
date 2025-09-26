import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from "@/lib/api-client'

export function useMemoryStats() {
  return useQuery({
    queryKey: ['memory', 'stats'],
    queryFn: () => apiClient.getMemoryStats()
  })
}

export function useStoreMemory() {
  return useMutation({
    mutationFn: (data: any) => apiClient.storeMemory(data)
  })
}

export function useSearchMemory() {
  return useMutation({
    mutationFn: (data: any) => apiClient.searchMemory(data)
  })
}
