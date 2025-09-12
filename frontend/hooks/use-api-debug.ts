import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

export function useSystemHealth() {
  console.log('🔍 useSystemHealth called')
  return useQuery({
    queryKey: ['system-health'],
    queryFn: async () => {
      console.log('🚀 useSystemHealth queryFn executing')
      const result = await apiClient.getSystemHealth()
      console.log('✅ useSystemHealth result:', result)
      return result
    },
    staleTime: 30000,
    refetchInterval: 60000,
  })
}

export function useSystemMetrics() {
  console.log('🔍 useSystemMetrics called')
  return useQuery({
    queryKey: ['system-metrics'],
    queryFn: async () => {
      console.log('🚀 useSystemMetrics queryFn executing')
      const result = await apiClient.getSystemMetrics()
      console.log('✅ useSystemMetrics result:', result)
      return result
    },
    staleTime: 30000,
    refetchInterval: 60000,
  })
}

export function useAgents() {
  console.log('🔍 useAgents called')
  return useQuery({
    queryKey: ['agents'],
    queryFn: async () => {
      console.log('🚀 useAgents queryFn executing')
      const result = await apiClient.getAgents()
      console.log('✅ useAgents result:', result)
      return result
    },
    staleTime: 30000,
    refetchInterval: 60000,
  })
}
