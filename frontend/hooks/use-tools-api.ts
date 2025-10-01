import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

export function useTools() {
  return useQuery({
    queryKey: ['tools'],
    queryFn: () => apiClient.getTools()
  })
}

export function useInstallTool() {
  return useMutation({
    mutationFn: ({ id, data }: any) => apiClient.installTool(id, data)
  })
}

export function useToolStatus(id: string) {
  return useQuery({
    queryKey: ['tools', id, 'status'],
    queryFn: () => apiClient.getToolStatus(id)
  })
}

export function useToolsHealth() {
  return useQuery({
    queryKey: ['tools', 'health'],
    queryFn: () => apiClient.getToolsHealth()
  })
}
