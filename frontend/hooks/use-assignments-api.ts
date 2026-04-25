import { useQuery } from '@tanstack/react-query'
import apiClient from '@/lib/api-client'

export interface RecommendedItem {
  id: number | string
  type: 'playbook' | 'mission'
  name: string
  description: string
  icon: string | null
  category: string
  use_count?: number
  install_count?: number
  source: 'workspace' | 'marketplace'
}

export interface RecommendedResponse {
  items: RecommendedItem[]
  workspace_count: number
  marketplace_count: number
}

export const assignmentKeys = {
  all: ['assignments'] as const,
  recommended: (limit?: number) => [...assignmentKeys.all, 'recommended', limit] as const,
}

export function useRecommendedAssignments(limit: number = 8) {
  return useQuery({
    queryKey: assignmentKeys.recommended(limit),
    queryFn: () => apiClient.getRecommendedAssignments(limit) as Promise<RecommendedResponse>,
    staleTime: 5 * 60 * 1000,
  })
}
