/**
 * Memory Explorer API hooks (PRD-77)
 * React Query integration for browsing, searching, and managing memories.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// ============= TYPES =============

export interface MemoryItem {
  id: string
  content: string
  score: number | null
  metadata: Record<string, any> | null
  created_at: string | null
  updated_at: string | null
}

export interface MemoryBrowseResponse {
  success: boolean
  memories: MemoryItem[]
  total: number
  source: string
  search_query: string | null
}

export interface MemoryHealthResponse {
  success: boolean
  mem0_available: boolean
  total_memories: number
  oldest_memory: string | null
  newest_memory: string | null
  search_effectiveness: {
    total_searches: number
    hits: number
    hit_rate: number
  }
  health_status: 'healthy' | 'degraded' | 'unavailable'
}

export interface MemoryStatsResponse {
  system_stats: {
    total_memories: number
    memory_levels: Record<string, number>
    memory_types: Record<string, number>
    agents_with_memories: number
    source: string
  }
  access_metrics: {
    total_accesses: number
    hit_rate: number
    cache_utilization: number
  }
  is_real_data: boolean
  timestamp: string
}

export interface MemoryConsolidateResponse {
  success: boolean
  merged_memory_id?: string
  deleted_count?: number
  message?: string
  error?: string
}

export interface MemoryFilters {
  query?: string
  limit?: number
}

// ============= QUERY KEYS =============

export const memoryQueryKeys = {
  all: ['memory-explorer'] as const,
  browse: (filters: MemoryFilters) => ['memory-explorer', 'browse', filters] as const,
  health: () => ['memory-explorer', 'health'] as const,
  stats: () => ['memory-explorer', 'stats'] as const,
}

// ============= QUERY HOOKS =============

/**
 * Browse/search memories with optional query.
 */
export function useMemoryBrowse(filters: MemoryFilters = {}) {
  const params = new URLSearchParams()
  if (filters.query) params.set('query', filters.query)
  if (filters.limit) params.set('limit', String(filters.limit))

  const qs = params.toString()

  return useQuery<MemoryBrowseResponse>({
    queryKey: memoryQueryKeys.browse(filters),
    queryFn: () =>
      apiClient.request<MemoryBrowseResponse>(`/api/v1/memory/browse${qs ? `?${qs}` : ''}`),
    staleTime: 15000,
    refetchInterval: 60000,
  })
}

/**
 * Get memory health report.
 */
export function useMemoryHealth() {
  return useQuery<MemoryHealthResponse>({
    queryKey: memoryQueryKeys.health(),
    queryFn: () => apiClient.request<MemoryHealthResponse>('/api/v1/memory/health'),
    staleTime: 30000,
    refetchInterval: 120000,
  })
}

/**
 * Get memory stats (total, by type, by level).
 */
export function useMemoryExplorerStats() {
  return useQuery<MemoryStatsResponse>({
    queryKey: memoryQueryKeys.stats(),
    queryFn: () => apiClient.request<MemoryStatsResponse>('/api/v1/memory/stats/real'),
    staleTime: 30000,
    refetchInterval: 60000,
  })
}

// ============= MUTATION HOOKS =============

/**
 * Delete a single memory by ID.
 */
export function useDeleteMemory() {
  const queryClient = useQueryClient()

  return useMutation<
    { success: boolean; message?: string },
    Error,
    { memoryId: string }
  >({
    mutationFn: ({ memoryId }) =>
      apiClient.request(`/api/v1/memory/${memoryId}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoryQueryKeys.all })
      toast.success('Memory deleted')
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to delete memory')
    },
  })
}

/**
 * Consolidate (merge) multiple memories into one.
 */
export function useConsolidateMemories() {
  const queryClient = useQueryClient()

  return useMutation<
    MemoryConsolidateResponse,
    Error,
    { memory_ids: string[]; strategy: 'merge' | 'summarise' }
  >({
    mutationFn: (body) =>
      apiClient.request('/api/v1/memory/consolidate', {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: memoryQueryKeys.all })
      toast.success(data.message || 'Memories consolidated')
    },
    onError: (error) => {
      toast.error(error.message || 'Consolidation failed')
    },
  })
}
