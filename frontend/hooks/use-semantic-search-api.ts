/**
 * Semantic Search API Hook
 * 
 * Provides semantic search capabilities across document chunks
 * using vector similarity search powered by pgvector and OpenAI embeddings.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

export interface SearchResult {
  chunk_id: number
  document_id: number
  chunk_index: number
  content: string
  similarity: number
  metadata: Record<string, any>
  source: {
    filename: string
    file_type: string
    file_size: number
    upload_date: string | null
  }
}

export interface SemanticSearchResponse {
  query: string
  total_results: number
  execution_time_ms: number
  results: SearchResult[]
}

export interface SemanticSearchParams {
  query: string
  limit?: number
  min_similarity?: number
  document_ids?: number[]
}

/**
 * Hook for performing semantic search across documents
 */
export function useSemanticSearch(
  params: SemanticSearchParams | null,
  options?: {
    enabled?: boolean
    staleTime?: number
  }
) {
  return useQuery({
    queryKey: ['documents', 'search', params],
    queryFn: async () => {
      if (!params || !params.query) {
        throw new Error('Query is required for semantic search')
      }

      const searchParams = new URLSearchParams({
        query: params.query,
        limit: String(params.limit || 10),
        min_similarity: String(params.min_similarity || 0.7),
      })

      if (params.document_ids && params.document_ids.length > 0) {
        params.document_ids.forEach(id => {
          searchParams.append('document_ids', String(id))
        })
      }

      // CRITICAL: Call backend directly to bypass Next.js proxy mock data
      const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.automatos.app'
      const url = `${BACKEND_URL}/api/documents/search?${searchParams.toString()}`
      console.log('[Semantic Search] Calling backend directly:', url)
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      console.log('[Semantic Search] Response status:', response.status)

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      return response.json() as Promise<SemanticSearchResponse>
    },
    enabled: options?.enabled !== false && !!params?.query,
    staleTime: options?.staleTime || 5 * 60 * 1000, // 5 minutes
    retry: 2,
  })
}

/**
 * Hook for tracking search queries (for analytics)
 */
export function useTrackSearchQuery() {
  return useMutation({
    mutationFn: async (data: { query: string; results_count: number }) => {
      // Track search event for analytics
      // TODO: Implement when analytics endpoint is ready
      console.log('Search tracked:', data)
      return data
    },
  })
}
