/**
 * Semantic Search API Hook
 *
 * Provides semantic search capabilities across document chunks
 * using vector similarity search powered by pgvector and OpenAI embeddings.
 */

import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

export interface SearchResult {
  document_id: number
  excerpt: string
  similarity: number
  chunk_index: number
  chunk_count: number
  metadata: Record<string, any>
  source: {
    filename: string
    file_type: string
    file_size: number
    upload_date: string | null
  }
  preview?: string
  preview_truncated?: boolean
  full_content_available?: boolean
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
 * Hook for performing semantic search across documents.
 * Uses apiClient for proper auth (Clerk JWT + workspace ID).
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

      // Use apiClient for proper auth headers (Clerk JWT + workspace ID)
      return apiClient.post<SemanticSearchResponse>(
        `/api/documents/search?${searchParams.toString()}`
      )
    },
    enabled: options?.enabled !== false && !!params?.query,
    staleTime: options?.staleTime || 5 * 60 * 1000,
    retry: 2,
  })
}

/**
 * Hook for fetching document content with optional chunk highlighting.
 */
export function useDocumentContent(
  documentId: number | null,
  highlightChunkIds?: number[],
  options?: { enabled?: boolean }
) {
  return useQuery({
    queryKey: ['documents', documentId, 'content', highlightChunkIds],
    queryFn: async () => {
      if (!documentId) throw new Error('Document ID required')
      const params = new URLSearchParams()
      if (highlightChunkIds?.length) {
        highlightChunkIds.forEach(id => params.append('highlight_chunk_ids', String(id)))
      }
      const qs = params.toString()
      return apiClient.get(`/api/documents/${documentId}/content${qs ? `?${qs}` : ''}`)
    },
    enabled: options?.enabled !== false && !!documentId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Hook for tracking search queries (for analytics)
 */
export function useTrackSearchQuery() {
  return useMutation({
    mutationFn: async (data: { query: string; results_count: number }) => {
      return apiClient.post('/api/documents/analytics/track', {
        event_type: 'document_searched',
        query: data.query,
        results_count: data.results_count,
      })
    },
  })
}
