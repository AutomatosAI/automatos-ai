/**
 * RAG (Retrieval Augmented Generation) API Hook
 *
 * Provides hooks for retrieving optimized context for LLM augmentation
 * using Maximal Marginal Relevance (MMR) for diversity
 */

import { useMutation, useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

export interface RAGChunk {
  chunk_id: number
  document_id: number
  chunk_index: number
  content: string
  similarity: number
  source: {
    filename: string
    file_type: string
    chunk_index: number
  }
  tokens: number
  truncated: boolean
}

export interface RAGRetrievalResponse {
  query: string
  chunks: RAGChunk[]
  context: string
  total_tokens: number
  diversity_score: number
  execution_time_ms: number
  settings: {
    max_chunks: number
    max_tokens: number
    diversity: number
    lambda: number
  }
}

export interface RAGRetrievalParams {
  query: string
  max_chunks?: number
  max_tokens?: number
  diversity?: number
}

/**
 * Hook for performing RAG retrieval
 */
export function useRAGRetrieval(
  params: RAGRetrievalParams | null,
  options?: {
    enabled?: boolean
  }
) {
  return useQuery({
    queryKey: ['rag', 'retrieve', params],
    queryFn: async () => {
      if (!params || !params.query) {
        throw new Error('Query is required for RAG retrieval')
      }
      return apiClient.ragRetrieve(params) as Promise<RAGRetrievalResponse>
    },
    enabled: options?.enabled !== false && !!params?.query,
    staleTime: 30000,
    retry: 2,
  })
}

/**
 * Mutation hook for RAG retrieval (when you want manual control)
 */
export function useRAGRetrievalMutation() {
  return useMutation({
    mutationFn: async (params: RAGRetrievalParams) => {
      return apiClient.ragRetrieve(params) as Promise<RAGRetrievalResponse>
    },
  })
}
