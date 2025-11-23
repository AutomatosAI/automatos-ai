/**
 * RAG (Retrieval Augmented Generation) API Hook
 * 
 * Provides hooks for retrieving optimized context for LLM augmentation
 * using Maximal Marginal Relevance (MMR) for diversity
 */

import { useMutation, useQuery } from '@tanstack/react-query'

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

      const searchParams = new URLSearchParams({
        query: params.query,
      })

      if (params.max_chunks) {
        searchParams.append('max_chunks', String(params.max_chunks))
      }
      if (params.max_tokens) {
        searchParams.append('max_tokens', String(params.max_tokens))
      }
      if (params.diversity !== undefined) {
        searchParams.append('diversity', String(params.diversity))
      }

      // CRITICAL: Call backend directly to bypass Next.js proxy mock data
      const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || ''
      const url = `${BACKEND_URL}/api/documents/rag/retrieve?${searchParams.toString()}`
      console.log('[RAG Retrieval] Calling backend directly:', url)
      console.log('[RAG Retrieval] Params:', params)
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      console.log('[RAG Retrieval] Response status:', response.status)

      if (!response.ok) {
        throw new Error(`RAG retrieval failed: ${response.statusText}`)
      }

      return response.json() as Promise<RAGRetrievalResponse>
    },
    enabled: options?.enabled !== false && !!params?.query,
    staleTime: 30000, // Cache for 30 seconds
    retry: 2,
  })
}

/**
 * Mutation hook for RAG retrieval (when you want manual control)
 */
export function useRAGRetrievalMutation() {
  return useMutation({
    mutationFn: async (params: RAGRetrievalParams) => {
      const searchParams = new URLSearchParams({
        query: params.query,
        max_chunks: String(params.max_chunks || 5),
        max_tokens: String(params.max_tokens || 2000),
        diversity: String(params.diversity || 0.3),
      })

      // CRITICAL: Call backend directly to bypass Next.js proxy mock data
      const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || ''
      const url = `${BACKEND_URL}/api/documents/rag/retrieve?${searchParams.toString()}`
      console.log('[RAG Mutation] Calling backend directly:', url)
      console.log('[RAG Mutation] Params:', params)

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      console.log('[RAG Mutation] Response status:', response.status, response.statusText)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('[RAG Hook] Error response:', errorText)
        throw new Error(`RAG retrieval failed (${response.status}): ${errorText || response.statusText}`)
      }

      const data = await response.json()
      console.log('[RAG Hook] Success! Data received:', data)
      return data as RAGRetrievalResponse
    },
  })
}