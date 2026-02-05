/**
 * Processing Queue API Hook
 * 
 * Provides real-time access to document processing queue status
 */

import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

export interface QueueDocument {
  document_id: number
  filename: string
  file_type: string
  file_size: number
  upload_date?: string
  position_in_queue?: number
}

export interface ProcessingDocument {
  document_id: number
  filename: string
  file_type: string
  file_size: number
  status: string
  progress: number
  current_step: string
  step_name: string
  started_at?: string
  eta_seconds: number
}

export interface FailedDocument {
  document_id: number
  filename: string
  file_type: string
  file_size: number
  failed_at?: string
  error: string
}

export interface QueueStatus {
  queue_depth: number
  currently_processing: number
  failed_count: number
  pending: QueueDocument[]
  processing: ProcessingDocument[]
  failed: FailedDocument[]
  timestamp: string
}

/**
 * Hook to fetch current processing queue status
 */
export function useProcessingQueue(options?: {
  enabled?: boolean
  refetchInterval?: number
}) {
  return useQuery({
    queryKey: ['documents', 'queue', 'status'],
    queryFn: async () => {
      // Use apiClient for proper auth headers and base URL
      return await apiClient.get<QueueStatus>('/api/documents/queue/status')
    },
    enabled: options?.enabled !== false,
    refetchInterval: options?.refetchInterval || 10000, // Refetch every 10 seconds (reduced API calls)
    staleTime: 8000, // Consider data stale after 8 seconds
    retry: 2,
    refetchOnWindowFocus: false, // Don't refetch when switching browser tabs
    refetchOnMount: false, // Don't refetch on every mount
  })
}

/**
 * Hook to start processing a document
 */
export function useStartProcessing() {
  return async (documentId: number) => {
    // This would trigger document reprocessing
    // For now, return success
    console.log('Starting processing for document:', documentId)
    return { success: true, document_id: documentId }
  }
}
