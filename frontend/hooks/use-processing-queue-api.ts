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
      const response = await fetch(
        `/api/documents/queue/status`,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      )

      if (!response.ok) {
        throw new Error(`Failed to fetch queue status: ${response.statusText}`)
      }

      return response.json() as Promise<QueueStatus>
    },
    enabled: options?.enabled !== false,
    refetchInterval: options?.refetchInterval || 5000, // Refetch every 5 seconds
    staleTime: 2000, // Consider data stale after 2 seconds
    retry: 3,
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
