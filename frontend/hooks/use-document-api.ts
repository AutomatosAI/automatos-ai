/**
 * Enhanced Document API hooks for React Query integration
 * Provides auto-caching, retries, and fallback data
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { logger } from '../lib/logger'
import apiClient from '../lib/api-client'

// Document query keys for consistent caching
export const documentQueryKeys = {
  documents: ['documents'],
  document: (id: string) => ['documents', id],
  documentStats: ['documents', 'stats'],
  documentCategories: ['documents', 'categories'],
  processingStatus: (id: string) => ['documents', id, 'processing'],
  processingQueue: ['documents', 'processing', 'queue'],
  documentSearch: (query: string) => ['documents', 'search', query],
  semanticSearch: (query: string) => ['documents', 'semantic', query],
  documentInsights: (id: string) => ['documents', id, 'insights'],
  analytics: (timeRange: string) => ['documents', 'analytics', timeRange],
  usageAnalytics: (timeRange: string) => ['documents', 'usage', timeRange],
  processingAnalytics: (timeRange: string) => ['documents', 'processing', 'analytics', timeRange],
  storageAnalytics: ['documents', 'storage'],
}

// Common fallback data for improved UX when API is unavailable
const FALLBACK_DATA = {
  documents: [
    { id: 'fallback-1', filename: 'Example Document 1.pdf', status: 'processed', file_size: 1240000, chunk_count: 12, upload_date: new Date().toISOString() },
    { id: 'fallback-2', filename: 'Example Document 2.docx', status: 'processing', file_size: 520000, chunk_count: 5, upload_date: new Date().toISOString() },
    { id: 'fallback-3', filename: 'Sample Report.txt', status: 'error', file_size: 45000, chunk_count: 2, upload_date: new Date().toISOString() }
  ],
  documentStats: {
    total_documents: 3,
    total_processed: 2,
    total_processing: 1,
    total_error: 0,
    total_size_mb: 1.8,
    total_chunks: 19,
    avg_processing_time_sec: 35
  },
  processingQueue: {
    pipeline_status: 'idle',
    total_documents: 0,
    processing_documents: 0,
    completed_documents: 0,
    failed_documents: 0,
    success_rate: 100,
    avg_processing_time: '0s',
    queue_status: {
      pending: 0,
      active_workers: 0,
      estimated_completion: 'N/A'
    },
    processing_stages: [
      { stage: 'Upload', status: 'idle', documents_count: 0, avg_duration: '0s', success_rate: 100 },
      { stage: 'Parse', status: 'idle', documents_count: 0, avg_duration: '0s', success_rate: 100 },
      { stage: 'Chunk', status: 'idle', documents_count: 0, avg_duration: '0s', success_rate: 100 },
      { stage: 'Embed', status: 'idle', documents_count: 0, avg_duration: '0s', success_rate: 100 },
      { stage: 'Index', status: 'idle', documents_count: 0, avg_duration: '0s', success_rate: 100 }
    ],
    recent_activity: [],
    active_jobs: [],
    last_updated: new Date().toISOString()
  },
  documentCategories: [
    { id: 'reports', name: 'Reports', count: 1 },
    { id: 'contracts', name: 'Contracts', count: 1 },
    { id: 'emails', name: 'Emails', count: 1 }
  ]
}

// Common error handler for mutations
const handleApiError = (error: any, message = 'API request failed'): void => {
  logger.error(message, error)
  toast.error(message)
}

/**
 * Get all documents
 */
export function useDocuments() {
  return useQuery({
    queryKey: documentQueryKeys.documents,
    queryFn: () => apiClient.getDocuments(),
    retry: 2,
    staleTime: 30000,
    placeholderData: FALLBACK_DATA.documents,
    onError: (error) => handleApiError(error, 'Failed to load documents')
  })
}
/**
 * Get a single document by ID
 */
export function useDocument(documentId: string | null) {
  return useQuery({
    queryKey: documentQueryKeys.document(documentId || ''),
    queryFn: () => apiClient.getDocument(documentId || ''),
    enabled: !!documentId,
    retry: 2,
    staleTime: 30000,
    onError: (error) => handleApiError(error, 'Failed to load document details')
  })
}

/**
 * Get document statistics
 */
export function useDocumentStats() {
  return useQuery({
    queryKey: documentQueryKeys.documentStats,
    queryFn: () => apiClient.getDocumentAnalytics(),
    retry: 2,
    staleTime: 60000,
    placeholderData: FALLBACK_DATA.documentStats,
    onError: (error) => handleApiError(error, 'Failed to load document statistics')
  })
}

/**
 * Get document categories
 */
export function useDocumentCategories() {
  return useQuery({
    queryKey: documentQueryKeys.documentCategories,
    queryFn: () => Promise.resolve(FALLBACK_DATA.documentCategories), // Placeholder until API is available
    retry: 2,
    staleTime: 60000 * 5, // 5 minutes
    placeholderData: FALLBACK_DATA.documentCategories,
  })
}

/**
 * Get processing status for a document
 */
export function useProcessingStatus(documentId: string | null) {
  return useQuery({
    queryKey: documentQueryKeys.processingStatus(documentId || ''),
    queryFn: () => apiClient.processDocument(documentId || ''),
    enabled: !!documentId,
    retry: 2,
    refetchInterval: 5000, // Poll every 5 seconds while document is processing
    onError: (error) => handleApiError(error, 'Failed to fetch processing status')
  })
}

/**
 * Get processing queue status
 */
export function useProcessingQueue() {
  return useQuery({
    queryKey: documentQueryKeys.processingQueue,
    queryFn: () => Promise.resolve(FALLBACK_DATA.processingQueue), // Placeholder until API is ready
    retry: 2,
    refetchInterval: 30000, // Poll every 30 seconds instead of 10 to reduce frequency
    staleTime: 15000, // Data is considered fresh for 15 seconds
    placeholderData: FALLBACK_DATA.processingQueue,
    onError: (error) => handleApiError(error, 'Failed to load processing queue')
  })
}

/**
 * Search documents
 */
export function useDocumentSearch(query: string, filters: any = {}, enabled: boolean = true) {
  return useQuery({
    queryKey: [...documentQueryKeys.documentSearch(query), filters],
    queryFn: () => apiClient.getDocuments(),
    enabled: enabled && query.length > 2,
    retry: 1,
    staleTime: 30000,
    placeholderData: [],
    onError: (error) => handleApiError(error, 'Document search failed')
  })
}
/**
 * Semantic search for documents
 */
export function useSemanticSearch(query: string, options: any = {}, enabled: boolean = true) {
  return useQuery({
    queryKey: [...documentQueryKeys.semanticSearch(query), options],
    queryFn: () => apiClient.getDocuments(), // Use regular API until semantic endpoint exists
    enabled: enabled && query.length > 2,
    retry: 1,
    staleTime: 30000,
    placeholderData: [],
    onError: (error) => handleApiError(error, 'Semantic search failed')
  })
}

/**
 * Get document insights
 */
export function useDocumentInsights(documentId: string | null) {
  return useQuery({
    queryKey: documentQueryKeys.documentInsights(documentId || ''),
    queryFn: () => apiClient.getDocument(documentId || ''), // Use regular document endpoint until insights API exists
    enabled: !!documentId,
    retry: 1,
    staleTime: 60000,
    onError: (error) => handleApiError(error, 'Failed to load document insights')
  })
}

/**
 * Get document analytics data
 */
export function useDocumentAnalytics(timeRange: string = '24h') {
  return useQuery({
    queryKey: documentQueryKeys.analytics(timeRange),
    queryFn: () => apiClient.getDocumentAnalytics(),
    retry: 2,
    staleTime: 60000, // 1 minute
    onError: (error) => handleApiError(error, 'Failed to load document analytics')
  })
}

/**
 * Get usage analytics
 */
export function useUsageAnalytics(timeRange: string = '24h') {
  return useQuery({
    queryKey: documentQueryKeys.usageAnalytics(timeRange),
    queryFn: () => Promise.resolve({}), // Placeholder until API is available
    retry: 2,
    staleTime: 60000, // 1 minute
    onError: (error) => handleApiError(error, 'Failed to load usage analytics')
  })
}

/**
 * Get processing analytics
 */
export function useProcessingAnalytics(timeRange: string = '24h') {
  return useQuery({
    queryKey: documentQueryKeys.processingAnalytics(timeRange),
    queryFn: () => Promise.resolve({}), // Placeholder until API is available
    retry: 2,
    staleTime: 60000, // 1 minute
    onError: (error) => handleApiError(error, 'Failed to load processing analytics')
  })
}

/**
 * Get storage analytics
 */
export function useStorageAnalytics() {
  return useQuery({
    queryKey: documentQueryKeys.storageAnalytics,
    queryFn: () => Promise.resolve({}), // Placeholder until API is available
    retry: 2,
    staleTime: 60000 * 5, // 5 minutes
    onError: (error) => handleApiError(error, 'Failed to load storage analytics')
  })
}

// MUTATIONS

/**
 * Upload a document
 */
export function useUploadDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: { file: File, metadata?: any }) => {
      return apiClient.uploadDocument(data.file, data.metadata)
    },
    onSuccess: () => {
      toast.success('Document uploaded successfully')
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentStats })
    },
    onError: (error) => handleApiError(error, 'Failed to upload document')
  })
}

/**
 * Update document
 */
export function useUpdateDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: { id: string, updates: any }) => {
      return apiClient.updateDocument(data.id, data.updates)
    },
    onSuccess: (_, variables) => {
      toast.success('Document updated successfully')
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(variables.id) })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
    },
    onError: (error) => handleApiError(error, 'Failed to update document')
  })
}

/**
 * Delete document
 */
export function useDeleteDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => {
      return apiClient.deleteDocument(id)
    },
    onSuccess: (_, id) => {
      toast.success('Document deleted successfully')
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentStats })
      queryClient.removeQueries({ queryKey: documentQueryKeys.document(id) })
    },
    onError: (error) => handleApiError(error, 'Failed to delete document')
  })
}

/**
 * Start processing a document or all documents
 */
export function useStartProcessing() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: string | {}) => {
      // Handle both single document ID and reprocess all (empty object)
      if (typeof data === 'string') {
        return apiClient.processDocument(data)
      } else {
        // For reprocessing all, we'll use a placeholder API call
        return Promise.resolve({ message: 'Reprocessing started' })
      }
    },
    onSuccess: (_, data) => {
      if (typeof data === 'string') {
        toast.success('Document processing started')
        queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(data) })
        queryClient.invalidateQueries({ queryKey: documentQueryKeys.processingStatus(data) })
        
        // Start polling for status updates
        queryClient.setQueryDefaults(
          documentQueryKeys.processingStatus(data),
          { refetchInterval: 5000 }
        )
      } else {
        toast.success('Reprocessing all documents started')
      }
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.processingQueue })
    },
    onError: (error) => handleApiError(error, 'Failed to start document processing')
  })
}

/**
 * Analyze document
 */
export function useAnalyzeDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => {
      return apiClient.processDocument(id)
    },
    onSuccess: (_, id) => {
      toast.success('Document analysis started')
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(id) })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentInsights(id) })
    },
    onError: (error) => handleApiError(error, 'Failed to analyze document')
  })
}

/**
 * Generate summary for document
 */
export function useGenerateSummary() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => {
      return Promise.resolve({}) // Placeholder until API is available
    },
    onSuccess: (_, id) => {
      toast.success('Document summary generated')
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(id) })
    },
    onError: (error) => handleApiError(error, 'Failed to generate document summary')
  })
}

/**
 * Index document
 */
export function useIndexDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: { id: string, options?: any }) => {
      return apiClient.processDocument(data.id)
    },
    onSuccess: (_, variables) => {
      toast.success('Document indexed successfully')
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(variables.id) })
    },
    onError: (error) => handleApiError(error, 'Failed to index document')
  })
}

/**
 * Tag document
 */
export function useTagDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: { id: string, tags: string[] }) => {
      return apiClient.updateDocument(data.id, { tags: data.tags })
    },
    onSuccess: (_, variables) => {
      toast.success('Document tags updated')
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(variables.id) })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
    },
    onError: (error) => handleApiError(error, 'Failed to update document tags')
  })
}

/**
 * Batch delete documents
 */
export function useBatchDelete() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (ids: string[]) => {
      return Promise.all(ids.map(id => apiClient.deleteDocument(id)))
    },
    onSuccess: (_, ids) => {
      toast.success()
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentStats })
      ids.forEach(id => {
        queryClient.removeQueries({ queryKey: documentQueryKeys.document(id) })
      })
    },
    onError: (error) => handleApiError(error, 'Failed to delete documents')
  })
}

/**
 * Batch process documents
 */
export function useBatchProcess() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (ids: string[]) => {
      return Promise.all(ids.map(id => apiClient.processDocument(id)))
    },
    onSuccess: (_, ids) => {
      toast.success()
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.processingQueue })
      ids.forEach(id => {
        queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(id) })
        queryClient.invalidateQueries({ queryKey: documentQueryKeys.processingStatus(id) })
        
        // Start polling for status updates
        queryClient.setQueryDefaults(
          documentQueryKeys.processingStatus(id),
          { refetchInterval: 5000 }
        )
      })
    },
    onError: (error) => handleApiError(error, 'Failed to process documents')
  })
}
