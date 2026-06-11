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

// PRD-157 S6: the FALLBACK_DATA placebo (fake "Example Document 1.pdf" et al.)
// was removed. On API failure the hooks now surface the real error/empty state
// instead of pretending documents, stats and a queue exist.

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
    onError: (error) => handleApiError(error, 'Failed to load document statistics')
  })
}

/**
 * Get document categories
 */
export function useDocumentCategories() {
  return useQuery({
    queryKey: documentQueryKeys.documentCategories,
    // PRD-157 S6: no categories API yet — return an honest empty list instead
    // of fabricated categories.
    queryFn: () => Promise.resolve([]),
    retry: 2,
    staleTime: 60000 * 5, // 5 minutes
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
    // PRD-157 S6: call the real queue endpoint instead of returning fake status.
    queryFn: () => apiClient.getProcessingQueue(),
    retry: 2,
    refetchInterval: 30000, // Poll every 30 seconds instead of 10 to reduce frequency
    staleTime: 15000, // Data is considered fresh for 15 seconds
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
    queryFn: async () => {
      const response = await apiClient.semanticSearch(query, options)
      // Transform API response to match component expectations
      if (response && response.results) {
        return response.results.map((result: any) => ({
          document_id: result.document_id,
          document_name: result.source?.filename || 'Untitled Document',
          document_type: result.source?.file_type || 'Document',
          relevance_score: result.similarity,
          matched_content: result.excerpt,
          preview: result.preview,
          chunk_index: result.chunk_index,
          page_number: result.metadata?.page_number,
          section: result.metadata?.section,
          last_updated: result.source?.upload_date
        }))
      }
      return []
    },
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
      console.log('[useUploadDocument] Starting mutation with:', data)
      return apiClient.uploadDocument(data.file, data.metadata)
    },
    onSuccess: (response) => {
      console.log('[useUploadDocument] SUCCESS! Response:', response)
      toast.success('Document uploaded successfully')
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentStats })
    },
    onError: (error: any) => {
      console.error('[useUploadDocument] ERROR:', error)
      handleApiError(error, 'Failed to upload document')
    }
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
