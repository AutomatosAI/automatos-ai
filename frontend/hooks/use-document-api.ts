
/**
 * Enhanced Document API hooks for real-time data fetching and mutations
 * Provides React Query integration with automatic caching, retries, and real-time updates
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'

// API client - you'll need to adjust the import path based on your project structure
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const API_PREFIX = process.env.NEXT_PUBLIC_API_PREFIX || '/api'

// Simple API client for demonstration - replace with your actual API client
const apiClient = {
  async request(endpoint: string, options: RequestInit = {}) {
    const url = `${API_BASE}${API_PREFIX}${endpoint}`
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  },

  // Document endpoints
  getDocuments: () => apiClient.request('/documents'),
  getDocument: (id: string) => apiClient.request(`/documents/${id}`),
  getDocumentStats: () => apiClient.request('/documents/stats'),
  getDocumentCategories: () => apiClient.request('/documents/categories'),
  createDocument: (data: any) => apiClient.request('/documents', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  updateDocument: (id: string, data: any) => apiClient.request(`/documents/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  deleteDocument: (id: string) => apiClient.request(`/documents/${id}`, {
    method: 'DELETE',
  }),

  // File upload endpoints
  uploadDocument: async (file: File, metadata: any = {}) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('metadata', JSON.stringify(metadata))

    const response = await fetch(`${API_BASE}${API_PREFIX}/documents/upload`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status} ${response.statusText}`)
    }

    return response.json()
  },

  // Processing endpoints
  getProcessingStatus: (documentId: string) => apiClient.request(`/documents/${documentId}/processing`),
  startProcessing: (documentId: string, options: any = {}) => apiClient.request(`/documents/${documentId}/process`, {
    method: 'POST',
    body: JSON.stringify(options),
  }),
  getProcessingQueue: () => apiClient.request('/documents/processing/queue'),

  // Search and analysis endpoints
  searchDocuments: (query: string, filters: any = {}) => apiClient.request('/documents/search', {
    method: 'POST',
    body: JSON.stringify({ query, filters }),
  }),
  analyzeDocument: (documentId: string) => apiClient.request(`/documents/${documentId}/analyze`, {
    method: 'POST',
  }),
  getDocumentInsights: (documentId: string) => apiClient.request(`/documents/${documentId}/insights`),
  semanticSearch: (query: string, options: any = {}) => apiClient.request('/documents/semantic-search', {
    method: 'POST',
    body: JSON.stringify({ query, options }),
  }),

  // Analytics endpoints
  getDocumentAnalytics: (timeRange: string) => apiClient.request(`/documents/analytics?timeRange=${timeRange}`),
  getUsageAnalytics: (timeRange: string) => apiClient.request(`/documents/analytics/usage?timeRange=${timeRange}`),
  getProcessingAnalytics: (timeRange: string) => apiClient.request(`/documents/analytics/processing?timeRange=${timeRange}`),
  getStorageAnalytics: () => apiClient.request('/documents/analytics/storage'),

  // Content extraction endpoints
  extractText: (documentId: string) => apiClient.request(`/documents/${documentId}/extract/text`),
  extractMetadata: (documentId: string) => apiClient.request(`/documents/${documentId}/extract/metadata`),
  extractEntities: (documentId: string) => apiClient.request(`/documents/${documentId}/extract/entities`),
  generateSummary: (documentId: string) => apiClient.request(`/documents/${documentId}/summarize`, {
    method: 'POST',
  }),

  // RAG integration endpoints
  indexDocument: (documentId: string) => apiClient.request(`/documents/${documentId}/index`, {
    method: 'POST',
  }),
  queryDocuments: (query: string, documentIds: string[] = []) => apiClient.request('/documents/query', {
    method: 'POST',
    body: JSON.stringify({ query, document_ids: documentIds }),
  }),
  getDocumentChunks: (documentId: string) => apiClient.request(`/documents/${documentId}/chunks`),

  // Management endpoints
  moveToFolder: (documentId: string, folderId: string) => apiClient.request(`/documents/${documentId}/move`, {
    method: 'POST',
    body: JSON.stringify({ folder_id: folderId }),
  }),
  tagDocument: (documentId: string, tags: string[]) => apiClient.request(`/documents/${documentId}/tags`, {
    method: 'POST',
    body: JSON.stringify({ tags }),
  }),
  shareDocument: (documentId: string, shareOptions: any) => apiClient.request(`/documents/${documentId}/share`, {
    method: 'POST',
    body: JSON.stringify(shareOptions),
  }),
  duplicateDocument: (documentId: string) => apiClient.request(`/documents/${documentId}/duplicate`, {
    method: 'POST',
  }),

  // Batch operations
  batchDelete: (documentIds: string[]) => apiClient.request('/documents/batch/delete', {
    method: 'POST',
    body: JSON.stringify({ document_ids: documentIds }),
  }),
  batchProcess: (documentIds: string[], options: any = {}) => apiClient.request('/documents/batch/process', {
    method: 'POST',
    body: JSON.stringify({ document_ids: documentIds, options }),
  }),
  batchTag: (documentIds: string[], tags: string[]) => apiClient.request('/documents/batch/tag', {
    method: 'POST',
    body: JSON.stringify({ document_ids: documentIds, tags }),
  }),
}

// Query keys for React Query
export const documentQueryKeys = {
  // Documents
  documents: ['documents'] as const,
  document: (id: string) => ['documents', id] as const,
  documentStats: ['documents', 'stats'] as const,
  documentCategories: ['documents', 'categories'] as const,
  
  // Processing
  processingStatus: (id: string) => ['documents', id, 'processing'] as const,
  processingQueue: ['documents', 'processing', 'queue'] as const,
  
  // Search
  search: (query: string, filters: any) => ['documents', 'search', query, filters] as const,
  semanticSearch: (query: string, options: any) => ['documents', 'semantic-search', query, options] as const,
  
  // Analytics
  analytics: (timeRange: string) => ['documents', 'analytics', timeRange] as const,
  usageAnalytics: (timeRange: string) => ['documents', 'analytics', 'usage', timeRange] as const,
  processingAnalytics: (timeRange: string) => ['documents', 'analytics', 'processing', timeRange] as const,
  storageAnalytics: ['documents', 'analytics', 'storage'] as const,
  
  // Content
  documentInsights: (id: string) => ['documents', id, 'insights'] as const,
  documentText: (id: string) => ['documents', id, 'text'] as const,
  documentMetadata: (id: string) => ['documents', id, 'metadata'] as const,
  documentEntities: (id: string) => ['documents', id, 'entities'] as const,
  documentChunks: (id: string) => ['documents', id, 'chunks'] as const,
}

// ============= QUERY HOOKS =============

// Get all documents
export function useDocuments() {
  return useQuery({
    queryKey: documentQueryKeys.documents,
    queryFn: apiClient.getDocuments,
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // Consider data stale after 15 seconds
  })
}

// Get single document
export function useDocument(documentId: string | null) {
  return useQuery({
    queryKey: documentQueryKeys.document(documentId!),
    queryFn: () => apiClient.getDocument(documentId!),
    enabled: !!documentId,
    refetchInterval: 10000,
  })
}

// Get document statistics
export function useDocumentStats() {
  return useQuery({
    queryKey: documentQueryKeys.documentStats,
    queryFn: apiClient.getDocumentStats,
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get document categories
export function useDocumentCategories() {
  return useQuery({
    queryKey: documentQueryKeys.documentCategories,
    queryFn: apiClient.getDocumentCategories,
    staleTime: 5 * 60 * 1000, // Categories don't change often
  })
}

// Get processing status
export function useProcessingStatus(documentId: string | null) {
  return useQuery({
    queryKey: documentQueryKeys.processingStatus(documentId!),
    queryFn: () => apiClient.getProcessingStatus(documentId!),
    enabled: !!documentId,
    refetchInterval: 2000, // Check processing status frequently
  })
}

// Get processing queue
export function useProcessingQueue() {
  return useQuery({
    queryKey: documentQueryKeys.processingQueue,
    queryFn: apiClient.getProcessingQueue,
    refetchInterval: 5000,
  })
}

// Search documents
export function useDocumentSearch(query: string, filters: any = {}, enabled: boolean = true) {
  return useQuery({
    queryKey: documentQueryKeys.search(query, filters),
    queryFn: () => apiClient.searchDocuments(query, filters),
    enabled: enabled && !!query,
    staleTime: 30000,
  })
}

// Semantic search
export function useSemanticSearch(query: string, options: any = {}, enabled: boolean = true) {
  return useQuery({
    queryKey: documentQueryKeys.semanticSearch(query, options),
    queryFn: () => apiClient.semanticSearch(query, options),
    enabled: enabled && !!query,
    staleTime: 30000,
  })
}

// Get document insights
export function useDocumentInsights(documentId: string | null) {
  return useQuery({
    queryKey: documentQueryKeys.documentInsights(documentId!),
    queryFn: () => apiClient.getDocumentInsights(documentId!),
    enabled: !!documentId,
  })
}

// Get document analytics
export function useDocumentAnalytics(timeRange: string = '24h') {
  return useQuery({
    queryKey: documentQueryKeys.analytics(timeRange),
    queryFn: () => apiClient.getDocumentAnalytics(timeRange),
    refetchInterval: 60000, // Refetch every minute
  })
}

// Get usage analytics
export function useUsageAnalytics(timeRange: string = '24h') {
  return useQuery({
    queryKey: documentQueryKeys.usageAnalytics(timeRange),
    queryFn: () => apiClient.getUsageAnalytics(timeRange),
    refetchInterval: 60000,
  })
}

// Get processing analytics
export function useProcessingAnalytics(timeRange: string = '24h') {
  return useQuery({
    queryKey: documentQueryKeys.processingAnalytics(timeRange),
    queryFn: () => apiClient.getProcessingAnalytics(timeRange),
    refetchInterval: 60000,
  })
}

// Get storage analytics
export function useStorageAnalytics() {
  return useQuery({
    queryKey: documentQueryKeys.storageAnalytics,
    queryFn: apiClient.getStorageAnalytics,
    refetchInterval: 5 * 60 * 1000, // Every 5 minutes
  })
}

// ============= MUTATION HOOKS =============

// Upload document
export function useUploadDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ file, metadata }: { file: File; metadata?: any }) =>
      apiClient.uploadDocument(file, metadata),
    onSuccess: (data) => {
      // Invalidate and refetch documents
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentStats })
      toast.success('Document uploaded successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to upload document: ${error.message}`)
      throw error
    },
  })
}

// Update document
export function useUpdateDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) => 
      apiClient.updateDocument(id, data),
    onSuccess: (data, variables) => {
      // Invalidate specific document and list
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(variables.id) })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      toast.success('Document updated successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to update document: ${error.message}`)
      throw error
    },
  })
}

// Delete document
export function useDeleteDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: apiClient.deleteDocument,
    onSuccess: (data, documentId) => {
      // Remove from cache and invalidate lists
      queryClient.removeQueries({ queryKey: documentQueryKeys.document(documentId) })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentStats })
      toast.success('Document deleted successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to delete document: ${error.message}`)
      throw error
    },
  })
}

// Start document processing
export function useStartProcessing() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ documentId, options }: { documentId: string; options?: any }) =>
      apiClient.startProcessing(documentId, options),
    onSuccess: (data, variables) => {
      // Invalidate processing status and queue
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.processingStatus(variables.documentId) })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.processingQueue })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      toast.success('Document processing started!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to start processing: ${error.message}`)
      throw error
    },
  })
}

// Analyze document
export function useAnalyzeDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: apiClient.analyzeDocument,
    onSuccess: (data, documentId) => {
      // Invalidate document insights and document data
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentInsights(documentId) })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(documentId) })
      toast.success('Document analysis completed!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to analyze document: ${error.message}`)
      throw error
    },
  })
}

// Generate summary
export function useGenerateSummary() {
  return useMutation({
    mutationFn: apiClient.generateSummary,
    onSuccess: (data) => {
      toast.success('Summary generated successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to generate summary: ${error.message}`)
      throw error
    },
  })
}

// Index document for RAG
export function useIndexDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: apiClient.indexDocument,
    onSuccess: (data, documentId) => {
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(documentId) })
      toast.success('Document indexed successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to index document: ${error.message}`)
      throw error
    },
  })
}

// Tag document
export function useTagDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ documentId, tags }: { documentId: string; tags: string[] }) =>
      apiClient.tagDocument(documentId, tags),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.document(variables.documentId) })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      toast.success('Document tagged successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to tag document: ${error.message}`)
      throw error
    },
  })
}

// Batch operations
export function useBatchDelete() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: apiClient.batchDelete,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documentStats })
      toast.success('Documents deleted successfully!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to delete documents: ${error.message}`)
      throw error
    },
  })
}

export function useBatchProcess() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ documentIds, options }: { documentIds: string[]; options?: any }) =>
      apiClient.batchProcess(documentIds, options),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.documents })
      queryClient.invalidateQueries({ queryKey: documentQueryKeys.processingQueue })
      toast.success('Batch processing started!')
      return data
    },
    onError: (error) => {
      toast.error(`Failed to start batch processing: ${error.message}`)
      throw error
    },
  })
}

