/**
 * Cloud Documents API hooks (PRD-42)
 * TanStack Query hooks for cloud storage connections, folder navigation,
 * sync operations, and cloud RAG queries.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import apiClient from '../lib/api-client'

// ---------------------------------------------------------------------------
// Types (matching backend Pydantic models)
// ---------------------------------------------------------------------------

export interface CloudConnection {
  id: number
  app_name: string
  status: string
  connected_at: string | null
  sync_enabled: boolean
  total_documents_synced: number
  last_successful_sync: string | null
  root_folder_path: string | null
}

export interface CloudFolder {
  name: string
  path: string
  has_children: boolean
}

export interface CloudFile {
  external_file_id: string
  name: string
  path: string
  mime_type: string
  size: number
  modified_at: string | null
  is_synced: boolean
  sync_status: string
  chunk_count: number
  last_synced_at: string | null
}

export interface SyncJob {
  id: number
  status: string
  started_at: string | null
  completed_at: string | null
  files_synced: number
  files_skipped: number
  files_errored: number
  total_chunks_created: number
  error_message: string | null
}

export interface SyncStatus {
  connection_id: number
  app_name: string
  sync_enabled: boolean
  last_sync_at: string | null
  total_documents: number
  synced: number
  pending: number
  errors: number
  total_chunks: number
}

export interface CloudRAGResult {
  query: string
  context: string
  chunks: Array<{
    document_id: number | null
    file_name: string
    app_name: string
    file_path: string
    chunk_index: number
    content: string
    similarity_score: number
  }>
  sources: Array<{
    document_id: number | null
    file_name: string
    app_name: string
    file_path: string
    chunks_used: number
  }>
  total_tokens: number
}

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------

export const cloudDocumentKeys = {
  connections: ['cloud-documents', 'connections'] as const,
  folders: (connectionId: number, path: string) =>
    ['cloud-documents', 'folders', connectionId, path] as const,
  files: (connectionId: number, path: string) =>
    ['cloud-documents', 'files', connectionId, path] as const,
  syncJob: (jobId: number) =>
    ['cloud-documents', 'sync-job', jobId] as const,
  syncStatus: (connectionId: number) =>
    ['cloud-documents', 'sync-status', connectionId] as const,
  ragQuery: (query: string) =>
    ['cloud-documents', 'rag', query] as const,
}

// ---------------------------------------------------------------------------
// Query hooks
// ---------------------------------------------------------------------------

/** List connected cloud storage providers for this workspace. */
export function useCloudConnections() {
  return useQuery({
    queryKey: cloudDocumentKeys.connections,
    queryFn: () => apiClient.get<CloudConnection[]>('/api/cloud-documents/connections'),
    retry: 2,
    staleTime: 30_000,
    onError: () => toast.error('Failed to load cloud connections'),
  })
}

/** List folders at a path (lazy-load tree). */
export function useCloudFolders(connectionId: number, path: string, enabled = true) {
  return useQuery({
    queryKey: cloudDocumentKeys.folders(connectionId, path),
    queryFn: () =>
      apiClient.get<CloudFolder[]>(
        `/api/cloud-documents/connections/${connectionId}/folders?path=${encodeURIComponent(path)}`,
      ),
    enabled: enabled && connectionId > 0,
    retry: 1,
    staleTime: 60_000,
    onError: () => toast.error('Failed to load folders'),
  })
}

/** List files at a path with sync status. */
export function useCloudFiles(connectionId: number, path: string, enabled = true) {
  return useQuery({
    queryKey: cloudDocumentKeys.files(connectionId, path),
    queryFn: () =>
      apiClient.get<CloudFile[]>(
        `/api/cloud-documents/connections/${connectionId}/files?path=${encodeURIComponent(path)}`,
      ),
    enabled: enabled && connectionId > 0,
    retry: 1,
    staleTime: 30_000,
    onError: () => toast.error('Failed to load files'),
  })
}

/** Poll a sync job for progress. */
export function useSyncJob(jobId: number | null, enabled = true) {
  return useQuery({
    queryKey: cloudDocumentKeys.syncJob(jobId ?? 0),
    queryFn: () =>
      apiClient.get<SyncJob>(`/api/cloud-documents/sync-jobs/${jobId}`),
    enabled: enabled && !!jobId,
    refetchInterval: (data) =>
      data?.status === 'running' ? 2_000 : false,
    retry: 1,
    onError: () => toast.error('Failed to get sync job status'),
  })
}

/** Get sync summary for a connection. */
export function useSyncStatus(connectionId: number, enabled = true) {
  return useQuery({
    queryKey: cloudDocumentKeys.syncStatus(connectionId),
    queryFn: () =>
      apiClient.get<SyncStatus>(
        `/api/cloud-documents/connections/${connectionId}/sync-status`,
      ),
    enabled: enabled && connectionId > 0,
    retry: 1,
    staleTime: 15_000,
    onError: () => toast.error('Failed to load sync status'),
  })
}

// ---------------------------------------------------------------------------
// Mutation hooks
// ---------------------------------------------------------------------------

/** Select root sync folder for a connection. */
export function useSelectFolder() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (args: { connectionId: number; rootFolderPath: string }) =>
      apiClient.post(
        `/api/cloud-documents/connections/${args.connectionId}/select-folder`,
        { root_folder_path: args.rootFolderPath },
      ),
    onSuccess: () => {
      toast.success('Root folder selected')
      queryClient.invalidateQueries({ queryKey: cloudDocumentKeys.connections })
    },
    onError: () => toast.error('Failed to select folder'),
  })
}

/** Trigger a manual sync. */
export function useTriggerSync() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (connectionId: number) =>
      apiClient.post<{ job_id: number; status: string; message: string }>(
        `/api/cloud-documents/connections/${connectionId}/sync`,
      ),
    onSuccess: (data, connectionId) => {
      toast.success(`Sync started (Job #${data.job_id})`)
      queryClient.invalidateQueries({
        queryKey: cloudDocumentKeys.syncStatus(connectionId),
      })
    },
    onError: () => toast.error('Failed to start sync'),
  })
}

/** Disconnect a cloud provider. */
export function useDisconnectConnection() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (args: { connectionId: number; deleteVectors: boolean }) =>
      apiClient.delete<{ status: string; vectors_deleted: boolean; documents_affected: number }>(
        `/api/cloud-documents/connections/${args.connectionId}?delete_vectors=${args.deleteVectors}`,
      ),
    onSuccess: () => {
      toast.success('Cloud storage disconnected')
      queryClient.invalidateQueries({ queryKey: cloudDocumentKeys.connections })
    },
    onError: () => toast.error('Failed to disconnect'),
  })
}

/** Cloud RAG query against S3 Vectors. */
export function useCloudRAGQuery() {
  return useMutation({
    mutationFn: (args: {
      query: string
      top_k?: number
      min_similarity?: number
    }) =>
      apiClient.post<CloudRAGResult>('/api/cloud-documents/rag/query', {
        query: args.query,
        top_k: args.top_k ?? 10,
        min_similarity: args.min_similarity ?? 0.5,
      }),
    onError: () => toast.error('Cloud RAG query failed'),
  })
}
