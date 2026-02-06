import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

interface CloudConnection {
  id: number
  app_name: string
  status: string
  connected_at?: string
  sync_enabled: boolean
  total_documents_synced: number
  last_successful_sync?: string
  root_folder_path?: string
}

interface CloudFolder {
  name: string
  path: string
  has_children: boolean
}

interface CloudFile {
  external_file_id: string
  name: string
  path: string
  mime_type?: string
  size?: number
  modified_at?: string
  is_synced: boolean
  sync_status: 'pending' | 'syncing' | 'synced' | 'error'
  chunk_count: number
  last_synced_at?: string
}

interface SyncJobResponse {
  job_id: number
  status: string
  message: string
}

interface SyncJob {
  id: number
  status: string
  started_at?: string
  completed_at?: string
  files_synced: number
  files_skipped: number
  files_errored: number
  total_chunks_created: number
  error_message?: string
}

// Get all cloud storage connections
export function useCloudConnections() {
  return useQuery<CloudConnection[]>({
    queryKey: ['cloud-connections'],
    queryFn: async () => {
      console.log('[useCloudConnections] Fetching connections...')
      const response = await apiClient.get('/api/cloud-documents/connections')
      console.log('[useCloudConnections] Response:', response)
      return response
    },
    staleTime: 30000 // 30 seconds
  })
}

// Get folders at a specific path
export function useCloudFolders(connectionId: number | null, path: string = '/') {
  return useQuery<CloudFolder[]>({
    queryKey: ['cloud-folders', connectionId, path],
    queryFn: async () => {
      if (!connectionId) return []
      const params = new URLSearchParams({ path })
      const response = await apiClient.get(
        `/api/cloud-documents/connections/${connectionId}/folders?${params}`
      )
      return response
    },
    enabled: connectionId !== null,
    staleTime: 60000 // 1 minute
  })
}

// Get files at a specific path
export function useCloudFiles(
  connectionId: number | null,
  path: string = '/',
  recursive: boolean = false
) {
  return useQuery<CloudFile[]>({
    queryKey: ['cloud-files', connectionId, path, recursive],
    queryFn: async () => {
      if (!connectionId) return []
      const params = new URLSearchParams({ path, recursive: String(recursive) })
      const response = await apiClient.get(
        `/api/cloud-documents/connections/${connectionId}/files?${params}`
      )
      return response
    },
    enabled: connectionId !== null,
    staleTime: 30000 // 30 seconds
  })
}

// Select root folder for syncing
export function useSelectRootFolder() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ connectionId, rootFolderPath }: {
      connectionId: number
      rootFolderPath: string
    }) => {
      const response = await apiClient.post(
        `/api/cloud-documents/connections/${connectionId}/select-folder`,
        { root_folder_path: rootFolderPath }
      )
      return response
    },
    onSuccess: () => {
      // Invalidate connections to refresh root folder info
      queryClient.invalidateQueries({ queryKey: ['cloud-connections'] })
    }
  })
}

// Trigger sync for a connection
export function useTriggerSync() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (connectionId: number) => {
      const response = await apiClient.post<SyncJobResponse>(
        `/api/cloud-documents/connections/${connectionId}/sync`
      )
      return response
    },
    onSuccess: (_, connectionId) => {
      // Invalidate files and connections to show updated sync status
      queryClient.invalidateQueries({ queryKey: ['cloud-files', connectionId] })
      queryClient.invalidateQueries({ queryKey: ['cloud-connections'] })
    }
  })
}

// Get sync job status
export function useSyncJob(jobId: number | null) {
  return useQuery<SyncJob>({
    queryKey: ['sync-job', jobId],
    queryFn: async () => {
      if (!jobId) throw new Error('No job ID')
      const response = await apiClient.get(`/api/cloud-documents/sync-jobs/${jobId}`)
      return response
    },
    enabled: jobId !== null,
    refetchInterval: (data) => {
      // Poll every 2 seconds while job is running
      return data?.status === 'running' ? 2000 : false
    }
  })
}

// Get sync status for a connection
export function useSyncStatus(connectionId: number | null) {
  return useQuery({
    queryKey: ['sync-status', connectionId],
    queryFn: async () => {
      if (!connectionId) return null
      const response = await apiClient.get(
        `/api/cloud-documents/connections/${connectionId}/sync-status`
      )
      return response
    },
    enabled: connectionId !== null,
    staleTime: 10000 // 10 seconds
  })
}

// Disconnect a cloud storage provider
export function useDisconnectProvider() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({
      connectionId,
      deleteVectors
    }: {
      connectionId: number
      deleteVectors: boolean
    }) => {
      const params = new URLSearchParams({ delete_vectors: String(deleteVectors) })
      const response = await apiClient.delete(
        `/api/cloud-documents/connections/${connectionId}?${params}`
      )
      return response
    },
    onSuccess: () => {
      // Refresh connections list
      queryClient.invalidateQueries({ queryKey: ['cloud-connections'] })
    }
  })
}
