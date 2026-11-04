/**
 * Tools API Hooks
 * ==============
 *
 * React Query hooks for the Tools/Integrations UI.
 * Uses the rewrite `/api/tools/*` endpoints (DB-backed cache + connections).
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useAuth } from '@/lib/auth-hooks'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

function shouldRetry(error: any) {
  const msg = String(error?.message || '')
  // Avoid retry storms when auth isn't ready / token missing.
  if (msg.includes('HTTP 401')) return false
  return true
}

export function useTools(params?: {
  status?: string
  category?: string
  provider?: string
  search?: string
  skip?: number
  limit?: number
}) {
  const { isLoaded, isSignedIn } = useAuth()
  return useQuery({
    queryKey: ['tools', params],
    queryFn: () => apiClient.getTools(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
    enabled: isLoaded && isSignedIn,
    retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
}

export function useToolsStats() {
  const { isLoaded, isSignedIn } = useAuth()
  return useQuery({
    queryKey: ['tools', 'stats'],
    queryFn: () => apiClient.getToolsStats(),
    // Avoid polling storms; rely on manual refresh + invalidations.
    refetchInterval: false,
    enabled: isLoaded && isSignedIn,
    retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
}

export function useToolCategories() {
  const { isLoaded, isSignedIn } = useAuth()
  return useQuery({
    queryKey: ['tools', 'categories'],
    queryFn: () => apiClient.getToolCategories(),
    staleTime: 10 * 60 * 1000, // 10 minutes
    enabled: isLoaded && isSignedIn,
    retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
}

export type ToolsSyncResult = Awaited<ReturnType<typeof apiClient.syncToolsCache>>

export const SYNC_ALREADY_RUNNING_TITLE = 'A catalogue sync is already running.'
export const SYNC_FAILED_TITLE = 'Sync failed'
export const SYNC_FAILED_FALLBACK = 'The catalogue sync did not complete. The backend log has the details.'

/**
 * The sync route answers 200 with the job, so a sync that failed (Composio rejected
 * the key, say) or one already running must not read as "finished".
 */
function announceSyncResult(result: ToolsSyncResult) {
  if (result?.already_running) {
    toast(SYNC_ALREADY_RUNNING_TITLE, { description: 'This Sync did not start another one. Refresh the page when it is done.' })
  } else if (result?.status === 'failed') {
    toast.error(SYNC_FAILED_TITLE, { description: result.error || SYNC_FAILED_FALLBACK })
  } else {
    toast('Sync started/completed', { description: 'Marketplace cache sync finished. Refreshing tools list…' })
  }
}

export function useSyncToolsCache() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (syncType: 'full' | 'incremental' = 'full') => apiClient.syncToolsCache(syncType),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['tools'] })
      queryClient.invalidateQueries({ queryKey: ['tools', 'stats'] })
      queryClient.invalidateQueries({ queryKey: ['tools', 'categories'] })
      announceSyncResult(result)
    },
    onError: (error: any) => {
      toast.error(SYNC_FAILED_TITLE, { description: error.message || 'Failed to sync tools cache.' })
    },
  })
}

export type IntegrationsStatus = Awaited<ReturnType<typeof apiClient.getIntegrationsStatus>>

/**
 * PRD-233 S2 — can Composio integrations run on this server? Drives the
 * "integrations disabled" card on the Tools page; `available` is the same
 * predicate the backend tool router uses, so the UI never claims more than
 * the agents can actually reach.
 */
export function useIntegrationsStatus() {
  const { isLoaded, isSignedIn } = useAuth()
  return useQuery({
    queryKey: ['tools', 'integrations-status'],
    queryFn: () => apiClient.getIntegrationsStatus(),
    staleTime: 5 * 60 * 1000, // 5 minutes — flips only on a backend restart
    enabled: isLoaded && isSignedIn,
    retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
}
