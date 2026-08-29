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

export function useSyncToolsCache() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (syncType: 'full' | 'incremental' = 'full') => apiClient.syncToolsCache(syncType),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tools'] })
      queryClient.invalidateQueries({ queryKey: ['tools', 'stats'] })
      queryClient.invalidateQueries({ queryKey: ['tools', 'categories'] })
      toast('Sync started/completed', { description: 'Marketplace cache sync finished. Refreshing tools list…' })
    },
    onError: (error: any) => {
      toast.error('Sync failed', { description: error.message || 'Failed to sync tools cache.' })
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
