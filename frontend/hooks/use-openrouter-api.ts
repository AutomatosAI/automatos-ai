/**
 * OpenRouter Marketplace API Hooks
 * =================================
 *
 * React Query hooks for the OpenRouter model cache.
 * Completely independent from Composio tools sync.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useAuth } from '@clerk/nextjs'
import { apiClient } from '@/lib/api-client'
import { useToast } from './use-toast'

export function useSyncOpenRouterCache() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: () => apiClient.syncOpenRouterCache(),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['openrouterModels'] })
      queryClient.invalidateQueries({ queryKey: ['openrouterStats'] })
      queryClient.invalidateQueries({ queryKey: ['openrouterProviders'] })
      toast({
        title: 'LLM sync complete',
        description: `Synced ${data?.models_synced ?? 0} models from OpenRouter (${data?.models_added ?? 0} new, ${data?.models_updated ?? 0} updated)`,
      })
    },
    onError: (error: any) => {
      toast({
        title: 'LLM sync failed',
        description: error.message || 'Failed to sync OpenRouter models.',
        variant: 'destructive',
      })
    },
  })
}

export function useOpenRouterStats() {
  const { isLoaded, isSignedIn } = useAuth()
  return useQuery({
    queryKey: ['openrouterStats'],
    queryFn: () => apiClient.get('/api/openrouter/stats'),
    staleTime: 5 * 60 * 1000,
    enabled: isLoaded && isSignedIn,
    refetchOnWindowFocus: false,
  })
}

export function useOpenRouterProviders() {
  const { isLoaded, isSignedIn } = useAuth()
  return useQuery<{ provider: string; count: number }[]>({
    queryKey: ['openrouterProviders'],
    queryFn: () => apiClient.get('/api/openrouter/providers'),
    staleTime: 5 * 60 * 1000,
    enabled: isLoaded && isSignedIn,
    refetchOnWindowFocus: false,
  })
}
