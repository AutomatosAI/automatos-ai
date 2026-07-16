/**
 * Watchlist React Query hooks -- PRD-204 S11 (Auto Watcher).
 *
 * Reads `GET /api/v1/watches` (live watches; `includeClosed` widens to the
 * full history) and drives the cancel verb. Same shape as
 * use-approval-grants.ts; polling mirrors use-activity-api.ts (60s interval,
 * 30s staleTime) -- the watcher tick runs on minutes, so a minute-fresh list
 * is honest without hammering.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { WatchDetailResponse, WatchesResponse } from '@/lib/api-client'

export const watchesQueryKeys = {
  all: ['watches'] as const,
  list: (includeClosed?: boolean) =>
    ['watches', 'list', includeClosed ? 'all' : 'live'] as const,
  detail: (watchId: string) => ['watches', 'detail', watchId] as const,
}

export function useWatches(includeClosed: boolean = false) {
  return useQuery<WatchesResponse>({
    queryKey: watchesQueryKeys.list(includeClosed),
    queryFn: () => apiClient.listWatches(includeClosed),
    refetchInterval: 60_000,
    staleTime: 30_000,
  })
}

export function useWatchDetail(watchId: string | null) {
  return useQuery<WatchDetailResponse>({
    queryKey: watchesQueryKeys.detail(watchId ?? 'none'),
    queryFn: () => apiClient.getWatch(watchId as string),
    enabled: !!watchId,
    refetchInterval: 60_000,
    staleTime: 30_000,
  })
}

export function useCancelWatch() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (watchId: string) => apiClient.cancelWatch(watchId),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: watchesQueryKeys.all }),
  })
}
