import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

/** PRD-239 S7b: whether this workspace's worker can run an SDK ("Auto session") at all.
 *  `undefined` while unknown (loading, or an older worker) — the Canvas then keeps the tab. */
export function useCanvasSdkAvailable(workspaceId: string | undefined): boolean | undefined {
  const { data } = useQuery({
    queryKey: ['canvas-sdk', workspaceId],
    queryFn: () => apiClient.request<{ sdk_available?: boolean }>(`/api/workspaces/${workspaceId}/canvas/sessions`),
    enabled: Boolean(workspaceId),
    staleTime: 60_000,
    retry: false,
  })
  return typeof data?.sdk_available === 'boolean' ? data.sdk_available : undefined
}
