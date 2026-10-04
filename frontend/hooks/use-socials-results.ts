/**
 * PRD-251C Wave 4 — React Query over what Socials learns once posts go out: Posted (US-C408).
 * Keys sit under the Socials key of the workspace (use-socials-api.ts), so a workspace switch
 * clears them with the rest.
 */
import { useQuery } from '@tanstack/react-query'

import { useWorkspace } from '@/components/workspace-provider'
import { apiClient } from '@/lib/api-client'
import type { SocialPostedFilters } from '@/lib/socials-results-types'
import { socialsQueryKeys } from './use-socials-api'

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

export const resultsQueryKeys = {
  posted: (ws: string | null, filters: SocialPostedFilters) =>
    [...socialsQueryKeys.all(ws), 'posted', filters.planId ?? '', filters.channel ?? '', filters.format ?? ''] as const,
}

/** Posted: what went out, newest first, with the filters given. */
export function useSocialPosted(filters: SocialPostedFilters) {
  const ws = useWorkspaceId()
  return useQuery({
    queryKey: resultsQueryKeys.posted(ws, filters),
    queryFn: () => apiClient.listSocialPosted(filters),
    enabled: !!ws,
  })
}
