/**
 * Socials channel and source hooks (PRD-251 S2.2, US-207..US-209)
 * ================================================================
 *
 * The connected channels (GET /api/socials/channels) and the claim source search
 * (GET /api/socials/sources). The composer's own hooks went with it: the post editor's
 * are in hooks/use-socials-editor.ts (PRD-251B US-B109). Every call goes through
 * apiClient; the server checks everything again.
 */
import { useQuery } from '@tanstack/react-query'

import { apiClient } from '@/lib/api-client'
import type { SocialChannel } from '@/lib/api-client'
import { useSocialsOn } from '@/hooks/use-socials-api'

export const socialsComposerKeys = {
  channels: (workspaceId: string | null) => ['socials', workspaceId, 'channels'] as const,
  sources: (workspaceId: string | null, q: string) => ['socials', workspaceId, 'sources', q] as const,
}

/** What a claim can be bound to (D7), searched as the person types. */
export function useSocialSourceSearch(q: string, enabled: boolean) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery({
    queryKey: socialsComposerKeys.sources(workspaceId, q),
    enabled: socialsOn && enabled,
    queryFn: () => apiClient.searchSocialSources({ q }),
    staleTime: 30_000,
  })
}

/** The workspace's connected channels, each with its post kinds and copy limits. */
export function useSocialChannels() {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialChannel[]>({
    queryKey: socialsComposerKeys.channels(workspaceId),
    enabled: socialsOn,
    queryFn: () => apiClient.listSocialChannels(),
    staleTime: 60_000,
  })
}
