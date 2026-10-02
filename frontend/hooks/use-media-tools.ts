'use client'

/**
 * PRD-251B US-B304 — the AI tools a workspace uses (GET/PUT /api/socials/media-tools): the
 * Brand kit tab's AI tools section and the Plan page's summary read it. It sits behind
 * Socials (D1), so it is only asked for while Socials is on for the workspace.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import type { SocialMediaToolsInput, SocialMediaToolsResponse } from '@/lib/brand-style-types'
import { useSocialsOn } from '@/hooks/use-socials-api'

export const mediaToolsKey = (workspaceId: string | null) => ['socials', workspaceId, 'media-tools'] as const

export function useSocialMediaTools() {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialMediaToolsResponse>({
    queryKey: mediaToolsKey(workspaceId),
    enabled: socialsOn,
    queryFn: () => apiClient.getSocialMediaTools(),
    staleTime: 30_000,
  })
}

export function useUpdateSocialMediaTools() {
  const { workspaceId } = useSocialsOn()
  const client = useQueryClient()
  return useMutation<SocialMediaToolsResponse, Error, SocialMediaToolsInput>({
    mutationFn: (input) => apiClient.updateSocialMediaTools(input),
    onSuccess: (tools) => {
      client.setQueryData(mediaToolsKey(workspaceId), tools)
      toast.success('AI tools saved')
    },
    onError: (error) => {
      toast.error(error.message || 'Could not save the AI tools')
    },
  })
}
