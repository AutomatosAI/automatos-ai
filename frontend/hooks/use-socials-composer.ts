/**
 * Socials composer hooks (PRD-251 S2.2, US-207..US-209)
 * =====================================================
 *
 * The connected channels (GET /api/socials/channels), the brief → proposal call
 * (POST /api/socials/compose, never saved), and "Save draft": the post is
 * created (POST /api/socials/posts), then its channels are set (PUT
 * /api/socials/posts/{id}/targets). Every call goes through apiClient; the server
 * checks everything again.
 */
import { useMutation, useQuery } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import type {
  CreateSocialPostInput,
  SocialChannel,
  SocialComposeInput,
  SocialComposeProposal,
  SocialPost,
  SocialPostTargetInput,
} from '@/lib/api-client'
import { useInvalidateSocials, useSocialsOn } from '@/hooks/use-socials-api'

export const socialsComposerKeys = {
  channels: (workspaceId: string | null) => ['socials', workspaceId, 'channels'] as const,
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

/** A brief becomes a draft proposal (not saved). The server's message shows on failure. */
export function useComposeSocialPost() {
  return useMutation<SocialComposeProposal, Error, SocialComposeInput>({
    mutationFn: (input) => apiClient.composeSocialPost(input),
    onError: (error) => {
      toast.error(error.message || 'Could not draft the post')
    },
  })
}

export interface SaveDraftInput {
  post: CreateSocialPostInput
  targets: SocialPostTargetInput[]
}

async function saveDraft({ post, targets }: SaveDraftInput): Promise<SocialPost> {
  const created = await apiClient.createSocialPost(post)
  return targets.length > 0 ? apiClient.setSocialPostTargets(created.id, targets) : created
}

/** "Save draft": create the post, then set its channels; the list refetches. */
export function useSaveComposedDraft() {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialPost, Error, SaveDraftInput>({
    mutationFn: saveDraft,
    onSuccess: async () => {
      await invalidate()
      toast.success('Draft saved')
    },
    onError: async (error) => {
      await invalidate()
      toast.error(error.message || 'Could not save the draft')
    },
  })
}
