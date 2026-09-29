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
  post: (workspaceId: string | null, postId: string | null) => ['socials', workspaceId, 'post', postId] as const,
  sources: (workspaceId: string | null, q: string) => ['socials', workspaceId, 'sources', q] as const,
}

/** How often the composer rereads its post while a render or a preview runs. */
export const SOCIALS_PREVIEW_POLL_MS = 5_000

/** Poll while the post renders or its preview does. */
export function previewPollInterval(post: SocialPost | undefined): number | false {
  const busy = post?.status === 'rendering' || post?.preview?.status === 'rendering'
  return busy ? SOCIALS_PREVIEW_POLL_MS : false
}

/** The post the composer saved, read again while it renders (US-208). */
export function useComposerPost(postId: string | null) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialPost>({
    queryKey: socialsComposerKeys.post(workspaceId, postId),
    enabled: socialsOn && !!postId,
    queryFn: () => apiClient.getSocialPost(postId as string),
    refetchInterval: (data) => previewPollInterval(data),
  })
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
  /** The post the composer saved already (a preview saves first): it is updated, not created again. */
  postId?: string | null
}

/** Create (or update) the post, then set its channels. */
export async function saveDraft({ post, targets, postId }: SaveDraftInput): Promise<SocialPost> {
  const saved = postId ? await apiClient.updateSocialPost(postId, post) : await apiClient.createSocialPost(post)
  return targets.length > 0 || postId ? apiClient.setSocialPostTargets(saved.id, targets) : saved
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

export interface PreviewInput extends SaveDraftInput {
  /** A video previews at half resolution; an image renders for real (US-104). */
  video: boolean
}

/** Save the draft, then render it: the preview for a video, the real render for an image.
 * 429 (no render minutes left) shows the server's reason, as a render does. */
export function usePreviewComposedDraft() {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialPost, Error, PreviewInput>({
    mutationFn: async ({ video, ...draft }) => {
      const saved = await saveDraft(draft)
      return apiClient.renderSocialPost(saved.id, video ? { preview: true } : {})
    },
    onSuccess: async () => {
      await invalidate()
    },
    onError: async (error) => {
      await invalidate()
      toast.error(error.message || 'Could not render the preview')
    },
  })
}
