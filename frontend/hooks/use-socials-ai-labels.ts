'use client'

/**
 * PRD-264 EUAI-05 — each channel's AI label decision for a post, for its receipt.
 *
 * `GET /api/socials/posts/{id}/ai-labels` (orchestrator/modules/socials/ai_media.py)
 * answers, for every target, the line its publish used once one ran, else the line its
 * publish will use: TikTok's AI-generated label, YouTube Studio's setting, or the
 * caption for a channel that takes no AI flag. The backend owns the rule (AI footage,
 * AI stills, a synthetic voice, read from the render record); the receipt only shows
 * its words. The query is keyed by the post's hash, status and last update, so it is
 * read again whenever the post changes, and it sits under the socials key, so every
 * socials refresh reads it again too.
 *
 * PRD-264 EUAI-08a — the same answer carries the post's "Realistic AI people, places,
 * products or voice: label as AI-generated" setting (`deepfake_label`,
 * orchestrator/modules/socials/deepfake_label.py): on by default while the post carries
 * AI media. The composer switches it with `PUT /api/socials/posts/{id}/ai-label`; the
 * setting is approved content, so an approved post goes back for approval.
 */
import { useMutation, useQuery } from '@tanstack/react-query'
import { toast } from 'sonner'

import { useInvalidateSocials, usePostWriteErrorHandler, useSocialsOn } from '@/hooks/use-socials-api'
import { apiClient, type SocialPost } from '@/lib/api-client'

/** What AI made for the post, as the render recorded it. */
export interface SocialPostAiMedia {
  carries: boolean
  footage: string[]
  stills: string[]
  /** The synthetic voice the render spoke with (`kokoro`, a voice toolkit), or null. */
  voice: string | null
  own: string[]
  /** True when the decision came from the render record. */
  recorded: boolean
}

/** One target's AI label line; `at_publish` when it is the line a publish attempt used. */
export interface SocialTargetAiLabel {
  id: string
  toolkit: string
  post_kind: string
  ai_label: string
  at_publish: boolean
}

/** PRD-264 EUAI-08a: the post's AI-generated label setting, in the backend's words. */
export interface SocialPostDeepfakeLabel {
  /** "Realistic AI people, places, products or voice: label as AI-generated" */
  name: string
  /** What a person chose for this post; null follows the media (`default`). */
  setting: boolean | null
  /** On by default: the post carries AI media. */
  default: boolean
  /** Whether the post is labelled now. */
  on: boolean
  /** The caption label in the post's language ("AI-generated."). */
  label: string
  /** Why switching it off is the customer's call. */
  reason: string
  /** Whether the post's rendered video shows the label (a render from before the label counts as none); null when it carries no rendered video. */
  video_labelled: boolean | null
  /** The setting as the receipt states it. */
  line: string
}

export interface SocialPostAiLabels {
  ai_media: SocialPostAiMedia
  deepfake_label?: SocialPostDeepfakeLabel
  targets: SocialTargetAiLabel[]
}

const AI_LABELS_STALE_MS = 30_000

export function aiLabelsPath(postId: string): string {
  return `/api/socials/posts/${postId}/ai-labels`
}

/** PRD-264 EUAI-08a: where a person switches the post's AI-generated label. */
export function aiLabelSettingPath(postId: string): string {
  return `/api/socials/posts/${postId}/ai-label`
}

/** The post's AI label decisions. The receipt reads them while the post has channels;
 * the composer (`always`) reads the setting before any channel is ticked. */
export function useSocialPostAiLabels(post: SocialPost, { always = false }: { always?: boolean } = {}) {
  const { workspaceId, socialsOn } = useSocialsOn()
  const hasTargets = (post.targets ?? []).length > 0
  return useQuery<SocialPostAiLabels>({
    queryKey: ['socials', workspaceId, 'ai-labels', post.id, post.content_hash, post.status, post.updated_at],
    enabled: socialsOn && (always || hasTargets),
    queryFn: () => apiClient.get<SocialPostAiLabels>(aiLabelsPath(post.id)),
    staleTime: AI_LABELS_STALE_MS,
    retry: false,
  })
}

/** Switch the post's AI-generated label on or off; the post as saved. */
export function useSetSocialPostAiLabel() {
  const invalidate = useInvalidateSocials()
  const onError = usePostWriteErrorHandler('Could not change the AI-generated label')
  return useMutation<SocialPost, Error, { postId: string; on: boolean }>({
    mutationFn: ({ postId, on }) => apiClient.put<SocialPost>(aiLabelSettingPath(postId), { on }),
    onSuccess: async (_saved, { on }) => {
      await invalidate()
      toast.success(on ? 'AI-generated label on' : 'AI-generated label off')
    },
    onError,
  })
}
