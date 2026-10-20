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
 */
import { useQuery } from '@tanstack/react-query'

import { useSocialsOn } from '@/hooks/use-socials-api'
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

export interface SocialPostAiLabels {
  ai_media: SocialPostAiMedia
  targets: SocialTargetAiLabel[]
}

const AI_LABELS_STALE_MS = 30_000

export function aiLabelsPath(postId: string): string {
  return `/api/socials/posts/${postId}/ai-labels`
}

/** The post's AI label decisions; read only while the post has channels. */
export function useSocialPostAiLabels(post: SocialPost) {
  const { workspaceId, socialsOn } = useSocialsOn()
  const hasTargets = (post.targets ?? []).length > 0
  return useQuery<SocialPostAiLabels>({
    queryKey: ['socials', workspaceId, 'ai-labels', post.id, post.content_hash, post.status, post.updated_at],
    enabled: socialsOn && hasTargets,
    queryFn: () => apiClient.get<SocialPostAiLabels>(aiLabelsPath(post.id)),
    staleTime: AI_LABELS_STALE_MS,
    retry: false,
  })
}
