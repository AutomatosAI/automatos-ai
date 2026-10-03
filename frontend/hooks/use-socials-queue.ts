/**
 * Socials Queue hooks (PRD-251B US-B111)
 * ======================================
 *
 * Another take (POST /api/socials/posts/{id}/retake), sending a post back to Auto (request
 * changes with the comment, then another take with it as guidance), and approving every
 * shown post by the content hash on screen: through the series path when the shown posts
 * are one series campaign's, otherwise one approve each. Nothing is approved implicitly: a
 * post whose hash moved (409) or whose claims need a second confirmation (422) is left, and
 * said so.
 */
import { useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient, type SocialCampaign, type SocialPost } from '@/lib/api-client'
import { useInvalidateSocials, usePostWriteErrorHandler } from '@/hooks/use-socials-api'

export const RETAKE_STARTED = 'Auto is making another take. It comes back here.'
export const SENT_BACK = 'Sent back to Auto. The new take comes back here.'

export function useRetakeSocialPost() {
  const invalidate = useInvalidateSocials()
  const onError = usePostWriteErrorHandler('Auto could not make another take')
  return useMutation<SocialPost, Error, string>({
    mutationFn: (postId) => apiClient.retakeSocialPost(postId),
    onSuccess: async () => {
      await invalidate()
      toast.success(RETAKE_STARTED)
    },
    onError,
  })
}

export function useSendBackToAuto() {
  const invalidate = useInvalidateSocials()
  const onError = usePostWriteErrorHandler('Could not send the post back')
  return useMutation<SocialPost, Error, { postId: string; comment: string }>({
    mutationFn: async ({ postId, comment }) => {
      await apiClient.requestSocialPostChanges(postId, comment)
      return apiClient.retakeSocialPost(postId, comment)
    },
    onSuccess: async () => {
      await invalidate()
      toast.success(SENT_BACK)
    },
    onError,
  })
}

export interface ShownApproval {
  approved: number
  /** Each post left unapproved, and why. */
  left: { title: string; why: string }[]
}

/** The series campaign every shown post belongs to, if they are one series' posts. */
export function sharedSeries(posts: ReadonlyArray<SocialPost>, campaigns: ReadonlyArray<SocialCampaign>): SocialCampaign | null {
  const ids = new Set(posts.map((post) => post.campaign_id ?? null))
  if (ids.size !== 1 || ids.has(null)) return null
  const campaign = campaigns.find((c) => c.id === posts[0].campaign_id)
  return campaign?.approval_mode === 'series' ? campaign : null
}

async function approveEach(posts: ReadonlyArray<SocialPost>): Promise<ShownApproval> {
  const result: ShownApproval = { approved: 0, left: [] }
  for (const post of posts) {
    try {
      await apiClient.approveSocialPost(post.id, post.content_hash)
      result.approved += 1
    } catch (error) {
      const status = (error as { status?: number }).status
      const why = status === 409 ? 'it changed since it was shown' : status === 422 ? 'its unsourced claims need a second confirmation' : (error as Error).message
      result.left.push({ title: post.title, why })
    }
  }
  return result
}

async function approveSeries(campaign: SocialCampaign, posts: ReadonlyArray<SocialPost>): Promise<ShownApproval> {
  const answer = await apiClient.approveSocialCampaignSeries(
    campaign.id,
    posts.map((post) => ({ post_id: post.id, content_hash: post.content_hash })),
  )
  const titles = new Map(posts.map((post) => [post.id, post.title]))
  return {
    approved: answer.approved.length,
    left: answer.left.map((entry) => ({ title: entry.title ?? titles.get(entry.post_id) ?? entry.post_id, why: entry.message })),
  }
}

/** Approve all shown: each post by the hash on screen (the series path for one series campaign). */
export function useApproveShown(campaigns: ReadonlyArray<SocialCampaign>) {
  const invalidate = useInvalidateSocials()
  return useMutation<ShownApproval, Error, ReadonlyArray<SocialPost>>({
    mutationFn: (posts) => {
      const series = sharedSeries(posts, campaigns)
      return series ? approveSeries(series, posts) : approveEach(posts)
    },
    onSuccess: async ({ approved, left }) => {
      await invalidate()
      toast.success(`Approved ${approved} ${approved === 1 ? 'post' : 'posts'}.`)
      left.forEach((entry) => toast.error(`Not approved: ${entry.title}, because ${entry.why}.`))
    },
    onError: async (error) => {
      await invalidate()
      toast.error(error.message || 'Could not approve the shown posts')
    },
  })
}
