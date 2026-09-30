'use client'

/**
 * PRD-251 Wave 3 — scheduling and publishing a post from the post view (US-306..US-308).
 *
 * Every call goes through apiClient with the backend's verb (POST). A schedule
 * moves the post's slot and keeps its approval; the calendar and the post view
 * both refetch after it. A 409 says the post changed since it was loaded and
 * refetches (as approve does); anything else shows the server's message.
 */
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { activityQueryKeys } from '@/hooks/use-activity-api'
import { useInvalidateSocials, usePostWriteErrorHandler } from '@/hooks/use-socials-api'
import { apiClient, type SocialPost } from '@/lib/api-client'

export interface ScheduleSocialPostInput {
  postId: string
  /** The slot, an ISO instant. */
  scheduledFor: string
  /** The IANA timezone the slot is shown in: the post's own. */
  timezone: string
}

/** After a schedule write: the posts and the calendar both show the new slot. */
function useAfterScheduleWrite() {
  const invalidateSocials = useInvalidateSocials()
  const queryClient = useQueryClient()
  return async () => {
    await Promise.all([invalidateSocials(), queryClient.invalidateQueries({ queryKey: activityQueryKeys.all })])
  }
}

export function useScheduleSocialPost() {
  const after = useAfterScheduleWrite()
  const onError = usePostWriteErrorHandler('The post could not be scheduled')
  return useMutation<SocialPost, Error, ScheduleSocialPostInput>({
    mutationFn: ({ postId, scheduledFor, timezone }) => apiClient.scheduleSocialPost(postId, scheduledFor, timezone),
    onSuccess: async () => {
      await after()
      toast.success('Scheduled. The approval stands.')
    },
    onError: async (error) => {
      await onError(error)
      await after()
    },
  })
}

/** Publish an approved, scheduled or missed post now (US-301): 202, the post in publishing. */
export function usePublishSocialPostNow() {
  const after = useAfterScheduleWrite()
  const onError = usePostWriteErrorHandler('The post could not be published')
  return useMutation<SocialPost, Error, { postId: string }>({
    mutationFn: ({ postId }) => apiClient.publishSocialPostNow(postId),
    onSuccess: async () => {
      await after()
      toast.success('Publishing. Each channel shows its receipt when it is done.')
    },
    onError,
  })
}

/** Publish again the targets that failed (US-301); a published target never runs again. */
export function useRetrySocialPost() {
  const after = useAfterScheduleWrite()
  const onError = usePostWriteErrorHandler('The retry could not start')
  return useMutation<SocialPost, Error, { postId: string }>({
    mutationFn: ({ postId }) => apiClient.retrySocialPost(postId),
    onSuccess: async () => {
      await after()
      toast.success('Retrying the channels that failed.')
    },
    onError,
  })
}

export function useUnscheduleSocialPost() {
  const after = useAfterScheduleWrite()
  const onError = usePostWriteErrorHandler('The post could not be unscheduled')
  return useMutation<SocialPost, Error, { postId: string }>({
    mutationFn: ({ postId }) => apiClient.unscheduleSocialPost(postId),
    onSuccess: async () => {
      await after()
      toast.success('Unscheduled. It stays approved.')
    },
    onError,
  })
}
