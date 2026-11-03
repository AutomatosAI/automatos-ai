'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// PRD-221 S13 — Auto's Read digest + feedback. Same 60s-poll family as
// use-activity-api; the backend caches per (workspace, state_hash) so the poll
// is cheap (an unchanged state returns the cached read without an LLM call).
//
// PRD-268 §6.4 — the feedback route keeps one rating per person per state_hash
// (a second press updates it), and GET answers the caller's own rating. The
// read and the post share one cache entry per state_hash: the brief card's
// thumbs and the SDK's useRateBrief read it, and a press in either updates it.

const FEEDBACK_PATH = '/api/activity/digest/feedback'
/** A remembered rating changes only when this person presses: a minute old is fresh enough. */
const RATING_STALE_MS = 60000

export interface WorkspaceDigest {
  text: string
  generated_at: string
  state_hash: string
  needs_attention_count: number
}

export type DigestRatingValue = 1 | -1

export interface DigestFeedbackVars {
  state_hash: string
  rating: DigestRatingValue
}

/** The caller's own rating of the read or brief keyed by state_hash: null while they haven't rated it. */
export interface DigestRating {
  readonly state_hash: string
  readonly rating: DigestRatingValue | null
}

export const digestFeedbackQueryKeys = {
  all: ['workspace-digest-feedback'] as const,
  rating: (stateHash: string) => ['workspace-digest-feedback', stateHash] as const,
}

/** The route's answer as the cache holds it: a rating that isn't 1 or -1 is none. */
function ratingAnswerOf(stateHash: string, raw: unknown): DigestRating {
  const rating = (raw as { rating?: unknown } | null)?.rating
  return { state_hash: stateHash, rating: rating === 1 || rating === -1 ? rating : null }
}

export function useWorkspaceDigest(period: string = '1d') {
  return useQuery<WorkspaceDigest>({
    queryKey: ['workspace-digest', period],
    queryFn: () =>
      apiClient.request<WorkspaceDigest>(`/api/activity/digest?period=${period}`),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

/** The signed-in person's own rating for a state_hash; nothing is asked without one. */
export function useDigestRating(stateHash: string | null | undefined) {
  const hash = stateHash ?? ''
  return useQuery<DigestRating>({
    queryKey: digestFeedbackQueryKeys.rating(hash),
    queryFn: async () =>
      ratingAnswerOf(hash, await apiClient.request<unknown>(`${FEEDBACK_PATH}?state_hash=${encodeURIComponent(hash)}`)),
    enabled: hash !== '',
    staleTime: RATING_STALE_MS,
  })
}

/** Post a thumbs rating; once the server has it, every reader of that state_hash's rating shows it. */
export function useSubmitDigestFeedback() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (vars: DigestFeedbackVars) => apiClient.post(FEEDBACK_PATH, vars),
    onSuccess: async (_answer, vars) => {
      // A read still in flight would land after this and put the old rating back: cancel it first.
      const key = digestFeedbackQueryKeys.rating(vars.state_hash)
      await queryClient.cancelQueries({ queryKey: key })
      queryClient.setQueryData<DigestRating>(key, { state_hash: vars.state_hash, rating: vars.rating })
    },
  })
}
