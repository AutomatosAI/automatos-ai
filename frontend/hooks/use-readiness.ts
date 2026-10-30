'use client'

/**
 * PRD-268 §6.11.3 (US-C21b) — what Auto depends on, and what couldn't be checked.
 *
 * `useReadiness` reads W1's GET /api/readiness (US-C18): Readiness@1, the workspace's last check
 * while the server holds it, else a check made now. `useCheckReadiness` is "Check again"
 * (POST /api/readiness/check): its answer replaces the one shown. The server limits the checks
 * per workspace; over the limit it refuses with `checked_recently` (429) and checks nothing, and
 * `checkedRecently(error)` says so, so the list can say "Checked a moment ago".
 *
 * The microphone and speech rows are the browser's (useVoice().available): the server never
 * claims them, and neither does this hook.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { readinessOf, refusalOf } from '@/lib/proactive/wire'
import type { Readiness } from '@/lib/proactive/types'

const READINESS_PATH = '/api/readiness'
const READINESS_STALE_MS = 60_000
/** The code of the refusal a check over the limit gets (services/readiness/strings.py). */
export const CHECKED_RECENTLY = 'checked_recently'

export const readinessQueryKeys = {
  all: ['proactive', 'readiness'] as const,
}

/** True when a Check again was refused because the workspace checked a moment ago. */
export function checkedRecently(error: unknown): boolean {
  return refusalOf(error)?.code === CHECKED_RECENTLY
}

/** Readiness@1 for the caller's workspace. */
export function useReadiness() {
  return useQuery<Readiness>({
    queryKey: readinessQueryKeys.all,
    queryFn: async () => readinessOf(await apiClient.request<unknown>(READINESS_PATH)),
    staleTime: READINESS_STALE_MS,
  })
}

/** Check again: every item now; the answer is the one every reader shows. */
export function useCheckReadiness() {
  const queryClient = useQueryClient()
  return useMutation<Readiness, Error, void>({
    mutationFn: async () => readinessOf(await apiClient.post<unknown>(`${READINESS_PATH}/check`)),
    onSuccess: (readiness) => {
      queryClient.setQueryData(readinessQueryKeys.all, readiness)
    },
  })
}
