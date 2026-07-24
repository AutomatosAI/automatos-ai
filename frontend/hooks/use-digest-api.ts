'use client'

import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// PRD-221 S13 — Auto's Read digest + feedback. Same 60s-poll family as
// use-activity-api; the backend caches per (workspace, state_hash) so the poll
// is cheap (an unchanged state returns the cached read without an LLM call).

export interface WorkspaceDigest {
  text: string
  generated_at: string
  state_hash: string
  needs_attention_count: number
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

export function useSubmitDigestFeedback() {
  return useMutation({
    mutationFn: (vars: { state_hash: string; rating: 1 | -1 }) =>
      apiClient.post('/api/activity/digest/feedback', vars),
  })
}
