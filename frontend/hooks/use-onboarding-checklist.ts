/**
 * PRD-222 US-020 (W2·S4) — the post-setup checklist hooks.
 *
 * The checklist is a SERVER read-model: item completion is derived from live
 * workspace counts on the backend (connections / missions / members), and only
 * the dismissal flags are persisted in `workspaces.onboarding.checklist` (D8 —
 * never localStorage). One query key backs both the chat card and the Command
 * Center card, so a dismissal on either surface updates both.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useAuth } from '@clerk/nextjs'
import { apiClient } from '@/lib/api-client'

export interface ChecklistItem {
  id: string
  label: string
  done: boolean
  href?: string
  manual?: boolean
}

export interface OnboardingChecklist {
  items: ChecklistItem[]
  dismissed: boolean
  completed_count: number
  total_count: number
}

const CHECKLIST_KEY = ['onboarding', 'checklist'] as const
const ENDPOINT = '/api/workspaces/current/onboarding/checklist'

function shouldRetry(error: unknown) {
  return !String((error as { message?: string })?.message || '').includes('HTTP 401')
}

/** Read the current workspace's post-setup checklist. */
export function useOnboardingChecklist(opts?: { enabled?: boolean }) {
  const { isLoaded, isSignedIn } = useAuth()
  return useQuery({
    queryKey: CHECKLIST_KEY,
    queryFn: async () => apiClient.get<OnboardingChecklist>(ENDPOINT),
    enabled: (opts?.enabled ?? true) && isLoaded && isSignedIn,
    staleTime: 30 * 1000,
    retry: (failureCount, error) => shouldRetry(error) && failureCount < 1,
    refetchOnWindowFocus: false,
  })
}

/** Persist a dismissal (the whole card, or the manual Academy item). */
export function useUpdateChecklist() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (patch: { dismissed?: boolean; academy_done?: boolean }) =>
      apiClient.patch<OnboardingChecklist>(ENDPOINT, patch),
    onSuccess: (fresh) => {
      // The PATCH returns the recomputed checklist — seed the cache with it so
      // both surfaces reflect the change without an extra round-trip.
      if (fresh) queryClient.setQueryData(CHECKLIST_KEY, fresh)
      queryClient.invalidateQueries({ queryKey: CHECKLIST_KEY })
    },
  })
}
