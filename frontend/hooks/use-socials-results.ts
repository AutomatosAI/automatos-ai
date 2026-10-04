/**
 * PRD-251C Wave 4 — React Query over what Socials learns once posts go out: Posted (US-C408),
 * Auto's proposals for a plan (US-C404, applied by one click through the plan's PUT) and the
 * plan's health (US-C407).
 * Keys sit under the Socials key of the workspace (use-socials-api.ts), so a workspace switch
 * clears them with the rest.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { useWorkspace } from '@/components/workspace-provider'
import { apiClient } from '@/lib/api-client'
import type { SocialPlan, SocialPlanInput } from '@/lib/socials-plan-types'
import type { SocialPlanProposal, SocialPostedFilters } from '@/lib/socials-results-types'
import { socialsQueryKeys } from './use-socials-api'
import { planQueryKeys, reasonOf } from './use-socials-plans'

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

export const resultsQueryKeys = {
  posted: (ws: string | null, filters: SocialPostedFilters) =>
    [...socialsQueryKeys.all(ws), 'posted', filters.planId ?? '', filters.channel ?? '', filters.format ?? ''] as const,
  proposals: (ws: string | null, planId: string) => [...socialsQueryKeys.all(ws), 'plans', planId, 'proposals'] as const,
  health: (ws: string | null, planId: string) => [...socialsQueryKeys.all(ws), 'plans', planId, 'health'] as const,
  voice: (ws: string | null) => [...socialsQueryKeys.all(ws), 'voice-examples'] as const,
}

/** Posted: what went out, newest first, with the filters given. */
export function useSocialPosted(filters: SocialPostedFilters) {
  const ws = useWorkspaceId()
  return useQuery({
    queryKey: resultsQueryKeys.posted(ws, filters),
    queryFn: () => apiClient.listSocialPosted(filters),
    enabled: !!ws,
  })
}

/** Auto's proposals for a saved plan. */
export function useSocialPlanProposals(planId: string | null) {
  const ws = useWorkspaceId()
  return useQuery({
    queryKey: resultsQueryKeys.proposals(ws, planId ?? ''),
    queryFn: () => apiClient.getSocialPlanProposals(planId as string),
    enabled: !!ws && !!planId,
  })
}

/** What needs the owner in a saved plan now. */
export function useSocialPlanHealth(planId: string | null) {
  const ws = useWorkspaceId()
  return useQuery({
    queryKey: resultsQueryKeys.health(ws, planId ?? ''),
    queryFn: () => apiClient.getSocialPlanHealth(planId as string),
    enabled: !!ws && !!planId,
  })
}

/** Apply one proposal: its changes through the plan's PUT, then the plan and its insights refresh. */
export function useApplyProposal(planId: string, onApplied: (plan: SocialPlan) => void) {
  const ws = useWorkspaceId()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (proposal: SocialPlanProposal) => apiClient.updateSocialPlan(planId, proposal.changes as SocialPlanInput),
    onSuccess: (plan) => {
      toast.success('Applied to the plan.')
      queryClient.invalidateQueries({ queryKey: planQueryKeys.plans(ws) })
      onApplied(plan)
    },
    onError: (error) => toast.error(reasonOf(error, 'The proposal could not be applied.')),
  })
}

/** The owner's voice examples (US-C406), newest first. */
export function useVoiceExamples() {
  const ws = useWorkspaceId()
  return useQuery({ queryKey: resultsQueryKeys.voice(ws), queryFn: () => apiClient.listSocialVoiceExamples(), enabled: !!ws })
}

/** Remove one voice example; the list refreshes. */
export function useRemoveVoiceExample() {
  const ws = useWorkspaceId()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (exampleId: string) => apiClient.deleteSocialVoiceExample(exampleId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: resultsQueryKeys.voice(ws) }),
    onError: (error) => toast.error(reasonOf(error, 'The example could not be removed.')),
  })
}
