/**
 * PRD-251B Wave 2 — React Query over a Socials plan (US-B202), its slots (US-B208), its
 * content bank (US-B203, US-B204) and the music library (a post's music). The server checks
 * every rule (the cadence, a fact's source, the never-say list); these hooks call it, show
 * its reason when it refuses, and refresh what changed. Keys sit under the Socials key of
 * the workspace (use-socials-api.ts), so a workspace switch clears them with the rest.
 */
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { useWorkspace } from '@/components/workspace-provider'
import { apiClient } from '@/lib/api-client'
import type { SocialPlan, SocialPlanInput, SocialPlanSlotsResponse, SocialTopicInput } from '@/lib/socials-plan-types'
import { socialsQueryKeys } from './use-socials-api'

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

export const planQueryKeys = {
  plans: (ws: string | null) => [...socialsQueryKeys.all(ws), 'plans'] as const,
  plan: (ws: string | null, planId: string) => [...socialsQueryKeys.all(ws), 'plans', planId] as const,
  slots: (ws: string | null, planId: string, start: string, end: string) =>
    [...socialsQueryKeys.all(ws), 'plans', planId, 'slots', start, end] as const,
  topics: (ws: string | null, planId: string) => [...socialsQueryKeys.all(ws), 'plans', planId, 'topics'] as const,
  music: (ws: string | null) => [...socialsQueryKeys.all(ws), 'music'] as const,
}

/** The server's reason for a refusal, else `fallback`. */
export function reasonOf(error: unknown, fallback: string): string {
  const message = (error as Error | null)?.message
  return message && message.trim() ? message : fallback
}

function useRefreshPlans() {
  const ws = useWorkspaceId()
  const queryClient = useQueryClient()
  return () => queryClient.invalidateQueries({ queryKey: planQueryKeys.plans(ws) })
}

export function useSocialPlans() {
  const ws = useWorkspaceId()
  return useQuery({ queryKey: planQueryKeys.plans(ws), queryFn: () => apiClient.listSocialPlans(), enabled: !!ws })
}

export function useSocialPlan(planId: string | null) {
  const ws = useWorkspaceId()
  return useQuery({
    queryKey: planQueryKeys.plan(ws, planId ?? ''),
    queryFn: () => apiClient.getSocialPlan(planId as string),
    enabled: !!ws && !!planId,
  })
}

/** Create the plan (no id) or save its changes; the answer is the plan as saved. */
export function useSaveSocialPlan() {
  const refresh = useRefreshPlans()
  return useMutation({
    mutationFn: ({ planId, input }: { planId: string | null; input: SocialPlanInput }) =>
      planId ? apiClient.updateSocialPlan(planId, input) : apiClient.createSocialPlan(input),
    onSuccess: () => {
      toast.success('Plan saved.')
      void refresh()
    },
    onError: (error) => toast.error(reasonOf(error, 'The plan could not be saved.')),
  })
}

export type PlanStatusAction = 'pause' | 'resume' | 'end'

const STATUS_CALLS: Record<PlanStatusAction, (planId: string) => Promise<SocialPlan>> = {
  pause: (planId) => apiClient.pauseSocialPlan(planId),
  resume: (planId) => apiClient.resumeSocialPlan(planId),
  end: (planId) => apiClient.endSocialPlan(planId),
}

export function useSetSocialPlanStatus() {
  const refresh = useRefreshPlans()
  return useMutation({
    mutationFn: ({ planId, action }: { planId: string; action: PlanStatusAction }) => STATUS_CALLS[action](planId),
    onSuccess: () => void refresh(),
    onError: (error) => toast.error(reasonOf(error, 'The plan could not change.')),
  })
}

export const PLAN_DELETED = 'Plan deleted. The posts it made stay in the calendar.'

/** Delete a plan (owners and admins): the plans and the posts (now unlinked) refresh. */
export function useDeleteSocialPlan() {
  const ws = useWorkspaceId()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ planId }: { planId: string }) => apiClient.deleteSocialPlan(planId),
    onSuccess: () => {
      toast.success(PLAN_DELETED)
      void queryClient.invalidateQueries({ queryKey: socialsQueryKeys.all(ws) })
    },
    onError: (error) => toast.error(reasonOf(error, 'The plan could not be deleted.')),
  })
}

export const RESEARCH_STARTED = 'Research started: new topics join the bank as Auto finds them.'

export function useResearchSocialPlan() {
  const refresh = useRefreshPlans()
  return useMutation({
    mutationFn: (planId: string) => apiClient.researchSocialPlan(planId),
    onSuccess: () => {
      toast.success(RESEARCH_STARTED)
      void refresh()
    },
    onError: (error) => toast.error(reasonOf(error, 'Research could not start.')),
  })
}

/** Each plan's slots in [start, end) (ISO): one query per plan; the answers in plan order. */
export function useSocialPlansSlots(planIds: ReadonlyArray<string>, start: string, end: string) {
  const ws = useWorkspaceId()
  const results = useQueries({
    queries: planIds.map((planId) => ({
      queryKey: planQueryKeys.slots(ws, planId, start, end),
      queryFn: () => apiClient.listSocialPlanSlots(planId, start, end),
      enabled: !!ws,
    })),
  })
  return results.map((result) => result.data).filter((data): data is SocialPlanSlotsResponse => !!data)
}

export function useSocialPlanTopics(planId: string | null) {
  const ws = useWorkspaceId()
  return useQuery({
    queryKey: planQueryKeys.topics(ws, planId ?? ''),
    queryFn: () => apiClient.listSocialPlanTopics(planId as string),
    enabled: !!ws && !!planId,
  })
}

type TopicWrite =
  | { kind: 'add'; input: SocialTopicInput }
  | { kind: 'edit'; topicId: string; input: SocialTopicInput }
  | { kind: 'pin'; topicId: string; pinnedOn: string | null }
  | { kind: 'delete'; topicId: string }

function writeTopic(planId: string, write: TopicWrite): Promise<unknown> {
  if (write.kind === 'add') return apiClient.addSocialPlanTopic(planId, write.input)
  if (write.kind === 'edit') return apiClient.updateSocialPlanTopic(planId, write.topicId, write.input)
  if (write.kind === 'pin') return apiClient.pinSocialPlanTopic(planId, write.topicId, write.pinnedOn)
  return apiClient.deleteSocialPlanTopic(planId, write.topicId)
}

/** The warning an added topic came back with (PRD-251C: the workspace already has one close to it). */
function warningOf(answer: unknown): string | null {
  const warning = (answer as { warning?: unknown } | null)?.warning
  return typeof warning === 'string' && warning.trim() ? warning : null
}

/** Add, edit, pin or delete one of the plan's topics; the bank and its counts refresh. A topic
 * close to one the workspace has is added, and the person told which (they may repeat on purpose). */
export function useWriteSocialTopic(planId: string) {
  const ws = useWorkspaceId()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (write: TopicWrite) => writeTopic(planId, write),
    onSuccess: (answer) => {
      const warning = warningOf(answer)
      if (warning) toast.warning(`Added. ${warning}`)
      void queryClient.invalidateQueries({ queryKey: planQueryKeys.topics(ws, planId) })
      void queryClient.invalidateQueries({ queryKey: planQueryKeys.plans(ws) })
    },
    onError: (error) => toast.error(reasonOf(error, 'The content bank refused that topic.')),
  })
}

export function useSocialMusic() {
  const ws = useWorkspaceId()
  return useQuery({ queryKey: planQueryKeys.music(ws), queryFn: () => apiClient.listSocialMusic(), enabled: !!ws, staleTime: 300_000 })
}
