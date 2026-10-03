/**
 * PRD-251B (3 Oct 2026 pass) — Plan with Auto over React Query: the draft (POST
 * /api/socials/plans/draft, nothing saved) and the save that adopts it. The plan is created
 * first; then each idea Auto heard joins its content bank, in Auto's order, and research
 * starts on the sources the person picked. An idea the bank refuses, or research that cannot
 * start, never undoes the plan: the person is told which, and the server's reason.
 */
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { adoptedNotes, type AdoptedDraft } from '@/components/deliverables/socials/plans/plan-auto-model'
import { useWorkspace } from '@/components/workspace-provider'
import { apiClient } from '@/lib/api-client'
import type { SocialPlan, SocialPlanAutoRequest, SocialPlanInput, SocialTopicInput } from '@/lib/socials-plan-types'
import { planQueryKeys, reasonOf } from './use-socials-plans'

export const DRAFT_FAILED = 'Auto could not draft the plan. Try again, or fill it in yourself.'
const PLAN_NOT_SAVED = 'The plan could not be saved.'

export function useDraftSocialPlan() {
  return useMutation({
    mutationFn: (input: SocialPlanAutoRequest) => apiClient.draftSocialPlan(input),
    onError: (error) => toast.error(reasonOf(error, DRAFT_FAILED)),
  })
}

export interface DraftedPlanSave {
  input: SocialPlanInput
  topics: SocialTopicInput[]
  research: boolean
}

async function addIdeas(planId: string, topics: ReadonlyArray<SocialTopicInput>): Promise<{ added: number; refused: string[] }> {
  const refused: string[] = []
  for (const topic of topics) {
    try {
      await apiClient.addSocialPlanTopic(planId, topic)
    } catch (error) {
      refused.push(`"${topic.title}": ${reasonOf(error, 'no reason given')}`)
    }
  }
  return { added: topics.length - refused.length, refused }
}

async function startResearch(planId: string): Promise<string | null> {
  try {
    await apiClient.researchSocialPlan(planId)
    return null
  } catch (error) {
    return reasonOf(error, 'Research could not start.')
  }
}

async function adopt({ input, topics, research }: DraftedPlanSave): Promise<{ plan: SocialPlan; adopted: AdoptedDraft }> {
  const plan = await apiClient.createSocialPlan(input)
  const ideas = await addIdeas(plan.id, topics)
  const researchError = research ? await startResearch(plan.id) : null
  return { plan, adopted: { ...ideas, researching: research && !researchError, researchError } }
}

/** Save Auto's draft as a new plan, with its ideas and a first research run; the answer is the plan. */
export function useSaveDraftedPlan() {
  const ws = useWorkspace().workspace?.id ?? null
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: adopt,
    onSuccess: ({ adopted }) => {
      const { said, warned } = adoptedNotes(adopted)
      toast.success(said)
      warned.forEach((line) => toast.error(line))
      void queryClient.invalidateQueries({ queryKey: planQueryKeys.plans(ws) })
    },
    onError: (error) => toast.error(reasonOf(error, PLAN_NOT_SAVED)),
  })
}
