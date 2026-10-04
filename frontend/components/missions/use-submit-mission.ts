'use client'

/**
 * The New mission form's submit: builds the request (create-mission-payload.ts), creates the
 * mission, then closes and resets the form and opens the new mission. Moved out of
 * create-mission-modal.tsx.
 */

import { useRouter } from 'next/navigation'
import { toast } from 'sonner'
import { useCreateMission } from '@/hooks/use-missions-api'
import { useMissionStore } from '@/stores/mission-store'
import type { MissionAttachment } from './create-mission-options'
import {
  createMissionRequest,
  missionConfig,
  missionGoal,
  type BusinessPlanValues,
  type MissionFormValues,
} from './create-mission-payload'

interface SubmitMissionArgs {
  form: MissionFormValues
  businessPlan: BusinessPlanValues
  attachments: MissionAttachment[]
  onOpenChange: (open: boolean) => void
  resetForm: () => void
}

export function useSubmitMission({ form, businessPlan, attachments, onOpenChange, resetForm }: SubmitMissionArgs) {
  const router = useRouter()
  const createMission = useCreateMission()
  const setActivePlanningMissionId = useMissionStore((s) => s.setActivePlanningMissionId)

  const submit = () => {
    const result = missionGoal(form, businessPlan)
    if ('error' in result) {
      toast.error(result.error)
      return
    }
    const config = missionConfig(form, businessPlan, attachments)
    createMission.mutate(createMissionRequest(result.goal, config, form.selectedTemplate), {
      onSuccess: (mission) => {
        setActivePlanningMissionId(mission.id)
        toast.success('Mission created — plan is being generated')
        onOpenChange(false)
        resetForm()
        router.push(`/missions/${mission.id}` as any)
      },
      onError: (err) => {
        toast.error(err.message || 'Failed to create mission')
      },
    })
  }

  return { submit, isSubmitting: createMission.isLoading }
}
