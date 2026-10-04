'use client'

import { Target, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { AttachmentsField } from './create-mission-attachments'
import { BusinessPlanFields } from './create-mission-business-plan'
import {
  MissionTextFields,
  PowerModePicker,
  SwitchRow,
  TagsField,
  TemplatePicker,
} from './create-mission-fields'
import { businessPlanComplete, isBusinessPlanTemplate } from './create-mission-payload'
import { useBusinessPlanFields, useMissionFormFields, useOrchestratorModel } from './use-create-mission-form'
import { useMissionAttachments } from './use-mission-attachments'
import { useSubmitMission } from './use-submit-mission'

interface CreateMissionModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialGoal?: string
  initialDescription?: string
}

interface CreateMissionFooterProps {
  isSubmitting: boolean
  submitDisabled: boolean
  attachmentCount: number
  onCancel: () => void
  onSubmit: () => void
}

function CreateMissionFooter({ isSubmitting, submitDisabled, attachmentCount, onCancel, onSubmit }: CreateMissionFooterProps) {
  return (
    <DialogFooter>
      <Button variant="outline" onClick={onCancel} disabled={isSubmitting}>
        Cancel
      </Button>
      <Button onClick={onSubmit} disabled={submitDisabled}>
        {isSubmitting ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Creating...
          </>
        ) : (
          <>
            <Target className="w-4 h-4 mr-2" />
            Create Mission
            {attachmentCount > 0 && (
              <span className="ml-1 text-[10px] opacity-70">
                ({attachmentCount} file{attachmentCount !== 1 ? 's' : ''})
              </span>
            )}
          </>
        )}
      </Button>
    </DialogFooter>
  )
}

export function CreateMissionModal({ open, onOpenChange, initialGoal, initialDescription }: CreateMissionModalProps) {
  const form = useMissionFormFields(initialGoal, initialDescription)
  const businessPlan = useBusinessPlanFields()
  const uploads = useMissionAttachments()
  const orchestratorModel = useOrchestratorModel(open)
  const isBusinessPlan = isBusinessPlanTemplate(form.selectedTemplate)

  const resetForm = () => {
    form.reset()
    businessPlan.reset()
    uploads.clearFiles()
  }
  const { submit, isSubmitting } = useSubmitMission({
    form, businessPlan, attachments: uploads.attachments, onOpenChange, resetForm,
  })
  const canSubmit = isBusinessPlan
    ? businessPlanComplete(businessPlan)
    : Boolean(form.name.trim() || form.description.trim())

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            New Mission
          </DialogTitle>
          <DialogDescription>
            Define a goal for your AI workforce. Attach files like PRDs, design docs,
            or data to give agents context.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <TemplatePicker selected={form.selectedTemplate} onSelect={form.setSelectedTemplate} />
          {isBusinessPlan && <BusinessPlanFields plan={businessPlan} />}
          <MissionTextFields form={form} isBusinessPlan={isBusinessPlan} />
          <PowerModePicker powerMode={form.powerMode} orchestratorModel={orchestratorModel} onSelect={form.setPowerMode} />
          <AttachmentsField uploads={uploads} />
          {/* Budget pause toggle */}
          <SwitchRow
            id="budget-pause"
            label="Pause on budget exceeded"
            hint="Pause the mission if token usage exceeds the estimated budget"
            checked={form.budgetPauseEnabled}
            onToggle={() => form.setBudgetPauseEnabled((v) => !v)}
          />
          {/* Check each step toggle (F282) */}
          <SwitchRow
            id="check-each-step"
            label="Check each step with me"
            hint="Every step waits in Review for your OK before the next one starts"
            checked={form.checkEachStep}
            onToggle={() => form.setCheckEachStep((v) => !v)}
          />
          <TagsField tags={form.tags} onChange={form.setTags} />
        </div>

        <CreateMissionFooter
          isSubmitting={isSubmitting}
          submitDisabled={isSubmitting || uploads.isUploading || !canSubmit}
          attachmentCount={uploads.attachments.length}
          onCancel={() => onOpenChange(false)}
          onSubmit={submit}
        />
      </DialogContent>
    </Dialog>
  )
}
