'use client'

import { useCallback, useState } from 'react'
import { useCreatePlaybook, useUpdatePlaybook } from './use-playbook-api'
import { toast } from 'sonner'
import type { PlaybookFormValues } from '@/components/workflows/create-playbook-modal'

/**
 * Transforms frontend PlaybookFormValues into the API request body
 * expected by POST /api/workflow-recipes
 */
function transformFormToApiPayload(data: PlaybookFormValues) {
  // Generate a slug-style template_id from the playbook name
  const templateId = data.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    + '-' + Date.now().toString(36)

  // Parse JSON string fields into objects
  let inputs: Record<string, unknown> | undefined
  let outputs: Record<string, unknown> | undefined

  try {
    const parsed = JSON.parse(data.inputs)
    if (parsed && Object.keys(parsed).length > 0) {
      inputs = parsed
    }
  } catch {
    // inputs stays undefined if invalid/empty
  }

  try {
    const parsed = JSON.parse(data.outputs)
    if (parsed && Object.keys(parsed).length > 0) {
      outputs = parsed
    }
  } catch {
    // outputs stays undefined if invalid/empty
  }

  // Build steps array with required fields
  const steps = data.steps.map((step, index) => ({
    step_id: step.step_id || `step-${index + 1}`,
    order: step.order ?? index + 1,
    agent_id: typeof step.agent_id === 'string' ? parseInt(step.agent_id, 10) : step.agent_id,
    prompt_template: step.prompt_template,
    ...(step.pass_to ? { pass_to: step.pass_to } : {}),
    ...(step.pre_exec ? { pre_exec: step.pre_exec } : {}),
    error_handling: step.error_handling || 'stop',
  }))

  // Build template_definition from steps (backend expects this)
  const templateDefinition = {
    steps: steps.map((s) => ({
      step_id: s.step_id,
      order: s.order,
      agent_id: s.agent_id,
      prompt_template: s.prompt_template,
      error_handling: s.error_handling,
      ...(s.pass_to ? { pass_to: s.pass_to } : {}),
      ...(s.pre_exec ? { pre_exec: s.pre_exec } : {}),
    })),
    version: '1.0',
  }

  // Build execution_config matching backend expectations (timeouts in seconds)
  const executionConfig = {
    mode: data.execution_config.mode,
    max_retries: data.execution_config.max_retries,
    per_step_timeout: Math.round(data.execution_config.timeout_per_step / 1000),
    total_timeout: Math.round(data.execution_config.total_timeout / 1000),
    auto_learning: data.execution_config.auto_learning,
    ...(data.execution_config.mode === 'parallel' ? { parallel_limit: data.execution_config.parallel_limit } : {}),
    memory_isolation: data.execution_config.memory_isolation,
  }

  // Build schedule_config (only include if not manual with empty config)
  const scheduleConfig = {
    type: data.schedule_config.type,
    ...(data.schedule_config.type === 'cron' && data.schedule_config.cron_expression
      ? { cron_expression: data.schedule_config.cron_expression }
      : {}),
    ...(data.schedule_config.type === 'trigger'
      ? { trigger_config: data.schedule_config.trigger_config || {} }
      : {}),
  }

  return {
    template_id: templateId,
    name: data.name.trim(),
    description: data.description.trim(),
    template_definition: templateDefinition,
    steps,
    ...(inputs ? { inputs } : {}),
    ...(outputs ? { outputs } : {}),
    execution_config: executionConfig,
    schedule_config: scheduleConfig,
    tags: [],
    is_public: true,
  }
}

/**
 * Validates form data before submission.
 * Returns null if valid, or an error message string if invalid.
 */
function validateFormData(data: PlaybookFormValues): string | null {
  // Name validation
  if (!data.name || data.name.trim().length < 3) {
    return 'Playbook name must be at least 3 characters'
  }

  // Steps validation
  if (!data.steps || data.steps.length === 0) {
    return 'At least one workflow step is required'
  }

  for (let i = 0; i < data.steps.length; i++) {
    const step = data.steps[i]
    if (!step.agent_id) {
      return `Step ${i + 1} must have an agent assigned`
    }
    if (!step.prompt_template || step.prompt_template.trim().length === 0) {
      return `Step ${i + 1} must have a prompt template`
    }
  }

  // JSON schema validation
  if (data.inputs && data.inputs.trim() !== '{}') {
    try {
      JSON.parse(data.inputs)
    } catch {
      return 'Input schema contains invalid JSON'
    }
  }

  if (data.outputs && data.outputs.trim() !== '{}') {
    try {
      JSON.parse(data.outputs)
    } catch {
      return 'Output schema contains invalid JSON'
    }
  }

  return null
}

export interface UsePlaybookFormReturn {
  isSubmitting: boolean
  lastSavedWebhookId: string | null
  submitPlaybook: (data: PlaybookFormValues, onSuccess?: () => void) => Promise<void>
  updatePlaybook: (playbookId: string, data: PlaybookFormValues, onSuccess?: () => void) => Promise<void>
}

/**
 * Hook for managing playbook form submission.
 * Handles validation, API call, toast notifications, and query invalidation.
 */
export function usePlaybookForm(): UsePlaybookFormReturn {
  const createPlaybookMutation = useCreatePlaybook()
  const updatePlaybookMutation = useUpdatePlaybook()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [lastSavedWebhookId, setLastSavedWebhookId] = useState<string | null>(null)

  const submitPlaybook = useCallback(
    async (data: PlaybookFormValues, onSuccess?: () => void) => {
      // Validate form data
      const validationError = validateFormData(data)
      if (validationError) {
        toast.error('Validation Error', { description: validationError })
        return
      }

      setIsSubmitting(true)

      try {
        const payload = transformFormToApiPayload(data)

        const result = await createPlaybookMutation.mutateAsync(payload) as any

        // Extract webhook_id from server response for display
        const webhookId = result?.playbook?.schedule_config?.webhook_id
        if (webhookId) {
          setLastSavedWebhookId(webhookId)
        }

        toast('Playbook Created', { description: `"${data.name}" has been created successfully.` })

        onSuccess?.()
      } catch (err: unknown) {
        const message =
          err instanceof Error
            ? err.message
            : typeof err === 'object' && err !== null && 'detail' in err
              ? String((err as { detail: unknown }).detail)
              : 'Failed to create playbook. Please try again.'

        toast.error('Error Creating Playbook', { description: message })
      } finally {
        setIsSubmitting(false)
      }
    },
    [createPlaybookMutation]
  )

  const updatePlaybook = useCallback(
    async (playbookId: string, data: PlaybookFormValues, onSuccess?: () => void) => {
      // Validate form data
      const validationError = validateFormData(data)
      if (validationError) {
        toast.error('Validation Error', { description: validationError })
        return
      }

      setIsSubmitting(true)

      try {
        const payload = transformFormToApiPayload(data)

        const result = await updatePlaybookMutation.mutateAsync({ playbookId, playbookData: payload }) as any

        // Extract webhook_id from server response for display
        const webhookId = result?.playbook?.schedule_config?.webhook_id
        if (webhookId) {
          setLastSavedWebhookId(webhookId)
        }

        toast('Playbook Updated', { description: `"${data.name}" has been updated successfully.` })

        onSuccess?.()
      } catch (err: unknown) {
        const message =
          err instanceof Error
            ? err.message
            : typeof err === 'object' && err !== null && 'detail' in err
              ? String((err as { detail: unknown }).detail)
              : 'Failed to update playbook. Please try again.'

        toast.error('Error Updating Playbook', { description: message })
      } finally {
        setIsSubmitting(false)
      }
    },
    [updatePlaybookMutation]
  )

  return {
    isSubmitting,
    lastSavedWebhookId,
    submitPlaybook,
    updatePlaybook,
  }
}
