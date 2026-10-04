'use client'

import { useCallback, useState } from 'react'
import { useCreatePlaybook, useUpdatePlaybook } from './use-playbook-api'
import { apiClient } from '@/lib/api-client'
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

/** The stored settings an edit is laid over (GET /api/workflow-recipes/{id}). */
interface StoredPlaybookConfigs {
  execution_config?: Record<string, unknown> | null
  schedule_config?: Record<string, unknown> | null
}

/**
 * The body of an edit (PUT /api/workflow-recipes/{id}): only what the editor edits,
 * laid over the playbook's stored settings (F242, night 7b). The editor rebuilt
 * execution_config and schedule_config from its own fields, and the server keeps
 * them as sent, so a save dropped the owner's "wait for me"
 * (execution_config.wait_for_me), switched a timer that was off back on
 * (schedule_config.enabled) and lost its timezone. It also sent a fresh
 * template_id, emptied the tags and made the playbook public again.
 */
function editPayload(data: PlaybookFormValues, stored: StoredPlaybookConfigs | null) {
  const made = transformFormToApiPayload(data)
  return {
    name: made.name,
    description: made.description,
    template_definition: made.template_definition,
    steps: made.steps,
    ...(made.inputs ? { inputs: made.inputs } : {}),
    ...(made.outputs ? { outputs: made.outputs } : {}),
    execution_config: { ...(stored?.execution_config ?? {}), ...made.execution_config },
    schedule_config: { ...(stored?.schedule_config ?? {}), ...made.schedule_config },
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

/** What a save says when it is done, or when it failed. */
interface SaveWords {
  done: string
  doneText: (name: string) => string
  failed: string
  fallback: string
}

const CREATED: SaveWords = {
  done: 'Playbook Created',
  doneText: (name) => `"${name}" has been created successfully.`,
  failed: 'Error Creating Playbook',
  fallback: 'Failed to create playbook. Please try again.',
}

const UPDATED: SaveWords = {
  done: 'Playbook Updated',
  doneText: (name) => `"${name}" has been updated successfully.`,
  failed: 'Error Updating Playbook',
  fallback: 'Failed to update playbook. Please try again.',
}

/** A failed save's words: the server's, or the fallback. */
function failureText(err: unknown, fallback: string): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'object' && err !== null && 'detail' in err) return String((err as { detail: unknown }).detail)
  return fallback
}

/** Sends one save and says how it went; keeps the webhook id the server gives back. */
async function saveAndSay(
  data: PlaybookFormValues,
  send: () => Promise<unknown>,
  words: SaveWords,
  keepWebhookId: (id: string) => void,
  onSuccess?: () => void,
): Promise<void> {
  try {
    const result = (await send()) as { playbook?: { schedule_config?: { webhook_id?: string } } } | null
    const webhookId = result?.playbook?.schedule_config?.webhook_id
    if (webhookId) keepWebhookId(webhookId)
    toast(words.done, { description: words.doneText(data.name) })
    onSuccess?.()
  } catch (err: unknown) {
    toast.error(words.failed, { description: failureText(err, words.fallback) })
  }
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

  const save = useCallback(
    async (data: PlaybookFormValues, send: () => Promise<unknown>, words: SaveWords, onSuccess?: () => void) => {
      const validationError = validateFormData(data)
      if (validationError) {
        toast.error('Validation Error', { description: validationError })
        return
      }
      setIsSubmitting(true)
      try {
        await saveAndSay(data, send, words, setLastSavedWebhookId, onSuccess)
      } finally {
        setIsSubmitting(false)
      }
    },
    []
  )

  const submitPlaybook = useCallback(
    (data: PlaybookFormValues, onSuccess?: () => void) =>
      save(data, () => createPlaybookMutation.mutateAsync(transformFormToApiPayload(data)), CREATED, onSuccess),
    [createPlaybookMutation, save]
  )

  // F242: an edit is laid over the playbook's settings as they are stored now.
  const updatePlaybook = useCallback(
    (playbookId: string, data: PlaybookFormValues, onSuccess?: () => void) =>
      save(data, async () => {
        const stored = (await apiClient.getWorkflowRecipeById(playbookId)) as StoredPlaybookConfigs | null
        return updatePlaybookMutation.mutateAsync({ playbookId, playbookData: editPayload(data, stored) })
      }, UPDATED, onSuccess),
    [updatePlaybookMutation, save]
  )

  return {
    isSubmitting,
    lastSavedWebhookId,
    submitPlaybook,
    updatePlaybook,
  }
}
