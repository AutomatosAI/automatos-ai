'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Plus,
  Trash2,
  ChevronUp,
  ChevronDown,
  Bot,
  AlertCircle,
  GripVertical,
} from 'lucide-react'
import { useFormContext } from 'react-hook-form'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { useAgents } from '@/hooks/use-agent-api'
import type { RecipeFormValues } from './create-recipe-modal'

interface Agent {
  id: number
  name: string
  agent_model_config?: {
    model_id?: string
    provider?: string
  }
  model_config?: any
  skills?: Array<{ id: string; name: string }>
  status: string
}

const getModelDisplayName = (modelId?: string): string => {
  if (!modelId) return 'Unknown'
  const modelNames: Record<string, string> = {
    'gpt-4': 'GPT-4',
    'gpt-4-turbo': 'GPT-4 Turbo',
    'gpt-4-turbo-preview': 'GPT-4 Turbo',
    'gpt-3.5-turbo': 'GPT-3.5',
    'claude-3-opus-20240229': 'Claude 3 Opus',
    'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
    'claude-3-haiku-20240307': 'Claude 3 Haiku',
  }
  return modelNames[modelId] || modelId.split('-')[0].toUpperCase()
}

const ERROR_HANDLING_OPTIONS = [
  { value: 'stop', label: 'Stop Execution' },
  { value: 'skip', label: 'Skip Step' },
  { value: 'retry', label: 'Retry Step' },
  { value: 'fallback', label: 'Use Fallback' },
]

function VariableHighlightTextarea({
  value,
  onChange,
  placeholder,
}: {
  value: string
  onChange: (val: string) => void
  placeholder?: string
}) {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const overlayRef = React.useRef<HTMLPreElement>(null)

  const highlighted = React.useMemo(() => {
    if (!value) return ''
    // Escape HTML entities first to prevent XSS via dangerouslySetInnerHTML
    const escaped = value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
    return escaped.replace(
      /\{(\w+)\}/g,
      '<span class="text-orange-400 font-semibold">{$1}</span>'
    )
  }, [value])

  const handleScroll = () => {
    if (textareaRef.current && overlayRef.current) {
      overlayRef.current.scrollTop = textareaRef.current.scrollTop
      overlayRef.current.scrollLeft = textareaRef.current.scrollLeft
    }
  }

  return (
    <div className="relative">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onScroll={handleScroll}
        placeholder={placeholder}
        className="w-full min-h-[80px] bg-secondary/50 rounded-xl border border-border/30 px-3 py-2 text-sm font-mono text-transparent caret-orange-400 resize-y focus:outline-none focus:ring-1 focus:ring-orange-400/50"
      />
      <pre
        ref={overlayRef}
        className="absolute inset-0 px-3 py-2 text-sm font-mono whitespace-pre-wrap break-words pointer-events-none overflow-hidden"
        aria-hidden="true"
        dangerouslySetInnerHTML={{ __html: highlighted || `<span class="text-muted-foreground/50">${(placeholder || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>` }}
      />
    </div>
  )
}

export function RecipeStepBuilder() {
  const methods = useFormContext<RecipeFormValues>()
  const steps = methods.watch('steps')
  const { data: agentsData, isLoading: agentsLoading } = useAgents()

  const agents: Agent[] = React.useMemo(() => {
    if (!agentsData) return []
    const list = Array.isArray(agentsData) ? agentsData : (agentsData as any).agents || []
    return list.filter((a: Agent) => a.status === 'active')
  }, [agentsData])

  const addStep = () => {
    const newStep = {
      step_id: `step-${Date.now()}`,
      order: steps.length + 1,
      agent_id: '',
      prompt_template: '',
      pass_to: undefined as string | undefined,
      error_handling: 'stop',
    }
    methods.setValue('steps', [...steps, newStep])
  }

  const removeStep = (index: number) => {
    const updated = steps.filter((_, i) => i !== index).map((step, i) => ({
      ...step,
      order: i + 1,
    }))
    // Clear pass_to references to removed step
    const removedId = steps[index].step_id
    const cleaned = updated.map((step) => ({
      ...step,
      pass_to: step.pass_to === removedId ? undefined : step.pass_to,
    }))
    methods.setValue('steps', cleaned)
  }

  const moveStep = (index: number, direction: 'up' | 'down') => {
    const newIndex = direction === 'up' ? index - 1 : index + 1
    if (newIndex < 0 || newIndex >= steps.length) return

    const updated = [...steps]
    const temp = updated[index]
    updated[index] = updated[newIndex]
    updated[newIndex] = temp

    // Re-number orders
    const reordered = updated.map((step, i) => ({ ...step, order: i + 1 }))
    methods.setValue('steps', reordered)
  }

  const updateStep = (index: number, field: keyof RecipeFormValues['steps'][0], value: string | undefined) => {
    const updated = [...steps]
    updated[index] = { ...updated[index], [field]: value }
    methods.setValue('steps', updated)
  }

  // Validate circular dependencies in pass_to
  const hasCircularDependency = React.useMemo(() => {
    const graph = new Map<string, string>()
    for (const step of steps) {
      if (step.pass_to) {
        graph.set(step.step_id, step.pass_to)
      }
    }
    // Simple cycle detection: follow each chain
    for (const startId of graph.keys()) {
      const visited = new Set<string>()
      let current: string | undefined = startId
      while (current && graph.has(current)) {
        if (visited.has(current)) return true
        visited.add(current)
        current = graph.get(current)
      }
    }
    return false
  }, [steps])

  const validationErrors = React.useMemo(() => {
    const errors: string[] = []
    if (steps.length === 0) {
      errors.push('At least one step is required')
    }
    for (const step of steps) {
      if (!step.agent_id) {
        errors.push(`Step ${step.order}: Agent is required`)
      }
    }
    if (hasCircularDependency) {
      errors.push('Circular dependency detected in pass_to chain')
    }
    return errors
  }, [steps, hasCircularDependency])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            Workflow Steps
          </h3>
          <p className="text-xs text-muted-foreground mt-1">
            Add steps and assign agents. Use {'{variable_name}'} in prompts for input interpolation.
          </p>
        </div>
        <span className="text-xs text-muted-foreground">
          {steps.length} step{steps.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Validation Errors */}
      {validationErrors.length > 0 && steps.length > 0 && (
        <div className="flex items-start gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-400">
          <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
          <div className="space-y-0.5">
            {validationErrors.map((err, i) => (
              <div key={i}>{err}</div>
            ))}
          </div>
        </div>
      )}

      {/* Steps List */}
      <AnimatePresence mode="popLayout">
        {steps.map((step, index) => (
          <motion.div
            key={step.step_id}
            layout
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.2 }}
            className="glass-card rounded-2xl p-4 space-y-3 border border-border/20 hover:border-orange-400/20 transition-colors"
          >
            {/* Step Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <GripVertical className="w-4 h-4 text-muted-foreground/30" />
                <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-orange-400/10 text-orange-400 text-xs font-bold">
                  {step.order}
                </div>
                <span className="text-sm font-medium text-foreground/80">
                  Step {step.order}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={index === 0}
                  onClick={() => moveStep(index, 'up')}
                >
                  <ChevronUp className="w-3.5 h-3.5" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={index === steps.length - 1}
                  onClick={() => moveStep(index, 'down')}
                >
                  <ChevronDown className="w-3.5 h-3.5" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                  onClick={() => removeStep(index)}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              </div>
            </div>

            {/* Agent Selector */}
            <div>
              <Label className="text-xs text-muted-foreground">
                Agent <span className="text-red-400">*</span>
              </Label>
              <select
                value={step.agent_id}
                onChange={(e) => updateStep(index, 'agent_id', e.target.value)}
                className="w-full mt-1 bg-secondary/50 rounded-xl border border-border/30 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-orange-400/50 appearance-none cursor-pointer"
              >
                <option value="">Select an agent...</option>
                {agentsLoading && <option disabled>Loading agents...</option>}
                {agents.map((agent) => {
                  const model = getModelDisplayName(
                    agent.agent_model_config?.model_id || agent.model_config?.model_id
                  )
                  const toolCount = agent.skills?.length || 0
                  return (
                    <option key={agent.id} value={String(agent.id)}>
                      {agent.name} ({model} • {toolCount} tool{toolCount !== 1 ? 's' : ''})
                    </option>
                  )
                })}
              </select>
              {/* Agent info badge */}
              {step.agent_id && (() => {
                const selected = agents.find((a) => String(a.id) === step.agent_id)
                if (!selected) return null
                const model = getModelDisplayName(
                  selected.agent_model_config?.model_id || selected.model_config?.model_id
                )
                const toolCount = selected.skills?.length || 0
                return (
                  <div className="flex items-center gap-2 mt-1.5">
                    <Bot className="w-3 h-3 text-orange-400" />
                    <span className="text-xs text-muted-foreground">
                      {selected.name} — {model} • {toolCount} tool{toolCount !== 1 ? 's' : ''}
                    </span>
                  </div>
                )
              })()}
            </div>

            {/* Prompt Template */}
            <div>
              <Label className="text-xs text-muted-foreground">Prompt Template</Label>
              <div className="mt-1">
                <VariableHighlightTextarea
                  value={step.prompt_template}
                  onChange={(val) => updateStep(index, 'prompt_template', val)}
                  placeholder="Enter the prompt for this step. Use {variable_name} for input variables..."
                />
              </div>
            </div>

            {/* Pass To & Error Handling Row */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">Pass Output To</Label>
                <select
                  value={step.pass_to || ''}
                  onChange={(e) => updateStep(index, 'pass_to', e.target.value || undefined)}
                  className="w-full mt-1 bg-secondary/50 rounded-xl border border-border/30 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-orange-400/50 appearance-none cursor-pointer"
                >
                  <option value="">Next step (default)</option>
                  {steps
                    .filter((s) => s.step_id !== step.step_id)
                    .map((s) => (
                      <option key={s.step_id} value={s.step_id}>
                        Step {s.order}
                        {s.agent_id
                          ? ` (${agents.find((a) => String(a.id) === s.agent_id)?.name || 'Agent'})`
                          : ''}
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <Label className="text-xs text-muted-foreground">Error Handling</Label>
                <select
                  value={step.error_handling}
                  onChange={(e) => updateStep(index, 'error_handling', e.target.value)}
                  className="w-full mt-1 bg-secondary/50 rounded-xl border border-border/30 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-orange-400/50 appearance-none cursor-pointer"
                >
                  {ERROR_HANDLING_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Add Step Button */}
      <motion.div whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
        <Button
          type="button"
          onClick={addStep}
          className="w-full bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200 rounded-2xl h-11"
        >
          <Plus className="w-4 h-4 mr-2" />
          Add Step
        </Button>
      </motion.div>

      {/* Empty State */}
      {steps.length === 0 && (
        <div className="p-6 border border-dashed border-border/50 rounded-2xl text-center text-muted-foreground text-sm">
          No steps configured. Click "Add Step" to begin building your workflow.
        </div>
      )}
    </div>
  )
}
