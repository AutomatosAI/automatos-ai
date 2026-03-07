'use client'

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CheckCircle,
  XCircle,
  Loader2,
  Clock,
  Bot,
  ChevronDown,
  ChevronRight,
  Zap,
  Download,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api-client'

export interface RecipeStepResult {
  step_id: string
  order: number
  agent_id: number
  agent_name: string
  status: 'pending' | 'running' | 'success' | 'failed'
  output?: string
  output_preview?: string
  tool_calls?: Array<{
    action: string
    params?: any
    result?: any
    duration_ms?: number
  }>
  tool_calls_summary?: string[]
  duration_ms?: number
  tokens_used?: number
  started_at?: string
  completed_at?: string
  error?: string
  retries?: number
  log_url?: string
}

interface RecipeStepProgressProps {
  currentStep: number
  totalSteps: number
  status: 'pending' | 'running' | 'completed' | 'failed'
  stepResults: RecipeStepResult[]
  steps?: Array<{ step_id: string; order: number; prompt_template: string; agent_id: number }>
  recipeId?: string
  executionId?: string
}

export function RecipeStepProgress({
  currentStep,
  totalSteps,
  status,
  stepResults,
  steps,
  recipeId,
  executionId,
}: RecipeStepProgressProps) {
  const [expandedSteps, setExpandedSteps] = React.useState<Set<number>>(new Set())
  const [fullLogs, setFullLogs] = React.useState<Record<number, any>>({})
  const [loadingLogs, setLoadingLogs] = React.useState<Set<number>>(new Set())

  const toggleStep = (order: number) => {
    setExpandedSteps(prev => {
      const next = new Set(prev)
      if (next.has(order)) next.delete(order)
      else next.add(order)
      return next
    })
  }

  const loadFullLogs = async (stepOrder: number) => {
    if (!recipeId || !executionId || fullLogs[stepOrder] || loadingLogs.has(stepOrder)) return
    setLoadingLogs(prev => new Set(prev).add(stepOrder))
    try {
      const data: any = await apiClient.getRecipeStepFullLogs(recipeId, executionId, stepOrder)
      setFullLogs(prev => ({ ...prev, [stepOrder]: data?.log_data || data }))
    } catch (err) {
      console.error(`Failed to load full logs for step ${stepOrder}:`, err)
    } finally {
      setLoadingLogs(prev => {
        const next = new Set(prev)
        next.delete(stepOrder)
        return next
      })
    }
  }

  // Build step data: merge recipe steps definition with execution results
  const stepData = React.useMemo(() => {
    const items: Array<{
      order: number
      stepId: string
      prompt: string
      agentName: string
      agentId: number
      status: 'pending' | 'running' | 'success' | 'failed'
      result?: RecipeStepResult
    }> = []

    for (let i = 0; i < totalSteps; i++) {
      const stepDef = steps?.[i]
      const result = stepResults.find(r => r.order === i + 1)

      items.push({
        order: i + 1,
        stepId: stepDef?.step_id || result?.step_id || `step-${i + 1}`,
        prompt: stepDef?.prompt_template || result?.prompt_template || '',
        agentName: result?.agent_name || `Agent ${stepDef?.agent_id || '?'}`,
        agentId: stepDef?.agent_id || result?.agent_id || 0,
        status: result?.status || (i + 1 < currentStep ? 'success' : i + 1 === currentStep && status === 'running' ? 'running' : 'pending'),
        result,
      })
    }
    return items
  }, [totalSteps, steps, stepResults, currentStep, status])

  const statusLabel = status === 'running'
    ? `Step ${currentStep} of ${totalSteps} — Running`
    : status === 'completed'
    ? `All ${totalSteps} steps completed`
    : status === 'failed'
    ? `Failed at step ${currentStep} of ${totalSteps}`
    : 'Pending'

  const progressPercent = status === 'completed'
    ? 100
    : totalSteps > 0
    ? Math.round(((stepResults.filter(r => r.status === 'success').length) / totalSteps) * 100)
    : 0

  return (
    <div className="glass-panel rounded-2xl p-4 space-y-4">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Zap className="w-5 h-5 text-primary" />
          <h3 className="font-semibold text-sm">Recipe Execution</h3>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className={cn(
              'text-xs',
              status === 'running' && 'border-orange-400/50 text-orange-400',
              status === 'completed' && 'border-emerald-400/50 text-emerald-400',
              status === 'failed' && 'border-red-400/50 text-red-400',
              status === 'pending' && 'border-gray-400/50 text-gray-400'
            )}
          >
            {statusLabel}
          </Badge>
        </div>
      </div>

      {/* Progress bar */}
      <div className="relative h-2 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          className={cn(
            'h-full rounded-full',
            status === 'completed' ? 'bg-emerald-500' :
            status === 'failed' ? 'bg-red-500' :
            'bg-gradient-to-r from-orange-500 to-primary'
          )}
          initial={{ width: 0 }}
          animate={{ width: `${progressPercent}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
        {status === 'running' && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-pulse" />
        )}
      </div>

      {/* Step timeline */}
      <div className="space-y-2">
        <AnimatePresence initial={false}>
          {stepData.map((step, idx) => {
            const isExpanded = expandedSteps.has(step.order)
            const hasOutput = step.result?.output || step.result?.output_preview || step.result?.error || (step.result?.tool_calls?.length || 0) > 0 || (step.result?.tool_calls_summary?.length || 0) > 0 || step.result?.log_url

            return (
              <motion.div
                key={step.stepId}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05 }}
                className={cn(
                  'rounded-xl border transition-all duration-200',
                  step.status === 'running' && 'border-orange-400/30 bg-orange-400/5 ring-1 ring-orange-400/20',
                  step.status === 'success' && 'border-emerald-400/20 bg-emerald-400/5',
                  step.status === 'failed' && 'border-red-400/20 bg-red-400/5',
                  step.status === 'pending' && 'border-border/30 bg-white/[0.02]',
                )}
              >
                {/* Step header */}
                <button
                  className="w-full flex items-center gap-3 p-3 text-left"
                  onClick={() => hasOutput && toggleStep(step.order)}
                >
                  {/* Step number + status icon */}
                  <div className="flex-shrink-0 relative">
                    <div className={cn(
                      'w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold',
                      step.status === 'running' && 'bg-orange-400/20 text-orange-400',
                      step.status === 'success' && 'bg-emerald-400/20 text-emerald-400',
                      step.status === 'failed' && 'bg-red-400/20 text-red-400',
                      step.status === 'pending' && 'bg-white/10 text-muted-foreground',
                    )}>
                      {step.status === 'running' ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : step.status === 'success' ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : step.status === 'failed' ? (
                        <XCircle className="w-4 h-4" />
                      ) : (
                        <span>{step.order}</span>
                      )}
                    </div>
                    {/* Connector line to next step */}
                    {idx < stepData.length - 1 && (
                      <div className={cn(
                        'absolute left-1/2 -translate-x-1/2 top-full w-px h-2',
                        step.status === 'success' ? 'bg-emerald-400/30' : 'bg-white/10'
                      )} />
                    )}
                  </div>

                  {/* Step info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-xs font-medium text-muted-foreground">
                        Step {step.order}
                      </span>
                      <div className="flex items-center gap-1">
                        <Bot className="w-3 h-3 text-primary/60" />
                        <span className="text-xs text-primary/80">{step.agentName}</span>
                      </div>
                    </div>
                    <p className="text-sm truncate text-foreground/80">
                      {step.prompt.length > 80 ? step.prompt.slice(0, 80) + '...' : step.prompt || 'No prompt defined'}
                    </p>
                  </div>

                  {/* Metrics + expand indicator */}
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {step.result?.duration_ms != null && (
                      <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {(step.result.duration_ms / 1000).toFixed(1)}s
                      </span>
                    )}
                    {step.result?.tokens_used != null && step.result.tokens_used > 0 && (
                      <span className="text-[10px] text-muted-foreground">
                        {step.result.tokens_used} tok
                      </span>
                    )}
                    {hasOutput && (
                      isExpanded
                        ? <ChevronDown className="w-4 h-4 text-muted-foreground" />
                        : <ChevronRight className="w-4 h-4 text-muted-foreground" />
                    )}
                  </div>
                </button>

                {/* Expanded content */}
                <AnimatePresence>
                  {isExpanded && step.result && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="px-3 pb-3 space-y-2 border-t border-border/30 pt-2">
                        {/* Error message */}
                        {step.result.error && (
                          <div className="bg-red-400/10 border border-red-400/20 rounded-lg p-2 text-xs text-red-400">
                            {step.result.error}
                          </div>
                        )}

                        {/* Tool calls summary (compact from DB) */}
                        {step.result.tool_calls_summary && step.result.tool_calls_summary.length > 0 && !fullLogs[step.order] && (
                          <div className="space-y-1">
                            <span className="text-[10px] uppercase text-muted-foreground font-medium">Tool Calls</span>
                            <div className="flex flex-wrap gap-1">
                              {step.result.tool_calls_summary.map((summary, tcIdx) => (
                                <Badge key={tcIdx} variant="outline" className="text-[10px] h-5 bg-primary/10 text-primary border-primary/30">
                                  {summary}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Detailed tool calls (from DB step_results or full logs) */}
                        {(() => {
                          const toolCalls = fullLogs[step.order]?.tool_calls || step.result.tool_calls
                          if (!toolCalls || toolCalls.length === 0) return null
                          return (
                            <div className="space-y-1">
                              <span className="text-[10px] uppercase text-muted-foreground font-medium">
                                {fullLogs[step.order] ? 'Full Tool Calls' : 'Tool Calls'}
                              </span>
                              {toolCalls.map((tc: any, tcIdx: number) => (
                                <div key={tcIdx} className="bg-white/5 rounded-lg p-2 text-xs">
                                  <div className="flex items-center gap-2 mb-1">
                                    <Badge variant="outline" className="text-[10px] h-4 bg-primary/10 text-primary border-primary/30">
                                      {tc.action || tc.action_name || 'action'}
                                    </Badge>
                                    {tc.duration_ms != null && (
                                      <span className="text-[10px] text-muted-foreground">
                                        {(tc.duration_ms / 1000).toFixed(1)}s
                                      </span>
                                    )}
                                  </div>
                                  {tc.result && (
                                    <pre className="text-[10px] text-muted-foreground max-h-20 overflow-auto whitespace-pre-wrap mt-1">
                                      {typeof tc.result === 'string' ? tc.result.slice(0, 500) : JSON.stringify(tc.result, null, 2).slice(0, 500)}
                                    </pre>
                                  )}
                                </div>
                              ))}
                            </div>
                          )
                        })()}

                        {/* Output — full from S3 or preview from DB */}
                        {(() => {
                          const fullOutput = fullLogs[step.order]?.agent_output
                          const output = fullOutput || step.result.output || step.result.output_preview
                          if (!output) return null
                          return (
                            <div className="space-y-1">
                              <span className="text-[10px] uppercase text-muted-foreground font-medium">
                                {fullOutput ? 'Full Output' : 'Output Preview'}
                              </span>
                              <div className="bg-white/5 rounded-lg p-2">
                                <pre className="text-xs text-foreground/80 max-h-48 overflow-auto whitespace-pre-wrap">
                                  {fullOutput ? fullOutput : output.slice(0, 500)}
                                  {!fullOutput && output.length > 500 && '...'}
                                </pre>
                              </div>
                            </div>
                          )
                        })()}

                        {/* Load full logs button — only when S3 log_url exists and not yet loaded */}
                        {step.result.log_url && !fullLogs[step.order] && recipeId && executionId && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-xs text-primary/80 hover:text-primary gap-1.5 h-7"
                            disabled={loadingLogs.has(step.order)}
                            onClick={(e) => {
                              e.stopPropagation()
                              loadFullLogs(step.order)
                            }}
                          >
                            {loadingLogs.has(step.order) ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <Download className="w-3 h-3" />
                            )}
                            {loadingLogs.has(step.order) ? 'Loading...' : 'Load full logs'}
                          </Button>
                        )}

                        {/* Retries indicator */}
                        {step.result.retries != null && step.result.retries > 0 && (
                          <span className="text-[10px] text-amber-400">
                            Retried {step.result.retries} time{step.result.retries > 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
    </div>
  )
}
