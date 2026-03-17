'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Lightbulb,
  ArrowUpRight,
  Zap,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  TrendingUp,
  XCircle,
  Loader2,
  Edit,
  Cpu,
  Wrench,
  BarChart3,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'

export interface Suggestion {
  type: 'prompt_rewrite' | 'model_upgrade' | 'tool_addition'
  description: string
  priority: 'high' | 'medium' | 'low'
  target_step?: number
  suggested_value?: string
}

export interface Pattern {
  type: string
  description: string
  severity: 'high' | 'medium' | 'low'
  step?: number
  stage?: number
}

export interface PerformanceMetrics {
  total_duration_ms?: number
  success_rate?: number
  total_retries?: number
  error_count?: number
  step_durations?: Record<string, number>
}

export interface SuggestionsData {
  recipe_id: string
  quality_score: number | null
  suggestions: Suggestion[]
  patterns: Pattern[]
  performance_metrics: PerformanceMetrics | null
  last_analyzed_at: string | null
  analysis_count: number
}

interface PlaybookSuggestionsPanelProps {
  data: SuggestionsData
  onApplyPromptRewrite?: (suggestion: Suggestion) => void
  onApplyModelUpgrade?: (suggestion: Suggestion) => void
  onApplyToolAddition?: (suggestion: Suggestion) => void
  playbookName?: string
}

const SUGGESTION_TYPE_CONFIG: Record<string, { icon: typeof Lightbulb; label: string; color: string }> = {
  prompt_rewrite: { icon: Edit, label: 'Prompt Rewrite', color: 'text-[hsl(var(--info))] bg-[hsl(var(--info))]/10 border-[hsl(var(--info))]/30' },
  model_upgrade: { icon: ArrowUpRight, label: 'Model Upgrade', color: 'text-[hsl(var(--agent))] bg-[hsl(var(--agent))]/10 border-[hsl(var(--agent))]/30' },
  tool_addition: { icon: Wrench, label: 'Tool Addition', color: 'text-[hsl(var(--success))] bg-[hsl(var(--success))]/10 border-[hsl(var(--success))]/30' },
}

const PRIORITY_CONFIG: Record<string, string> = {
  high: 'bg-[hsl(var(--destructive))]/10 text-[hsl(var(--destructive))] border-[hsl(var(--destructive))]/30',
  medium: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/30',
  low: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/30',
}

const SEVERITY_CONFIG: Record<string, { color: string; icon: typeof AlertTriangle }> = {
  high: { color: 'text-[hsl(var(--destructive))] bg-[hsl(var(--destructive))]/10 border-[hsl(var(--destructive))]/30', icon: XCircle },
  medium: { color: 'text-[hsl(var(--warning))] bg-[hsl(var(--warning))]/10 border-[hsl(var(--warning))]/30', icon: AlertTriangle },
  low: { color: 'text-[hsl(var(--info))] bg-[hsl(var(--info))]/10 border-[hsl(var(--info))]/30', icon: TrendingUp },
}

export function PlaybookSuggestionsPanel({
  data,
  onApplyPromptRewrite,
  onApplyModelUpgrade,
  onApplyToolAddition,
  playbookName,
}: PlaybookSuggestionsPanelProps) {
  const [showPatterns, setShowPatterns] = useState(false)
  const [showPerformance, setShowPerformance] = useState(false)
  const [applyingIndex, setApplyingIndex] = useState<number | null>(null)

  const hasSuggestions = data.suggestions.length > 0
  const hasPatterns = data.patterns.length > 0
  const hasPerformance = data.performance_metrics !== null

  const handleApply = async (suggestion: Suggestion, index: number) => {
    setApplyingIndex(index)
    try {
      if (suggestion.type === 'prompt_rewrite' && onApplyPromptRewrite) {
        onApplyPromptRewrite(suggestion)
      } else if (suggestion.type === 'model_upgrade' && onApplyModelUpgrade) {
        onApplyModelUpgrade(suggestion)
      } else if (suggestion.type === 'tool_addition' && onApplyToolAddition) {
        onApplyToolAddition(suggestion)
      }
    } finally {
      setApplyingIndex(null)
    }
  }

  const getApplyHandler = (suggestion: Suggestion): (() => void) | undefined => {
    if (suggestion.type === 'prompt_rewrite' && onApplyPromptRewrite) return () => {}
    if (suggestion.type === 'model_upgrade' && onApplyModelUpgrade) return () => {}
    if (suggestion.type === 'tool_addition' && onApplyToolAddition) return () => {}
    return undefined
  }

  if (!hasSuggestions && !hasPatterns) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Lightbulb className="w-8 h-8 mx-auto mb-3 opacity-50" />
        <p className="text-sm">No suggestions yet</p>
        <p className="text-xs mt-1">Execute this playbook to generate improvement suggestions</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Quality Score Header */}
      {data.quality_score !== null && (
        <div className="flex items-center gap-3 p-3 rounded-xl bg-secondary/30 border border-border/20">
          <div className="relative w-12 h-12">
            <svg className="w-12 h-12 -rotate-90" viewBox="0 0 36 36">
              <circle cx="18" cy="18" r="15.91" fill="none" stroke="currentColor" strokeWidth="2" className="text-secondary" />
              <circle
                cx="18" cy="18" r="15.91" fill="none" strokeWidth="2"
                strokeDasharray={`${data.quality_score * 100} ${100 - data.quality_score * 100}`}
                strokeLinecap="round"
                className={data.quality_score >= 0.75 ? 'text-[hsl(var(--success))]' : data.quality_score >= 0.5 ? 'text-[hsl(var(--warning))]' : 'text-[hsl(var(--destructive))]'}
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs font-bold">
              {Math.round(data.quality_score * 100)}
            </span>
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium">Quality Score</div>
            <div className="text-xs text-muted-foreground">
              {data.analysis_count} {data.analysis_count === 1 ? 'analysis' : 'analyses'} performed
              {data.last_analyzed_at && (
                <> &middot; Last: {new Date(data.last_analyzed_at).toLocaleDateString()}</>
              )}
            </div>
          </div>
          <Badge variant="outline" className="text-xs">
            {data.suggestions.length} suggestion{data.suggestions.length !== 1 ? 's' : ''}
          </Badge>
        </div>
      )}

      {/* Suggestions List */}
      {hasSuggestions && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Lightbulb className="w-4 h-4 text-primary" />
            Improvement Suggestions
          </div>
          <div className="space-y-2">
            {data.suggestions.map((suggestion, index) => {
              const typeConfig = SUGGESTION_TYPE_CONFIG[suggestion.type] || SUGGESTION_TYPE_CONFIG.prompt_rewrite
              const TypeIcon = typeConfig.icon
              const hasHandler = getApplyHandler(suggestion) !== undefined

              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="p-3 bg-secondary/20 border-border/20 hover:border-border/40 transition-colors">
                    <div className="flex items-start gap-3">
                      <div className={`p-1.5 rounded-lg border ${typeConfig.color}`}>
                        <TypeIcon className="w-3.5 h-3.5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="outline" className={`text-[10px] h-4 ${typeConfig.color}`}>
                            {typeConfig.label}
                          </Badge>
                          <Badge variant="outline" className={`text-[10px] h-4 ${PRIORITY_CONFIG[suggestion.priority]}`}>
                            {suggestion.priority}
                          </Badge>
                          {suggestion.target_step !== undefined && (
                            <span className="text-[10px] text-muted-foreground">
                              Step {suggestion.target_step}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          {suggestion.description}
                        </p>
                        {suggestion.suggested_value && (
                          <div className="mt-2 p-2 rounded-lg bg-black/20 border border-border/10">
                            <pre className="text-[10px] text-muted-foreground whitespace-pre-wrap font-mono">
                              {suggestion.suggested_value}
                            </pre>
                          </div>
                        )}
                      </div>
                      {hasHandler && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 text-xs shrink-0"
                          onClick={() => handleApply(suggestion, index)}
                          disabled={applyingIndex === index}
                        >
                          {applyingIndex === index ? (
                            <Loader2 className="w-3 h-3 animate-spin" />
                          ) : (
                            'Apply'
                          )}
                        </Button>
                      )}
                    </div>
                  </Card>
                </motion.div>
              )
            })}
          </div>
        </div>
      )}

      {/* Patterns Section (collapsible) */}
      {hasPatterns && (
        <div>
          <button
            className="flex items-center gap-2 text-sm font-medium w-full text-left hover:text-foreground transition-colors"
            onClick={() => setShowPatterns(!showPatterns)}
          >
            <AlertTriangle className="w-4 h-4 text-[hsl(var(--warning))]" />
            Detected Patterns ({data.patterns.length})
            {showPatterns ? <ChevronUp className="w-3.5 h-3.5 ml-auto" /> : <ChevronDown className="w-3.5 h-3.5 ml-auto" />}
          </button>
          <AnimatePresence>
            {showPatterns && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="space-y-2 mt-2">
                  {data.patterns.map((pattern, index) => {
                    const sevConfig = SEVERITY_CONFIG[pattern.severity] || SEVERITY_CONFIG.low
                    const SevIcon = sevConfig.icon

                    return (
                      <div key={index} className="flex items-start gap-2 p-2 rounded-lg bg-secondary/10 border border-border/10">
                        <SevIcon className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${sevConfig.color.split(' ')[0]}`} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5 mb-0.5">
                            <Badge variant="outline" className={`text-[10px] h-4 ${sevConfig.color}`}>
                              {pattern.severity}
                            </Badge>
                            <span className="text-[10px] text-muted-foreground">{pattern.type}</span>
                            {pattern.step !== undefined && (
                              <span className="text-[10px] text-muted-foreground">Step {pattern.step}</span>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground">{pattern.description}</p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Performance Metrics (collapsible) */}
      {hasPerformance && data.performance_metrics && (
        <div>
          <button
            className="flex items-center gap-2 text-sm font-medium w-full text-left hover:text-foreground transition-colors"
            onClick={() => setShowPerformance(!showPerformance)}
          >
            <BarChart3 className="w-4 h-4 text-[hsl(var(--info))]" />
            Performance Metrics
            {showPerformance ? <ChevronUp className="w-3.5 h-3.5 ml-auto" /> : <ChevronDown className="w-3.5 h-3.5 ml-auto" />}
          </button>
          <AnimatePresence>
            {showPerformance && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="grid grid-cols-2 gap-2 mt-2">
                  {data.performance_metrics.total_duration_ms !== undefined && (
                    <div className="p-2 rounded-lg bg-secondary/10 border border-border/10">
                      <div className="text-[10px] text-muted-foreground">Total Duration</div>
                      <div className="text-sm font-medium">{(data.performance_metrics.total_duration_ms / 1000).toFixed(1)}s</div>
                    </div>
                  )}
                  {data.performance_metrics.success_rate !== undefined && (
                    <div className="p-2 rounded-lg bg-secondary/10 border border-border/10">
                      <div className="text-[10px] text-muted-foreground">Success Rate</div>
                      <div className="text-sm font-medium">{Math.round(data.performance_metrics.success_rate * 100)}%</div>
                    </div>
                  )}
                  {data.performance_metrics.total_retries !== undefined && (
                    <div className="p-2 rounded-lg bg-secondary/10 border border-border/10">
                      <div className="text-[10px] text-muted-foreground">Total Retries</div>
                      <div className="text-sm font-medium">{data.performance_metrics.total_retries}</div>
                    </div>
                  )}
                  {data.performance_metrics.error_count !== undefined && (
                    <div className="p-2 rounded-lg bg-secondary/10 border border-border/10">
                      <div className="text-[10px] text-muted-foreground">Errors</div>
                      <div className="text-sm font-medium">{data.performance_metrics.error_count}</div>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
