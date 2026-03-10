'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain,
  Target,
  Database,
  ChevronDown,
  ChevronRight,
  Lightbulb,
  AlertTriangle,
  TrendingUp,
  Award,
  Zap,
  CheckCircle,
  XCircle,
  BarChart3,
  ArrowUpRight,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

// --- Types ---

export interface LearningPattern {
  type: string
  description: string
  severity?: 'high' | 'medium' | 'low'
  step?: number | string | null
  stage?: number | string | null
}

export interface LearningSuggestion {
  type: 'prompt_rewrite' | 'model_upgrade' | 'tool_addition'
  description: string
  priority: 'high' | 'medium' | 'low'
  target_step?: number | string | null
  details?: string
}

export interface LearningPerformanceMetrics {
  total_duration_ms?: number
  step_durations?: Record<string, number>
  success_rate?: number
  total_retries?: number
  error_count?: number
}

export interface LearningData {
  patterns: LearningPattern[]
  suggestions: LearningSuggestion[]
  performance_metrics: LearningPerformanceMetrics
}

export interface QualityBreakdown {
  completeness: number
  accuracy: number
  efficiency: number
  reliability: number
  cost: number
}

export interface QualityBottleneck {
  type: 'slow_step' | 'failed_step' | 'high_retry_step'
  step_index: number
  value?: number
  description?: string
}

export interface QualityData {
  quality_score: number
  breakdown: QualityBreakdown
  grade: string
  bottlenecks: QualityBottleneck[]
}

export interface MemoryEntry {
  scope: string
  type: string
  agent_id?: string | number
  step_index?: number
}

export interface MemoryData {
  stored_memories: number
  scopes: string[]
  memories: MemoryEntry[]
  errors?: string[]
}

export interface TheaterSelfLearningPanelProps {
  learningData?: LearningData | null
  qualityData?: QualityData | null
  memoryData?: MemoryData | null
  defaultOpen?: boolean
}

// --- Helpers ---

const formatDuration = (ms?: number) => {
  if (!ms) return '—'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

const getSeverityColor = (severity?: string) => {
  switch (severity) {
    case 'high':
      return 'text-red-400 border-red-400/30 bg-red-400/10'
    case 'medium':
      return 'text-amber-400 border-amber-400/30 bg-amber-400/10'
    case 'low':
      return 'text-blue-400 border-blue-400/30 bg-blue-400/10'
    default:
      return 'text-gray-400 border-gray-400/30 bg-gray-400/10'
  }
}

const getPriorityColor = (priority?: string) => {
  switch (priority) {
    case 'high':
      return 'text-red-400 border-red-400/40'
    case 'medium':
      return 'text-amber-400 border-amber-400/40'
    case 'low':
      return 'text-emerald-400 border-emerald-400/40'
    default:
      return 'text-gray-400 border-gray-400/40'
  }
}

const getSuggestionIcon = (type: LearningSuggestion['type']) => {
  switch (type) {
    case 'prompt_rewrite':
      return <Lightbulb className="w-3.5 h-3.5" />
    case 'model_upgrade':
      return <ArrowUpRight className="w-3.5 h-3.5" />
    case 'tool_addition':
      return <Zap className="w-3.5 h-3.5" />
    default:
      return <Lightbulb className="w-3.5 h-3.5" />
  }
}

const getSuggestionLabel = (type: LearningSuggestion['type']) => {
  switch (type) {
    case 'prompt_rewrite':
      return 'Prompt Rewrite'
    case 'model_upgrade':
      return 'Model Upgrade'
    case 'tool_addition':
      return 'Tool Addition'
    default:
      return type
  }
}

const getGradeColor = (grade: string) => {
  switch (grade) {
    case 'A':
      return 'text-emerald-400 bg-emerald-400/20 border-emerald-400/40'
    case 'B':
      return 'text-blue-400 bg-blue-400/20 border-blue-400/40'
    case 'C':
      return 'text-amber-400 bg-amber-400/20 border-amber-400/40'
    case 'D':
      return 'text-orange-400 bg-orange-400/20 border-orange-400/40'
    case 'F':
      return 'text-red-400 bg-red-400/20 border-red-400/40'
    default:
      return 'text-gray-400 bg-gray-400/20 border-gray-400/40'
  }
}

const getBottleneckIcon = (type: QualityBottleneck['type']) => {
  switch (type) {
    case 'slow_step':
      return <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
    case 'failed_step':
      return <XCircle className="w-3.5 h-3.5 text-red-400" />
    case 'high_retry_step':
      return <TrendingUp className="w-3.5 h-3.5 text-orange-400" />
    default:
      return <AlertTriangle className="w-3.5 h-3.5 text-gray-400" />
  }
}

// --- Sub-components ---

function CollapsibleSection({
  title,
  icon,
  count,
  defaultOpen = false,
  children,
}: {
  title: string
  icon: React.ReactNode
  count?: number
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border border-border/30 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 p-3 text-left hover:bg-white/5 transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-muted-foreground flex-shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />
        )}
        <span className="text-muted-foreground flex-shrink-0">{icon}</span>
        <span className="text-sm font-medium">{title}</span>
        {count !== undefined && (
          <Badge variant="outline" className="text-[10px] h-5 ml-auto">
            {count}
          </Badge>
        )}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-3 pt-0 space-y-2">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function QualityBar({
  label,
  value,
  color,
}: {
  label: string
  value: number
  color: string
}) {
  const percentage = Math.round(value * 100)

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{percentage}%</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}

function LearnSection({ data }: { data: LearningData }) {
  const { patterns, suggestions, performance_metrics } = data

  return (
    <div className="space-y-3">
      {/* Patterns */}
      {patterns.length > 0 && (
        <CollapsibleSection
          title="Extracted Patterns"
          icon={<Brain className="w-4 h-4" />}
          count={patterns.length}
          defaultOpen={true}
        >
          <div className="space-y-2">
            {patterns.map((pattern, i) => (
              <div
                key={i}
                className={cn(
                  'rounded-lg border p-2.5 text-sm',
                  getSeverityColor(pattern.severity)
                )}
              >
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant="outline" className="text-[10px] h-5">
                    {pattern.type}
                  </Badge>
                  {pattern.severity && (
                    <Badge
                      variant="outline"
                      className={cn('text-[10px] h-5', getPriorityColor(pattern.severity))}
                    >
                      {pattern.severity}
                    </Badge>
                  )}
                  {pattern.step != null && (
                    <span className="text-[10px] text-muted-foreground">
                      Step {pattern.step}
                    </span>
                  )}
                </div>
                <p className="text-xs break-words">{pattern.description}</p>
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <CollapsibleSection
          title="Improvement Suggestions"
          icon={<Lightbulb className="w-4 h-4" />}
          count={suggestions.length}
          defaultOpen={true}
        >
          <div className="space-y-2">
            {suggestions.map((suggestion, i) => (
              <div
                key={i}
                className="rounded-lg border border-border/30 bg-black/20 p-2.5"
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-orange-400">
                    {getSuggestionIcon(suggestion.type)}
                  </span>
                  <Badge
                    variant="outline"
                    className="text-[10px] h-5 text-orange-400 border-orange-400/30"
                  >
                    {getSuggestionLabel(suggestion.type)}
                  </Badge>
                  <Badge
                    variant="outline"
                    className={cn('text-[10px] h-5', getPriorityColor(suggestion.priority))}
                  >
                    {suggestion.priority}
                  </Badge>
                  {suggestion.target_step != null && (
                    <span className="text-[10px] text-muted-foreground ml-auto">
                      Step {suggestion.target_step}
                    </span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground break-words">
                  {suggestion.description}
                </p>
                {suggestion.details && (
                  <p className="text-xs text-muted-foreground/70 mt-1 break-words italic">
                    {suggestion.details}
                  </p>
                )}
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* Performance Metrics */}
      {performance_metrics && (
        <CollapsibleSection
          title="Performance Metrics"
          icon={<BarChart3 className="w-4 h-4" />}
          defaultOpen={false}
        >
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-lg border border-border/30 bg-black/20 p-2.5">
              <div className="text-[10px] uppercase text-muted-foreground mb-0.5">
                Total Duration
              </div>
              <div className="text-sm font-medium">
                {formatDuration(performance_metrics.total_duration_ms)}
              </div>
            </div>
            <div className="rounded-lg border border-border/30 bg-black/20 p-2.5">
              <div className="text-[10px] uppercase text-muted-foreground mb-0.5">
                Success Rate
              </div>
              <div className="text-sm font-medium">
                {performance_metrics.success_rate != null
                  ? `${Math.round(performance_metrics.success_rate * 100)}%`
                  : '—'}
              </div>
            </div>
            <div className="rounded-lg border border-border/30 bg-black/20 p-2.5">
              <div className="text-[10px] uppercase text-muted-foreground mb-0.5">
                Total Retries
              </div>
              <div className="text-sm font-medium">
                {performance_metrics.total_retries ?? 0}
              </div>
            </div>
            <div className="rounded-lg border border-border/30 bg-black/20 p-2.5">
              <div className="text-[10px] uppercase text-muted-foreground mb-0.5">
                Errors
              </div>
              <div className="text-sm font-medium">
                {performance_metrics.error_count ?? 0}
              </div>
            </div>
          </div>
        </CollapsibleSection>
      )}

      {/* Empty state */}
      {patterns.length === 0 && suggestions.length === 0 && (
        <div className="text-center text-sm text-muted-foreground py-4">
          No patterns or suggestions extracted from this execution.
        </div>
      )}
    </div>
  )
}

function QualitySection({ data }: { data: QualityData }) {
  const { quality_score, breakdown, grade, bottlenecks } = data
  const scorePercent = Math.round(quality_score * 100)

  const dimensionColors: Record<string, string> = {
    completeness: '#10b981',
    accuracy: '#3b82f6',
    efficiency: '#f59e0b',
    reliability: '#8b5cf6',
    cost: '#f97316',
  }

  return (
    <div className="space-y-3">
      {/* Score and Grade header */}
      <div className="flex items-center gap-4">
        {/* Score circle */}
        <div className="relative w-16 h-16 flex-shrink-0">
          <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
            <circle
              cx="32"
              cy="32"
              r="28"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
              className="text-gray-800"
            />
            <motion.circle
              cx="32"
              cy="32"
              r="28"
              stroke={quality_score >= 0.75 ? '#10b981' : quality_score >= 0.5 ? '#f59e0b' : '#ef4444'}
              strokeWidth="4"
              fill="none"
              strokeLinecap="round"
              strokeDasharray={`${2 * Math.PI * 28}`}
              initial={{ strokeDashoffset: 2 * Math.PI * 28 }}
              animate={{ strokeDashoffset: 2 * Math.PI * 28 * (1 - quality_score) }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-sm font-bold">{scorePercent}%</span>
          </div>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold">Quality Score</span>
            <Badge
              variant="outline"
              className={cn('text-xs font-bold px-2', getGradeColor(grade))}
            >
              {grade}
            </Badge>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{
                background: `linear-gradient(90deg, ${
                  quality_score >= 0.75 ? '#10b981' : quality_score >= 0.5 ? '#f59e0b' : '#ef4444'
                }, ${quality_score >= 0.9 ? '#10b981' : '#f97316'})`,
              }}
              initial={{ width: 0 }}
              animate={{ width: `${scorePercent}%` }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
            />
          </div>
        </div>
      </div>

      {/* 5-Dimensional Breakdown */}
      <div className="space-y-2">
        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Quality Dimensions
        </div>
        {Object.entries(breakdown).map(([key, value]) => (
          <QualityBar
            key={key}
            label={key.charAt(0).toUpperCase() + key.slice(1)}
            value={value}
            color={dimensionColors[key] || '#f97316'}
          />
        ))}
      </div>

      {/* Bottlenecks */}
      {bottlenecks.length > 0 && (
        <CollapsibleSection
          title="Bottlenecks"
          icon={<AlertTriangle className="w-4 h-4" />}
          count={bottlenecks.length}
          defaultOpen={true}
        >
          <div className="space-y-2">
            {bottlenecks.map((bottleneck, i) => (
              <div
                key={i}
                className="flex items-start gap-2 rounded-lg border border-border/30 bg-black/20 p-2.5"
              >
                {getBottleneckIcon(bottleneck.type)}
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[10px] h-5">
                      {bottleneck.type.replace(/_/g, ' ')}
                    </Badge>
                    <span className="text-[10px] text-muted-foreground">
                      Step {bottleneck.step_index}
                    </span>
                  </div>
                  {bottleneck.description && (
                    <p className="text-xs text-muted-foreground mt-0.5 break-words">
                      {bottleneck.description}
                    </p>
                  )}
                  {bottleneck.value != null && (
                    <span className="text-[10px] text-muted-foreground">
                      {bottleneck.type === 'slow_step'
                        ? formatDuration(bottleneck.value)
                        : `${bottleneck.value} retries`}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}
    </div>
  )
}

function MemorySection({ data }: { data: MemoryData }) {
  const { stored_memories, scopes, memories, errors } = data

  return (
    <div className="space-y-3">
      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-lg border border-border/30 bg-black/20 p-2.5">
          <div className="text-[10px] uppercase text-muted-foreground mb-0.5">
            Memories Stored
          </div>
          <div className="text-sm font-medium flex items-center gap-1">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
            {stored_memories}
          </div>
        </div>
        <div className="rounded-lg border border-border/30 bg-black/20 p-2.5">
          <div className="text-[10px] uppercase text-muted-foreground mb-0.5">
            Scopes Used
          </div>
          <div className="text-sm font-medium">{scopes?.length ?? 0}</div>
        </div>
      </div>

      {/* Memory entries */}
      {memories && memories.length > 0 && (
        <CollapsibleSection
          title="Stored Memories"
          icon={<Database className="w-4 h-4" />}
          count={memories.length}
          defaultOpen={false}
        >
          <div className="space-y-1.5">
            {memories.map((entry, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-lg border border-border/30 bg-black/20 p-2 text-xs"
              >
                {entry.type === 'recipe_execution' ? (
                  <Target className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" />
                ) : (
                  <Brain className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" />
                )}
                <div className="min-w-0 flex-1">
                  <Badge
                    variant="outline"
                    className="text-[10px] h-5 mr-1"
                  >
                    {entry.type.replace(/_/g, ' ')}
                  </Badge>
                  {entry.agent_id != null && (
                    <span className="text-muted-foreground">
                      Agent {entry.agent_id}
                    </span>
                  )}
                  {entry.step_index != null && (
                    <span className="text-muted-foreground ml-1">
                      (Step {entry.step_index})
                    </span>
                  )}
                </div>
                <span className="text-[10px] text-muted-foreground truncate max-w-[120px]" title={entry.scope}>
                  {entry.scope}
                </span>
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* Scopes list */}
      {scopes && scopes.length > 0 && (
        <CollapsibleSection
          title="Memory Scopes"
          icon={<Database className="w-4 h-4" />}
          count={scopes.length}
          defaultOpen={false}
        >
          <div className="space-y-1">
            {scopes.map((scope, i) => (
              <div
                key={i}
                className="text-xs text-muted-foreground font-mono bg-black/20 rounded-lg px-2.5 py-1.5 break-all"
              >
                {scope}
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* Errors */}
      {errors && errors.length > 0 && (
        <div className="space-y-1.5">
          {errors.map((error, i) => (
            <div
              key={i}
              className="flex items-start gap-2 rounded-lg border border-red-400/30 bg-red-400/10 p-2.5 text-xs text-red-300"
            >
              <XCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span className="break-words">{error}</span>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {stored_memories === 0 && (!errors || errors.length === 0) && (
        <div className="text-center text-sm text-muted-foreground py-4">
          No memories stored for this execution.
        </div>
      )}
    </div>
  )
}

// --- Main Component ---

export function TheaterSelfLearningPanel({
  learningData,
  qualityData,
  memoryData,
  defaultOpen = false,
}: TheaterSelfLearningPanelProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  const hasData = learningData || qualityData || memoryData

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="glass-card rounded-2xl overflow-hidden border border-orange-400/20"
    >
      {/* Panel header - always visible */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-3 p-4 text-left hover:bg-white/5 transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="w-5 h-5 text-orange-400 flex-shrink-0" />
        ) : (
          <ChevronRight className="w-5 h-5 text-orange-400 flex-shrink-0" />
        )}
        <div className="flex items-center gap-2">
          <Award className="w-5 h-5 text-orange-400" />
          <span className="text-sm font-semibold">Self-Learning Analysis</span>
        </div>

        {/* Summary badges */}
        <div className="flex items-center gap-2 ml-auto">
          {qualityData && (
            <Badge
              variant="outline"
              className={cn('text-[10px] h-5', getGradeColor(qualityData.grade))}
            >
              Grade {qualityData.grade}
            </Badge>
          )}
          {learningData && learningData.patterns.length > 0 && (
            <Badge variant="outline" className="text-[10px] h-5 text-orange-400 border-orange-400/30">
              {learningData.patterns.length} patterns
            </Badge>
          )}
          {learningData && learningData.suggestions.length > 0 && (
            <Badge variant="outline" className="text-[10px] h-5 text-amber-400 border-amber-400/30">
              {learningData.suggestions.length} suggestions
            </Badge>
          )}
          {memoryData && memoryData.stored_memories > 0 && (
            <Badge variant="outline" className="text-[10px] h-5 text-purple-400 border-purple-400/30">
              {memoryData.stored_memories} memories
            </Badge>
          )}
        </div>
      </button>

      {/* Panel body - collapsible */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-4">
              {!hasData && (
                <div className="text-center text-sm text-muted-foreground py-6">
                  Self-learning analysis will appear here after the Learn stage completes.
                </div>
              )}

              {/* Learn Section (Stage 6) */}
              {learningData && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4 text-orange-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-orange-400">
                      Stage 6: Learn
                    </span>
                  </div>
                  <LearnSection data={learningData} />
                </div>
              )}

              {/* Quality Section (Stage 7) */}
              {qualityData && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Target className="w-4 h-4 text-orange-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-orange-400">
                      Stage 7: Quality
                    </span>
                  </div>
                  <QualitySection data={qualityData} />
                </div>
              )}

              {/* Memory Section (Stage 8) */}
              {memoryData && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Database className="w-4 h-4 text-orange-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-orange-400">
                      Stage 8: Memory
                    </span>
                  </div>
                  <MemorySection data={memoryData} />
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
