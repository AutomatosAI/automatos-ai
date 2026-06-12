'use client'

import { motion } from 'framer-motion'
import {
  GitBranch,
  Bot,
  Search,
  Zap,
  Layers,
  Brain,
  Target,
  Database,
  FileText,
  CheckCircle,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface StageInfo {
  id: number
  name: string
  shortName: string
  icon: React.ReactNode
  description: string
  color: string
}

const STAGES: StageInfo[] = [
  { id: 1, name: 'Task Decomposition', shortName: 'Decompose', icon: <GitBranch className="w-4 h-4" />, description: 'Breaking task into subtasks', color: '#ff6b35' },
  { id: 2, name: 'Agent Selection', shortName: 'Select', icon: <Bot className="w-4 h-4" />, description: 'Choosing best agents', color: '#ff8c5a' },
  { id: 3, name: 'Context Engineering', shortName: 'Context', icon: <Search className="w-4 h-4" />, description: 'Optimizing context', color: '#fbbf24' },
  { id: 4, name: 'Agent Execution', shortName: 'Execute', icon: <Zap className="w-4 h-4" />, description: 'Running agents', color: '#ff6b35' },
  { id: 5, name: 'Result Aggregation', shortName: 'Aggregate', icon: <Layers className="w-4 h-4" />, description: 'Combining results', color: '#10b981' },
  { id: 6, name: 'Learning Update', shortName: 'Learn', icon: <Brain className="w-4 h-4" />, description: 'Extracting patterns', color: '#f97316' },
  { id: 7, name: 'Quality Assessment', shortName: 'Quality', icon: <Target className="w-4 h-4" />, description: 'Validating output', color: '#ff6b35' },
  { id: 8, name: 'Memory Storage', shortName: 'Memory', icon: <Database className="w-4 h-4" />, description: 'Storing learnings', color: '#f97316' },
  { id: 9, name: 'Response Generation', shortName: 'Response', icon: <FileText className="w-4 h-4" />, description: 'Final output', color: '#10b981' },
]

export interface TheaterStageProgressProps {
  /** Current active stage (1-9) */
  currentStage: number
  /** Optional callback when a stage is clicked */
  onStageClick?: (stage: number) => void
}

export function TheaterStageProgress({ currentStage, onStageClick }: TheaterStageProgressProps) {
  const completedStages = STAGES.filter((s) => s.id < currentStage).map((s) => s.id)
  const progress = ((currentStage - 1) / 9) * 100

  return (
    <div className="glass-panel rounded-2xl p-4">
      {/* Stage buttons row */}
      <div className="flex items-center justify-between gap-2">
        {STAGES.map((stage, index) => {
          const isCompleted = completedStages.includes(stage.id)
          const isActive = currentStage === stage.id
          const isPending = !isCompleted && !isActive

          return (
            <div key={stage.id} className="flex items-center flex-1">
              <motion.button
                onClick={() => onStageClick?.(stage.id)}
                className={cn(
                  'relative flex flex-col items-center justify-center p-3 rounded-xl border-2 transition-all min-w-[80px]',
                  isCompleted && 'stage-completed',
                  isActive && 'stage-active',
                  isPending && 'stage-pending'
                )}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {/* Icon */}
                <div
                  className={cn(
                    'mb-1 transition-colors',
                    isCompleted && 'text-success',
                    isActive && 'text-primary',
                    isPending && 'text-muted-foreground'
                  )}
                  style={{ color: isActive ? stage.color : undefined }}
                >
                  {isCompleted ? <CheckCircle className="w-5 h-5" /> : stage.icon}
                </div>

                {/* Label */}
                <span
                  className={cn(
                    'text-xs font-medium',
                    isCompleted && 'text-success/80',
                    isActive && 'text-white',
                    isPending && 'text-muted-foreground'
                  )}
                >
                  {stage.shortName}
                </span>

                {/* Stage number badge */}
                <div
                  className={cn(
                    'absolute -top-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold',
                    isCompleted && 'bg-success text-white',
                    isActive && 'bg-primary text-white',
                    isPending && 'bg-secondary text-muted-foreground'
                  )}
                >
                  {stage.id}
                </div>

                {/* Active pulse indicator */}
                {isActive && (
                  <motion.div
                    className="absolute -bottom-1 left-1/2 w-2 h-2 rounded-full bg-primary"
                    animate={{ scale: [1, 1.5, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                    style={{ marginLeft: -4 }}
                  />
                )}
              </motion.button>

              {/* Connector between stages */}
              {index < STAGES.length - 1 && (
                <div className="flex-1 mx-1 relative h-[2px]">
                  <div
                    className={cn(
                      'absolute inset-0 rounded-full',
                      isCompleted ? 'stage-connector-completed' : isActive ? 'stage-connector-active' : 'stage-connector'
                    )}
                  />
                  {isActive && (
                    <div className="absolute inset-0 overflow-hidden">
                      <div className="flow-particle w-4 h-full bg-gradient-to-r from-transparent via-primary to-transparent" />
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Progress bar */}
      <div className="mt-4 h-1.5 bg-secondary rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-emerald-500 via-primary to-warning relative"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5 }}
        >
          <div className="absolute inset-0 progress-shimmer" />
        </motion.div>
      </div>

      {/* Stage info text */}
      <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
        <span>Stage {currentStage} of 9</span>
        <span>{STAGES[currentStage - 1]?.description}</span>
      </div>
    </div>
  )
}

export { STAGES }
export type { StageInfo }
