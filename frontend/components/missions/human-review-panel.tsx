'use client'

import { useState } from 'react'
import { CheckCircle2, XCircle, AlertTriangle, ChevronDown, MessageSquare } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'
import { useMissionStore } from '@/stores/mission-store'
import { useReviewMission } from '@/hooks/use-missions-api'
import type { TaskResponse } from '@/types/missions'
import { TASK_STATE_CONFIG } from '@/types/missions'
import { toast } from 'sonner'

const proseClass =
  'prose prose-sm max-w-none dark:prose-invert prose-headings:text-foreground prose-p:text-foreground prose-a:text-orange-500 dark:prose-a:text-orange-300 prose-li:text-foreground prose-strong:text-foreground'

interface HumanReviewPanelProps {
  missionId: string
  tasks: TaskResponse[]
  className?: string
}

export function HumanReviewPanel({ missionId, tasks, className }: HumanReviewPanelProps) {
  const { taskFeedback, setTaskFeedback, removeTaskFeedback, clearTaskFeedback } = useMissionStore()
  const reviewMutation = useReviewMission()
  const [showRejectAll, setShowRejectAll] = useState(false)
  const [generalFeedback, setGeneralFeedback] = useState('')

  // Only show verified/completed tasks for review
  const reviewableTasks = tasks.filter((t) =>
    ['verified', 'completed', 'failed'].includes(t.state),
  )

  const flaggedCount = Object.keys(taskFeedback).length
  const canRejectFlagged = flaggedCount > 0

  const handleAccept = () => {
    reviewMutation.mutate(
      { id: missionId, body: { verdict: 'accept' } },
      {
        onSuccess: () => {
          clearTaskFeedback()
          toast.success('Mission accepted')
        },
        onError: (err) => toast.error(err.message || 'Failed to accept mission'),
      },
    )
  }

  const handleRejectFlagged = () => {
    if (!canRejectFlagged) return
    reviewMutation.mutate(
      {
        id: missionId,
        body: {
          verdict: 'reject',
          task_feedback: taskFeedback,
        },
      },
      {
        onSuccess: () => {
          clearTaskFeedback()
          toast.success('Mission rejected — flagged tasks will be retried')
        },
        onError: (err) => toast.error(err.message || 'Failed to reject mission'),
      },
    )
  }

  const handleRejectWithFeedback = () => {
    if (!generalFeedback.trim()) return
    reviewMutation.mutate(
      {
        id: missionId,
        body: {
          verdict: 'reject',
          feedback: generalFeedback.trim(),
        },
      },
      {
        onSuccess: () => {
          clearTaskFeedback()
          setGeneralFeedback('')
          setShowRejectAll(false)
          toast.success('Mission rejected with feedback — tasks will be retried')
        },
        onError: (err) => toast.error(err.message || 'Failed to reject mission'),
      },
    )
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="p-4 border-b border-border">
        <h3 className="text-sm font-semibold">Review Mission</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          {reviewableTasks.length} tasks to review
        </p>
      </div>

      {/* Task list */}
      <ScrollArea className="flex-1">
        <div className="p-3 space-y-2">
          {reviewableTasks.map((task) => (
            <TaskReviewItem
              key={task.id}
              task={task}
              feedback={taskFeedback[task.id] ?? null}
              onFeedbackChange={(feedback) => {
                if (feedback) {
                  setTaskFeedback(task.id, feedback)
                } else {
                  removeTaskFeedback(task.id)
                }
              }}
            />
          ))}
        </div>
      </ScrollArea>

      {/* Footer */}
      <div className="p-4 border-t border-border space-y-3">
        {flaggedCount > 0 && (
          <div className="flex items-center gap-2 text-xs text-[hsl(var(--warning))]">
            <AlertTriangle className="w-3.5 h-3.5" />
            {flaggedCount} task{flaggedCount !== 1 ? 's' : ''} flagged for revision
          </div>
        )}

        {showRejectAll && (
          <div className="space-y-2">
            <Textarea
              id="general-rejection-feedback"
              aria-label="General rejection feedback"
              value={generalFeedback}
              onChange={(e) => setGeneralFeedback(e.target.value)}
              placeholder="What needs to change? Provide general feedback for the mission..."
              className="text-xs min-h-[80px] bg-secondary/20"
            />
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="flex-1 border-destructive/30 text-destructive hover:bg-destructive/10"
                onClick={handleRejectWithFeedback}
                disabled={!generalFeedback.trim() || reviewMutation.isLoading}
              >
                Send Rejection
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => { setShowRejectAll(false); setGeneralFeedback('') }}
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        <div className="flex gap-2">
          <Button
            variant="outline"
            className="flex-1 border-[hsl(var(--success))]/30 text-[hsl(var(--success))] hover:bg-[hsl(var(--success))]/10"
            onClick={handleAccept}
            disabled={reviewMutation.isLoading}
          >
            <CheckCircle2 className="w-4 h-4 mr-1.5" />
            Accept All
          </Button>
          <Button
            variant="outline"
            className="flex-1 border-destructive/30 text-destructive hover:bg-destructive/10"
            onClick={handleRejectFlagged}
            disabled={!canRejectFlagged || reviewMutation.isLoading}
          >
            <XCircle className="w-4 h-4 mr-1.5" />
            Reject Flagged
          </Button>
          <Button
            variant="outline"
            className="shrink-0 border-destructive/30 text-destructive hover:bg-destructive/10"
            onClick={() => setShowRejectAll(!showRejectAll)}
            disabled={reviewMutation.isLoading}
            aria-label="Reject with general feedback"
            aria-pressed={showRejectAll}
          >
            <MessageSquare className="w-4 h-4" />
          </Button>
        </div>

        <p className="text-[10px] text-muted-foreground text-center">
          Flag individual tasks or use the feedback button to reject with general notes
        </p>
      </div>
    </div>
  )
}

// ── Per-task review item ──────────────────────────────────────

interface TaskReviewItemProps {
  task: TaskResponse
  feedback: string | null
  onFeedbackChange: (feedback: string | null) => void
}

function TaskReviewItem({ task, feedback, onFeedbackChange }: TaskReviewItemProps) {
  const [isOpen, setIsOpen] = useState(false)
  const isFlagged = feedback !== null
  const stateConfig = TASK_STATE_CONFIG[task.state]

  return (
    <div
      className={cn(
        'rounded-lg border p-3 transition-colors',
        isFlagged
          ? 'border-[hsl(var(--warning))]/30 bg-[hsl(var(--warning))]/5'
          : 'border-border bg-secondary/10',
      )}
    >
      {/* Task header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold line-clamp-1">{task.title}</p>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="outline" className={cn('text-[10px]', stateConfig.color)}>
              {stateConfig.label}
            </Badge>
            {task.agent_role && (
              <span className="text-[10px] text-muted-foreground">{task.agent_role}</span>
            )}
          </div>
        </div>
      </div>

      {/* Output preview */}
      {task.output && (
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <CollapsibleTrigger className="flex items-center gap-1 mt-2 text-[10px] text-muted-foreground hover:text-foreground cursor-pointer">
            <ChevronDown className={cn('w-3 h-3 transition-transform', isOpen && 'rotate-180')} />
            {isOpen ? 'Hide output' : 'Show output'}
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className={cn(proseClass, 'mt-2 rounded bg-secondary/30 p-2 text-[11px] leading-relaxed max-h-[300px] overflow-auto')}>
              <ReactMarkdown remarkPlugins={[remarkGfm]} disallowedElements={['img']} unwrapDisallowed>
                {task.output}
              </ReactMarkdown>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Flag toggle + feedback */}
      <div className="mt-2">
        {!isFlagged ? (
          <button
            onClick={() => onFeedbackChange('')}
            className="text-[10px] text-[hsl(var(--warning))] hover:underline cursor-pointer"
          >
            Flag for revision
          </button>
        ) : (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-[hsl(var(--warning))] font-medium">Flagged for revision</span>
              <button
                onClick={() => onFeedbackChange(null)}
                className="text-[10px] text-muted-foreground hover:text-foreground cursor-pointer"
              >
                Remove flag
              </button>
            </div>
            <Textarea
              value={feedback ?? ''}
              onChange={(e) => onFeedbackChange(e.target.value)}
              placeholder="What needs to change?"
              className="text-xs min-h-[60px] bg-secondary/20"
            />
          </div>
        )}
      </div>
    </div>
  )
}
