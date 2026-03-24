'use client'

import { useState, useMemo } from 'react'
import { ChevronDown, CheckCircle2, XCircle, Copy, Check, FileText, Download } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'
import type { TaskResponse, MissionDetailResponse } from '@/types/missions'
import { TASK_STATE_CONFIG } from '@/types/missions'
import { toast } from 'sonner'

interface MissionResultsPanelProps {
  mission: MissionDetailResponse
  className?: string
}

const proseClass =
  'prose prose-sm max-w-none dark:prose-invert prose-headings:text-foreground prose-p:text-foreground prose-a:text-orange-500 dark:prose-a:text-orange-300 prose-li:text-foreground prose-strong:text-foreground'

type ResultsView = 'combined' | 'per-task'

export function MissionResultsPanel({ mission, className }: MissionResultsPanelProps) {
  const isTerminal = ['completed', 'failed', 'cancelled'].includes(mission.state)
  const [view, setView] = useState<ResultsView>(isTerminal ? 'combined' : 'per-task')

  const completedTasks = mission.tasks
    .filter((t) => ['verified', 'completed', 'failed'].includes(t.state))
    .sort((a, b) => a.sequence_number - b.sequence_number)

  const summary = mission.output_summary as Record<string, unknown> | null

  const combinedMarkdown = useMemo(() => {
    return completedTasks
      .filter((t) => t.output)
      .map((t) => `# ${t.title}\n\n${t.output}`)
      .join('\n\n---\n\n')
  }, [completedTasks])

  const handleCopyAll = async () => {
    await navigator.clipboard.writeText(combinedMarkdown)
    toast.success('All results copied to clipboard')
  }

  const handleDownload = () => {
    const blob = new Blob([combinedMarkdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `mission-results-${mission.id.slice(0, 8)}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="p-4 border-b border-border space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-1.5">
              <FileText className="w-4 h-4" />
              Mission Results
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              {completedTasks.filter((t) => t.state === 'verified').length} of{' '}
              {mission.tasks.length} tasks verified
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="w-3.5 h-3.5 mr-1.5" />
              Download .md
            </Button>
            <Button variant="outline" size="sm" onClick={handleCopyAll}>
              <Copy className="w-3.5 h-3.5 mr-1.5" />
              Copy All
            </Button>
          </div>
        </div>

        {/* View toggle */}
        <div className="flex gap-1 bg-secondary/30 rounded-md p-0.5 w-fit">
          <button
            onClick={() => setView('combined')}
            className={cn(
              'px-3 py-1 text-xs rounded transition-colors cursor-pointer',
              view === 'combined'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Combined
          </button>
          <button
            onClick={() => setView('per-task')}
            className={cn(
              'px-3 py-1 text-xs rounded transition-colors cursor-pointer',
              view === 'per-task'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Per Task
          </button>
        </div>
      </div>

      {/* Summary stats */}
      {summary && (
        <div className="px-4 py-3 border-b border-border bg-secondary/10">
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <p className="text-lg font-semibold">{(summary.tasks_completed as number) ?? 0}</p>
              <p className="text-[10px] text-muted-foreground">Completed</p>
            </div>
            <div>
              <p className="text-lg font-semibold">{(summary.tasks_failed as number) ?? 0}</p>
              <p className="text-[10px] text-muted-foreground">Failed</p>
            </div>
            <div>
              <p className="text-lg font-semibold">
                {summary.total_duration_seconds
                  ? `${Math.round((summary.total_duration_seconds as number) / 60)}m`
                  : '-'}
              </p>
              <p className="text-[10px] text-muted-foreground">Duration</p>
            </div>
          </div>
        </div>
      )}

      {/* Task outputs */}
      <ScrollArea className="flex-1">
        {view === 'combined' ? (
          <div className="p-4">
            {combinedMarkdown ? (
              <div className={cn(proseClass, 'text-xs leading-relaxed')}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {combinedMarkdown}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground text-center py-8">
                No completed tasks yet
              </p>
            )}
          </div>
        ) : (
          <div className="p-3 space-y-2">
            {completedTasks.map((task) => (
              <TaskResultItem key={task.id} task={task} />
            ))}
            {completedTasks.length === 0 && (
              <p className="text-xs text-muted-foreground text-center py-8">
                No completed tasks yet
              </p>
            )}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}

// ── Per-task result ─────────────────────────────────────────────

function TaskResultItem({ task }: { task: TaskResponse }) {
  const [isOpen, setIsOpen] = useState(task.state === 'verified')
  const [copied, setCopied] = useState(false)
  const stateConfig = TASK_STATE_CONFIG[task.state]
  const isVerified = task.state === 'verified'
  const isFailed = task.state === 'failed'

  const handleCopy = async () => {
    if (!task.output) return
    await navigator.clipboard.writeText(task.output)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div
        className={cn(
          'rounded-lg border transition-colors',
          isVerified && 'border-[hsl(var(--success))]/20',
          isFailed && 'border-destructive/20',
          !isVerified && !isFailed && 'border-border',
        )}
      >
        {/* Task header */}
        <CollapsibleTrigger className="w-full p-3 flex items-center justify-between gap-2 hover:bg-secondary/20 transition-colors rounded-t-lg cursor-pointer">
          <div className="flex items-center gap-2 min-w-0">
            {isVerified ? (
              <CheckCircle2 className="w-4 h-4 text-[hsl(var(--success))] shrink-0" />
            ) : isFailed ? (
              <XCircle className="w-4 h-4 text-destructive shrink-0" />
            ) : null}
            <span className="text-xs font-medium truncate">{task.title}</span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Badge variant="outline" className={cn('text-[10px]', stateConfig.color)}>
              {stateConfig.label}
            </Badge>
            <ChevronDown
              className={cn(
                'w-3.5 h-3.5 text-muted-foreground transition-transform',
                isOpen && 'rotate-180',
              )}
            />
          </div>
        </CollapsibleTrigger>

        {/* Task output */}
        <CollapsibleContent>
          <div className="px-3 pb-3 border-t border-border/50">
            {/* Copy button */}
            {task.output && (
              <div className="flex justify-end pt-2 pb-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 text-[10px] text-muted-foreground"
                  onClick={handleCopy}
                >
                  {copied ? (
                    <Check className="w-3 h-3 mr-1" />
                  ) : (
                    <Copy className="w-3 h-3 mr-1" />
                  )}
                  {copied ? 'Copied' : 'Copy'}
                </Button>
              </div>
            )}

            {/* Markdown output */}
            {task.output ? (
              <div className={cn(proseClass, 'text-xs leading-relaxed')}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {task.output}
                </ReactMarkdown>
              </div>
            ) : task.failure_detail ? (
              <div className="text-xs text-destructive/80 bg-destructive/5 rounded p-2 mt-2">
                {task.failure_detail}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground italic mt-2">No output</p>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}
