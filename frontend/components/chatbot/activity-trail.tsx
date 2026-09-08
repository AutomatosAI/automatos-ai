'use client'

/**
 * PRD-238 S3 — what Auto did this turn, one line per tool call, in order.
 *
 * Replaces the running/error-only chips: completed calls stay visible with
 * their one-line result and duration, a skipped call closes its line, and
 * a cap that ended the turn is said out loud. Click a line for its input.
 */
import { CheckCircle2, ChevronRight, Loader2, MinusCircle, XCircle } from 'lucide-react'
import type { LimitReached, ToolCall } from '@/types'
import { formatDuration } from '@/lib/chat/tool-calls'

export interface ActivityTrailProps {
  toolCalls: ToolCall[]
  formatLabel: (tc: ToolCall) => string
  /** PRD-238 S4: progress lines from a long-running tool call, newest last. */
  progress?: string[]
}

function stateIcon(tc: ToolCall) {
  if (tc.state === 'running') return <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-[hsl(var(--info))]" aria-label="Running" />
  if (tc.state === 'error') return <XCircle className="h-3.5 w-3.5 shrink-0 text-destructive/80" aria-label="Failed" />
  if (tc.skipped) return <MinusCircle className="h-3.5 w-3.5 shrink-0 text-muted-foreground/70" aria-label="Skipped" />
  return <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-success/80" aria-label="Done" />
}

export function ActivityTrail({ toolCalls, formatLabel, progress = [] }: ActivityTrailProps) {
  if (toolCalls.length === 0 && progress.length === 0) return null
  return (
    <ol className="space-y-0.5 text-xs" aria-label="Activity">
      {toolCalls.map((tc) => {
        const label = formatLabel(tc)
        const text = tc.state === 'error' ? `${label} failed` : label
        const hasInput = tc.input && typeof tc.input === 'object' && Object.keys(tc.input).length > 0
        return (
          <li key={tc.toolCallId}>
            <details className="group/trail">
              <summary
                className="flex cursor-pointer list-none items-center gap-1.5 rounded-md px-1.5 py-0.5 text-muted-foreground transition-colors hover:bg-secondary/40 hover:text-foreground [&::-webkit-details-marker]:hidden"
                title={tc.error || tc.summary || label}
              >
                {stateIcon(tc)}
                <span className={tc.state === 'running' ? 'text-foreground' : ''}>{text}</span>
                {tc.summary && tc.state !== 'error' && (
                  <span className="min-w-0 truncate text-muted-foreground/80">· {tc.summary}</span>
                )}
                {tc.state === 'error' && tc.error && (
                  <span className="min-w-0 truncate text-destructive/70">· {tc.error}</span>
                )}
                {tc.durationMs !== undefined && tc.state !== 'running' && (
                  <span className="ml-auto shrink-0 tabular-nums text-muted-foreground/60">{formatDuration(tc.durationMs)}</span>
                )}
                {hasInput && (
                  <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground/50 transition-transform group-open/trail:rotate-90" />
                )}
              </summary>
              {hasInput && (
                <pre className="mt-1 max-h-40 overflow-auto rounded-md bg-secondary/30 px-2 py-1.5 text-[11px] text-muted-foreground">
                  {JSON.stringify(tc.input, null, 2)}
                </pre>
              )}
            </details>
          </li>
        )
      })}
      {progress.map((line, index) => (
        <li key={`progress-${index}`} className="flex items-center gap-1.5 px-1.5 py-0.5 text-muted-foreground/80" data-testid="progress-line">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary/60" aria-hidden />
          <span className="truncate">{line}</span>
        </li>
      ))}
    </ol>
  )
}

export function LimitReachedNote({ limit }: { limit: LimitReached }) {
  return (
    <p className="rounded-md border border-warning/30 bg-warning/10 px-2 py-1 text-xs text-warning" role="status">
      {limit.message}
    </p>
  )
}
