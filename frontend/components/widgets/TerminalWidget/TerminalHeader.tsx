'use client'

/**
 * TerminalHeader Component for PRD-38.2 Extended Widgets
 *
 * Displays command info, exit code, and execution time
 */

import { Terminal, FolderOpen, Clock, CheckCircle2, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface TerminalHeaderProps {
  command: string
  workingDirectory?: string
  exitCode?: number
  executionTime?: number
  isStreaming?: boolean
}

export function TerminalHeader({
  command,
  workingDirectory,
  exitCode,
  executionTime,
  isStreaming,
}: TerminalHeaderProps) {
  // Determine success/error state
  const isSuccess = exitCode === 0
  const isError = exitCode !== undefined && exitCode !== 0

  // Format execution time
  const formatTime = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    const mins = Math.floor(ms / 60000)
    const secs = ((ms % 60000) / 1000).toFixed(1)
    return `${mins}m ${secs}s`
  }

  return (
    <div className="flex flex-col gap-1 px-3 py-2 bg-background border-b border-border text-sm">
      {/* Command line */}
      <div className="flex items-center gap-2">
        <span className="text-success font-mono">$</span>
        <code className="text-gray-200 font-mono flex-1 truncate">
          {command}
        </code>
      </div>

      {/* Info row */}
      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        {/* Working directory */}
        {workingDirectory && (
          <div className="flex items-center gap-1">
            <FolderOpen className="h-3 w-3" />
            <span className="font-mono truncate max-w-[200px]">
              {workingDirectory}
            </span>
          </div>
        )}

        {/* Exit code badge */}
        {exitCode !== undefined && (
          <span
            className={cn(
              'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
              isSuccess && 'bg-green-900/60 text-green-300',
              isError && 'bg-destructive/60 text-destructive/80'
            )}
          >
            {isSuccess ? (
              <CheckCircle2 className="h-3 w-3" />
            ) : (
              <XCircle className="h-3 w-3" />
            )}
            exit {exitCode}
          </span>
        )}

        {/* Streaming indicator */}
        {isStreaming && (
          <div className="flex items-center gap-1 text-warning">
            <div className="w-2 h-2 rounded-full bg-warning animate-pulse" />
            <span>Running...</span>
          </div>
        )}

        {/* Execution time */}
        {executionTime !== undefined && !isStreaming && (
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            <span>{formatTime(executionTime)}</span>
          </div>
        )}
      </div>
    </div>
  )
}
