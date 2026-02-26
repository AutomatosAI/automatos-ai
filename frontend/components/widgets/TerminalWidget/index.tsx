'use client'

/**
 * TerminalWidget — PRD-38.2 Extended Widgets
 *
 * Two modes:
 * 1. Read-only: Displays command output with ANSI colors (existing behavior)
 * 2. Interactive: VS Code-style terminal with command input + history
 *    (activated when data.interactive=true and data.workspaceId is set)
 */

import { useState, useCallback } from 'react'
import { Terminal, Copy, Check, RotateCcw } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { TerminalHeader } from './TerminalHeader'
import { TerminalOutput } from './TerminalOutput'
import { InteractiveTerminal } from './InteractiveTerminal'
import type {
  WidgetBaseProps,
  TerminalWidgetData,
  WidgetDefinition,
} from '../types'
import { toast } from 'sonner'

// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?(?:\x07|\x1b\\)/g
function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, '')
}

export function TerminalWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
}: WidgetBaseProps<TerminalWidgetData>) {
  const [copied, setCopied] = useState(false)
  const isInteractive = data.interactive && data.workspaceId

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(stripAnsi(data.output || ''))
      setCopied(true)
      toast.success('Output copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Failed to copy output')
    }
  }, [data.output])

  const handleRerun = useCallback(() => {
    toast.info('Re-run functionality requires backend integration')
    onRefresh?.()
  }, [onRefresh])

  const lineCount = data.output ? data.output.split('\n').length : 0
  const displayTitle = title || (isInteractive ? 'Terminal' : `$ ${data.command?.split(' ')[0] || 'terminal'}`)

  return (
    <WidgetBase
      title={displayTitle}
      icon={<Terminal className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading || data.isStreaming}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onCopy={!isInteractive ? handleCopy : undefined}
      canCopy={!isInteractive}
      canMaximize
      widgetId={id}
      widgetType="terminal"
      contentClassName="p-0"
      customActions={isInteractive ? [] : [
        {
          label: copied ? 'Copied!' : 'Copy Output',
          icon: copied ? (
            <Check className="h-4 w-4 mr-2 text-green-500" />
          ) : (
            <Copy className="h-4 w-4 mr-2" />
          ),
          onClick: handleCopy,
        },
        {
          label: 'Re-run',
          icon: <RotateCcw className="h-4 w-4 mr-2" />,
          onClick: handleRerun,
        },
      ]}
    >
      {isInteractive ? (
        <InteractiveTerminal
          workspaceId={data.workspaceId!}
          className="h-full"
        />
      ) : (
        <div className="flex flex-col h-full bg-gray-900 rounded-b-lg overflow-hidden">
          <TerminalHeader
            command={data.command}
            workingDirectory={data.workingDirectory}
            exitCode={data.exitCode}
            executionTime={data.executionTime}
            isStreaming={data.isStreaming}
          />
          <TerminalOutput
            output={data.output}
            isStreaming={data.isStreaming}
            className="flex-1"
          />
          <div className="flex items-center justify-between px-3 py-1.5 border-t border-gray-700 bg-gray-800 text-xs text-gray-400">
            <span>{lineCount} lines</span>
            <span className="font-mono">Ctrl+F to search</span>
          </div>
        </div>
      )}
    </WidgetBase>
  )
}

export const TerminalWidgetDef: WidgetDefinition<TerminalWidgetData> = {
  type: 'terminal',
  displayName: 'Terminal',
  description: 'Interactive terminal or command output with ANSI support',
  icon: Terminal,
  component: TerminalWidget,
  defaultSize: { width: 6, height: 4 },
  minSize: { width: 4, height: 3 },
  capabilities: ['copyable', 'refreshable', 'fullscreen'],
}

registerWidget(TerminalWidgetDef)
