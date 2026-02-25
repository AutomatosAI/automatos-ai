'use client'

/**
 * TerminalWidget Component for PRD-38.2 Extended Widgets
 *
 * Displays command execution output with ANSI color support,
 * exit code display, and execution time.
 */

import { useState, useCallback } from 'react'
import { Terminal, Copy, Check, RotateCcw, Search } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { TerminalHeader } from './TerminalHeader'
import { TerminalOutput } from './TerminalOutput'
import type {
  WidgetBaseProps,
  TerminalWidgetData,
  WidgetDefinition,
} from '../types'
import { toast } from 'sonner'

/**
 * Strip ANSI escape codes from a string for plain-text clipboard copy.
 */
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

  // Copy output to clipboard (ANSI codes stripped for plain text)
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(stripAnsi(data.output))
      setCopied(true)
      toast.success('Output copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      toast.error('Failed to copy output')
    }
  }, [data.output])

  // Re-run command (placeholder - would need backend integration)
  const handleRerun = useCallback(() => {
    toast.info('Re-run functionality requires backend integration')
    onRefresh?.()
  }, [onRefresh])

  // Calculate line count
  const lineCount = data.output ? data.output.split('\n').length : 0

  // Determine widget title
  const displayTitle = title || `$ ${data.command.split(' ')[0]}`

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
      onCopy={handleCopy}
      canCopy
      customActions={[
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
      <div className="flex flex-col h-full bg-gray-900 rounded-b-lg overflow-hidden">
        {/* Command header */}
        <TerminalHeader
          command={data.command}
          workingDirectory={data.workingDirectory}
          exitCode={data.exitCode}
          executionTime={data.executionTime}
          isStreaming={data.isStreaming}
        />

        {/* Output area */}
        <TerminalOutput
          output={data.output}
          isStreaming={data.isStreaming}
          className="flex-1"
        />

        {/* Footer with stats */}
        <div className="flex items-center justify-between px-3 py-1.5 border-t border-gray-700 bg-gray-800 text-xs text-gray-400">
          <span>{lineCount} lines</span>
          <div className="flex items-center gap-2">
            <span className="font-mono">Ctrl+F to search</span>
          </div>
        </div>
      </div>
    </WidgetBase>
  )
}

/**
 * Widget definition for registry
 */
export const TerminalWidgetDef: WidgetDefinition<TerminalWidgetData> = {
  type: 'terminal',
  displayName: 'Terminal',
  description: 'Display command execution output with ANSI support',
  icon: Terminal,
  component: TerminalWidget,
  defaultSize: { width: 6, height: 4 },
  minSize: { width: 4, height: 3 },
  capabilities: ['copyable', 'refreshable', 'fullscreen'],
}

// Register the widget
registerWidget(TerminalWidgetDef)
