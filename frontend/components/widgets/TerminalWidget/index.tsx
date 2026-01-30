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

  // Copy output to clipboard
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(data.output)
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
      <div className="flex flex-col h-full bg-[#1a1a1a] rounded-b-lg overflow-hidden">
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
        <div className="flex items-center justify-between px-3 py-1.5 border-t border-[#333] bg-[#252525] text-xs text-gray-400">
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
