'use client'

/**
 * CodeWidget Component for PRD-38.1 Widget Architecture
 *
 * Displays code snippets with syntax highlighting, line numbers,
 * and file path information. Migrated from code-artifact.tsx.
 */

import { useCallback, useState, useEffect, useRef } from 'react'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-sql'
import 'prismjs/components/prism-yaml'
import 'prismjs/components/prism-markdown'
import 'prismjs/components/prism-css'
import 'prismjs/components/prism-go'
import 'prismjs/components/prism-rust'
import 'prismjs/components/prism-java'
import { Code2, FileCode, Copy, Check } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import type { WidgetBaseProps, CodeWidgetData, WidgetDefinition } from '../types'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

/**
 * Language mapping for Prism
 */
const LANGUAGE_MAP: Record<string, string> = {
  python: 'python',
  py: 'python',
  javascript: 'javascript',
  js: 'javascript',
  typescript: 'typescript',
  ts: 'typescript',
  json: 'json',
  bash: 'bash',
  shell: 'bash',
  sh: 'bash',
  sql: 'sql',
  html: 'html',
  css: 'css',
  yaml: 'yaml',
  yml: 'yaml',
  markdown: 'markdown',
  md: 'markdown',
  go: 'go',
  golang: 'go',
  rust: 'rust',
  rs: 'rust',
  java: 'java',
  text: 'text',
  txt: 'text',
}

export function CodeWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
}: WidgetBaseProps<CodeWidgetData>) {
  const [copied, setCopied] = useState(false)
  const codeRef = useRef<HTMLElement>(null)

  // Get normalized language
  const language = LANGUAGE_MAP[data.language?.toLowerCase() || 'text'] || 'text'

  // Apply Prism highlighting
  useEffect(() => {
    if (codeRef.current && data.code) {
      Prism.highlightElement(codeRef.current)
    }
  }, [data.code, language])

  // Copy to clipboard handler
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(data.code)
      setCopied(true)
      toast.success('Code copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      toast.error('Failed to copy code')
    }
  }, [data.code])

  // Count lines for display
  const lineCount = data.code.split('\n').length
  const showLineNumbers = lineCount > 3

  return (
    <WidgetBase
      title={title}
      icon={<Code2 className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onCopy={handleCopy}
      canCopy
    >
      <div className="flex flex-col h-full">
        {/* File info bar */}
        {(data.filePath || data.lineNumber || data.symbolName) && (
          <div className="flex items-center gap-2 px-3 py-2 bg-muted/30 border-b border-border/30 text-xs text-muted-foreground">
            <FileCode className="h-3.5 w-3.5 flex-shrink-0" />
            {data.filePath && (
              <span className="font-mono truncate">{data.filePath}</span>
            )}
            {data.lineNumber && (
              <span className="text-primary font-medium">
                :L{data.lineNumber}
              </span>
            )}
            {data.symbolName && (
              <span className="ml-2 px-1.5 py-0.5 bg-primary/10 rounded text-primary font-medium">
                {data.symbolName}
              </span>
            )}
          </div>
        )}

        {/* Code block */}
        <div className="flex-1 overflow-auto bg-[#2d2d2d]">
          <div className="flex min-h-full">
            {/* Line numbers - separate column */}
            {showLineNumbers && (
              <div className="flex-shrink-0 w-12 bg-[#252525] border-r border-[#3a3a3a] text-right pr-3 py-4 text-xs text-muted-foreground select-none font-mono">
                {Array.from({ length: lineCount }, (_, i) => (
                  <div key={i} className="leading-relaxed text-sm">
                    {i + 1}
                  </div>
                ))}
              </div>
            )}
            {/* Code content */}
            <pre className="flex-1 p-4 text-sm leading-relaxed m-0 overflow-x-auto">
              <code
                ref={codeRef}
                className={cn(
                  `language-${language}`,
                  'font-mono text-sm'
                )}
              >
                {data.code}
              </code>
            </pre>
          </div>
        </div>

        {/* Explanation */}
        {data.explanation && (
          <div className="px-3 py-2 bg-muted/20 border-t border-border/30">
            <p className="text-sm text-muted-foreground">
              <span className="font-medium text-foreground">Explanation:</span>{' '}
              {data.explanation}
            </p>
          </div>
        )}

        {/* Footer with stats */}
        <div className="flex items-center justify-between px-3 py-1.5 border-t border-border/30 bg-muted/20 text-xs text-muted-foreground">
          <span>{lineCount} lines</span>
          <span className="font-mono uppercase">{data.language || 'text'}</span>
        </div>
      </div>
    </WidgetBase>
  )
}

/**
 * Widget definition for registry
 */
export const CodeWidgetDef: WidgetDefinition<CodeWidgetData> = {
  type: 'code',
  displayName: 'Code',
  description: 'Display code with syntax highlighting',
  icon: Code2,
  component: CodeWidget,
  defaultSize: { width: 6, height: 4 },
  minSize: { width: 3, height: 2 },
  capabilities: ['copyable', 'fullscreen'],
}

// Register the widget
registerWidget(CodeWidgetDef)
