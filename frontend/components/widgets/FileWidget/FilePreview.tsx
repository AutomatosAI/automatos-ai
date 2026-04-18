'use client'

/**
 * FilePreview Component for PRD-38.2 Extended Widgets
 *
 * Content preview for different file types:
 * - text/code: syntax-highlighted pre/code block with font-mono styling
 * - image: img tag with constrained max dimensions
 * - binary: not-available message
 */

import { useMemo, useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { FileQuestion } from 'lucide-react'
import { cn } from '@/lib/utils'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'

interface FilePreviewProps {
  content?: string
  previewType?: 'text' | 'image' | 'pdf' | 'code' | 'binary' | 'html'
  filename?: string
  className?: string
}

/**
 * Detect language from filename extension
 */
function getLanguageFromFilename(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() || ''
  const langMap: Record<string, string> = {
    js: 'javascript',
    jsx: 'javascript',
    ts: 'typescript',
    tsx: 'typescript',
    py: 'python',
    rb: 'ruby',
    go: 'go',
    rs: 'rust',
    java: 'java',
    c: 'c',
    cpp: 'cpp',
    h: 'c',
    cs: 'csharp',
    php: 'php',
    swift: 'swift',
    kt: 'kotlin',
    sql: 'sql',
    sh: 'bash',
    bash: 'bash',
    zsh: 'bash',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
    xml: 'xml',
    html: 'html',
    htm: 'html',
    css: 'css',
    scss: 'scss',
    less: 'less',
    md: 'markdown',
    markdown: 'markdown',
  }
  return langMap[ext] || 'text'
}

/**
 * Highlight source code using Prism.js.
 * Returns highlighted HTML string or null if grammar not found.
 * Note: Prism.highlight encodes special characters during tokenization,
 * so the output is safe for innerHTML rendering.
 */
function highlightCode(code: string, language: string): string | null {
  const grammar = Prism.languages[language]
  if (!grammar) return null
  return Prism.highlight(code, grammar, language)
}

export function FilePreview({
  content,
  previewType,
  filename,
  className,
}: FilePreviewProps) {
  // HTML preview vs source toggle
  const [htmlView, setHtmlView] = useState<'preview' | 'source'>('preview')

  // Compute syntax-highlighted HTML for code previews
  const highlightedHtml = useMemo(() => {
    if (previewType !== 'code' || !content || !filename) return null
    const language = getLanguageFromFilename(filename)
    return highlightCode(content, language)
  }, [content, previewType, filename])

  // No content
  if (!content) {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center h-full text-muted-foreground',
          className
        )}
      >
        <FileQuestion className="h-12 w-12 mb-2 opacity-50" />
        <p className="text-sm">No preview available</p>
      </div>
    )
  }

  // Binary content
  if (previewType === 'binary') {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center h-full text-muted-foreground',
          className
        )}
      >
        <FileQuestion className="h-12 w-12 mb-2 opacity-50" />
        <p className="text-sm">Binary file - preview not available</p>
      </div>
    )
  }

  // Image preview with constrained max dimensions
  if (previewType === 'image') {
    return (
      <div className={cn('flex items-center justify-center h-full p-4', className)}>
        <img
          src={content}
          alt={filename || 'Image preview'}
          style={{ maxWidth: '100%', maxHeight: '100%' }}
          className="object-contain rounded-md"
        />
      </div>
    )
  }

  // HTML preview — rendered in sandboxed iframe, with source toggle
  if (previewType === 'html') {
    return (
      <div className={cn('flex flex-col h-full', className)}>
        <div className="flex items-center gap-1 border-b bg-muted/30 px-2 py-1">
          <button
            type="button"
            onClick={() => setHtmlView('preview')}
            className={cn(
              'rounded px-2 py-1 text-xs font-medium transition-colors',
              htmlView === 'preview'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            Preview
          </button>
          <button
            type="button"
            onClick={() => setHtmlView('source')}
            className={cn(
              'rounded px-2 py-1 text-xs font-medium transition-colors',
              htmlView === 'source'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            Source
          </button>
        </div>
        {htmlView === 'preview' ? (
          <iframe
            // Sandboxed: allow-scripts only. No allow-same-origin → iframe
            // cannot access parent cookies, storage, or DOM. Agent-generated
            // HTML is untrusted input.
            sandbox="allow-scripts"
            srcDoc={content}
            title={filename || 'HTML preview'}
            className="flex-1 w-full border-0 bg-white"
          />
        ) : (
          <ScrollArea className="flex-1">
            <pre className="p-4 text-sm font-mono whitespace-pre-wrap text-gray-200 bg-[#2d2d2d] min-h-full">
              <code>{content}</code>
            </pre>
          </ScrollArea>
        )}
      </div>
    )
  }

  // Code preview with syntax highlighting and font-mono styling
  if (previewType === 'code' && filename) {
    const language = getLanguageFromFilename(filename)

    return (
      <ScrollArea className={cn('h-full', className)}>
        <div className="bg-[#2d2d2d] p-4 min-h-full">
          {highlightedHtml ? (
            <pre className="text-sm font-mono leading-relaxed">
              {/* Safe: Prism.highlight encodes HTML entities during tokenization */}
              <code
                className={`language-${language}`}
                dangerouslySetInnerHTML={{ __html: highlightedHtml }}
              />
            </pre>
          ) : (
            <pre className="text-sm font-mono leading-relaxed text-gray-200">
              <code>{content}</code>
            </pre>
          )}
        </div>
      </ScrollArea>
    )
  }

  // Default text preview with font-mono
  return (
    <ScrollArea className={cn('h-full', className)}>
      <pre className="p-4 text-sm font-mono whitespace-pre-wrap">{content}</pre>
    </ScrollArea>
  )
}
