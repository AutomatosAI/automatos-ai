'use client'

/**
 * FilePreview Component for PRD-38.2 Extended Widgets
 *
 * Content preview for different file types
 */

import { useMemo } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { FileQuestion, ImageIcon, FileCode, FileText } from 'lucide-react'
import { cn } from '@/lib/utils'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'

interface FilePreviewProps {
  content?: string
  previewType?: 'text' | 'image' | 'pdf' | 'code' | 'binary'
  filename?: string
  className?: string
}

/**
 * Detect language from filename
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

export function FilePreview({
  content,
  previewType,
  filename,
  className,
}: FilePreviewProps) {
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

  // Image preview
  if (previewType === 'image') {
    return (
      <div className={cn('flex items-center justify-center h-full p-4', className)}>
        <img
          src={content}
          alt={filename || 'Image preview'}
          className="max-w-full max-h-full object-contain rounded-md"
        />
      </div>
    )
  }

  // Code preview with syntax highlighting
  if (previewType === 'code' && filename) {
    const language = getLanguageFromFilename(filename)

    return (
      <ScrollArea className={cn('h-full', className)}>
        <div className="bg-[#2d2d2d] p-4 min-h-full">
          <pre className="text-sm leading-relaxed">
            <code className={`language-${language}`}>{content}</code>
          </pre>
        </div>
      </ScrollArea>
    )
  }

  // Default text preview
  return (
    <ScrollArea className={cn('h-full', className)}>
      <pre className="p-4 text-sm font-mono whitespace-pre-wrap">{content}</pre>
    </ScrollArea>
  )
}
