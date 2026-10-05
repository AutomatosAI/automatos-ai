'use client'

/**
 * The pieces of the chat's Document widget that need no widget state: the body (JSON
 * pretty-printed, anything else as markdown), the info bar and the markdown download.
 * Moved out of DocumentWidget unchanged (F358) so the component stays within its size limit.
 */
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { DocumentWidgetData } from '../types'
import { DOCUMENT_MARKDOWN_COMPONENTS } from './markdown-components'

const JSON_PRE_CLASS =
  'rounded-md bg-[#161b22] border border-[#30363d] p-4 overflow-x-auto text-[13px] font-mono text-[#e6edf3]'

function isJsonDocument(data: DocumentWidgetData): boolean {
  if ((data.format as string) === 'json') return true
  if (data.filename?.endsWith('.json')) return true
  const trimmed = data.content?.trim() || ''
  return (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']'))
}

function prettyJson(content: string): string {
  try {
    return JSON.stringify(JSON.parse(content), null, 2)
  } catch {
    return content
  }
}

export function DocumentContent({ data }: { data: DocumentWidgetData }) {
  if (isJsonDocument(data)) {
    return (
      <pre className={JSON_PRE_CLASS}>
        <code>{prettyJson(data.content)}</code>
      </pre>
    )
  }
  return (
    <article className="prose prose-sm prose-invert max-w-none break-words [overflow-wrap:anywhere]">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={DOCUMENT_MARKDOWN_COMPONENTS}>
        {data.content}
      </ReactMarkdown>
    </article>
  )
}

/** Similarity or relevance as a whole percentage (a 0–1 score is scaled up); null if neither. */
export function relevancePercent(data: DocumentWidgetData): number | null {
  const score = data.similarity ?? data.relevance
  if (score === undefined) return null
  return Math.round(score < 1 ? score * 100 : score)
}

export function DocumentInfoBar({ data }: { data: DocumentWidgetData }) {
  const percent = relevancePercent(data)
  return (
    <div className="flex items-center flex-wrap gap-2 px-3 py-2 bg-[#252525] border-b border-[#3a3a3a]">
      {percent !== null && (
        <Badge
          variant="secondary"
          className={cn(
            'text-xs',
            percent >= 80 && 'bg-success/10 text-success',
            percent >= 50 && percent < 80 && 'bg-warning/10 text-warning',
            percent < 50 && 'bg-destructive/10 text-destructive'
          )}
        >
          {percent}% match
        </Badge>
      )}
      {data.chunkCount !== undefined && (
        <Badge variant="outline" className="text-xs">
          {data.chunkCount} chunks
        </Badge>
      )}
      {data.filename && (
        <span className="text-xs text-muted-foreground font-mono truncate max-w-[200px]">
          {data.filename}
        </span>
      )}
    </div>
  )
}

/** Save text the widget already holds (a knowledge document with no file route) as markdown. */
export function downloadMarkdownFile(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}
