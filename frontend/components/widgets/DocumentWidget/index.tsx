'use client'

/**
 * DocumentWidget Component for PRD-38.1 Widget Architecture
 *
 * Displays RAG results, markdown documents, and text content
 * with chunk inspection. Migrated from text-artifact.tsx.
 */

import { useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  FileText,
  Download,
  List,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  ExternalLink,
  Eye,
} from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import type { WidgetBaseProps, DocumentWidgetData, WidgetDefinition } from '../types'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

export function DocumentWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
}: WidgetBaseProps<DocumentWidgetData>) {
  const [activeTab, setActiveTab] = useState<'content' | 'chunks'>('content')
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)
  const [highlightedChunkIndex, setHighlightedChunkIndex] = useState<number | null>(null)

  // View in Document: switch to content tab and scroll to highlighted chunk
  const handleViewInDocument = useCallback((chunkIndex: number) => {
    setHighlightedChunkIndex(chunkIndex)
    setActiveTab('content')
    // Scroll to the chunk marker after a short delay for tab switch
    setTimeout(() => {
      const el = document.querySelector(`[data-chunk-index="${chunkIndex}"]`)
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
  }, [])

  // Handle download - create blob from content since we have it in memory
  const handleDownload = useCallback(() => {
    try {
      const blob = new Blob([data.content], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = data.filename || title || 'document.md'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success('Document downloaded')
    } catch (error) {
      toast.error('Failed to download document')
    }
  }, [data.content, data.filename, title])

  // Copy content to clipboard
  const handleCopyContent = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(data.content)
      toast.success('Content copied to clipboard')
    } catch {
      toast.error('Failed to copy content')
    }
  }, [data.content])

  // Copy chunk to clipboard
  const handleCopyChunk = useCallback(async (chunkContent: string, index: number) => {
    try {
      await navigator.clipboard.writeText(chunkContent)
      setCopiedIndex(index)
      toast.success('Chunk copied to clipboard')
      setTimeout(() => setCopiedIndex(null), 2000)
    } catch {
      toast.error('Failed to copy chunk')
    }
  }, [])

  // Calculate relevance percentage
  const relevancePercent = data.similarity !== undefined
    ? Math.round((data.similarity < 1 ? data.similarity * 100 : data.similarity))
    : data.relevance !== undefined
    ? Math.round((data.relevance < 1 ? data.relevance * 100 : data.relevance))
    : null

  // Detect if content is JSON
  const isJsonContent = (() => {
    if (data.format === 'json') return true
    if (data.filename?.endsWith('.json')) return true
    const trimmed = data.content?.trim() || ''
    return (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
           (trimmed.startsWith('[') && trimmed.endsWith(']'))
  })()

  // Format JSON nicely
  const formattedContent = (() => {
    if (!isJsonContent) return data.content
    try {
      const parsed = JSON.parse(data.content)
      return JSON.stringify(parsed, null, 2)
    } catch {
      return data.content
    }
  })()

  // GitHub-style markdown components
  const markdownComponents = {
    // Links
    a: ({ href, children, ...props }: any) => (
      <a
        {...props}
        href={href}
        target="_blank"
        rel="noreferrer"
        className="text-[#58a6ff] hover:underline"
      >
        {children}
      </a>
    ),
    // Inline code
    code: ({ inline, className, children, ...props }: any) => {
      if (inline) {
        return (
          <code className="rounded-md bg-[#343942] px-1.5 py-0.5 text-[13px] font-mono text-[#e6edf3]" {...props}>
            {children}
          </code>
        )
      }
      // Code blocks (inside pre)
      const match = /language-(\w+)/.exec(className || '')
      const lang = match?.[1] || ''
      return (
        <code className={cn("text-[13px] font-mono text-[#e6edf3]", className)} {...props}>
          {children}
        </code>
      )
    },
    // Code blocks wrapper
    pre: ({ children }: any) => (
      <pre className="rounded-md bg-[#161b22] border border-[#30363d] p-4 overflow-x-auto my-4">
        {children}
      </pre>
    ),
    // Headings - GitHub style with bottom border
    h1: ({ children }: any) => (
      <h1 className="text-[2em] font-semibold text-[#e6edf3] border-b border-[#30363d] pb-2 mt-6 mb-4">{children}</h1>
    ),
    h2: ({ children }: any) => (
      <h2 className="text-[1.5em] font-semibold text-[#e6edf3] border-b border-[#30363d] pb-2 mt-6 mb-4">{children}</h2>
    ),
    h3: ({ children }: any) => (
      <h3 className="text-[1.25em] font-semibold text-[#e6edf3] mt-6 mb-4">{children}</h3>
    ),
    h4: ({ children }: any) => (
      <h4 className="text-[1em] font-semibold text-[#e6edf3] mt-6 mb-4">{children}</h4>
    ),
    h5: ({ children }: any) => (
      <h5 className="text-[0.875em] font-semibold text-[#e6edf3] mt-6 mb-4">{children}</h5>
    ),
    h6: ({ children }: any) => (
      <h6 className="text-[0.85em] font-semibold text-[#8b949e] mt-6 mb-4">{children}</h6>
    ),
    // Paragraphs
    p: ({ children }: any) => (
      <p className="text-[#e6edf3] leading-[1.6] mb-4">{children}</p>
    ),
    // Lists
    ul: ({ children }: any) => (
      <ul className="list-disc pl-8 mb-4 space-y-1 text-[#e6edf3]">{children}</ul>
    ),
    ol: ({ children }: any) => (
      <ol className="list-decimal pl-8 mb-4 space-y-1 text-[#e6edf3]">{children}</ol>
    ),
    li: ({ children }: any) => (
      <li className="text-[#e6edf3] leading-[1.6]">{children}</li>
    ),
    // Blockquote - GitHub style with left border
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-[#30363d] pl-4 my-4 text-[#8b949e]">{children}</blockquote>
    ),
    // Horizontal rule
    hr: () => (
      <hr className="border-t border-[#30363d] my-6" />
    ),
    // Tables - GitHub style
    table: ({ children }: any) => (
      <div className="overflow-x-auto my-4">
        <table className="min-w-full border-collapse border border-[#30363d] text-sm">{children}</table>
      </div>
    ),
    thead: ({ children }: any) => (
      <thead className="bg-[#161b22]">{children}</thead>
    ),
    tbody: ({ children }: any) => (
      <tbody className="divide-y divide-[#30363d]">{children}</tbody>
    ),
    tr: ({ children }: any) => (
      <tr className="even:bg-[#161b22]/50">{children}</tr>
    ),
    th: ({ children }: any) => (
      <th className="px-4 py-3 text-left text-[#e6edf3] font-semibold border border-[#30363d]">{children}</th>
    ),
    td: ({ children }: any) => (
      <td className="px-4 py-3 text-[#e6edf3] border border-[#30363d]">{children}</td>
    ),
    // Strong/bold
    strong: ({ children }: any) => (
      <strong className="font-semibold text-[#e6edf3]">{children}</strong>
    ),
    // Emphasis/italic
    em: ({ children }: any) => (
      <em className="italic text-[#e6edf3]">{children}</em>
    ),
    // Images
    img: ({ src, alt, ...props }: any) => (
      <img src={src} alt={alt} className="max-w-full rounded-md border border-[#30363d] my-4" {...props} />
    ),
  }

  return (
    <WidgetBase
      title={title}
      icon={<FileText className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onDownload={data.content ? handleDownload : undefined}
      onCopy={handleCopyContent}
      canDownload={!!data.content}
      canCopy
    >
      <div className="flex flex-col h-full">
        {/* Info bar */}
        <div className="flex items-center flex-wrap gap-2 px-3 py-2 bg-[#252525] border-b border-[#3a3a3a]">
          {relevancePercent !== null && (
            <Badge
              variant="secondary"
              className={cn(
                'text-xs',
                relevancePercent >= 80 && 'bg-green-500/10 text-green-600',
                relevancePercent >= 50 && relevancePercent < 80 && 'bg-yellow-500/10 text-yellow-600',
                relevancePercent < 50 && 'bg-red-500/10 text-red-600'
              )}
            >
              {relevancePercent}% match
            </Badge>
          )}
          {data.chunkCount !== undefined && (
            <Badge variant="outline" className="text-xs">
              {data.chunkCount} chunks
            </Badge>
          )}
          {data.filename && (
            <span className="text-xs text-gray-400 font-mono truncate max-w-[200px]">
              {data.filename}
            </span>
          )}
        </div>

        {/* Tabs for content vs chunks */}
        {data.chunks && data.chunks.length > 0 ? (
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as 'content' | 'chunks')}
            className="flex flex-col flex-1 min-h-0"
          >
            <TabsList className="mx-3 mt-2 h-8 bg-[#252525] border border-[#3a3a3a]">
              <TabsTrigger value="content" className="text-xs h-7 px-3 data-[state=active]:bg-[#1e1e1e] data-[state=active]:text-gray-100 text-gray-400">
                <FileText className="h-3 w-3 mr-1.5" />
                Content
              </TabsTrigger>
              <TabsTrigger value="chunks" className="text-xs h-7 px-3 data-[state=active]:bg-[#1e1e1e] data-[state=active]:text-gray-100 text-gray-400">
                <List className="h-3 w-3 mr-1.5" />
                Chunks ({data.chunks.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="content" className="flex-1 m-0 min-h-0 bg-[#0d1117]">
              <ScrollArea className="h-full">
                <div className="px-6 py-4">
                  {isJsonContent ? (
                    <pre className="rounded-md bg-[#161b22] border border-[#30363d] p-4 overflow-x-auto text-[13px] font-mono text-[#e6edf3]">
                      <code>{formattedContent}</code>
                    </pre>
                  ) : (
                    <article className="prose prose-sm prose-invert max-w-none break-words [overflow-wrap:anywhere]">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={markdownComponents}
                      >
                        {data.content}
                      </ReactMarkdown>
                    </article>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="chunks" className="flex-1 m-0 min-h-0 bg-[#0d1117]">
              <ScrollArea className="h-full">
                <div className="divide-y divide-[#3a3a3a]">
                  {data.chunks.map((chunk, i) => (
                    <ChunkItem
                      key={i}
                      index={i}
                      chunk={chunk}
                      onCopy={() => handleCopyChunk(chunk.content, i)}
                      onViewInDocument={handleViewInDocument}
                      isCopied={copiedIndex === i}
                    />
                  ))}
                </div>
              </ScrollArea>
            </TabsContent>
          </Tabs>
        ) : (
          <ScrollArea className="flex-1 bg-[#0d1117]">
            <div className="px-6 py-4">
              {isJsonContent ? (
                <pre className="rounded-md bg-[#161b22] border border-[#30363d] p-4 overflow-x-auto text-[13px] font-mono text-[#e6edf3]">
                  <code>{formattedContent}</code>
                </pre>
              ) : (
                <article className="prose prose-sm prose-invert max-w-none break-words [overflow-wrap:anywhere]">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {data.content}
                  </ReactMarkdown>
                </article>
              )}
            </div>
          </ScrollArea>
        )}
      </div>
    </WidgetBase>
  )
}

/**
 * Chunk item component
 */
interface ChunkItemProps {
  index: number
  chunk: {
    content: string
    excerpt?: string
    similarity?: number
    document_id?: number
    page?: number
  }
  onCopy: () => void
  onViewInDocument?: (index: number) => void
  isCopied: boolean
}

function ChunkItem({ index, chunk, onCopy, onViewInDocument, isCopied }: ChunkItemProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const excerpt = chunk.excerpt || chunk.content.slice(0, 150)
  const isLong = chunk.content.length > 150

  return (
    <div className="p-3 hover:bg-[#2a2a2a] transition-colors">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs font-mono bg-[#2d2d2d] border-[#3a3a3a] text-gray-300">
            #{index + 1}
          </Badge>
          {chunk.similarity !== undefined && (
            <span className="text-xs text-gray-500">
              {Math.round(chunk.similarity * 100)}% relevant
            </span>
          )}
          {chunk.page !== undefined && (
            <span className="text-xs text-gray-500">
              Page {chunk.page}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {onViewInDocument && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs text-blue-400 hover:text-blue-300 hover:bg-[#3a3a3a]"
              onClick={() => onViewInDocument(index)}
              title="View in Document"
            >
              <Eye className="h-3 w-3" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs text-gray-400 hover:text-gray-200 hover:bg-[#3a3a3a]"
            onClick={onCopy}
          >
            {isCopied ? (
              <Check className="h-3 w-3 text-green-500" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        </div>
      </div>

      <div className="text-sm text-gray-300">
        {isExpanded || !isLong ? (
          <p className="whitespace-pre-wrap">{chunk.content}</p>
        ) : (
          <p className="whitespace-pre-wrap">{excerpt}...</p>
        )}
      </div>

      {isLong && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs mt-2 text-gray-400 hover:text-gray-200 hover:bg-[#3a3a3a]"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              Show less
            </>
          ) : (
            <>
              <ChevronRight className="h-3 w-3 mr-1" />
              Show more
            </>
          )}
        </Button>
      )}
    </div>
  )
}

/**
 * Widget definition for registry
 */
export const DocumentWidgetDef: WidgetDefinition<DocumentWidgetData> = {
  type: 'document',
  displayName: 'Document',
  description: 'Display RAG results and markdown documents',
  icon: FileText,
  component: DocumentWidget,
  defaultSize: { width: 6, height: 5 },
  minSize: { width: 3, height: 2 },
  capabilities: ['downloadable', 'fullscreen', 'copyable'],
}

// Register the widget
registerWidget(DocumentWidgetDef)
