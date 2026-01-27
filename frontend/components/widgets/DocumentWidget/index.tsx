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
  ExternalLink,
  Copy,
  Check,
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

  // Handle download
  const handleDownload = useCallback(() => {
    if (data.downloadUrl) {
      window.open(data.downloadUrl, '_blank')
    }
  }, [data.downloadUrl])

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
      onDownload={data.downloadUrl ? handleDownload : undefined}
      onCopy={handleCopyContent}
      canDownload={!!data.downloadUrl}
      canCopy
    >
      <div className="flex flex-col h-full">
        {/* Info bar */}
        <div className="flex items-center flex-wrap gap-2 px-3 py-2 bg-muted/30 border-b border-border/30">
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
            <span className="text-xs text-muted-foreground font-mono truncate max-w-[200px]">
              {data.filename}
            </span>
          )}
          {data.downloadUrl && (
            <a
              href={data.downloadUrl}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-primary hover:underline inline-flex items-center gap-1"
            >
              <ExternalLink className="h-3 w-3" />
              Open
            </a>
          )}
        </div>

        {/* Tabs for content vs chunks */}
        {data.chunks && data.chunks.length > 0 ? (
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as 'content' | 'chunks')}
            className="flex flex-col flex-1 min-h-0"
          >
            <TabsList className="mx-3 mt-2 h-8">
              <TabsTrigger value="content" className="text-xs h-7 px-3">
                <FileText className="h-3 w-3 mr-1.5" />
                Content
              </TabsTrigger>
              <TabsTrigger value="chunks" className="text-xs h-7 px-3">
                <List className="h-3 w-3 mr-1.5" />
                Chunks ({data.chunks.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="content" className="flex-1 m-0 min-h-0">
              <ScrollArea className="h-full">
                <div className="px-4 py-3">
                  <article className="prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: ({ href, children, ...props }) => (
                          <a
                            {...props}
                            href={href}
                            target="_blank"
                            rel="noreferrer"
                            className="text-primary hover:underline"
                          >
                            {children}
                          </a>
                        ),
                        code: ({ children }) => (
                          <code className="rounded bg-muted/50 px-1.5 py-0.5 text-xs">
                            {children}
                          </code>
                        ),
                        pre: ({ children }) => (
                          <pre className="rounded-lg bg-muted/30 p-4 text-xs overflow-x-auto border border-border/50">
                            {children}
                          </pre>
                        ),
                      }}
                    >
                      {data.content}
                    </ReactMarkdown>
                  </article>
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="chunks" className="flex-1 m-0 min-h-0">
              <ScrollArea className="h-full">
                <div className="divide-y divide-border/30">
                  {data.chunks.map((chunk, i) => (
                    <ChunkItem
                      key={i}
                      index={i}
                      chunk={chunk}
                      onCopy={() => handleCopyChunk(chunk.content, i)}
                      isCopied={copiedIndex === i}
                    />
                  ))}
                </div>
              </ScrollArea>
            </TabsContent>
          </Tabs>
        ) : (
          <ScrollArea className="flex-1">
            <div className="px-4 py-3">
              <article className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: ({ href, children, ...props }) => (
                      <a
                        {...props}
                        href={href}
                        target="_blank"
                        rel="noreferrer"
                        className="text-primary hover:underline"
                      >
                        {children}
                      </a>
                    ),
                    code: ({ children }) => (
                      <code className="rounded bg-muted/50 px-1.5 py-0.5 text-xs">
                        {children}
                      </code>
                    ),
                    pre: ({ children }) => (
                      <pre className="rounded-lg bg-muted/30 p-4 text-xs overflow-x-auto border border-border/50">
                        {children}
                      </pre>
                    ),
                  }}
                >
                  {data.content}
                </ReactMarkdown>
              </article>
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
  }
  onCopy: () => void
  isCopied: boolean
}

function ChunkItem({ index, chunk, onCopy, isCopied }: ChunkItemProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const excerpt = chunk.excerpt || chunk.content.slice(0, 150)
  const isLong = chunk.content.length > 150

  return (
    <div className="p-3 hover:bg-muted/20 transition-colors">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs font-mono">
            #{index + 1}
          </Badge>
          {chunk.similarity !== undefined && (
            <span className="text-xs text-muted-foreground">
              {Math.round(chunk.similarity * 100)}% relevant
            </span>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs"
          onClick={onCopy}
        >
          {isCopied ? (
            <Check className="h-3 w-3 text-green-500" />
          ) : (
            <Copy className="h-3 w-3" />
          )}
        </Button>
      </div>

      <div className="text-sm text-foreground/80">
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
          className="h-6 px-2 text-xs mt-2"
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
