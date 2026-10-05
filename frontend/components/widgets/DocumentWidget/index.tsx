'use client'

/**
 * DocumentWidget Component for PRD-38.1 Widget Architecture
 *
 * Displays RAG results, markdown documents, and text content
 * with chunk inspection. Migrated from text-artifact.tsx.
 */

import { useState, useCallback } from 'react'
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
import { downloadFilename, useApiFileDownload } from '@/hooks/use-api-file-download'
import { DocumentContent, DocumentInfoBar, downloadMarkdownFile } from './document-content'

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

  // A file route (generated document, knowledge file) downloads with auth (F358);
  // a knowledge document with no route saves the text the widget holds.
  const { download } = useApiFileDownload()
  const handleDownload = useCallback(() => {
    if (data.downloadUrl) {
      void download(data.downloadUrl, downloadFilename(data.downloadUrl, data.filename || title))
      return
    }
    try {
      downloadMarkdownFile(data.content, data.filename || title || 'document.md')
      toast.success('Document downloaded')
    } catch (err) {
      console.error('[DocumentWidget] markdown download failed', err)
      toast.error('Failed to download document')
    }
  }, [data.content, data.filename, data.downloadUrl, title, download])

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
      onDownload={data.content || data.downloadUrl ? handleDownload : undefined}
      onCopy={handleCopyContent}
      canDownload={!!(data.content || data.downloadUrl)}
      canCopy
    >
      <div className="flex flex-col h-full">
        <DocumentInfoBar data={data} />

        {/* Tabs for content vs chunks */}
        {data.chunks && data.chunks.length > 0 ? (
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as 'content' | 'chunks')}
            className="flex flex-col flex-1 min-h-0"
          >
            <TabsList className="mx-3 mt-2 h-8 bg-[#252525] border border-[#3a3a3a]">
              <TabsTrigger value="content" className="text-xs h-7 px-3 data-[state=active]:bg-[#1e1e1e] data-[state=active]:text-gray-100 text-muted-foreground">
                <FileText className="h-3 w-3 mr-1.5" />
                Content
              </TabsTrigger>
              <TabsTrigger value="chunks" className="text-xs h-7 px-3 data-[state=active]:bg-[#1e1e1e] data-[state=active]:text-gray-100 text-muted-foreground">
                <List className="h-3 w-3 mr-1.5" />
                Chunks ({data.chunks.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="content" className="flex-1 m-0 min-h-0 bg-[#0d1117]">
              <ScrollArea className="h-full">
                <div className="px-6 py-4">
                  <DocumentContent data={data} />
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
              <DocumentContent data={data} />
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
          <Badge variant="outline" className="text-xs font-mono bg-[#2d2d2d] border-[#3a3a3a] text-foreground/90">
            #{index + 1}
          </Badge>
          {chunk.similarity !== undefined && (
            <span className="text-xs text-muted-foreground">
              {Math.round(chunk.similarity * 100)}% relevant
            </span>
          )}
          {chunk.page !== undefined && (
            <span className="text-xs text-muted-foreground">
              Page {chunk.page}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {onViewInDocument && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs text-info hover:text-info/80 hover:bg-[#3a3a3a]"
              onClick={() => onViewInDocument(index)}
              title="View in Document"
            >
              <Eye className="h-3 w-3" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs text-muted-foreground hover:text-gray-200 hover:bg-[#3a3a3a]"
            onClick={onCopy}
          >
            {isCopied ? (
              <Check className="h-3 w-3 text-success" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        </div>
      </div>

      <div className="text-sm text-foreground/90">
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
          className="h-6 px-2 text-xs mt-2 text-muted-foreground hover:text-gray-200 hover:bg-[#3a3a3a]"
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
