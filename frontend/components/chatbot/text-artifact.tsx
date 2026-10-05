'use client'

import { useMemo } from 'react'
import { toast } from 'sonner'
import { Download, Copy, Loader2 } from 'lucide-react'
import type { PandasAIChart } from '@/types/chat'
import { MarkdownView } from '@/components/shared/markdown-view'
import { downloadFilename, useApiFileDownload } from '@/hooks/use-api-file-download'
import { PandasAICharts } from './pandas-ai-charts'

export interface TextArtifactProps {
  content: string
  metadata?: Record<string, any>
}

interface PandasAIInsight {
  summary?: string
  charts?: PandasAIChart[]
  error?: string
}

/**
 * The file behind an API route (F358): an <a href> to the relative path went to the
 * frontend's own origin without the Authorization header, so it fetches with auth.
 */
function DownloadChip({ url, filename }: { url: string; filename?: string }) {
  const { download, downloading } = useApiFileDownload()
  return (
    <button
      type="button"
      onClick={() => download(url, downloadFilename(url, filename))}
      disabled={downloading}
      className="inline-flex items-center gap-1 rounded-full border border-info/30 bg-info/10 px-3 py-1 text-xs font-semibold uppercase text-info/70 hover:bg-info/20"
    >
      {downloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
      Download
    </button>
  )
}

export function TextArtifact({ content, metadata }: TextArtifactProps) {
  const pandasAI = (metadata?.pandas_ai ?? null) as PandasAIInsight | null
  const chunks = Array.isArray(metadata?.chunks) ? (metadata?.chunks as Array<{ content: string; excerpt?: string }>) : null
  const downloadUrl = metadata?.download_url as string | undefined

  /** Sandbox links are runtime artifacts, not destinations — show the label only. */
  const artifactLinkComponents = {
    a: ({ href, children, ...props }: any) =>
      href?.startsWith('sandbox://') ? (
        <span className="inline-flex items-center text-primary/80">
          {children ?? href.replace('sandbox://', '')}
        </span>
      ) : (
        <a {...props} href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      ),
  }

  const renderMarkdown = (markdown: string) => (
    <MarkdownView components={artifactLinkComponents}>{markdown}</MarkdownView>
  )

  return (
    <div className="space-y-6">
      {metadata && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            {metadata.database && (
              <span className="inline-flex items-center rounded-full border border-primary/40 bg-primary/10 px-3 py-1 text-xs font-semibold uppercase text-primary">
                {metadata.database}
              </span>
            )}
            {metadata.model && (
              <span className="inline-flex items-center rounded-full border border-info/40 bg-info/10 px-3 py-1 text-xs font-semibold uppercase text-info dark:text-info/70">
                {metadata.model}
              </span>
            )}
            {downloadUrl && <DownloadChip url={downloadUrl} filename={metadata.filename} />}
          </div>

          <div className="grid grid-cols-1 gap-3 text-sm text-muted-foreground md:grid-cols-2">
            {metadata.row_count !== undefined && (
              <div className="rounded-xl border border-border/60 bg-card/50 p-3 dark:border-gray-800/60 dark:bg-background/40">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">Rows</div>
                <div className="text-lg font-semibold text-foreground dark:text-gray-100">{metadata.row_count}</div>
              </div>
            )}
            {metadata.execution_time_ms !== undefined && (
              <div className="rounded-xl border border-border/60 bg-card/50 p-3 dark:border-gray-800/60 dark:bg-background/40">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">Execution Time</div>
                <div className="text-lg font-semibold text-foreground dark:text-gray-100">{Number(metadata.execution_time_ms).toFixed(0)} ms</div>
              </div>
            )}
            {metadata.similarity !== undefined && (
              <div className="rounded-xl border border-border/60 bg-card/50 p-3 dark:border-gray-800/60 dark:bg-background/40">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">Similarity</div>
                <div className="text-lg font-semibold text-foreground dark:text-gray-100">{(metadata.similarity * 100).toFixed(1)}%</div>
              </div>
            )}
            {metadata.document_id && (
              <div className="rounded-xl border border-border/60 bg-card/50 p-3 dark:border-gray-800/60 dark:bg-background/40">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">Document</div>
                <div className="text-base font-semibold text-foreground dark:text-gray-100">{metadata.document_id}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* RAG chunk inspector (when provided) */}
      {chunks && chunks.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-foreground/80 dark:text-foreground/90 uppercase tracking-wide">
            Relevant Chunks ({chunks.length})
          </h4>
          <div className="space-y-2">
            {chunks.map((chunk, idx) => (
              <details
                key={idx}
                className="rounded-xl border border-info/20 bg-info/5 p-4"
              >
                <summary className="cursor-pointer text-sm font-medium text-foreground dark:text-gray-200">
                  Chunk {idx + 1}: {chunk.excerpt ? chunk.excerpt.slice(0, 120) : 'Open'}
                  {chunk.excerpt && chunk.excerpt.length > 120 ? '…' : ''}
                </summary>
                <div className="mt-3 space-y-3">
                  <div className="flex items-center justify-end">
                    <button
                      className="inline-flex items-center gap-2 rounded border border-border/60 px-2 py-1 text-[11px] uppercase tracking-wide text-muted-foreground hover:border-primary/60 hover:text-primary/80"
                      onClick={async () => {
                        if (!navigator.clipboard) {
                          toast.error('Clipboard API is not available')
                          return
                        }
                        try {
                          await navigator.clipboard.writeText(chunk.content || '')
                          toast.success('Chunk copied')
                        } catch (error) {
                          toast.error('Failed to copy chunk')
                        }
                      }}
                      type="button"
                    >
                      <Copy className="h-3.5 w-3.5" />
                      Copy chunk
                    </button>
                  </div>
                  <pre className="rounded-lg bg-muted/40 p-4 text-xs overflow-x-auto border border-border/60 whitespace-pre-wrap text-foreground dark:bg-background/70 dark:border-gray-800/60 dark:text-gray-100">
                    {chunk.content}
                  </pre>
                </div>
              </details>
            ))}
          </div>
        </div>
      )}

      {content && renderMarkdown(content)}

      {pandasAI?.charts && pandasAI.charts.length > 0 && <PandasAICharts charts={pandasAI.charts} />}

      {pandasAI?.error && (
        <div className="text-sm text-destructive">
          PandasAI warning: {pandasAI.error}
        </div>
      )}
    </div>
  )
}

