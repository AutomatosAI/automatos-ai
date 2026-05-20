'use client'

import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { toast } from 'sonner'
import { ExternalLink, Copy } from 'lucide-react'

export interface TextArtifactProps {
  content: string
  metadata?: Record<string, any>
}

interface PandasAIChart {
  filename: string
  mime_type: string
  base64: string
}

interface PandasAIInsight {
  summary?: string
  charts?: PandasAIChart[]
  error?: string
}

export function TextArtifact({ content, metadata }: TextArtifactProps) {
  const pandasAI = (metadata?.pandas_ai ?? null) as PandasAIInsight | null
  const chunks = Array.isArray(metadata?.chunks) ? (metadata?.chunks as Array<{ content: string; excerpt?: string }>) : null
  const downloadUrl = metadata?.download_url as string | undefined

  const renderMarkdown = (markdown: string) => (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      className="prose prose-sm max-w-none dark:prose-invert"
      components={{
        a: ({ href, children, ...props }) => {
          if (href?.startsWith('sandbox://')) {
            const label = children ?? href.replace('sandbox://', '')
            return (
              <span className="inline-flex items-center text-primary/80">
                {label}
              </span>
            )
          }
          return (
            <a
              {...props}
              href={href}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:text-primary/80 underline"
            >
              {children}
            </a>
          )
        },
        code: ({ children }) => (
          <code className="rounded bg-secondary/40 px-1.5 py-0.5 text-xs text-foreground dark:bg-background/60 dark:text-gray-100">
            {children}
          </code>
        ),
        pre: ({ children }) => (
          <pre className="rounded-lg bg-muted/40 p-4 text-xs overflow-x-auto border border-border/60 text-foreground dark:bg-background/70 dark:border-gray-800/60 dark:text-gray-100">
            {children}
          </pre>
        ),
        ul: ({ children }) => (
          <ul className="list-disc space-y-2 pl-5 text-foreground dark:text-gray-100">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal space-y-2 pl-5 text-foreground dark:text-gray-100">{children}</ol>
        ),
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        table: ({ children }) => (
          <div className="overflow-x-auto rounded-xl border border-border/60 bg-card/50 dark:border-gray-800/60 dark:bg-background/40">
            <table className="min-w-full divide-y divide-border/60 text-sm text-foreground dark:divide-gray-800/70 dark:text-gray-100">
              {children}
            </table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-secondary/40 text-xs uppercase tracking-wide text-muted-foreground dark:bg-background/60 dark:text-muted-foreground">
            {children}
          </thead>
        ),
        tbody: ({ children }) => (
          <tbody className="divide-y divide-border/50 dark:divide-gray-800/70">{children}</tbody>
        ),
        tr: ({ children }) => (
          <tr className="hover:bg-secondary/40 transition-colors dark:hover:bg-background/60">{children}</tr>
        ),
        th: ({ children }) => (
          <th className="px-4 py-3 text-left font-semibold text-foreground/80 dark:text-foreground/90">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-4 py-3 align-top text-foreground dark:text-gray-200">{children}</td>
        ),
      }}
    >
      {markdown}
    </ReactMarkdown>
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
            {downloadUrl && (
              <a
                href={downloadUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 rounded-full border border-info/30 bg-info/10 px-3 py-1 text-xs font-semibold uppercase text-info/70 hover:bg-info/20"
              >
                <ExternalLink className="h-3.5 w-3.5" />
                Download
              </a>
            )}
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

      {pandasAI?.charts && pandasAI.charts.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-semibold text-foreground/90 uppercase tracking-wide">
            PandasAI Charts
          </h4>
          <div className="grid gap-4 md:grid-cols-2">
            {pandasAI.charts.map((chart, idx) => (
              <div
                key={`${chart.filename}-${idx}`}
                className="rounded-lg border border-gray-800/60 bg-background/40 p-4 flex flex-col items-center gap-3"
              >
                <img
                  src={`data:${chart.mime_type};base64,${chart.base64}`}
                  alt={chart.filename}
                  className="rounded-md border border-gray-800/40 max-h-72 w-full object-contain"
                />
                <div className="flex w-full items-center justify-between text-xs text-muted-foreground">
                  <span className="truncate">{chart.filename}</span>
                  <div className="flex items-center gap-2">
                    <button
                      className="rounded border border-border/60 px-2 py-1 text-[11px] uppercase tracking-wide text-foreground/90 hover:border-primary/60 hover:text-primary/80"
                      onClick={() => {
                        const link = document.createElement('a')
                        link.href = `data:${chart.mime_type};base64,${chart.base64}`
                        link.download = chart.filename
                        document.body.appendChild(link)
                        link.click()
                        document.body.removeChild(link)
                      }}
                    >
                      Download
                    </button>
                    <button
                      className="rounded border border-border/60 px-2 py-1 text-[11px] uppercase tracking-wide text-foreground/90 hover:border-primary/60 hover:text-primary/80"
                      onClick={async () => {
                        if (!navigator.clipboard) {
                          toast.error('Clipboard API is not available')
                          return
                        }
                        try {
                          await navigator.clipboard.writeText(`data:${chart.mime_type};base64,${chart.base64}`)
                          toast.success('Copied to clipboard')
                        } catch (error) {
                          toast.error('Failed to copy to clipboard')
                        }
                      }}
                    >
                      Copy
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {pandasAI?.error && (
        <div className="text-sm text-destructive">
          PandasAI warning: {pandasAI.error}
        </div>
      )}
    </div>
  )
}

