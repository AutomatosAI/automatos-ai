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
      className="prose prose-invert prose-sm max-w-none"
      components={{
        a: ({ href, children, ...props }) => {
          if (href?.startsWith('sandbox://')) {
            const label = children ?? href.replace('sandbox://', '')
            return (
              <span className="inline-flex items-center text-orange-300/80">
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
              className="text-orange-300 hover:text-orange-200 underline"
            >
              {children}
            </a>
          )
        },
        code: ({ children }) => (
          <code className="rounded bg-gray-900/60 px-1.5 py-0.5 text-xs">
            {children}
          </code>
        ),
        pre: ({ children }) => (
          <pre className="rounded-lg bg-gray-900/70 p-4 text-xs overflow-x-auto border border-gray-800/60">
            {children}
          </pre>
        ),
        ul: ({ children }) => (
          <ul className="list-disc space-y-2 pl-5 text-gray-100">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal space-y-2 pl-5 text-gray-100">{children}</ol>
        ),
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        table: ({ children }) => (
          <div className="overflow-x-auto rounded-xl border border-gray-800/60 bg-gray-900/40">
            <table className="min-w-full divide-y divide-gray-800/70 text-sm text-gray-100">
              {children}
            </table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-gray-900/60 text-xs uppercase tracking-wide text-gray-400">
            {children}
          </thead>
        ),
        tbody: ({ children }) => (
          <tbody className="divide-y divide-gray-800/70">{children}</tbody>
        ),
        tr: ({ children }) => (
          <tr className="hover:bg-gray-900/60 transition-colors">{children}</tr>
        ),
        th: ({ children }) => (
          <th className="px-4 py-3 text-left font-semibold text-gray-300">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-4 py-3 align-top text-gray-200">{children}</td>
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
              <span className="inline-flex items-center rounded-full border border-orange-500/40 bg-orange-500/10 px-3 py-1 text-xs font-semibold uppercase text-orange-200">
                {metadata.database}
              </span>
            )}
            {metadata.model && (
              <span className="inline-flex items-center rounded-full border border-blue-500/40 bg-blue-500/10 px-3 py-1 text-xs font-semibold uppercase text-blue-200">
                {metadata.model}
              </span>
            )}
            {downloadUrl && (
              <a
                href={downloadUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 rounded-full border border-blue-500/30 bg-blue-500/10 px-3 py-1 text-xs font-semibold uppercase text-blue-200 hover:bg-blue-500/20"
              >
                <ExternalLink className="h-3.5 w-3.5" />
                Download
              </a>
            )}
          </div>

          <div className="grid grid-cols-1 gap-3 text-sm text-gray-300 md:grid-cols-2">
            {metadata.row_count !== undefined && (
              <div className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Rows</div>
                <div className="text-lg font-semibold text-gray-100">{metadata.row_count}</div>
              </div>
            )}
            {metadata.execution_time_ms !== undefined && (
              <div className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Execution Time</div>
                <div className="text-lg font-semibold text-gray-100">{Number(metadata.execution_time_ms).toFixed(0)} ms</div>
              </div>
            )}
            {metadata.similarity !== undefined && (
              <div className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Similarity</div>
                <div className="text-lg font-semibold text-gray-100">{(metadata.similarity * 100).toFixed(1)}%</div>
              </div>
            )}
            {metadata.document_id && (
              <div className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Document</div>
                <div className="text-base font-semibold text-gray-100">{metadata.document_id}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* RAG chunk inspector (when provided) */}
      {chunks && chunks.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
            Relevant Chunks ({chunks.length})
          </h4>
          <div className="space-y-2">
            {chunks.map((chunk, idx) => (
              <details
                key={idx}
                className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4"
              >
                <summary className="cursor-pointer text-sm font-medium text-gray-200">
                  Chunk {idx + 1}: {chunk.excerpt ? chunk.excerpt.slice(0, 120) : 'Open'}
                  {chunk.excerpt && chunk.excerpt.length > 120 ? '…' : ''}
                </summary>
                <div className="mt-3 space-y-3">
                  <div className="flex items-center justify-end">
                    <button
                      className="inline-flex items-center gap-2 rounded border border-gray-700/60 px-2 py-1 text-[11px] uppercase tracking-wide text-gray-300 hover:border-orange-400/60 hover:text-orange-300"
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
                  <pre className="rounded-lg bg-gray-900/70 p-4 text-xs overflow-x-auto border border-gray-800/60 whitespace-pre-wrap">
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
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
            PandasAI Charts
          </h4>
          <div className="grid gap-4 md:grid-cols-2">
            {pandasAI.charts.map((chart, idx) => (
              <div
                key={`${chart.filename}-${idx}`}
                className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-4 flex flex-col items-center gap-3"
              >
                <img
                  src={`data:${chart.mime_type};base64,${chart.base64}`}
                  alt={chart.filename}
                  className="rounded-md border border-gray-800/40 max-h-72 w-full object-contain"
                />
                <div className="flex w-full items-center justify-between text-xs text-gray-500">
                  <span className="truncate">{chart.filename}</span>
                  <div className="flex items-center gap-2">
                    <button
                      className="rounded border border-gray-700/60 px-2 py-1 text-[11px] uppercase tracking-wide text-gray-300 hover:border-orange-400/60 hover:text-orange-300"
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
                      className="rounded border border-gray-700/60 px-2 py-1 text-[11px] uppercase tracking-wide text-gray-300 hover:border-orange-400/60 hover:text-orange-300"
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
        <div className="text-sm text-red-400">
          PandasAI warning: {pandasAI.error}
        </div>
      )}
    </div>
  )
}

