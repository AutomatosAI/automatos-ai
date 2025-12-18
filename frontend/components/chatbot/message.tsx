'use client'

import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { User, ThumbsUp, ThumbsDown, Copy, RotateCw, Code, FileText, Database, ChevronRight, Wrench, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { voteMessage } from '@/lib/chat/api'
import { copyToClipboard, formatTimestamp } from '@/lib/utils'
import type { ChatMessage, Artifact, CodeSnippet, DocumentReference, DatabaseResult, ToolCall, UseChatHelpers } from '@/types'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export interface MessageProps {
  chatId: string
  message: ChatMessage
  isLoading?: boolean
  setMessages: UseChatHelpers<ChatMessage>['setMessages']
  regenerate: () => void
  isReadonly: boolean
  onArtifactSelect?: (artifact: Artifact) => void
  onCodeSelect?: (code: CodeSnippet) => void
  onDocumentSelect?: (doc: DocumentReference) => void
  onDatabaseSelect?: (db: DatabaseResult) => void
}

export function Message({
  chatId,
  message,
  isLoading,
  setMessages,
  regenerate,
  isReadonly,
  onArtifactSelect,
  onCodeSelect,
  onDocumentSelect,
  onDatabaseSelect,
}: MessageProps) {
  const [mode, setMode] = useState<'view' | 'edit'>('view')
  const [isUpvoted, setIsUpvoted] = useState<boolean | undefined>()

  const handleCopy = async () => {
    const textParts = message.parts?.filter(p => p.type === 'text').map(p => 'text' in p ? p.text : '')
    const content = textParts?.join('\n') || message.content || ''

    if (await copyToClipboard(content)) {
      toast.success('Copied to clipboard')
    }
  }

  const handleVote = async (upvote: boolean) => {
    try {
      await voteMessage(chatId, message.id, upvote)
      setIsUpvoted(upvote)
      toast.success(upvote ? 'Upvoted' : 'Downvoted')
    } catch (error) {
      toast.error('Failed to vote')
    }
  }

  const markdownComponents = useMemo(() => ({
    p: ({ children }: any) => (
      <p className="text-foreground leading-relaxed tracking-[0.01em] dark:text-gray-100">{children}</p>
    ),
    strong: ({ children }: any) => (
      <strong className="text-foreground font-semibold dark:text-gray-100">{children}</strong>
    ),
    em: ({ children }: any) => <em className="text-muted-foreground italic dark:text-gray-300">{children}</em>,
    a: ({ href, children }: any) => (
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="text-orange-300 hover:text-orange-200 underline"
      >
        {children}
      </a>
    ),
    ul: ({ children }: any) => (
      <ul className="list-disc pl-6 space-y-3 text-foreground dark:text-gray-100">{children}</ul>
    ),
    ol: ({ children }: any) => (
      <ol className="list-decimal pl-6 space-y-3 text-foreground dark:text-gray-100">{children}</ol>
    ),
    li: ({ children }: any) => (
      <li className="bg-card/60 border border-border/60 rounded-lg px-4 py-3 shadow-sm dark:bg-gray-900/40 dark:border-gray-800/60">
        <div className="space-y-1 text-foreground dark:text-gray-100">{children}</div>
      </li>
    ),
    code: ({ inline, children }: any) => (
      inline ? (
        <code className="rounded bg-secondary/40 px-1.5 py-0.5 text-xs text-foreground dark:bg-gray-900/60 dark:text-orange-200">
          {children}
        </code>
      ) : (
        <pre className="rounded-lg bg-muted/40 p-4 text-xs overflow-x-auto border border-border/60 text-foreground dark:bg-gray-900/70 dark:border-gray-800/60 dark:text-gray-100">
          <code className="text-inherit">{children}</code>
        </pre>
      )
    ),
    table: ({ children }: any) => (
      <div className="overflow-x-auto rounded-xl border border-border/60 bg-card/50 dark:border-gray-800/60 dark:bg-gray-900/40">
        <table className="min-w-full divide-y divide-border/60 text-sm text-foreground dark:divide-gray-800/70 dark:text-gray-100">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }: any) => (
      <thead className="bg-secondary/40 text-xs uppercase tracking-wide text-muted-foreground dark:bg-gray-900/60 dark:text-gray-400">
        {children}
      </thead>
    ),
    tbody: ({ children }: any) => (
      <tbody className="divide-y divide-border/50 dark:divide-gray-800/70">{children}</tbody>
    ),
    tr: ({ children }: any) => (
      <tr className="hover:bg-secondary/40 transition-colors dark:hover:bg-gray-900/60">{children}</tr>
    ),
    th: ({ children }: any) => (
      <th className="px-4 py-3 text-left font-semibold text-foreground/80 dark:text-gray-300">
        {children}
      </th>
    ),
    td: ({ children }: any) => (
      <td className="px-4 py-3 align-top text-foreground dark:text-gray-200">{children}</td>
    ),
  }), [])

  const renderMessageContent = () => {
    // Handle AI SDK format (content field)
    if ('content' in message && message.content && (!message.parts || message.parts.length === 0)) {
      return (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          className="prose prose-sm md:prose-base max-w-none space-y-4 dark:prose-invert prose-headings:text-foreground dark:prose-headings:text-gray-100 prose-p:text-foreground dark:prose-p:text-gray-100 prose-a:text-orange-500 dark:prose-a:text-orange-300"
          components={markdownComponents}
        >
          {message.content}
        </ReactMarkdown>
      )
    }

    // Handle custom format (parts array)
    if (!message.parts || message.parts.length === 0) {
      return null
    }

    return (
      <div className="space-y-3">
        {message.parts.map((part, index) => {
          if (part.type === 'text' && 'text' in part) {
            return (
              <ReactMarkdown
                key={index}
                remarkPlugins={[remarkGfm]}
                className="prose prose-sm md:prose-base max-w-none space-y-4 dark:prose-invert prose-headings:text-foreground dark:prose-headings:text-gray-100 prose-p:text-foreground dark:prose-p:text-gray-100 prose-a:text-orange-500 dark:prose-a:text-orange-300"
                components={markdownComponents}
              >
                {part.text}
              </ReactMarkdown>
            )
          }

          if (part.type === 'file' && 'filename' in part) {
            return (
              <div key={index} className="flex items-center space-x-2 p-2 bg-secondary/30 rounded-lg">
                <FileText className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-muted-foreground">{part.filename}</span>
              </div>
            )
          }

          if (part.type === 'artifact' && 'artifact' in part) {
            return (
              <button
                key={index}
                onClick={() => onArtifactSelect?.(part.artifact)}
                className="w-full text-left p-3 rounded-xl bg-secondary/30 border border-border/60 hover:bg-secondary/50 hover:border-border transition-all"
              >
                <div className="flex items-center space-x-2">
                  <Code className="w-4 h-4 text-purple-400" />
                  <span className="text-sm text-foreground/80 dark:text-gray-300">{part.artifact.title}</span>
                  <Badge variant="outline" className="bg-purple-500/10 border-purple-500/20 text-purple-400 text-xs">
                    {part.artifact.kind}
                  </Badge>
                </div>
              </button>
            )
          }

          return null
        })}
      </div>
    )
  }

  const renderAssistantState = () => {
    if (message.role !== 'assistant') return null

    // Only show state for the actively streaming assistant message (the one passed isLoading)
    if (isLoading) {
      return (
        <div className="inline-flex items-center gap-2 rounded-full border border-orange-500/25 bg-orange-500/10 px-3 py-1 text-xs text-orange-200">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          <span className="font-medium">Thinking</span>
          <span className="opacity-70">…</span>
        </div>
      )
    }

    // When finished, keep it subtle (only show if this message has any content)
    const hasText =
      (typeof message.content === 'string' && message.content.trim().length > 0) ||
      (Array.isArray(message.parts) && message.parts.some((p: any) => p?.type === 'text' && p?.text))

    if (!hasText) return null

    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-gray-800/60 bg-gray-900/30 px-3 py-1 text-xs text-gray-300">
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
        <span className="font-medium text-muted-foreground dark:text-gray-300">Completed</span>
      </div>
    )
  }

  const renderToolCalls = () => {
    if (message.role !== 'assistant') return null
    const toolCalls = message.toolCalls || []
    if (toolCalls.length === 0) return null

    const getStatus = (toolCall: ToolCall) => {
      if (toolCall.state === 'running') {
        return { label: 'Running', icon: <Loader2 className="w-4 h-4 animate-spin" />, className: 'bg-orange-500/10 border-orange-500/30 text-orange-200' }
      }
      if (toolCall.state === 'error') {
        return { label: 'Error', icon: <XCircle className="w-4 h-4 text-red-400" />, className: 'bg-red-500/10 border-red-500/30 text-red-200' }
      }
      return { label: 'Completed', icon: <CheckCircle2 className="w-4 h-4 text-emerald-400" />, className: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200' }
    }

    return (
      <div className="space-y-2">
        {toolCalls.map((tc) => {
          const status = getStatus(tc)
          const durationText = typeof tc.durationMs === 'number' ? `${tc.durationMs.toFixed(0)}ms` : null

          return (
            <details
              key={tc.toolCallId}
              className="rounded-xl border border-border/60 bg-card/50 dark:border-gray-800/60 dark:bg-gray-900/30"
              open={tc.state === 'running'}
            >
              <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3">
                <div className="flex min-w-0 items-center gap-2">
                  <Wrench className="w-4 h-4 text-muted-foreground" />
                  <span className="truncate text-sm font-medium text-foreground dark:text-gray-200">{tc.toolName}</span>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {durationText && (
                    <span className="text-xs text-muted-foreground font-mono">{durationText}</span>
                  )}
                  <span
                    className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-xs ${status.className}`}
                  >
                    {status.icon}
                    <span>{status.label}</span>
                  </span>
                </div>
              </summary>

              <div className="space-y-3 border-t border-border/60 px-4 py-3 dark:border-gray-800/60">
                {tc.input && (
                  <div>
                    <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Parameters</div>
                    <pre className="mt-2 overflow-x-auto rounded-lg border border-border/60 bg-muted/30 p-3 text-xs text-foreground dark:border-gray-800/60 dark:bg-gray-900/60 dark:text-gray-200">
                      {JSON.stringify(tc.input, null, 2)}
                    </pre>
                  </div>
                )}

                {tc.error && (
                  <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">
                    {tc.error}
                  </div>
                )}
              </div>
            </details>
          )
        })}
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex items-start space-x-3 max-w-[85%] ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
        }`}>
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${message.role === 'user'
          ? 'bg-blue-600'
          : 'bg-gradient-to-br from-orange-500/40 to-red-500/30 ring-1 ring-orange-500/25'
          }`}>
          {message.role === 'user' ? (
            <User className="w-5 h-5 text-white" />
          ) : (
            <img
              src="/brand/automatos-mark-white.png"
              alt="Automatos"
              className="w-5 h-5 object-contain"
              draggable={false}
            />
          )}
        </div>

        {/* Message Content */}
        <div className="flex-1 space-y-3">
          <div className="space-y-2">
            {renderMessageContent()}

            {/* Assistant state (Thinking / Completed) */}
            {renderAssistantState()}

            {/* Tool calls (lifecycle transparency) */}
            {renderToolCalls()}

            {/* Metadata */}
            {message.metadata && message.role === 'assistant' && (
              <div className="mt-3 pt-3 border-t border-border/50 flex items-center justify-between text-xs dark:border-gray-700/50">
                <div className="flex items-center space-x-3 text-muted-foreground">
                  {message.metadata.source && (
                    <Badge variant="outline" className={`
                      ${message.metadata.source === 'codegraph' ? 'bg-purple-500/10 border-purple-500/20 text-purple-400' : ''}
                      ${message.metadata.source === 'rag' ? 'bg-blue-500/10 border-blue-500/20 text-blue-400' : ''}
                      ${message.metadata.source === 'semantic' ? 'bg-green-500/10 border-green-500/20 text-green-400' : ''}
                      ${message.metadata.source === 'database' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : ''}
                      ${message.metadata.source === 'llm' ? 'bg-orange-500/10 border-orange-500/20 text-orange-400' : ''}
                    `}>
                      {message.metadata.source.toUpperCase()}
                    </Badge>
                  )}
                  {message.metadata.processing_time !== undefined && (
                    <span>{(message.metadata.processing_time).toFixed(2)}s</span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Code Snippets */}
          {message.codeSnippets && message.codeSnippets.length > 0 && (
            <div className="space-y-2">
              {message.codeSnippets.map((snippet, idx) => (
                <button
                  key={idx}
                  onClick={() => onCodeSelect?.(snippet)}
                  className="w-full text-left p-3 rounded-xl bg-secondary/30 border border-border/60 hover:bg-secondary/50 hover:border-border transition-all group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Code className="w-4 h-4 text-purple-400" />
                      <span className="text-sm text-foreground/80 dark:text-gray-300 font-mono">{snippet.symbol_name || 'Code'}</span>
                      <span className="text-xs text-muted-foreground">{snippet.file_path}</span>
                    </div>
                    <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground/80 dark:group-hover:text-gray-300" />
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Documents */}
          {message.documents && message.documents.length > 0 && (
            <div className="space-y-2">
              {message.documents.map((doc, idx) => {
                // Support both old and new format
                const title = doc.title || doc.filename || 'Unknown Document'
                const relevance = doc.relevance !== undefined ? doc.relevance : (doc.similarity ? doc.similarity * 100 : 0)
                const preview = doc.preview || doc.excerpt || ''
                const chunkCount = doc.chunk_count
                const downloadUrl = doc.download_url
                const fullContent = doc.full_content
                const chunks = doc.chunks

                return (
                  <button
                    key={idx}
                    onClick={async () => {
                      // Prefer streaming/tool payloads (fast) before fetching
                      if (onArtifactSelect && (fullContent || (chunks && chunks.length > 0))) {
                        onArtifactSelect({
                          id: `doc-${Date.now()}-${idx}`,
                          title,
                          kind: 'text',
                          language: 'markdown',
                          content: fullContent || preview || doc.excerpt || '',
                          metadata: {
                            source: doc.filename,
                            relevance,
                            chunk_count: chunkCount,
                            download_url: downloadUrl,
                            chunks,
                          },
                        })
                        return
                      }

                      // Fallback: fetch full document content from API by file_path
                      if (doc.file_path && onArtifactSelect) {
                        try {
                          const response = await fetch(`/api/documents/content?path=${encodeURIComponent(doc.file_path)}`)
                          if (response.ok) {
                            const data = await response.json()
                            onArtifactSelect({
                              id: `doc-${Date.now()}-${idx}`,
                              title: title,
                              kind: 'text',
                              language: 'markdown',
                              content: data.content,
                              metadata: {
                                source: data.filename,
                                relevance: relevance,
                                chunk_count: chunkCount,
                                download_url: downloadUrl
                              }
                            })
                          } else {
                            console.error('Failed to fetch document:', response.statusText)
                          }
                        } catch (error) {
                          console.error('Error fetching document:', error)
                        }
                      } else if (onDocumentSelect) {
                        onDocumentSelect(doc)
                      }
                    }}
                    className="w-full text-left rounded-xl border border-blue-500/30 bg-blue-500/5 p-4 hover:border-blue-400/60 hover:bg-blue-500/10 transition-all group"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <FileText className="w-4 h-4 text-blue-400" />
                        <span className="text-sm font-medium text-foreground dark:text-gray-200">{title}</span>
                        <Badge variant="outline" className="bg-green-500/10 border-green-500/20 text-green-400 text-xs">
                          {relevance.toFixed(0)}%
                        </Badge>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground/80 dark:group-hover:text-gray-300" />
                    </div>

                    {chunkCount !== undefined && (
                      <div className="mt-2 text-xs text-muted-foreground">
                        {chunkCount} relevant section{chunkCount !== 1 ? 's' : ''} found
                      </div>
                    )}

                    {preview && (
                      <p className="mt-2 line-clamp-2 text-sm text-foreground/80 dark:text-gray-300 opacity-90">
                        {preview}
                      </p>
                    )}

                    <div className="mt-3 text-xs text-blue-400 flex items-center gap-1">
                      <span>📖</span>
                      <span>Click to view full document</span>
                    </div>
                  </button>
                )
              })}
            </div>
          )}

          {/* Database Results */}
          {message.database_results && message.database_results.length > 0 && (
            <div className="space-y-3">
              {message.database_results.map((dbResult, idx) => (
                <button
                  key={idx}
                  onClick={() => onDatabaseSelect?.(dbResult)}
                  className="w-full text-left p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/25 hover:border-emerald-500/40 transition-all group"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Database className="w-4 h-4 text-green-400" />
                      <span className="text-sm font-medium text-emerald-700 dark:text-green-300">{dbResult.database}</span>
                      <Badge variant="outline" className="bg-green-500/10 border-green-500/20 text-green-400 text-xs">
                        {dbResult.row_count} rows • {dbResult.execution_time_ms?.toFixed(0)}ms
                      </Badge>
                    </div>
                    <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-emerald-700 dark:group-hover:text-green-300" />
                  </div>

                  {dbResult.sql && (
                    <div className="mt-2 p-2 bg-muted/40 rounded-lg text-xs font-mono text-foreground/80 dark:bg-gray-900/50 dark:text-gray-300 overflow-x-auto">
                      {dbResult.sql.substring(0, 100)}{dbResult.sql.length > 100 ? '...' : ''}
                    </div>
                  )}

                  {dbResult.status === 'needs_clarification' && dbResult.clarifications && dbResult.clarifications.length > 0 ? (
                    <div className="mt-3 rounded-lg border border-orange-500/30 bg-orange-500/10 p-3 text-left">
                      <div className="text-sm font-semibold text-orange-200">Clarification needed</div>
                      {dbResult.message && (
                        <div className="mt-1 text-sm text-orange-100/80">{dbResult.message}</div>
                      )}
                      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-orange-100/90">
                        {dbResult.clarifications.slice(0, 3).map((q, qIdx) => (
                          <li key={qIdx}>{q}</li>
                        ))}
                      </ul>
                      <div className="mt-2 text-xs text-orange-100/70">
                        Click to open the Data Explorer and copy questions.
                      </div>
                    </div>
                  ) : (
                    <>
                      {dbResult.pandas_ai?.summary && (
                        <div className="mt-3 text-sm text-gray-200 bg-gray-900/40 border border-gray-800/60 rounded p-3 text-left">
                          {dbResult.pandas_ai.summary}
                        </div>
                      )}

                      {dbResult.pandas_ai?.charts && dbResult.pandas_ai.charts.length > 0 && (
                        <div className="mt-3 grid gap-2 md:grid-cols-2">
                          {dbResult.pandas_ai.charts.slice(0, 2).map((chart, chartIdx) => (
                            <div
                              key={`${chart.filename}-${chartIdx}`}
                              className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-2 flex flex-col items-center"
                            >
                              <img
                                src={`data:${chart.mime_type};base64,${chart.base64}`}
                                alt={chart.filename}
                                className="rounded-md border border-gray-800/40 max-h-32 object-contain"
                              />
                              <span className="mt-1 text-xs text-gray-500 truncate w-full text-center">
                                {chart.filename}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </button>
              ))}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="text-gray-500 hover:text-gray-300 p-1 h-auto"
            >
              <Copy className="w-3 h-3" />
            </Button>

            {message.role === 'assistant' && !isReadonly && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleVote(true)}
                  className={`p-1 h-auto ${isUpvoted === true ? 'text-green-400' : 'text-gray-500 hover:text-green-400'
                    }`}
                >
                  <ThumbsUp className="w-3 h-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleVote(false)}
                  className={`p-1 h-auto ${isUpvoted === false ? 'text-red-400' : 'text-gray-500 hover:text-red-400'
                    }`}
                >
                  <ThumbsDown className="w-3 h-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={regenerate}
                  className="text-gray-500 hover:text-gray-300 p-1 h-auto"
                >
                  <RotateCw className="w-3 h-3" />
                </Button>
              </>
            )}

            <span className="text-xs text-gray-500">
              {formatTimestamp(message.createdAt || new Date().toISOString())}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

