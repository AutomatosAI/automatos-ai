'use client'

import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { User, Code, FileText, Database, ChevronRight, CheckCircle2, XCircle, Loader2, Zap } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { ChatMessage, Artifact, CodeSnippet, DocumentReference, DatabaseResult, ToolCall, UseChatHelpers } from '@/types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { chatMarkdownComponents } from './markdown-components'
import { ImageGallery, type ChatImage } from './image-gallery'
import { MessageActions } from './message-actions'
import { VoiceMessage } from '@/components/voice/VoiceMessage'

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
  onLaunchAsMission?: (content: string) => void
}

// Extract images from markdown content
const IMAGE_RE = /!\[([^\]]*)\]\(([^)]+)\)/g

function extractImages(content: string): { text: string; images: ChatImage[] } {
  const images: ChatImage[] = []
  const text = content.replace(IMAGE_RE, (_, alt, src) => {
    images.push({ src, alt: alt || 'Generated image' })
    return ''
  })
  return { text: text.trim(), images }
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
  onLaunchAsMission,
}: MessageProps) {
  const isUser = message.role === 'user'

  // Get plain text content for actions
  const plainContent = useMemo(() => {
    const textParts = message.parts?.filter(p => p.type === 'text').map(p => 'text' in p ? p.text : '')
    return textParts?.join('\n') || message.content || ''
  }, [message.parts, message.content])

  const proseClass = "prose prose-sm md:prose-base max-w-none space-y-3 break-words [overflow-wrap:anywhere] dark:prose-invert prose-headings:text-foreground dark:prose-headings:text-gray-100 prose-p:text-foreground dark:prose-p:text-gray-100 prose-a:text-primary dark:prose-a:text-primary"

  const renderMarkdownWithImages = (text: string, keyPrefix = '') => {
    const { text: cleanText, images } = extractImages(text)
    return (
      <>
        {cleanText && (
          <ReactMarkdown
            key={`${keyPrefix}md`}
            remarkPlugins={[remarkGfm]}
            className={proseClass}
            components={chatMarkdownComponents}
          >
            {cleanText}
          </ReactMarkdown>
        )}
        {images.length > 0 && <ImageGallery images={images} />}
      </>
    )
  }

  const renderMessageContent = () => {
    // Handle AI SDK format (content field)
    if ('content' in message && message.content && (!message.parts || message.parts.length === 0)) {
      return renderMarkdownWithImages(message.content)
    }

    // Handle custom format (parts array)
    if (!message.parts || message.parts.length === 0) {
      return null
    }

    return (
      <div className="space-y-3">
        {message.parts.map((part, index) => {
          if (part.type === 'text' && 'text' in part) {
            return <div key={index}>{renderMarkdownWithImages(part.text, `p${index}`)}</div>
          }

          if (part.type === 'file' && 'filename' in part) {
            return (
              <div key={index} className="flex items-center space-x-2 p-2 bg-secondary/30 rounded-lg">
                <FileText className="w-4 h-4 text-info" />
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
                  <Code className="w-4 h-4 text-agent" />
                  <span className="text-sm text-foreground/80 dark:text-foreground/90">{part.artifact.title}</span>
                  <Badge variant="outline" className="bg-agent/10 border-agent/20 text-agent text-xs">
                    {part.artifact.kind}
                  </Badge>
                </div>
              </button>
            )
          }

          if (part.type === 'voice' && 'transcript' in part) {
            return (
              <VoiceMessage
                key={index}
                transcript={part.transcript}
                audioUrl={'audioUrl' in part ? part.audioUrl : undefined}
                audioBase64={'audioBase64' in part ? part.audioBase64 : undefined}
                isUser={isUser}
                durationMs={'durationMs' in part ? part.durationMs : undefined}
              />
            )
          }

          return null
        })}
      </div>
    )
  }

  const renderAssistantState = () => {
    if (message.role !== 'assistant') return null

    if (isLoading) {
      return (
        <div className="inline-flex items-center gap-2">
          <div className="flex space-x-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-2 h-2 rounded-full bg-primary/60 animate-pulse"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </div>
          <span className="text-xs text-primary/80 font-medium">Thinking</span>
        </div>
      )
    }

    return null
  }

  // PRD-180 S4 (F037): resolve a human label for a tool chip. For
  // ``composio_execute`` we surface the real external-app action where the
  // executor logged it (input.action / tool_slug / app), so a running Gmail or
  // Slack action reads e.g. "Composio · GMAIL_SEND_EMAIL" instead of being
  // invisible. Falls back to a plain "Composio" when no action is resolvable.
  const formatToolLabel = (tc: { toolName: string; input?: any }): string => {
    const toolLabels: Record<string, string> = {
      'composio_execute': 'Composio',
      'search_knowledge': 'Searching docs',
      'search_codebase': 'Searching code',
      'query_database': 'Querying database',
      'smart_query_database': 'Querying database',
      'get_memories': 'Checking memory',
      'search_memories': 'Searching memory',
      'execute_command': 'Running command',
    }
    const base = toolLabels[tc.toolName] || tc.toolName.replace(/_/g, ' ')
    if (tc.toolName === 'composio_execute') {
      const action =
        tc.input?.action || tc.input?.tool_slug || tc.input?.app || tc.input?.action_name
      if (action && typeof action === 'string') return `${base} · ${action}`
    }
    return base
  }

  const renderToolCalls = () => {
    if (message.role !== 'assistant') return null
    const toolCalls = message.toolCalls || []
    if (toolCalls.length === 0) return null

    // PRD-180 S4 (F037): no longer filter out ``composio_execute`` — every
    // external-app action must be visible while it runs and when it errors.
    const runningTools = toolCalls.filter(tc => tc.state === 'running')
    const errorTools = toolCalls.filter(tc => tc.state === 'error')

    if (runningTools.length === 0 && errorTools.length === 0) return null

    return (
      <div className="flex flex-wrap items-center gap-2">
        {runningTools.length > 0 && (
          <div className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs bg-primary/10 border border-primary/30 text-primary">
            <Loader2 className="w-3 h-3 animate-spin text-[hsl(var(--info))]" />
            <span>
              {runningTools.length === 1
                ? formatToolLabel(runningTools[0])
                : `Working (${runningTools.length} tools)`}
            </span>
          </div>
        )}
        {errorTools.map((tc) => (
          <div
            key={tc.toolCallId}
            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs bg-destructive/10 border border-destructive/30 text-destructive/80"
            title={tc.error || 'Tool failed'}
          >
            <XCircle className="w-3 h-3" />
            <span>{formatToolLabel(tc)} failed</span>
          </div>
        ))}
      </div>
    )
  }

  const renderRoutingIndicator = () => {
    if (message.role !== 'assistant' || !message.routingInfo) return null
    const { agentName, agentId, confidence } = message.routingInfo
    const displayName = agentName || `Agent #${agentId}`
    return (
      <div className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs bg-primary/10 border border-primary/20 text-primary">
        <Zap className="w-3 h-3" />
        <span>Routed to: <span className="font-medium">{displayName}</span></span>
        {confidence > 0 && confidence < 1 && (
          <span className="text-primary/60 ml-1">{(confidence * 100).toFixed(0)}%</span>
        )}
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      data-testid={isUser ? 'msg-user' : 'msg-assistant'}
      className={`group/msg flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex items-start space-x-3 max-w-[92%] min-w-0 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser
            ? 'bg-info'
            : 'bg-gradient-to-br from-warning/40 to-red-500/30 ring-1 ring-warning/25 shadow-[0_0_12px_rgba(249,115,22,0.15)]'
        }`}>
          {isUser ? (
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
        <div className="flex-1 min-w-0 space-y-3">
          {/* Message bubble */}
          <div className={`rounded-2xl px-4 py-3 space-y-2 ${
            isUser
              ? 'bg-primary/10 border border-primary/10 rounded-tr-sm'
              : 'bg-card/40 backdrop-blur-sm border border-border/30 rounded-tl-sm'
          }`}>
            {renderMessageContent()}

            {/* Assistant state (Thinking animation) */}
            {renderAssistantState()}

            {/* Tool calls (lifecycle transparency) */}
            {renderToolCalls()}

            {/* Routing indicator (auto-routed messages) */}
            {renderRoutingIndicator()}

            {/* Metadata */}
            {message.metadata && message.role === 'assistant' && (
              <div className="mt-2 pt-2 border-t border-border/30 flex items-center justify-between text-xs dark:border-border/30">
                <div className="flex items-center space-x-3 text-muted-foreground">
                  {message.metadata.source && (
                    typeof message.metadata.source === 'object' ? (
                      // PRD-205 S7: persisted background provenance -- render
                      // the server's label as-is (the escape keeps the
                      // fallback ASCII-only in source).
                      <Badge variant="outline" className="text-[10px] bg-warning/10 border-warning/20 text-warning">
                        {message.metadata.source.label || 'Auto \u00b7 background'}
                      </Badge>
                    ) : (
                      <Badge variant="outline" className={`text-[10px] ${
                        message.metadata.source === 'codegraph' ? 'bg-agent/10 border-agent/20 text-agent' :
                        message.metadata.source === 'rag' ? 'bg-info/10 border-info/20 text-info' :
                        message.metadata.source === 'semantic' ? 'bg-success/10 border-success/20 text-success' :
                        message.metadata.source === 'database' ? 'bg-success/10 border-success/20 text-success' :
                        message.metadata.source === 'llm' ? 'bg-primary/10 border-primary/20 text-primary' : ''
                      }`}>
                        {message.metadata.source.toUpperCase()}
                      </Badge>
                    )
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
                      <Code className="w-4 h-4 text-agent" />
                      <span className="text-sm text-foreground/80 dark:text-foreground/90 font-mono">{snippet.symbol_name || 'Code'}</span>
                      <span className="text-xs text-muted-foreground">{snippet.file_path}</span>
                    </div>
                    <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground/80 dark:group-hover:text-foreground/90" />
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Documents */}
          {message.documents && message.documents.length > 0 && (
            <div className="space-y-2">
              {message.documents.map((doc, idx) => {
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
                    onClick={() => {
                      if (onDocumentSelect) {
                        onDocumentSelect({
                          ...doc,
                          title,
                          full_content: fullContent || preview || doc.excerpt || '',
                          relevance,
                          chunk_count: chunkCount,
                          download_url: downloadUrl,
                          chunks,
                        })
                      }
                    }}
                    className="w-full text-left rounded-xl border border-info/30 bg-info/5 p-4 hover:border-info/60 hover:bg-info/10 transition-all group"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <FileText className="w-4 h-4 text-info" />
                        <span className="text-sm font-medium text-foreground dark:text-gray-200">{title}</span>
                        <Badge variant="outline" className="bg-success/10 border-success/20 text-success text-xs">
                          {relevance.toFixed(0)}%
                        </Badge>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground/80 dark:group-hover:text-foreground/90" />
                    </div>

                    {chunkCount !== undefined && (
                      <div className="mt-2 text-xs text-muted-foreground">
                        {chunkCount} relevant section{chunkCount !== 1 ? 's' : ''} found
                      </div>
                    )}

                    {preview && (
                      <p className="mt-2 line-clamp-2 text-sm text-foreground/80 dark:text-foreground/90 opacity-90">
                        {preview}
                      </p>
                    )}

                    <div className="mt-3 text-xs text-info flex items-center gap-1">
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
                  className="w-full text-left p-4 rounded-xl bg-success/5 border border-success/25 hover:border-success/40 transition-all group"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Database className="w-4 h-4 text-success" />
                      <span className="text-sm font-medium text-success">{dbResult.database}</span>
                      <Badge variant="outline" className="bg-success/10 border-success/20 text-success text-xs">
                        {dbResult.row_count} rows {dbResult.execution_time_ms && `• ${dbResult.execution_time_ms.toFixed(0)}ms`}
                      </Badge>
                    </div>
                    <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-success" />
                  </div>

                  {dbResult.sql && (
                    <div className="mt-2 p-2 bg-muted/40 rounded-lg text-xs font-mono text-foreground/80 dark:bg-background/50 dark:text-foreground/90 overflow-x-auto">
                      {dbResult.sql.substring(0, 100)}{dbResult.sql.length > 100 ? '...' : ''}
                    </div>
                  )}

                  {dbResult.status === 'needs_clarification' && dbResult.clarifications && dbResult.clarifications.length > 0 ? (
                    <div className="mt-3 rounded-lg border border-primary/30 bg-primary/10 p-3 text-left">
                      <div className="text-sm font-semibold text-primary">Clarification needed</div>
                      {dbResult.message && (
                        <div className="mt-1 text-sm text-foreground/80">{dbResult.message}</div>
                      )}
                      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-foreground">
                        {dbResult.clarifications.slice(0, 3).map((q, qIdx) => (
                          <li key={qIdx}>{q}</li>
                        ))}
                      </ul>
                      <div className="mt-2 text-xs text-muted-foreground">
                        Click to open the Data Explorer and copy questions.
                      </div>
                    </div>
                  ) : (
                    <>
                      {dbResult.pandas_ai?.summary && (
                        <div className="mt-3 text-sm text-gray-200 bg-background/40 border border-gray-800/60 rounded p-3 text-left">
                          {dbResult.pandas_ai.summary}
                        </div>
                      )}

                      {dbResult.pandas_ai?.charts && dbResult.pandas_ai.charts.length > 0 && (
                        <div className="mt-3 grid gap-2 md:grid-cols-2">
                          {dbResult.pandas_ai.charts.slice(0, 2).map((chart, chartIdx) => (
                            <div
                              key={`${chart.filename}-${chartIdx}`}
                              className="rounded-lg border border-gray-800/60 bg-background/40 p-2 flex flex-col items-center"
                            >
                              <img
                                src={`data:${chart.mime_type};base64,${chart.base64}`}
                                alt={chart.filename}
                                className="rounded-md border border-gray-800/40 max-h-32 object-contain"
                              />
                              <span className="mt-1 text-xs text-muted-foreground truncate w-full text-center">
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

          {/* Actions (show on hover) */}
          <MessageActions
            chatId={chatId}
            messageId={message.id}
            role={message.role as 'user' | 'assistant'}
            content={plainContent}
            isReadonly={isReadonly}
            regenerate={regenerate}
            createdAt={message.createdAt}
            onLaunchAsMission={onLaunchAsMission}
          />
        </div>
      </div>
    </motion.div>
  )
}
