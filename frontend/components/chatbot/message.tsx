'use client'

import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Bot, User, ThumbsUp, ThumbsDown, Copy, RotateCw, Code, FileText, Database, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { voteMessage } from '@/lib/chat/api'
import { copyToClipboard, formatTimestamp } from '@/lib/utils'
import type { ChatMessage, Artifact, CodeSnippet, DocumentReference, DatabaseResult, UseChatHelpers } from '@/types'
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
      <p className="text-gray-200 leading-relaxed">{children}</p>
    ),
    strong: ({ children }: any) => (
      <strong className="text-gray-100 font-semibold">{children}</strong>
    ),
    em: ({ children }: any) => <em className="text-gray-300 italic">{children}</em>,
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
      <ul className="list-disc pl-6 space-y-3 text-gray-100">{children}</ul>
    ),
    ol: ({ children }: any) => (
      <ol className="list-decimal pl-6 space-y-3 text-gray-100">{children}</ol>
    ),
    li: ({ children }: any) => (
      <li className="bg-gray-900/40 border border-gray-800/60 rounded-lg px-4 py-3 shadow-sm">
        <div className="space-y-1 text-gray-100">{children}</div>
      </li>
    ),
    code: ({ inline, children }: any) => (
      inline ? (
        <code className="rounded bg-gray-900/60 px-1.5 py-0.5 text-xs text-orange-200">
          {children}
        </code>
      ) : (
        <pre className="rounded-lg bg-gray-900/70 p-4 text-xs overflow-x-auto border border-gray-800/60">
          <code>{children}</code>
        </pre>
      )
    ),
    table: ({ children }: any) => (
      <div className="overflow-x-auto rounded-xl border border-gray-800/60 bg-gray-900/40">
        <table className="min-w-full divide-y divide-gray-800/70 text-sm text-gray-100">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }: any) => (
      <thead className="bg-gray-900/60 text-xs uppercase tracking-wide text-gray-400">
        {children}
      </thead>
    ),
    tbody: ({ children }: any) => (
      <tbody className="divide-y divide-gray-800/70">{children}</tbody>
    ),
    tr: ({ children }: any) => (
      <tr className="hover:bg-gray-900/60 transition-colors">{children}</tr>
    ),
    th: ({ children }: any) => (
      <th className="px-4 py-3 text-left font-semibold text-gray-300">
        {children}
      </th>
    ),
    td: ({ children }: any) => (
      <td className="px-4 py-3 align-top text-gray-200">{children}</td>
    ),
  }), [])

  const renderMessageContent = () => {
    // Handle AI SDK format (content field)
    if ('content' in message && message.content && (!message.parts || message.parts.length === 0)) {
      return (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          className="prose prose-invert prose-sm max-w-none space-y-3"
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
                className="prose prose-invert prose-sm max-w-none space-y-3"
                components={markdownComponents}
              >
                {part.text}
              </ReactMarkdown>
            )
          }

          if (part.type === 'file' && 'filename' in part) {
            return (
              <div key={index} className="flex items-center space-x-2 p-2 bg-gray-800/30 rounded">
                <FileText className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-gray-400">{part.filename}</span>
              </div>
            )
          }

          if (part.type === 'artifact' && 'artifact' in part) {
            return (
              <button
                key={index}
                onClick={() => onArtifactSelect?.(part.artifact)}
                className="w-full text-left p-3 rounded-lg bg-gray-800/30 border border-gray-700/50 hover:bg-gray-800/50 hover:border-gray-600 transition-all"
              >
                <div className="flex items-center space-x-2">
                  <Code className="w-4 h-4 text-purple-400" />
                  <span className="text-sm text-gray-300">{part.artifact.title}</span>
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex items-start space-x-3 max-w-[85%] ${
        message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
      }`}>
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          message.role === 'user' 
            ? 'bg-blue-600' 
            : 'bg-gradient-to-br from-orange-500 to-red-500'
        }`}>
          {message.role === 'user' ? (
            <User className="w-5 h-5 text-white" />
          ) : (
            <Bot className="w-5 h-5 text-white" />
          )}
        </div>

        {/* Message Content */}
        <div className="flex-1 space-y-3">
          <div className="space-y-2">
            {renderMessageContent()}
            
            {/* Metadata */}
            {message.metadata && message.role === 'assistant' && (
              <div className="mt-3 pt-3 border-t border-gray-700/50 flex items-center justify-between text-xs">
                <div className="flex items-center space-x-3 text-gray-400">
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
                  className="w-full text-left p-3 rounded-lg bg-gray-800/30 border border-gray-700/50 hover:bg-gray-800/50 hover:border-gray-600 transition-all group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Code className="w-4 h-4 text-purple-400" />
                      <span className="text-sm text-gray-300 font-mono">{snippet.symbol_name || 'Code'}</span>
                      <span className="text-xs text-gray-500">{snippet.file_path}</span>
                    </div>
                    <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-gray-300" />
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Documents */}
          {message.documents && message.documents.length > 0 && (
            <div className="space-y-2">
              {message.documents.map((doc, idx) => {
                return (
                <button
                  key={idx}
                  onClick={() => onDocumentSelect?.(doc)}
                  className="w-full text-left rounded-xl border border-blue-500/30 bg-blue-500/5 p-4 hover:border-blue-400/60 hover:bg-blue-500/10 transition-all group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <FileText className="w-4 h-4 text-blue-400" />
                      <span className="text-sm text-gray-300">{doc.filename || 'Unknown Document'}</span>
                      <Badge variant="outline" className="bg-green-500/10 border-green-500/20 text-green-400 text-xs">
                        {(doc.similarity * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-gray-300" />
                  </div>
                  {doc.excerpt && (
                    <p className="mt-2 line-clamp-3 text-sm text-gray-200 opacity-90">
                      {doc.excerpt}
                    </p>
                  )}
                  {doc.chunk_count !== undefined && (
                    <div className="mt-3 flex items-center gap-2 text-xs text-gray-400">
                      <span>{doc.chunk_count} chunks indexed</span>
                      {doc.chunk_index !== undefined && (
                        <span className="inline-flex items-center rounded-full border border-blue-500/30 bg-blue-500/10 px-2 py-0.5 text-[11px] uppercase tracking-wide text-blue-200">
                          match @ chunk {doc.chunk_index}
                        </span>
                      )}
                    </div>
                  )}
                </button>
              )})}
            </div>
          )}

          {/* Database Results */}
          {message.database_results && message.database_results.length > 0 && (
            <div className="space-y-3">
              {message.database_results.map((dbResult, idx) => (
                <button
                  key={idx}
                  onClick={() => onDatabaseSelect?.(dbResult)}
                  className="w-full text-left p-4 rounded-lg bg-gradient-to-br from-green-900/20 to-emerald-900/20 border border-green-500/30 hover:border-green-500/50 transition-all group"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Database className="w-4 h-4 text-green-400" />
                      <span className="text-sm font-medium text-green-300">{dbResult.database}</span>
                      <Badge variant="outline" className="bg-green-500/10 border-green-500/20 text-green-400 text-xs">
                        {dbResult.row_count} rows • {dbResult.execution_time_ms?.toFixed(0)}ms
                      </Badge>
                    </div>
                    <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-green-300" />
                  </div>

                  <div className="mt-2 p-2 bg-gray-900/50 rounded text-xs font-mono text-gray-300 overflow-x-auto">
                    {dbResult.sql.substring(0, 100)}{dbResult.sql.length > 100 ? '...' : ''}
                  </div>

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
                  className={`p-1 h-auto ${
                    isUpvoted === true ? 'text-green-400' : 'text-gray-500 hover:text-green-400'
                  }`}
                >
                  <ThumbsUp className="w-3 h-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleVote(false)}
                  className={`p-1 h-auto ${
                    isUpvoted === false ? 'text-red-400' : 'text-gray-500 hover:text-red-400'
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

