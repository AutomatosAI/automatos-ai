'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Bot, User, ThumbsUp, ThumbsDown, Copy, RotateCw, Code, FileText, Database, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { voteMessage } from '@/lib/chat/api'
import { copyToClipboard, formatTimestamp } from '@/lib/utils'
import type { ChatMessage, Artifact, CodeSnippet, DocumentReference, DatabaseResult, UseChatHelpers } from '@/types'
import { toast } from 'sonner'

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

  const renderMessageContent = () => {
    // Handle AI SDK format (content field)
    if ('content' in message && message.content && (!message.parts || message.parts.length === 0)) {
      return (
        <div className="space-y-3">
          <p className="text-gray-300 whitespace-pre-wrap">{message.content}</p>
        </div>
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
              <p key={index} className="text-gray-300 whitespace-pre-wrap">
                {part.text}
              </p>
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
          {(() => {
            console.log('[Message] Rendering - has documents?', !!message.documents, 'count:', message.documents?.length)
            console.log('[Message] Documents array:', message.documents)
            return null
          })()}
          {message.documents && message.documents.length > 0 && (
            <div className="space-y-2">
              {message.documents.map((doc, idx) => {
                console.log(`[Message] Document ${idx}:`, JSON.stringify(doc, null, 2))
                return (
                <button
                  key={idx}
                  onClick={() => onDocumentSelect?.(doc)}
                  className="w-full text-left p-3 rounded-lg bg-gray-800/30 border border-gray-700/50 hover:bg-gray-800/50 hover:border-gray-600 transition-all group"
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

