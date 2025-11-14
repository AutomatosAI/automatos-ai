// Type definitions for PRD-27 AI Chatbot Upgrade

import type { UIMessage } from '@ai-sdk/react'

/**
 * Artifact types supported by the system
 */
export type ArtifactKind = 'code' | 'text' | 'image' | 'sheet'

/**
 * Artifact data structure
 */
export interface Artifact {
  id: string
  kind: ArtifactKind
  title: string
  content: string
  language?: string // For code artifacts
  metadata?: Record<string, any>
  createdAt?: string
}

export interface PandasAIChart {
  filename: string
  mime_type: string
  base64: string
}

export interface PandasAIInsight {
  summary: string
  charts?: PandasAIChart[]
  error?: string
}

/**
 * Database query result
 */
export interface DatabaseResult {
  database: string
  sql: string
  row_count: number
  data: any[]
  columns: string[]
  execution_time_ms: number
  pandas_ai?: PandasAIInsight
}

/**
 * Code snippet from CodeGraph
 */
export interface CodeSnippet {
  language: string
  code: string
  file_path: string
  line_number?: number
  symbol_name?: string
  explanation?: string
}

/**
 * Document reference from RAG
 */
export interface DocumentReference {
  id: number
  filename: string
  excerpt: string
  content?: string
  preview?: string
  chunk_count?: number
  chunk_index?: number
  preview_chunk_start?: number
  preview_chunk_end?: number
  similarity: number
  has_full_content?: boolean
}

/**
 * Message metadata
 */
export interface MessageMetadata {
  intent?: string
  confidence?: number
  source?: 'rag' | 'semantic' | 'codegraph' | 'llm' | 'database'
  processing_time?: number
  tools_used?: string[]
  database_count?: number
}

/**
 * Custom UI data types for streaming
 */
export interface CustomUIDataTypes {
  textDelta?: string
  codeDelta?: string
  documentDelta?: string
  imageDelta?: string
  artifact?: Artifact
  usage?: AppUsage
  codeSnippets?: CodeSnippet[]
  documents?: DocumentReference[]
  database_results?: DatabaseResult[]
}

/**
 * Extended chat message with custom types
 */
export type ChatMessage = UIMessage<MessageMetadata> & {
  parts: MessagePart[]
  codeSnippets?: CodeSnippet[]
  documents?: DocumentReference[]
  database_results?: DatabaseResult[]
}

/**
 * Message part types (text, file, tool result)
 */
export type MessagePart = 
  | { type: 'text'; text: string }
  | { type: 'file'; filename: string; mediaType: string; url: string }
  | { type: 'tool-result'; toolName: string; result: any }
  | { type: 'artifact'; artifact: Artifact }

/**
 * App usage/token tracking
 */
export interface AppUsage {
  promptTokens: number
  completionTokens: number
  totalTokens: number
  cost?: number
}

/**
 * Chat visibility type
 */
export type VisibilityType = 'private' | 'public'

/**
 * Chat session
 */
export interface Chat {
  id: string
  userId: string
  title: string
  createdAt: string
  updatedAt: string
  visibility: VisibilityType
  lastContext?: AppUsage
}

/**
 * Vote on a message
 */
export interface Vote {
  chatId: string
  messageId: string
  isUpvoted: boolean
}

/**
 * Attachment type
 */
export interface Attachment {
  name: string
  contentType: string
  url: string
}

/**
 * Suggestion for document editing
 */
export interface Suggestion {
  id: string
  artifactId: string
  originalText: string
  suggestedText: string
  description: string
  isResolved: boolean
  createdAt: string
}

/**
 * Chat request payload
 */
export interface ChatRequest {
  id?: string
  message: {
    role: 'user' | 'assistant'
    parts: MessagePart[]
  }
  selectedChatModel?: string
  selectedVisibilityType?: VisibilityType
  context?: {
    currentPage?: string
    selectedItems?: any[]
    userRole?: string
    recentActions?: any[]
  }
}

/**
 * Chat response payload
 */
export interface ChatResponse {
  message: ChatMessage
  chat?: Chat
  usage?: AppUsage
}

/**
 * Stream data types
 */
export interface StreamData {
  type: 'text-delta' | 'code-delta' | 'artifact' | 'usage' | 'tool-result'
  data: any
}

/**
 * Model information
 */
export interface Model {
  id: string
  name: string
  provider: string
  description?: string
  contextWindow?: number
  pricing?: {
    input: number
    output: number
  }
}

/**
 * Use chat helpers type
 */
export interface UseChatHelpers<T> {
  messages: T[]
  setMessages: (messages: T[] | ((prev: T[]) => T[])) => void
  input: string
  setInput: (input: string) => void
  append: (message: T) => Promise<void>
  reload: () => void
  stop: () => void
  status: 'idle' | 'streaming' | 'awaiting_message'
}

