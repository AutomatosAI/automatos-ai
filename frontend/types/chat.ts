// Type definitions for PRD-27 AI Chatbot Upgrade

import type { UIMessage } from '@ai-sdk/react'

/**
 * Artifact types supported by the system
 */
export type ArtifactKind = 'code' | 'text' | 'image' | 'sheet' | 'document'

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

export type ToolCallState = 'running' | 'completed' | 'error'

/**
 * Tool call lifecycle state for UI transparency
 */
export interface ToolCall {
  toolCallId: string
  toolName: string
  state: ToolCallState
  input?: any
  error?: string
  durationMs?: number
  startedAt?: string
  endedAt?: string
}

/**
 * Database query result
 */
export interface DatabaseResult {
  database: string
  status?: string
  sql: string
  row_count: number
  data: any[]
  columns: string[]
  execution_time_ms: number
  pandas_ai?: PandasAIInsight
  explanation?: string
  rephrased_query?: string
  visualization?: any
  follow_up_questions?: string[]
  clarifications?: string[]
  clarification_answers?: Record<string, any>
  original_query?: string
  message?: string
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
  title?: string
  excerpt: string
  content?: string
  preview?: string
  chunk_count?: number
  chunk_index?: number
  preview_chunk_start?: number
  preview_chunk_end?: number
  similarity: number
  has_full_content?: boolean
  file_path?: string
  download_url?: string
  relevance?: number
  full_content?: string
  chunks?: Array<{ content: string; excerpt: string }>
}

/**
 * Routing decision metadata from universal router
 */
export interface RoutingInfo {
  requestId?: string
  agentId: number
  agentName?: string
  confidence: number
  routeType: string
  reasoning: string
}

/**
 * Message metadata
 */
export interface MessageMetadata {
  intent?: string
  confidence?: number
  // PRD-205 S7: background-authored messages carry their persisted
  // source.label ("Auto · background") through this slot — any string
  // renders as a neutral badge; the literals keep their colour branches.
  source?: 'rag' | 'semantic' | 'codegraph' | 'llm' | 'database' | (string & {})
  processing_time?: number
  tools_used?: string[]
  database_count?: number
  routing?: RoutingInfo
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
  toolCalls?: ToolCall[]
  routingInfo?: RoutingInfo
}

/**
 * Message part types (text, file, tool result)
 */
export type MessagePart =
  | { type: 'text'; text: string }
  | { type: 'file'; filename: string; mediaType: string; url: string }
  | { type: 'tool-result'; toolName: string; result: any }
  | { type: 'artifact'; artifact: Artifact }
  | { type: 'voice'; transcript: string; audioUrl?: string; audioBase64?: string; durationMs?: number }

/**
 * Voice message metadata from STT/TTS pipeline
 */
export interface VoiceMetadata {
  transcript: string
  audioUrl?: string | null
  sttModel?: string
  ttsModel?: string
  ttsVoice?: string
  sttLatencyMs?: number
  ttsLatencyMs?: number
}

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
  /** PRD-220: latest message text (truncated server-side) for thread lists. */
  lastMessagePreview?: string | null
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

// ---------------------------------------------------------------------------
// US-015: SSE Widget Events
// ---------------------------------------------------------------------------

/**
 * A memory record surfaced from the backend memory system.
 */
export interface MemorySummary {
  id: string
  memory: string
  tier: 'global' | 'agent' | string
}

/**
 * Payload of the `memory-injected` SSE data event.
 * Emitted when relevant memories are retrieved and injected into the LLM context.
 */
export interface MemoryInjectedEvent {
  memories: MemorySummary[]
  totalMatched: number
}

/**
 * Payload of the `memory-stored` SSE data event.
 * Emitted after a conversation exchange is persisted to memory.
 */
export interface MemoryStoredEvent {
  memory: {
    userMessage: string
    assistantResponse: string
    chatId?: string
  }
  reason: string
}

/**
 * Payload of the `workflow-update` SSE data event.
 * Emitted when a workflow execution changes state.
 */
export interface WorkflowUpdateEvent {
  workflowId: string
  status: string
  currentStep?: string
  progress?: number
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

