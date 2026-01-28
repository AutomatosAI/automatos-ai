/**
 * Widget Type Definitions for PRD-38.1 Widget Architecture
 *
 * This file defines the core types for the widget system that powers
 * the canvas-based workspace. All widgets share these common interfaces.
 */

import type { ComponentType, ReactNode } from 'react'

/**
 * All supported widget types
 * Phase 1: code, data, document, image
 * Phase 2+: email, terminal, workflow, memory, file, chart, form, chat
 */
export type WidgetType =
  // Phase 1 (migrate existing artifacts)
  | 'code'        // Code viewer/editor
  | 'data'        // Tables, charts (NL2SQL)
  | 'document'    // RAG results, markdown
  | 'image'       // Generated/uploaded images
  // Phase 2+ (future widgets)
  | 'email'       // Email viewer/composer
  | 'terminal'    // Command output
  | 'workflow'    // Workflow status/control
  | 'memory'      // Memory inspector
  | 'file'        // File preview/download
  | 'chart'       // Standalone charts
  | 'form'        // Input forms
  | 'chat'        // Embedded chat (for SDK)

/**
 * Widget position in the canvas grid
 */
export interface WidgetPosition {
  x: number
  y: number
}

/**
 * Widget size in grid units
 */
export interface WidgetSize {
  width: number
  height: number
}

/**
 * Widget loading/ready states
 */
export type WidgetState = 'loading' | 'ready' | 'error' | 'stale'

/**
 * Source of widget creation
 */
export interface WidgetSource {
  type: 'tool' | 'user' | 'system' | 'workflow'
  name: string           // Tool name or action that created it
  provider?: string      // e.g., 'rag', 'nl2sql', 'composio', 'codegraph'
}

/**
 * Widget metadata tracks origin and context
 */
export interface WidgetMetadata {
  source: WidgetSource
  createdAt: Date
  updatedAt?: Date
  conversationId?: string
  toolCallId?: string
  workspaceId?: string
  // For external widgets (Phase 4+)
  tenantId?: string
  apiKey?: string
}

/**
 * Widget error structure
 */
export interface WidgetError {
  message: string
  code?: string
  details?: unknown
}

/**
 * Widget action definition
 */
export interface WidgetAction {
  id: string
  label: string
  icon?: string | ReactNode
  shortcut?: string
  handler: () => void | Promise<void>
  disabled?: boolean
  requiresConfirm?: boolean
}

/**
 * Base widget data structure
 */
export interface Widget<TData = unknown> {
  id: string
  type: WidgetType
  title: string
  data: TData
  metadata: WidgetMetadata
  state: WidgetState
  position?: WidgetPosition
  size?: WidgetSize
  error?: WidgetError | null
  createdAt: string
  updatedAt?: string
}

/**
 * Base props that ALL widget components receive
 */
export interface WidgetBaseProps<TData = unknown> {
  // Identity
  id: string
  type?: WidgetType

  // Content
  title: string
  data: TData

  // Metadata
  metadata: WidgetMetadata

  // State
  isActive?: boolean
  isLoading?: boolean
  error?: WidgetError | Error | null

  // Layout (for canvas mode)
  position?: WidgetPosition
  size?: WidgetSize

  // Actions
  onClose?: () => void
  onMaximize?: () => void
  onMinimize?: () => void
  onRefresh?: () => void
  onAction?: (action: WidgetAction) => void
}

/**
 * Widget capability flags
 */
export type WidgetCapability =
  | 'editable'      // Content can be edited
  | 'exportable'    // Can export (CSV, PDF, etc.)
  | 'refreshable'   // Can refresh data
  | 'resizable'     // Can be resized
  | 'fullscreen'    // Can go fullscreen
  | 'copyable'      // Content can be copied
  | 'downloadable'  // Content can be downloaded
  | 'shareable'     // Can be shared

/**
 * Widget definition for registry
 */
export interface WidgetDefinition<TData = unknown> {
  type: WidgetType
  displayName: string
  description?: string
  icon: string | ComponentType<{ className?: string }>
  component: ComponentType<WidgetBaseProps<TData>>
  capabilities: WidgetCapability[]
  validateData?: (data: unknown) => data is TData
  defaultSize: WidgetSize
  minSize?: WidgetSize
  maxSize?: WidgetSize
  actions?: WidgetAction[]
}

// ============================================
// Widget-specific data types
// ============================================

/**
 * Code widget data (from CodeGraph, code artifacts)
 */
export interface CodeWidgetData {
  code: string
  language: string
  filePath?: string
  lineNumber?: number
  explanation?: string
  symbolName?: string
  highlights?: Array<{
    start: number
    end: number
    className?: string
  }>
}

/**
 * Chart data for DataWidget
 */
export interface ChartData {
  filename: string
  mimeType?: string
  base64: string
}

/**
 * Data widget data (from NL2SQL, database queries)
 */
export interface DataWidgetData {
  columns: string[]
  rows: Record<string, unknown>[]
  sql?: string
  database?: string
  rowCount: number
  executionTime?: number
  charts?: ChartData[]
  pandasAiSummary?: string
  explanation?: string
  rephrased_query?: string
  follow_up_questions?: string[]
}

/**
 * Document chunk for DocumentWidget
 */
export interface DocumentChunk {
  content: string
  excerpt?: string
  similarity?: number
  chunkIndex?: number
}

/**
 * Document widget data (from RAG, semantic search)
 */
export interface DocumentWidgetData {
  content: string
  format: 'markdown' | 'text' | 'html'
  filename?: string
  filePath?: string
  similarity?: number
  relevance?: number
  chunkCount?: number
  chunks?: DocumentChunk[]
  downloadUrl?: string
  hasFullContent?: boolean
}

/**
 * Image widget data (from image generation, uploads)
 */
export interface ImageWidgetData {
  src: string
  alt?: string
  width?: number
  height?: number
  mimeType?: string
  prompt?: string  // For generated images
  model?: string
  base64?: string
}

// ============================================
// Phase 2 Widget Data Types
// ============================================

/**
 * Email address structure
 */
export interface EmailAddress {
  email: string
  name?: string
}

/**
 * Email attachment structure
 */
export interface EmailAttachment {
  id: string
  filename: string
  mimeType: string
  size: number
  downloadUrl?: string
}

/**
 * Email summary (for list view)
 */
export interface EmailSummary {
  id: string
  from: EmailAddress
  to: EmailAddress[]
  subject: string
  snippet: string
  date: string
  isRead: boolean
  hasAttachments: boolean
  labels?: string[]
}

/**
 * Full email data (extends summary)
 */
export interface EmailFull extends EmailSummary {
  body: string
  bodyHtml?: string  // ⚠️ SECURITY: Must be sanitized before rendering
  cc?: EmailAddress[]
  bcc?: EmailAddress[]
  attachments?: EmailAttachment[]
  threadId?: string
}

/**
 * Email draft for compose mode
 */
export interface EmailDraft {
  to: string[]
  cc?: string[]
  bcc?: string[]
  subject: string
  body: string
  attachments?: File[]
}

/**
 * Email widget data (from Composio Gmail/Outlook)
 */
export interface EmailWidgetData {
  mode: 'list' | 'view' | 'compose'
  // For list mode
  emails?: EmailSummary[]
  totalCount?: number
  unreadCount?: number
  // For view mode
  email?: EmailFull
  // For compose mode
  draft?: EmailDraft
  replyTo?: EmailFull
}

/**
 * Terminal widget data (from execute_command, run_script)
 */
export interface TerminalWidgetData {
  command: string
  output: string
  exitCode?: number
  executionTime?: number
  workingDirectory?: string
  environment?: Record<string, string>
  isStreaming?: boolean
}

/**
 * Workflow status type
 */
export type WorkflowStatus =
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'

/**
 * Workflow step structure
 */
export interface WorkflowStep {
  id: string
  name: string
  type: 'action' | 'condition' | 'loop' | 'parallel'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  startedAt?: string
  completedAt?: string
  duration?: number
  result?: unknown
  error?: string
  children?: WorkflowStep[]  // For nested/parallel steps
}

/**
 * Workflow widget data (from workflow orchestrator)
 */
export interface WorkflowWidgetData {
  workflowId: string
  workflowName: string
  status: WorkflowStatus
  steps: WorkflowStep[]
  startedAt?: string
  completedAt?: string
  error?: string
  result?: unknown
  variables?: Record<string, unknown>
}

/**
 * Memory type categories
 */
export type MemoryType = 'fact' | 'preference' | 'context' | 'instruction'

/**
 * Memory item structure
 */
export interface Memory {
  id: string
  type: MemoryType
  content: string
  source: {
    conversationId?: string
    timestamp: string
    trigger?: string
  }
  relevance?: number  // For injected memories
  metadata?: Record<string, unknown>
}

/**
 * Memory widget data (from memory system)
 */
export interface MemoryWidgetData {
  mode: 'injected' | 'stored' | 'all'
  // Memories injected into this conversation
  injectedMemories?: Memory[]
  // Memories stored from this conversation
  storedMemories?: Memory[]
  // Search results
  searchResults?: Memory[]
  searchQuery?: string
  // Stats
  totalMemories?: number
  conversationMemories?: number
}

/**
 * File info structure
 */
export interface FileInfo {
  name: string
  path: string
  type: 'file' | 'directory'
  size: number
  mimeType?: string
  createdAt?: string
  modifiedAt?: string
  permissions?: string
}

/**
 * File widget data (from file operations)
 */
export interface FileWidgetData {
  mode: 'single' | 'list' | 'preview'
  // Single file
  file?: FileInfo
  // File list
  files?: FileInfo[]
  currentPath?: string
  // Preview content
  previewContent?: string
  previewType?: 'text' | 'image' | 'pdf' | 'code' | 'binary'
}

// ============================================
// Grid Layout Types
// ============================================

/**
 * Layout mode for the workspace canvas
 */
export type LayoutMode = 'grid' | 'freeform' | 'split' | 'focus'

/**
 * Grid configuration
 */
export interface GridConfig {
  columns: number
  rowHeight: number
  margin?: [number, number]
  containerPadding?: [number, number]
}

/**
 * Workspace layout state
 */
export interface WorkspaceLayout {
  mode: LayoutMode
  gridConfig: GridConfig
  positions: Record<string, WidgetPosition>
  sizes: Record<string, WidgetSize>
}
