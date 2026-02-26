/**
 * Widget Router for PRD-38.1 Widget Architecture
 *
 * Routes tool results to appropriate widget types and transforms
 * the data into widget-compatible formats.
 */

import type {
  WidgetType,
  Widget,
  WidgetMetadata,
  CodeWidgetData,
  DataWidgetData,
  DocumentWidgetData,
  ImageWidgetData,
  EmailWidgetData,
  TerminalWidgetData,
  WorkflowWidgetData,
  MemoryWidgetData,
  FileWidgetData,
  CodingCanvasWidgetData,
  ChartData,
  DocumentChunk,
  EmailSummary,
  WorkflowStep,
  Memory,
  FileInfo,
  WorkflowStatus,
} from './types'

/**
 * Mapping of tool names to widget types
 */
const TOOL_WIDGET_MAP: Record<string, WidgetType> = {
  // RAG Tools → Document Widget
  search_knowledge: 'document',
  semantic_search: 'document',
  search_documents: 'document',
  search_multimodal: 'document',
  get_document: 'document',
  get_document_content: 'document',

  // Database Tools → Data Widget
  smart_query_database: 'data',
  query_database: 'data',
  execute_sql: 'data',
  nl2sql: 'data',

  // Code Tools → Code Widget
  search_codebase: 'code',
  get_code_context: 'code',
  analyze_code: 'code',
  search_code: 'code',
  get_function: 'code',
  get_class: 'code',

  // Image Tools → Image Widget
  generate_image: 'image',
  edit_image: 'image',
  analyze_image: 'image',
  search_images: 'image',

  // Phase 2: Email Tools → Email Widget
  GMAIL_SEND_EMAIL: 'email',
  GMAIL_LIST_EMAILS: 'email',
  GMAIL_GET_EMAIL: 'email',
  GMAIL_REPLY_EMAIL: 'email',
  OUTLOOK_SEND_EMAIL: 'email',
  OUTLOOK_LIST_EMAILS: 'email',
  OUTLOOK_GET_EMAIL: 'email',
  send_email: 'email',
  list_emails: 'email',
  get_email: 'email',

  // Phase 2: Terminal Tools → Terminal Widget
  execute_command: 'terminal',
  run_script: 'terminal',
  shell_execute: 'terminal',
  run_bash: 'terminal',
  exec: 'terminal',

  // Phase 2: Workflow Tools → Workflow Widget
  run_workflow: 'workflow',
  get_workflow_status: 'workflow',
  pause_workflow: 'workflow',
  resume_workflow: 'workflow',
  cancel_workflow: 'workflow',
  start_workflow: 'workflow',

  // Phase 2: Memory Tools → Memory Widget
  store_memory: 'memory',
  recall_memory: 'memory',
  search_memory: 'memory',
  delete_memory: 'memory',
  list_memories: 'memory',
  get_memory: 'memory',

  // Phase 2: File Tools → File Widget
  read_file: 'file',
  write_file: 'file',
  list_files: 'file',
  delete_file: 'file',
  move_file: 'file',
  copy_file: 'file',
  get_file_info: 'file',

  // PRD-66: Workspace Tools → Coding Canvas Widget
  workspace_file_read: 'coding_canvas',
  workspace_file_write: 'coding_canvas',
  workspace_bash: 'coding_canvas',
  workspace_shell: 'coding_canvas',
  workspace_git_clone: 'coding_canvas',
  workspace_git_commit: 'coding_canvas',
  workspace_git_push: 'coding_canvas',
  workspace_git_pull: 'coding_canvas',
  workspace_git_status: 'coding_canvas',
}

/**
 * Route a tool result to the appropriate widget type
 * @param toolName - The name of the tool that generated the result
 * @param result - The tool result data
 * @returns The widget type or null if no mapping found
 */
export function routeToolToWidget(
  toolName: string,
  result: unknown
): WidgetType | null {
  // Direct mapping check
  if (TOOL_WIDGET_MAP[toolName]) {
    return TOOL_WIDGET_MAP[toolName]
  }

  // Type-safe result inspection
  const data = result as Record<string, unknown> | null

  if (!data || typeof data !== 'object') {
    return null
  }

  // Infer from result structure
  if (Array.isArray(data.database_results) && data.database_results.length > 0) {
    return 'data'
  }
  if (Array.isArray(data.documents) && data.documents.length > 0) {
    return 'document'
  }
  if (Array.isArray(data.code_snippets) && data.code_snippets.length > 0) {
    return 'code'
  }
  if (Array.isArray(data.images) && data.images.length > 0) {
    return 'image'
  }

  // Check for specific data patterns
  if (data.sql || data.columns) return 'data'
  if (data.code || data.language) return 'code'
  if (data.content && (data.chunks || data.similarity !== undefined)) return 'document'
  if (data.base64 || (data.src && typeof data.src === 'string' && data.src.startsWith('data:image'))) {
    return 'image'
  }

  // No silent default - warn and return null for explicit handling
  console.warn(
    `[Widget Router] Unknown tool result format. Tool: "${toolName}", ` +
    `Result keys: ${Object.keys(data).join(', ')}. ` +
    `Add mapping to TOOL_WIDGET_MAP or update inference logic.`
  )
  return null
}

/**
 * Generate a unique widget ID
 */
function generateWidgetId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `widget-${crypto.randomUUID()}`
  }
  // Fallback for environments without crypto.randomUUID
  const randomBytes = new Uint8Array(8)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(randomBytes)
  } else {
    // Last resort fallback
    for (let i = 0; i < randomBytes.length; i++) {
      randomBytes[i] = Math.floor(Math.random() * 256)
    }
  }
  const hex = Array.from(randomBytes, (b) => b.toString(16).padStart(2, '0')).join('')
  return `widget-${Date.now().toString(36)}-${hex}`
}

/**
 * Transform tool result to Code widget data
 */
function transformToCodeWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<CodeWidgetData>, 'id'> {
  const data: CodeWidgetData = {
    code: (result.code as string) || (result.content as string) || '',
    language: (result.language as string) || 'python',
    filePath: result.file_path as string | undefined,
    lineNumber: result.line_number as number | undefined,
    explanation: result.explanation as string | undefined,
    symbolName: result.symbol_name as string | undefined,
  }

  return {
    type: 'code',
    title: (result.symbol_name as string) || (result.file_path as string) || 'Code',
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: 'codegraph',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform tool result to Data widget data
 */
function transformToDataWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<DataWidgetData>, 'id'> {
  // Handle pandas_ai charts
  const pandasAi = result.pandas_ai as Record<string, unknown> | undefined
  const charts: ChartData[] = []

  if (pandasAi?.charts && Array.isArray(pandasAi.charts)) {
    pandasAi.charts.forEach((chart: Record<string, unknown>) => {
      if (chart.base64) {
        charts.push({
          filename: (chart.filename as string) || 'chart.png',
          mimeType: (chart.mime_type as string) || 'image/png',
          base64: chart.base64 as string,
        })
      }
    })
  }

  const data: DataWidgetData = {
    columns: (result.columns as string[]) || [],
    rows: (result.data as Record<string, unknown>[]) || (result.rows as Record<string, unknown>[]) || [],
    sql: result.sql as string | undefined,
    database: result.database as string | undefined,
    rowCount: (result.row_count as number) || (result.data as unknown[])?.length || 0,
    executionTime: result.execution_time_ms as number | undefined,
    charts,
    pandasAiSummary: pandasAi?.summary as string | undefined,
    explanation: result.explanation as string | undefined,
    rephrased_query: result.rephrased_query as string | undefined,
    follow_up_questions: result.follow_up_questions as string[] | undefined,
  }

  return {
    type: 'data',
    title: (result.database as string) || 'Query Result',
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: 'nl2sql',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform tool result to Document widget data
 */
function transformToDocumentWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<DocumentWidgetData>, 'id'> {
  // Handle chunks
  const chunks: DocumentChunk[] = []
  if (Array.isArray(result.chunks)) {
    result.chunks.forEach((chunk: Record<string, unknown>) => {
      chunks.push({
        content: (chunk.content as string) || (chunk.excerpt as string) || '',
        excerpt: chunk.excerpt as string | undefined,
        similarity: chunk.similarity as number | undefined,
        chunkIndex: chunk.chunk_index as number | undefined,
      })
    })
  }

  const data: DocumentWidgetData = {
    content:
      (result.full_content as string) ||
      (result.content as string) ||
      (result.preview as string) ||
      (result.excerpt as string) ||
      '',
    format: 'markdown',
    filename: result.filename as string | undefined,
    filePath: result.file_path as string | undefined,
    similarity: (result.relevance as number) ?? (result.similarity as number),
    chunkCount: result.chunk_count as number | undefined,
    chunks: chunks.length > 0 ? chunks : undefined,
    downloadUrl: result.download_url as string | undefined,
    hasFullContent: result.has_full_content as boolean | undefined,
  }

  return {
    type: 'document',
    title: (result.title as string) || (result.filename as string) || 'Document',
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: 'rag',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform tool result to Image widget data
 */
function transformToImageWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<ImageWidgetData>, 'id'> {
  // Determine image source
  let src = ''
  if (result.base64) {
    const mimeType = (result.mime_type as string) || 'image/png'
    src = `data:${mimeType};base64,${result.base64}`
  } else if (result.url) {
    src = result.url as string
  } else if (result.src) {
    src = result.src as string
  }

  const data: ImageWidgetData = {
    src,
    alt: (result.alt as string) || (result.prompt as string) || 'Image',
    width: result.width as number | undefined,
    height: result.height as number | undefined,
    mimeType: result.mime_type as string | undefined,
    prompt: result.prompt as string | undefined,
    model: result.model as string | undefined,
  }

  return {
    type: 'image',
    title: (result.filename as string) || 'Image',
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: result.model ? 'image-gen' : 'upload',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform tool result to Email widget data
 */
function transformToEmailWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<EmailWidgetData>, 'id'> {
  // Determine mode based on result structure
  let mode: 'list' | 'view' | 'compose' = 'list'
  if (result.email || result.message) {
    mode = 'view'
  } else if (result.draft) {
    mode = 'compose'
  }

  // Parse emails for list mode
  const emails: EmailSummary[] = []
  if (Array.isArray(result.emails) || Array.isArray(result.messages)) {
    const emailList = (result.emails || result.messages) as Record<string, unknown>[]
    emailList.forEach((email) => {
      emails.push({
        id: (email.id as string) || '',
        from: {
          email: (email.from_email as string) || (email.from as string) || '',
          name: email.from_name as string | undefined,
        },
        to: Array.isArray(email.to)
          ? email.to.map((t: unknown) => ({
              email: typeof t === 'string' ? t : (t as Record<string, unknown>).email as string,
              name: typeof t === 'object' ? (t as Record<string, unknown>).name as string : undefined,
            }))
          : [{ email: (email.to as string) || '' }],
        subject: (email.subject as string) || '(No subject)',
        snippet: (email.snippet as string) || (email.preview as string) || '',
        date: (email.date as string) || (email.received_at as string) || new Date().toISOString(),
        isRead: (email.is_read as boolean) ?? true,
        hasAttachments: (email.has_attachments as boolean) ?? false,
        labels: email.labels as string[] | undefined,
      })
    })
  }

  const data: EmailWidgetData = {
    mode,
    emails: emails.length > 0 ? emails : undefined,
    totalCount: (result.total_count as number) || emails.length,
    unreadCount: result.unread_count as number | undefined,
  }

  return {
    type: 'email',
    title: mode === 'list' ? 'Inbox' : (result.subject as string) || 'Email',
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: toolName.includes('GMAIL') ? 'gmail' : 'email',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform tool result to Terminal widget data
 */
function transformToTerminalWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<TerminalWidgetData>, 'id'> {
  const data: TerminalWidgetData = {
    command: (result.command as string) || (result.cmd as string) || '',
    output: (result.output as string) || (result.stdout as string) || '',
    exitCode: (result.exit_code as number) ?? (result.exitCode as number) ?? (result.return_code as number),
    executionTime: (result.execution_time as number) || (result.duration_ms as number),
    workingDirectory: (result.working_directory as string) || (result.cwd as string),
    isStreaming: (result.is_streaming as boolean) ?? false,
  }

  // Combine stdout and stderr if both present
  if (result.stderr && typeof result.stderr === 'string') {
    data.output = data.output + (data.output ? '\n' : '') + result.stderr
  }

  return {
    type: 'terminal',
    title: `$ ${data.command.split(' ')[0]}`,
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: 'terminal',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform tool result to Workflow widget data
 */
function transformToWorkflowWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<WorkflowWidgetData>, 'id'> {
  // Parse workflow steps
  const steps: WorkflowStep[] = []
  if (Array.isArray(result.steps)) {
    result.steps.forEach((step: Record<string, unknown>) => {
      steps.push({
        id: (step.id as string) || `step-${steps.length}`,
        name: (step.name as string) || 'Step',
        type: (step.type as WorkflowStep['type']) || 'action',
        status: (step.status as WorkflowStep['status']) || 'pending',
        startedAt: step.started_at as string | undefined,
        completedAt: step.completed_at as string | undefined,
        duration: step.duration as number | undefined,
        result: step.result,
        error: step.error as string | undefined,
      })
    })
  }

  const data: WorkflowWidgetData = {
    workflowId: (result.workflow_id as string) || (result.id as string) || '',
    workflowName: (result.workflow_name as string) || (result.name as string) || 'Workflow',
    status: (result.status as WorkflowStatus) || 'pending',
    steps,
    startedAt: result.started_at as string | undefined,
    completedAt: result.completed_at as string | undefined,
    error: result.error as string | undefined,
    result: result.result,
    variables: result.variables as Record<string, unknown> | undefined,
  }

  return {
    type: 'workflow',
    title: data.workflowName,
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: 'workflow',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform tool result to Memory widget data
 */
function transformToMemoryWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<MemoryWidgetData>, 'id'> {
  // Parse memories
  const parseMemory = (m: Record<string, unknown>): Memory => ({
    id: (m.id as string) || `mem-${Date.now()}`,
    type: (m.type as Memory['type']) || 'fact',
    content: (m.content as string) || '',
    source: {
      conversationId: m.conversation_id as string | undefined,
      timestamp: (m.timestamp as string) || new Date().toISOString(),
      trigger: m.trigger as string | undefined,
    },
    relevance: m.relevance as number | undefined,
    metadata: m.metadata as Record<string, unknown> | undefined,
  })

  const injectedMemories: Memory[] = []
  const storedMemories: Memory[] = []

  if (Array.isArray(result.injected_memories)) {
    result.injected_memories.forEach((m: Record<string, unknown>) => {
      injectedMemories.push(parseMemory(m))
    })
  }
  if (Array.isArray(result.stored_memories)) {
    result.stored_memories.forEach((m: Record<string, unknown>) => {
      storedMemories.push(parseMemory(m))
    })
  }
  if (Array.isArray(result.memories)) {
    result.memories.forEach((m: Record<string, unknown>) => {
      if (m.relevance !== undefined) {
        injectedMemories.push(parseMemory(m))
      } else {
        storedMemories.push(parseMemory(m))
      }
    })
  }

  const data: MemoryWidgetData = {
    mode: 'all',
    injectedMemories: injectedMemories.length > 0 ? injectedMemories : undefined,
    storedMemories: storedMemories.length > 0 ? storedMemories : undefined,
    totalMemories: (result.total_count as number) || injectedMemories.length + storedMemories.length,
  }

  return {
    type: 'memory',
    title: 'Memory',
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: 'memory',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform tool result to File widget data
 */
function transformToFileWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<FileWidgetData>, 'id'> {
  // Parse file info
  const parseFileInfo = (f: Record<string, unknown>): FileInfo => ({
    name: (f.name as string) || (f.filename as string) || 'Unknown',
    path: (f.path as string) || (f.file_path as string) || '',
    type: (f.is_directory as boolean) || (f.type as string) === 'directory' ? 'directory' : 'file',
    size: (f.size as number) || 0,
    mimeType: f.mime_type as string | undefined,
    createdAt: f.created_at as string | undefined,
    modifiedAt: f.modified_at as string | undefined,
    permissions: f.permissions as string | undefined,
  })

  // Determine mode
  let mode: 'single' | 'list' | 'preview' = 'single'
  if (Array.isArray(result.files) || Array.isArray(result.entries)) {
    mode = 'list'
  } else if (result.content || result.preview) {
    mode = 'preview'
  }

  const files: FileInfo[] = []
  if (Array.isArray(result.files)) {
    result.files.forEach((f: Record<string, unknown>) => {
      files.push(parseFileInfo(f))
    })
  }
  if (Array.isArray(result.entries)) {
    result.entries.forEach((f: Record<string, unknown>) => {
      files.push(parseFileInfo(f))
    })
  }

  const file = result.file
    ? parseFileInfo(result.file as Record<string, unknown>)
    : result.path
    ? parseFileInfo(result as Record<string, unknown>)
    : undefined

  // Determine preview type
  let previewType: FileWidgetData['previewType']
  if (result.content) {
    const mimeType = (result.mime_type as string) || ''
    if (mimeType.startsWith('image/')) previewType = 'image'
    else if (mimeType === 'application/pdf') previewType = 'pdf'
    else if (mimeType.includes('text') || mimeType.includes('json') || mimeType.includes('xml'))
      previewType = 'text'
    else previewType = 'code'
  }

  const data: FileWidgetData = {
    mode,
    file,
    files: files.length > 0 ? files : undefined,
    currentPath: result.current_path as string | undefined,
    previewContent: result.content as string | undefined,
    previewType,
  }

  return {
    type: 'file',
    title: file?.name || 'Files',
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: 'file',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Transform a tool result into widget data
 * @param type - The widget type to transform to
 * @param toolName - The name of the tool
 * @param result - The tool result data
 * @param conversationId - The conversation ID
 * @param toolCallId - The tool call ID
 * @returns Widget data without ID, or null if transformation fails
 */
export function transformToolResultToWidget(
  type: WidgetType,
  toolName: string,
  result: unknown,
  conversationId: string,
  toolCallId: string
): Omit<Widget, 'id'> | null {
  if (!result || typeof result !== 'object') {
    return null
  }

  const data = result as Record<string, unknown>

  const metadata: WidgetMetadata = {
    source: {
      type: 'tool',
      name: toolName,
    },
    createdAt: new Date(),
    conversationId,
    toolCallId,
  }

  switch (type) {
    case 'code':
      return transformToCodeWidget(toolName, data, metadata)

    case 'data':
      return transformToDataWidget(toolName, data, metadata)

    case 'document':
      return transformToDocumentWidget(toolName, data, metadata)

    case 'image':
      return transformToImageWidget(toolName, data, metadata)

    // Phase 2 widgets
    case 'email':
      return transformToEmailWidget(toolName, data, metadata)

    case 'terminal':
      return transformToTerminalWidget(toolName, data, metadata)

    case 'workflow':
      return transformToWorkflowWidget(toolName, data, metadata)

    case 'memory':
      return transformToMemoryWidget(toolName, data, metadata)

    case 'file':
      return transformToFileWidget(toolName, data, metadata)

    case 'coding_canvas':
      return transformToCodingCanvasWidget(toolName, data, metadata)

    default:
      console.warn(`[Widget Router] No transformer for widget type: ${type}`)
      return null
  }
}

/**
 * Create a complete widget from a tool result
 * Combines routing, transformation, and ID generation
 */
export function createWidgetFromToolResult(
  toolName: string,
  result: unknown,
  conversationId: string,
  toolCallId: string
): Widget | null {
  const widgetType = routeToolToWidget(toolName, result)

  if (!widgetType) {
    return null
  }

  const widgetData = transformToolResultToWidget(
    widgetType,
    toolName,
    result,
    conversationId,
    toolCallId
  )

  if (!widgetData) {
    return null
  }

  return {
    ...widgetData,
    id: generateWidgetId(),
  } as Widget
}

/**
 * Transform tool result to CodingCanvas widget data (PRD-66)
 *
 * Convergence logic: if a widget already exists for this taskId,
 * the caller should update it instead of creating a new one.
 */
function transformToCodingCanvasWidget(
  toolName: string,
  result: Record<string, unknown>,
  metadata: WidgetMetadata
): Omit<Widget<CodingCanvasWidgetData>, 'id'> {
  const workspaceId =
    (result.workspace_id as string) || metadata.workspaceId || ''
  const taskId = (result.task_id as string) || (result.correlation_id as string)

  const data: CodingCanvasWidgetData = {
    workspaceId,
    taskId,
    activeFilePath: result.file_path as string | undefined,
  }

  // If the tool result includes a file event, attach it for live updating
  if (result.file_path || result.path) {
    const eventType = toolName.includes('write') ? 'file_write' as const
      : toolName.includes('git') ? 'git_operation' as const
      : toolName.includes('bash') || toolName.includes('shell') ? 'stdout_chunk' as const
      : 'file_read' as const

    data.lastEvent = {
      type: eventType,
      path: (result.file_path as string) || (result.path as string),
      timestamp: new Date().toISOString(),
    }
  }

  return {
    type: 'coding_canvas',
    title: taskId ? `Workspace` : 'Code Canvas',
    data,
    metadata: {
      ...metadata,
      source: {
        ...metadata.source,
        provider: 'workspace',
      },
    },
    state: 'ready',
    createdAt: new Date().toISOString(),
  }
}

/**
 * Add a tool to widget mapping at runtime
 * Useful for plugins or dynamic tool registration
 */
export function addToolMapping(toolName: string, widgetType: WidgetType): void {
  TOOL_WIDGET_MAP[toolName] = widgetType
}

/**
 * Remove a tool mapping
 */
export function removeToolMapping(toolName: string): boolean {
  if (TOOL_WIDGET_MAP[toolName]) {
    delete TOOL_WIDGET_MAP[toolName]
    return true
  }
  return false
}

/**
 * Get the current tool to widget mappings
 */
export function getToolMappings(): Readonly<Record<string, WidgetType>> {
  return { ...TOOL_WIDGET_MAP }
}
