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
  ChartData,
  DocumentChunk,
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

  // Phase 2+ tools (placeholders)
  GMAIL_SEND_EMAIL: 'email',
  GMAIL_LIST_EMAILS: 'email',
  OUTLOOK_SEND_EMAIL: 'email',
  execute_command: 'terminal',
  run_script: 'terminal',
  run_workflow: 'workflow',
  get_workflow_status: 'workflow',
  read_file: 'file',
  write_file: 'file',
  list_files: 'file',
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
