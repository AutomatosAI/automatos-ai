/**
 * PRD-260 D6 — which state a tool call shows Auto in, ported from Automatos
 * Play's lib/mascot/tools.ts (the "Tool names" section of its mascot spec).
 * This file is the only place that knows tool names: the backend adds tools,
 * so the lists stay here. Names from orchestrator/modules/tools/ (13 Oct 2026).
 */

export type ToolActivity = 'searching' | 'reading' | 'working'

/** Web and marketplace tools. */
export const SEARCHING_TOOLS: ReadonlySet<string> = new Set([
  'platform_web_search',
  'platform_web_fetch',
  'search_images',
  'platform_search_packages',
  'platform_browse_marketplace_agents',
  'platform_browse_marketplace_plugins',
  'platform_browse_marketplace_skills',
])

/** Document, knowledge and memory lookups. */
export const READING_TOOLS: ReadonlySet<string> = new Set([
  'platform_read_document',
  'platform_search_documents',
  'platform_grep_documents',
  'platform_list_documents',
  'search_knowledge',
  'semantic_search',
  'search_multimodal',
  'search_tables',
  'search_formulas',
  'platform_search_memory',
  'platform_browse_memories',
  'platform_search_chat_history',
  'platform_query_graph',
  'platform_codegraph_search',
])

/**
 * Tools that write, although their names match the reading pattern. They work:
 * creating or deleting a document is not reading one.
 */
export const WORKING_TOOLS: ReadonlySet<string> = new Set([
  'generate_document',
  'platform_upload_document',
  'platform_delete_document',
  'platform_reprocess_document',
  'platform_store_memory',
  'platform_delete_memory',
  'platform_codegraph_index',
  'platform_codegraph_reindex',
  'platform_codegraph_set_auto_reindex',
])

export const SEARCHING_PATTERN = /web_|browse_marketplace/
export const READING_PATTERN = /document|knowledge|memor|semantic|graph/

/** Exact names first, then the patterns; anything else is working. */
export function toolActivity(toolName: string): ToolActivity {
  if (SEARCHING_TOOLS.has(toolName)) return 'searching'
  if (READING_TOOLS.has(toolName)) return 'reading'
  if (WORKING_TOOLS.has(toolName)) return 'working'
  if (SEARCHING_PATTERN.test(toolName)) return 'searching'
  if (READING_PATTERN.test(toolName)) return 'reading'
  return 'working'
}
