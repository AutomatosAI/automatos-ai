/**
 * Widget System Entry Point for PRD-38.1 Widget Architecture
 *
 * This file exports all widget-related modules and initializes
 * the widget registry with default widgets.
 */

// Types
export * from './types'

// Registry
export {
  registerWidget,
  unregisterWidget,
  getWidget,
  getAllWidgets,
  getRegisteredTypes,
  isWidgetRegistered,
  getWidgetsByCapability,
  clearRegistry,
  initializeDefaultWidgets,
} from './registry'

// Router
export {
  routeToolToWidget,
  transformToolResultToWidget,
  createWidgetFromToolResult,
  addToolMapping,
  removeToolMapping,
  getToolMappings,
} from './router'

// Base Components
export { WidgetBase } from './WidgetBase'
export { WidgetWrapper, WidgetRenderer } from './WidgetWrapper'

// Phase 1 Widget Components - importing these will auto-register them
export { CodeWidget, CodeWidgetDef } from './CodeWidget'
export { DataWidget, DataWidgetDef } from './DataWidget'
export { DocumentWidget, DocumentWidgetDef } from './DocumentWidget'
export { ImageWidget, ImageWidgetDef } from './ImageWidget'

// Phase 2 Widget Components (PRD-38.2)
export { EmailWidget, EmailWidgetDef } from './EmailWidget'
export { TerminalWidget, TerminalWidgetDef } from './TerminalWidget'
export { WorkflowWidget, WorkflowWidgetDef } from './WorkflowWidget'
export { MemoryWidget, MemoryWidgetDef } from './MemoryWidget'
export { FileWidget, FileWidgetDef } from './FileWidget'

// Re-export store selectors for convenience
export {
  useWorkspaceStore,
  useWidgets,
  useWidgetIds,
  useActiveWidgetId,
  useLayoutMode,
  useIsChatCollapsed,
  useChatPanelWidth,
  useIsWidgetTrayOpen,
  useWidget,
  useWidgetPosition,
  useWidgetSize,
} from '@/stores/workspace-store'
