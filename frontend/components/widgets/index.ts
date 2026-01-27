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

// Widget Components - importing these will auto-register them
export { CodeWidget, CodeWidgetDef } from './CodeWidget'
export { DataWidget, DataWidgetDef } from './DataWidget'
export { DocumentWidget, DocumentWidgetDef } from './DocumentWidget'
export { ImageWidget, ImageWidgetDef } from './ImageWidget'

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
