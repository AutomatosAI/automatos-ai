/**
 * Widget Registry for PRD-38.1 Widget Architecture
 *
 * Central registry for all widget types. This allows dynamic widget
 * registration and lookup, enabling extensibility for future phases.
 */

import type { WidgetType, WidgetDefinition } from './types'

/**
 * Internal registry map
 */
const widgetRegistry = new Map<WidgetType, WidgetDefinition>()

/**
 * Register a widget definition
 * @param definition - The widget definition to register
 */
export function registerWidget<TData = unknown>(
  definition: WidgetDefinition<TData>
): void {
  if (widgetRegistry.has(definition.type)) {
    console.warn(
      `[Widget Registry] Widget type "${definition.type}" is already registered. Overwriting.`
    )
  }
  widgetRegistry.set(definition.type, definition as WidgetDefinition)
}

/**
 * Unregister a widget type
 * @param type - The widget type to unregister
 */
export function unregisterWidget(type: WidgetType): boolean {
  return widgetRegistry.delete(type)
}

/**
 * Get a widget definition by type
 * @param type - The widget type to look up
 * @returns The widget definition or undefined if not found
 */
export function getWidget(type: WidgetType): WidgetDefinition | undefined {
  return widgetRegistry.get(type)
}

/**
 * Get all registered widget definitions
 * @returns Array of all registered widget definitions
 */
export function getAllWidgets(): WidgetDefinition[] {
  return Array.from(widgetRegistry.values())
}

/**
 * Get all registered widget types
 * @returns Array of all registered widget type strings
 */
export function getRegisteredTypes(): WidgetType[] {
  return Array.from(widgetRegistry.keys())
}

/**
 * Check if a widget type is registered
 * @param type - The widget type to check
 * @returns True if the widget type is registered
 */
export function isWidgetRegistered(type: WidgetType): boolean {
  return widgetRegistry.has(type)
}

/**
 * Get widgets by capability
 * @param capability - The capability to filter by
 * @returns Array of widget definitions that have the specified capability
 */
export function getWidgetsByCapability(
  capability: string
): WidgetDefinition[] {
  return getAllWidgets().filter((def) =>
    def.capabilities.includes(capability as any)
  )
}

/**
 * Clear all registered widgets (useful for testing)
 */
export function clearRegistry(): void {
  widgetRegistry.clear()
}

/**
 * Initialize default widgets
 * This is called at app startup to register the Phase 1 widgets
 */
export function initializeDefaultWidgets(): void {
  // Widget registrations are done via imports in index.ts
  // This function can be used for any additional initialization
  console.log(
    `[Widget Registry] Initialized with ${widgetRegistry.size} widget types`
  )
}
