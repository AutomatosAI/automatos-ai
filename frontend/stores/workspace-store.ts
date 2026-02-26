/**
 * Workspace Store for PRD-38.1 Widget Architecture
 *
 * Zustand store for managing widget state in the workspace canvas.
 * Handles widget CRUD, layout management, and chat panel integration.
 */

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type {
  Widget,
  WidgetPosition,
  WidgetSize,
  LayoutMode,
  GridConfig,
} from '@/components/widgets/types'
import {
  routeToolToWidget,
  transformToolResultToWidget,
} from '@/components/widgets/router'

/**
 * Workspace state interface
 */
interface WorkspaceState {
  // Widgets
  widgets: Record<string, Widget>
  widgetIds: string[] // Ordered list for rendering
  activeWidgetId: string | null

  // Layout
  layoutMode: LayoutMode
  gridConfig: GridConfig
  positions: Record<string, WidgetPosition>
  sizes: Record<string, WidgetSize>

  // Chat integration
  chatPanelWidth: number
  isChatCollapsed: boolean
  widgetsInContext: string[] // Widget IDs mentioned in chat

  // UI state
  isWidgetTrayOpen: boolean
  selectedWidgetForContext: string | null

  // Persistence tracking
  isDirty: boolean
  lastSaved: Date | null
}

/**
 * Workspace actions interface
 */
interface WorkspaceActions {
  // Widget CRUD
  addWidget: (widget: Omit<Widget, 'id'>) => string
  updateWidget: (id: string, updates: Partial<Widget>) => void
  removeWidget: (id: string) => void
  clearWidgets: () => void

  // Selection
  setActiveWidget: (id: string | null) => void
  bringToFront: (id: string) => void
  sendToBack: (id: string) => void

  // Layout
  setLayoutMode: (mode: LayoutMode) => void
  updatePosition: (id: string, position: WidgetPosition) => void
  updateSize: (id: string, size: WidgetSize) => void
  resetLayout: () => void

  // Chat panel
  setChatPanelWidth: (width: number) => void
  toggleChatCollapsed: () => void

  // Context (for chat references)
  addToContext: (widgetId: string) => void
  removeFromContext: (widgetId: string) => void
  clearContext: () => void

  // Tray
  toggleWidgetTray: () => void
  setWidgetTrayOpen: (open: boolean) => void

  // From tool results
  handleToolResult: (
    toolName: string,
    toolCallId: string,
    result: unknown,
    conversationId: string
  ) => string | null

  // Batch operations
  addWidgets: (widgets: Omit<Widget, 'id'>[]) => string[]
  removeWidgets: (ids: string[]) => void
}

/**
 * Generate a unique widget ID
 */
function generateWidgetId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `widget-${crypto.randomUUID()}`
  }
  const randomBytes = new Uint8Array(8)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(randomBytes)
  } else {
    for (let i = 0; i < randomBytes.length; i++) {
      randomBytes[i] = Math.floor(Math.random() * 256)
    }
  }
  const hex = Array.from(randomBytes, (b) => b.toString(16).padStart(2, '0')).join('')
  return `widget-${Date.now().toString(36)}-${hex}`
}

/**
 * Default grid configuration
 */
const DEFAULT_GRID_CONFIG: GridConfig = {
  columns: 12,
  rowHeight: 100,
  margin: [12, 12],
  containerPadding: [12, 12],
}

/**
 * Default widget sizes by type
 */
const DEFAULT_WIDGET_SIZES: Record<string, WidgetSize> = {
  code: { width: 6, height: 4 },
  data: { width: 6, height: 5 },
  document: { width: 6, height: 5 },
  image: { width: 4, height: 4 },
  default: { width: 6, height: 4 },
}

/**
 * Calculate next available position in grid
 */
function calculateNextPosition(
  existingPositions: Record<string, WidgetPosition>,
  gridColumns: number
): WidgetPosition {
  const widgetCount = Object.keys(existingPositions).length

  // Simple column-based placement
  const col = (widgetCount % 2) * 6 // Alternates between 0 and 6
  const row = Math.floor(widgetCount / 2) * 4

  return { x: col, y: row }
}

/**
 * Create the workspace store
 */
export const useWorkspaceStore = create<WorkspaceState & WorkspaceActions>()(
  persist(
    (set, get) => ({
      // Initial state
      widgets: {},
      widgetIds: [],
      activeWidgetId: null,
      layoutMode: 'grid',
      gridConfig: DEFAULT_GRID_CONFIG,
      positions: {},
      sizes: {},
      chatPanelWidth: 400,
      isChatCollapsed: false,
      widgetsInContext: [],
      isWidgetTrayOpen: true,
      selectedWidgetForContext: null,
      isDirty: false,
      lastSaved: null,

      // Widget CRUD
      addWidget: (widgetData) => {
        const id = generateWidgetId()
        const state = get()

        // Calculate position if not provided
        const position = widgetData.position || calculateNextPosition(
          state.positions,
          state.gridConfig.columns
        )

        // Get default size for widget type
        const size = widgetData.size ||
          DEFAULT_WIDGET_SIZES[widgetData.type] ||
          DEFAULT_WIDGET_SIZES.default

        const widget: Widget = {
          ...widgetData,
          id,
          state: widgetData.state || 'ready',
          createdAt: widgetData.createdAt || new Date().toISOString(),
        } as Widget

        set((state) => ({
          widgets: { ...state.widgets, [id]: widget },
          widgetIds: [...state.widgetIds, id],
          positions: { ...state.positions, [id]: position },
          sizes: { ...state.sizes, [id]: size },
          activeWidgetId: id,
          isDirty: true,
        }))

        return id
      },

      updateWidget: (id, updates) => {
        set((state) => {
          if (!state.widgets[id]) return state

          return {
            widgets: {
              ...state.widgets,
              [id]: {
                ...state.widgets[id],
                ...updates,
                updatedAt: new Date().toISOString(),
              },
            },
            isDirty: true,
          }
        })
      },

      removeWidget: (id) => {
        set((state) => {
          const { [id]: removed, ...remainingWidgets } = state.widgets
          const { [id]: removedPos, ...remainingPositions } = state.positions
          const { [id]: removedSize, ...remainingSizes } = state.sizes

          return {
            widgets: remainingWidgets,
            widgetIds: state.widgetIds.filter((wid) => wid !== id),
            positions: remainingPositions,
            sizes: remainingSizes,
            activeWidgetId: state.activeWidgetId === id ? null : state.activeWidgetId,
            widgetsInContext: state.widgetsInContext.filter((wid) => wid !== id),
            isDirty: true,
          }
        })
      },

      clearWidgets: () => {
        set({
          widgets: {},
          widgetIds: [],
          positions: {},
          sizes: {},
          activeWidgetId: null,
          widgetsInContext: [],
          isDirty: true,
        })
      },

      // Selection
      setActiveWidget: (id) => {
        set({ activeWidgetId: id })
      },

      bringToFront: (id) => {
        set((state) => {
          const index = state.widgetIds.indexOf(id)
          if (index === -1) return state

          const newOrder = [
            ...state.widgetIds.filter((wid) => wid !== id),
            id,
          ]
          return { widgetIds: newOrder }
        })
      },

      sendToBack: (id) => {
        set((state) => {
          const index = state.widgetIds.indexOf(id)
          if (index === -1) return state

          const newOrder = [
            id,
            ...state.widgetIds.filter((wid) => wid !== id),
          ]
          return { widgetIds: newOrder }
        })
      },

      // Layout
      setLayoutMode: (mode) => {
        set({ layoutMode: mode, isDirty: true })
      },

      updatePosition: (id, position) => {
        set((state) => ({
          positions: { ...state.positions, [id]: position },
          isDirty: true,
        }))
      },

      updateSize: (id, size) => {
        set((state) => ({
          sizes: { ...state.sizes, [id]: size },
          isDirty: true,
        }))
      },

      resetLayout: () => {
        const state = get()
        const newPositions: Record<string, WidgetPosition> = {}
        const newSizes: Record<string, WidgetSize> = {}

        state.widgetIds.forEach((id, index) => {
          const widget = state.widgets[id]
          const col = (index % 2) * 6
          const row = Math.floor(index / 2) * 4

          newPositions[id] = { x: col, y: row }
          newSizes[id] = DEFAULT_WIDGET_SIZES[widget?.type || 'default'] || DEFAULT_WIDGET_SIZES.default
        })

        set({
          positions: newPositions,
          sizes: newSizes,
          isDirty: true,
        })
      },

      // Chat panel
      setChatPanelWidth: (width) => {
        set({ chatPanelWidth: Math.max(300, Math.min(800, width)) })
      },

      toggleChatCollapsed: () => {
        set((state) => ({ isChatCollapsed: !state.isChatCollapsed }))
      },

      // Context
      addToContext: (widgetId) => {
        set((state) => {
          if (state.widgetsInContext.includes(widgetId)) return state
          return {
            widgetsInContext: [...state.widgetsInContext, widgetId],
          }
        })
      },

      removeFromContext: (widgetId) => {
        set((state) => ({
          widgetsInContext: state.widgetsInContext.filter((id) => id !== widgetId),
        }))
      },

      clearContext: () => {
        set({ widgetsInContext: [] })
      },

      // Tray
      toggleWidgetTray: () => {
        set((state) => ({ isWidgetTrayOpen: !state.isWidgetTrayOpen }))
      },

      setWidgetTrayOpen: (open) => {
        set({ isWidgetTrayOpen: open })
      },

      // Tool result handling
      handleToolResult: (toolName, toolCallId, result, conversationId) => {
        const widgetType = routeToolToWidget(toolName, result)

        if (!widgetType) {
          console.info(
            `[Workspace Store] No widget type for tool "${toolName}". Result not rendered.`
          )
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
          console.warn(
            `[Workspace Store] Failed to transform result for widget type "${widgetType}"`
          )
          return null
        }

        return get().addWidget(widgetData)
      },

      // Batch operations
      addWidgets: (widgetsData) => {
        const ids: string[] = []

        widgetsData.forEach((widgetData) => {
          const id = get().addWidget(widgetData)
          ids.push(id)
        })

        return ids
      },

      removeWidgets: (ids) => {
        ids.forEach((id) => get().removeWidget(id))
      },
    }),
    {
      name: 'automatos-workspace',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        // Persist layout preferences
        layoutMode: state.layoutMode,
        chatPanelWidth: state.chatPanelWidth,
        isChatCollapsed: state.isChatCollapsed,
        isWidgetTrayOpen: state.isWidgetTrayOpen,
        gridConfig: state.gridConfig,
        // Note: Widget data is NOT persisted by default
        // This can be enabled in Phase 3 for workspace persistence
      }),
    }
  )
)

/**
 * Selector hooks for common state slices
 */
export const useWidgets = () => useWorkspaceStore((state) => state.widgets)
export const useWidgetIds = () => useWorkspaceStore((state) => state.widgetIds)
export const useActiveWidgetId = () => useWorkspaceStore((state) => state.activeWidgetId)
export const useLayoutMode = () => useWorkspaceStore((state) => state.layoutMode)
export const useIsChatCollapsed = () => useWorkspaceStore((state) => state.isChatCollapsed)
export const useChatPanelWidth = () => useWorkspaceStore((state) => state.chatPanelWidth)
export const useIsWidgetTrayOpen = () => useWorkspaceStore((state) => state.isWidgetTrayOpen)

/**
 * Get a specific widget by ID
 */
export const useWidget = (id: string) =>
  useWorkspaceStore((state) => state.widgets[id])

/**
 * Get widget position
 */
export const useWidgetPosition = (id: string) =>
  useWorkspaceStore((state) => state.positions[id])

/**
 * Get widget size
 */
export const useWidgetSize = (id: string) =>
  useWorkspaceStore((state) => state.sizes[id])
