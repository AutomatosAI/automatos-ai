'use client'

/**
 * WidgetWrapper Component for PRD-38.1 Widget Architecture
 *
 * Wrapper component that resolves widget type from registry and renders
 * the appropriate widget component with all necessary props.
 */

import { useCallback, useMemo } from 'react'
import { getWidget } from './registry'
import { WidgetBase } from './WidgetBase'
import { useWorkspaceStore } from '@/stores/workspace-store'
import type { Widget } from './types'
import { AlertCircle } from 'lucide-react'

interface WidgetWrapperProps {
  widget: Widget
  isActive?: boolean
}

export function WidgetWrapper({ widget, isActive }: WidgetWrapperProps) {
  const removeWidget = useWorkspaceStore((s) => s.removeWidget)
  const setActiveWidget = useWorkspaceStore((s) => s.setActiveWidget)
  const bringToFront = useWorkspaceStore((s) => s.bringToFront)

  // Get widget definition from registry
  const definition = useMemo(() => getWidget(widget.type), [widget.type])

  // Event handlers
  const handleClose = useCallback(() => {
    removeWidget(widget.id)
  }, [widget.id, removeWidget])

  const handleClick = useCallback(() => {
    setActiveWidget(widget.id)
    bringToFront(widget.id)
  }, [widget.id, setActiveWidget, bringToFront])

  const handleMaximize = useCallback(() => {
    // TODO: Implement maximize behavior (Phase 2)
    console.log('[WidgetWrapper] Maximize not yet implemented')
  }, [])

  // Handle unknown widget type
  if (!definition) {
    return (
      <div onClick={handleClick} className="h-full">
        <WidgetBase
          title="Unknown Widget"
          icon={<AlertCircle className="h-4 w-4 text-destructive" />}
          isActive={isActive}
          error={{ message: `Unknown widget type: ${widget.type}` }}
          onClose={handleClose}
          showDragHandle
        >
          <div className="p-4 text-center text-muted-foreground">
            <p className="text-sm">
              Widget type &quot;{widget.type}&quot; is not registered.
            </p>
            <p className="text-xs mt-1">
              This may be a widget from a newer version or a missing plugin.
            </p>
          </div>
        </WidgetBase>
      </div>
    )
  }

  const WidgetComponent = definition.component

  return (
    <div onClick={handleClick} className="h-full">
      <WidgetComponent
        id={widget.id}
        type={widget.type}
        title={widget.title}
        data={widget.data}
        metadata={widget.metadata}
        isActive={isActive}
        isLoading={widget.state === 'loading'}
        error={widget.state === 'error' ? widget.error : null}
        onClose={handleClose}
        onMaximize={handleMaximize}
      />
    </div>
  )
}

/**
 * WidgetRenderer - Alternative component that can render widgets by ID
 * Useful when you only have the widget ID and need to look it up from store
 */
interface WidgetRendererProps {
  widgetId: string
}

export function WidgetRenderer({ widgetId }: WidgetRendererProps) {
  const widget = useWorkspaceStore((s) => s.widgets[widgetId])
  const activeWidgetId = useWorkspaceStore((s) => s.activeWidgetId)

  if (!widget) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <p className="text-sm">Widget not found</p>
      </div>
    )
  }

  return (
    <WidgetWrapper
      widget={widget}
      isActive={activeWidgetId === widgetId}
    />
  )
}
