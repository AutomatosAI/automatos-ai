'use client'

/**
 * WidgetWrapper Component for PRD-38.1 Widget Architecture
 *
 * Wrapper component that resolves widget type from registry and renders
 * the appropriate widget component with all necessary props.
 */

import { useCallback, useMemo, useState, useEffect } from 'react'
import { getWidget } from './registry'
import { WidgetBase } from './WidgetBase'
import { useWorkspaceStore } from '@/stores/workspace-store'
import type { Widget } from './types'
import { AlertCircle, X, Minimize2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface WidgetWrapperProps {
  widget: Widget
  isActive?: boolean
}

export function WidgetWrapper({ widget, isActive }: WidgetWrapperProps) {
  const removeWidget = useWorkspaceStore((s) => s.removeWidget)
  const setActiveWidget = useWorkspaceStore((s) => s.setActiveWidget)
  const bringToFront = useWorkspaceStore((s) => s.bringToFront)
  const [isFullscreen, setIsFullscreen] = useState(false)

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
    setIsFullscreen(true)
  }, [])

  const handleMinimize = useCallback(() => {
    setIsFullscreen(false)
  }, [])

  // Reset fullscreen when widget becomes inactive or component unmounts
  useEffect(() => {
    return () => {
      setIsFullscreen(false)
    }
  }, [])

  // Handle unknown widget type
  if (!definition) {
    return (
      <div onClick={handleClick} className="h-full">
        <WidgetBase
          title="Unknown Widget"
          icon={<AlertCircle className="h-4 w-4 text-destructive" />}
          isActive={isActive}
          error={{ message: `This panel type isn't available in this build: ${widget.type}` }}
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

  // Single render path — CSS-based fullscreen so the component never unmounts
  return (
    <div
      onClick={!isFullscreen ? handleClick : undefined}
      className={cn(
        "h-full",
        isFullscreen && "fixed inset-0 z-[100] bg-background flex flex-col"
      )}
    >
      {isFullscreen && (
        <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
          <h2 className="text-sm font-medium">{widget.title}</h2>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={handleMinimize}
              title="Exit fullscreen"
            >
              <Minimize2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 hover:bg-destructive/10 hover:text-destructive"
              onClick={handleClose}
              title="Close"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
      <div className={cn(isFullscreen ? "flex-1 overflow-hidden" : "h-full")}>
        <WidgetComponent
          id={widget.id}
          type={widget.type}
          title={widget.title}
          data={widget.data}
          metadata={widget.metadata}
          isActive={isFullscreen || isActive}
          isLoading={widget.state === 'loading'}
          error={widget.state === 'error' ? widget.error : null}
          onClose={handleClose}
          onMaximize={isFullscreen ? handleMinimize : handleMaximize}
        />
      </div>
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
