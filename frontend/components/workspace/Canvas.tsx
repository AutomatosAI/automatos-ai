'use client'

/**
 * Canvas Component for PRD-38.1 Widget Architecture
 *
 * Grid-based canvas for rendering and arranging widgets.
 * Uses react-grid-layout for drag-and-drop functionality.
 */

import { useCallback, useMemo } from 'react'
import GridLayout, { Layout } from 'react-grid-layout'
import { useWorkspaceStore } from '@/stores/workspace-store'
import { WidgetWrapper } from '@/components/widgets/WidgetWrapper'
import { cn } from '@/lib/utils'
import { LayoutGrid, Inbox } from 'lucide-react'

import 'react-grid-layout/css/styles.css'
import 'react-resizable/css/styles.css'

interface CanvasProps {
  className?: string
  width?: number
}

export function Canvas({ className, width = 1200 }: CanvasProps) {
  const widgets = useWorkspaceStore((s) => s.widgets)
  const widgetIds = useWorkspaceStore((s) => s.widgetIds)
  const activeWidgetId = useWorkspaceStore((s) => s.activeWidgetId)
  const positions = useWorkspaceStore((s) => s.positions)
  const sizes = useWorkspaceStore((s) => s.sizes)
  const updatePosition = useWorkspaceStore((s) => s.updatePosition)
  const updateSize = useWorkspaceStore((s) => s.updateSize)
  const setActiveWidget = useWorkspaceStore((s) => s.setActiveWidget)

  // Convert our state to react-grid-layout format
  const layout: Layout[] = useMemo(() => {
    return widgetIds.map((id, index) => {
      const pos = positions[id] || {
        x: (index % 2) * 6,
        y: Math.floor(index / 2) * 4,
      }
      const size = sizes[id] || { width: 6, height: 4 }

      return {
        i: id,
        x: pos.x,
        y: pos.y,
        w: size.width,
        h: size.height,
        minW: 3,
        minH: 2,
      }
    })
  }, [widgetIds, positions, sizes])

  // Handle layout changes from drag/resize
  const handleLayoutChange = useCallback(
    (newLayout: Layout[]) => {
      newLayout.forEach((item) => {
        const currentPos = positions[item.i]
        const currentSize = sizes[item.i]

        // Only update if position changed
        if (!currentPos || currentPos.x !== item.x || currentPos.y !== item.y) {
          updatePosition(item.i, { x: item.x, y: item.y })
        }

        // Only update if size changed
        if (
          !currentSize ||
          currentSize.width !== item.w ||
          currentSize.height !== item.h
        ) {
          updateSize(item.i, { width: item.w, height: item.h })
        }
      })
    },
    [positions, sizes, updatePosition, updateSize]
  )

  // Empty state
  if (widgetIds.length === 0) {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center h-full',
          'text-muted-foreground',
          className
        )}
      >
        <div className="flex flex-col items-center gap-4 p-8 text-center max-w-md">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted/50">
            <Inbox className="h-8 w-8 text-muted-foreground/50" />
          </div>
          <div>
            <h3 className="text-lg font-medium text-foreground">
              No widgets yet
            </h3>
            <p className="text-sm mt-1 text-muted-foreground">
              Ask a question about your data, documents, or code. Results will
              appear here as interactive widgets.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground/70 mt-2">
            <LayoutGrid className="h-4 w-4" />
            <span>Widgets can be dragged, resized, and arranged</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={cn('h-full overflow-auto bg-muted/20', className)}>
      <GridLayout
        className="layout"
        layout={layout}
        cols={12}
        rowHeight={100}
        width={width}
        onLayoutChange={handleLayoutChange}
        draggableHandle=".widget-drag-handle"
        resizeHandles={['se', 'sw', 'ne', 'nw', 'e', 'w', 's', 'n']}
        compactType="vertical"
        preventCollision={false}
        isResizable
        isDraggable
        margin={[12, 12]}
        containerPadding={[16, 16]}
        useCSSTransforms
      >
        {widgetIds.map((id) => {
          const widget = widgets[id]
          if (!widget) return null

          return (
            <div
              key={id}
              className={cn(
                'rounded-lg overflow-hidden transition-shadow duration-200',
                activeWidgetId === id && 'ring-2 ring-primary shadow-lg'
              )}
              onClick={() => setActiveWidget(id)}
            >
              <WidgetWrapper widget={widget} isActive={activeWidgetId === id} />
            </div>
          )
        })}
      </GridLayout>
    </div>
  )
}
