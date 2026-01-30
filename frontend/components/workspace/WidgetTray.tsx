'use client'

/**
 * WidgetTray Component for PRD-38.1 Widget Architecture
 *
 * Bottom tray showing thumbnails of all active widgets.
 * Allows quick navigation and widget management.
 */

import { useWorkspaceStore } from '@/stores/workspace-store'
import { getWidget } from '@/components/widgets/registry'
import { Button } from '@/components/ui/button'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import {
  ChevronUp,
  ChevronDown,
  X,
  Code2,
  Database,
  FileText,
  Image,
  Trash2,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { WidgetType } from '@/components/widgets/types'

/**
 * Get icon for widget type
 */
function getWidgetIcon(type: WidgetType) {
  switch (type) {
    case 'code':
      return <Code2 className="h-3.5 w-3.5" />
    case 'data':
      return <Database className="h-3.5 w-3.5" />
    case 'document':
      return <FileText className="h-3.5 w-3.5" />
    case 'image':
      return <Image className="h-3.5 w-3.5" />
    default:
      return <FileText className="h-3.5 w-3.5" />
  }
}

export function WidgetTray() {
  const widgets = useWorkspaceStore((s) => s.widgets)
  const widgetIds = useWorkspaceStore((s) => s.widgetIds)
  const activeWidgetId = useWorkspaceStore((s) => s.activeWidgetId)
  const isWidgetTrayOpen = useWorkspaceStore((s) => s.isWidgetTrayOpen)
  const toggleWidgetTray = useWorkspaceStore((s) => s.toggleWidgetTray)
  const setActiveWidget = useWorkspaceStore((s) => s.setActiveWidget)
  const removeWidget = useWorkspaceStore((s) => s.removeWidget)
  const clearWidgets = useWorkspaceStore((s) => s.clearWidgets)

  // Don't render if no widgets
  if (widgetIds.length === 0) return null

  return (
    <div className="border-t border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      {/* Header with toggle */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border/30">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-foreground">
            Widgets
          </span>
          <span className="text-xs text-muted-foreground bg-muted/50 px-2 py-0.5 rounded-full">
            {widgetIds.length}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {/* Clear all button */}
          {widgetIds.length > 1 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs text-muted-foreground hover:text-destructive"
              onClick={() => {
                if (confirm('Clear all widgets?')) {
                  clearWidgets()
                }
              }}
            >
              <Trash2 className="h-3.5 w-3.5 mr-1" />
              Clear
            </Button>
          )}

          {/* Toggle button */}
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            onClick={toggleWidgetTray}
          >
            {isWidgetTrayOpen ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronUp className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Widget list */}
      {isWidgetTrayOpen && (
        <ScrollArea className="h-20">
          <div className="flex gap-2 p-3">
            {widgetIds.map((id) => {
              const widget = widgets[id]
              if (!widget) return null

              const isActive = activeWidgetId === id

              return (
                <div
                  key={id}
                  className={cn(
                    'group relative flex items-center gap-2 px-3 py-2 rounded-lg',
                    'bg-muted/40 hover:bg-muted/70 cursor-pointer transition-all duration-150',
                    'min-w-[140px] max-w-[200px] flex-shrink-0',
                    'border border-transparent',
                    isActive && 'ring-2 ring-primary/50 bg-primary/5 border-primary/20'
                  )}
                  onClick={() => setActiveWidget(id)}
                >
                  {/* Icon */}
                  <span
                    className={cn(
                      'flex-shrink-0 text-muted-foreground',
                      isActive && 'text-primary'
                    )}
                  >
                    {getWidgetIcon(widget.type)}
                  </span>

                  {/* Title */}
                  <span
                    className={cn(
                      'text-sm truncate flex-1',
                      isActive ? 'text-foreground font-medium' : 'text-foreground/80'
                    )}
                  >
                    {widget.title}
                  </span>

                  {/* Close button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    className={cn(
                      'h-5 w-5 p-0 opacity-0 group-hover:opacity-100 transition-opacity',
                      'hover:bg-destructive/10 hover:text-destructive'
                    )}
                    onClick={(e) => {
                      e.stopPropagation()
                      removeWidget(id)
                    }}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              )
            })}
          </div>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      )}
    </div>
  )
}
