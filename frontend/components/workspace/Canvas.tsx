'use client'

/**
 * Canvas Component for PRD-38.1 Widget Architecture
 *
 * VS Code-style tabbed interface for viewing widgets.
 * Shows one widget at a time with tabs to switch between them.
 */

import { useCallback } from 'react'
import { useWorkspaceStore } from '@/stores/workspace-store'
import { WidgetWrapper } from '@/components/widgets/WidgetWrapper'
import { cn } from '@/lib/utils'
import { Inbox, X, FileText, Database, Code2, Image, Mail, Terminal, GitBranch, Brain, File, PanelLeftClose } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'

interface CanvasProps {
  className?: string
  width?: number
  onClose?: () => void
}

// Icon mapping for widget types
const WIDGET_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  code: Code2,
  data: Database,
  document: FileText,
  image: Image,
  email: Mail,
  terminal: Terminal,
  workflow: GitBranch,
  memory: Brain,
  file: File,
}

export function Canvas({ className, width = 1200, onClose }: CanvasProps) {
  const widgets = useWorkspaceStore((s) => s.widgets)
  const widgetIds = useWorkspaceStore((s) => s.widgetIds)
  const activeWidgetId = useWorkspaceStore((s) => s.activeWidgetId)
  const setActiveWidget = useWorkspaceStore((s) => s.setActiveWidget)
  const removeWidget = useWorkspaceStore((s) => s.removeWidget)

  // Get active widget, fallback to first if none selected
  const effectiveActiveId = activeWidgetId && widgets[activeWidgetId]
    ? activeWidgetId
    : widgetIds[0] || null
  const activeWidget = effectiveActiveId ? widgets[effectiveActiveId] : null

  // Handle tab close
  const handleCloseTab = useCallback((e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    removeWidget(id)

    // If closing active tab, switch to another
    if (id === effectiveActiveId) {
      const currentIndex = widgetIds.indexOf(id)
      const nextId = widgetIds[currentIndex + 1] || widgetIds[currentIndex - 1] || null
      setActiveWidget(nextId)
    }
  }, [effectiveActiveId, widgetIds, removeWidget, setActiveWidget])

  // Keyboard navigation across the tab list (WAI-ARIA tabs, automatic activation)
  const handleTabKeyDown = useCallback((e: React.KeyboardEvent) => {
    const idx = widgetIds.indexOf(effectiveActiveId as string)
    if (idx < 0) return
    let nextIdx: number | null = null
    if (e.key === 'ArrowRight') nextIdx = (idx + 1) % widgetIds.length
    else if (e.key === 'ArrowLeft') nextIdx = (idx - 1 + widgetIds.length) % widgetIds.length
    else if (e.key === 'Home') nextIdx = 0
    else if (e.key === 'End') nextIdx = widgetIds.length - 1
    if (nextIdx === null) return
    e.preventDefault()
    const nextId = widgetIds[nextIdx]
    setActiveWidget(nextId)
    requestAnimationFrame(() => document.getElementById(`canvas-tab-${nextId}`)?.focus())
  }, [widgetIds, effectiveActiveId, setActiveWidget])

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
        </div>
      </div>
    )
  }

  return (
    <div className={cn('h-full flex flex-col bg-background', className)}>
      {/* Tab Bar - VS Code style */}
      <div className="flex-shrink-0 border-b border-border bg-muted/30">
        <div className="flex items-center">
          <ScrollArea className="flex-1">
            <div className="flex" role="tablist" aria-label="Open widgets" onKeyDown={handleTabKeyDown}>
            {widgetIds.map((id) => {
              const widget = widgets[id]
              if (!widget) return null

              const isActive = id === effectiveActiveId
              const Icon = WIDGET_ICONS[widget.type] || FileText

              return (
                <div
                  key={id}
                  role="tab"
                  id={`canvas-tab-${id}`}
                  aria-selected={isActive}
                  aria-controls="canvas-tabpanel"
                  tabIndex={isActive ? 0 : -1}
                  onClick={() => setActiveWidget(id)}
                  className={cn(
                    'group flex items-center gap-2 px-3 py-2 border-r border-border cursor-pointer',
                    'hover:bg-muted/50 transition-colors min-w-[120px] max-w-[200px]',
                    'focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-inset',
                    isActive
                      ? 'bg-background border-b-2 border-b-primary'
                      : 'bg-muted/20'
                  )}
                >
                  <Icon className={cn(
                    'h-4 w-4 flex-shrink-0',
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  )} />
                  <span className={cn(
                    'text-sm truncate flex-1',
                    isActive ? 'text-foreground font-medium' : 'text-muted-foreground'
                  )}>
                    {widget.title}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Close ${widget.title}`}
                    className={cn(
                      'h-5 w-5 p-0 opacity-60 hover:opacity-100 hover:bg-destructive/20',
                      'transition-opacity'
                    )}
                    onClick={(e) => handleCloseTab(e, id)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              )
            })}
            </div>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>

          {/* Close Canvas Button */}
          {onClose && (
            <div className="flex-shrink-0 border-l border-border px-2 py-1">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-muted-foreground hover:text-foreground hover:bg-destructive/10"
                onClick={onClose}
                title="Close all widgets and return to chat"
              >
                <PanelLeftClose className="h-4 w-4 mr-1" />
                <span className="text-xs">Close</span>
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Active Widget Content - Full Size */}
      <div
        className="flex-1 overflow-hidden"
        role="tabpanel"
        id="canvas-tabpanel"
        aria-labelledby={effectiveActiveId ? `canvas-tab-${effectiveActiveId}` : undefined}
        tabIndex={0}
      >
        {activeWidget ? (
          <div className="h-full">
            <WidgetWrapper
              widget={activeWidget}
              isActive={true}
            />
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            <p>Select a tab to view content</p>
          </div>
        )}
      </div>
    </div>
  )
}
