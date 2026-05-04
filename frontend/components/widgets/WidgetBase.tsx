'use client'

/**
 * WidgetBase Component for PRD-38.1 Widget Architecture
 *
 * Base component that provides common widget chrome (header, actions, error handling).
 * All widget types use this as their wrapper for consistent look and behavior.
 */

import { ReactNode, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  X,
  Maximize2,
  Minimize2,
  RefreshCw,
  Copy,
  Download,
  MoreVertical,
  GripVertical,
  Settings,
} from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import type { WidgetMetadata, WidgetType, WidgetConfig } from './types'
import { WidgetConfigDialog } from '@/components/workspace/WidgetConfigDialog'

interface CustomAction {
  label: string
  icon?: ReactNode
  onClick: () => void
  disabled?: boolean
  destructive?: boolean
}

interface WidgetBaseProps {
  title: string
  icon?: ReactNode
  metadata?: WidgetMetadata
  isActive?: boolean
  isLoading?: boolean
  error?: Error | { message: string } | null
  children: ReactNode

  // Actions
  onClose?: () => void
  onMaximize?: () => void
  onMinimize?: () => void
  onRefresh?: () => void
  onCopy?: () => void
  onDownload?: () => void

  // Capabilities
  canCopy?: boolean
  canDownload?: boolean
  canRefresh?: boolean
  canMaximize?: boolean
  showDragHandle?: boolean

  // Custom actions
  customActions?: CustomAction[]

  // US-011: Per-widget config dialog
  widgetId?: string
  widgetType?: WidgetType
  currentConfig?: WidgetConfig

  className?: string
  contentClassName?: string
  headerClassName?: string
}

export function WidgetBase({
  title,
  icon,
  metadata,
  isActive,
  isLoading,
  error,
  children,
  onClose,
  onMaximize,
  onMinimize,
  onRefresh,
  onCopy,
  onDownload,
  canCopy,
  canDownload,
  canRefresh,
  canMaximize = true,
  showDragHandle = true,
  customActions,
  widgetId,
  widgetType,
  currentConfig,
  className,
  contentClassName,
  headerClassName,
}: WidgetBaseProps) {
  const [configOpen, setConfigOpen] = useState(false)

  const hasDropdownActions = (customActions && customActions.length > 0) ||
    (canCopy && onCopy) ||
    (canDownload && onDownload)

  return (
    <Card
      className={cn(
        'flex flex-col h-full overflow-hidden transition-all duration-220',
        'border-border/50 bg-card/95 backdrop-blur-sm',
        'shadow-sm hover:shadow-md',
        isActive && 'ring-2 ring-primary/50 shadow-lg',
        className
      )}
    >
      {/* Header */}
      <CardHeader
        className={cn(
          'flex-shrink-0 flex flex-row items-center justify-between',
          'p-3 pb-2 border-b border-border/30 bg-muted/30',
          headerClassName
        )}
      >
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {/* Drag handle */}
          {showDragHandle && (
            <div className="widget-drag-handle cursor-grab active:cursor-grabbing p-1 -ml-1 rounded hover:bg-muted/50 transition-colors">
              <GripVertical className="h-4 w-4 text-muted-foreground/50" />
            </div>
          )}

          {/* Icon */}
          {icon && (
            <span className="text-muted-foreground flex-shrink-0">{icon}</span>
          )}

          {/* Title */}
          <CardTitle className="text-sm font-medium truncate">
            {title}
          </CardTitle>

          {/* Source badge */}
          {metadata?.source && (
            <span className="hidden sm:inline-flex text-[10px] text-muted-foreground bg-muted/50 px-1.5 py-0.5 rounded font-mono">
              {metadata.source.name}
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-0.5 ml-2">
          {/* Settings gear (US-011) */}
          {widgetId && widgetType && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setConfigOpen(true)}
              title="Widget settings"
            >
              <Settings className="h-3.5 w-3.5" />
            </Button>
          )}

          {/* Refresh button */}
          {canRefresh && onRefresh && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={onRefresh}
              disabled={isLoading}
              title="Refresh"
            >
              <RefreshCw
                className={cn('h-3.5 w-3.5', isLoading && 'animate-spin')}
              />
            </Button>
          )}

          {/* Dropdown menu for additional actions */}
          {hasDropdownActions && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-7 w-7">
                  <MoreVertical className="h-3.5 w-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-40">
                {canCopy && onCopy && (
                  <DropdownMenuItem onClick={onCopy}>
                    <Copy className="h-4 w-4 mr-2" />
                    Copy
                  </DropdownMenuItem>
                )}
                {canDownload && onDownload && (
                  <DropdownMenuItem onClick={onDownload}>
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </DropdownMenuItem>
                )}
                {((canCopy && onCopy) || (canDownload && onDownload)) &&
                  customActions &&
                  customActions.length > 0 && <DropdownMenuSeparator />}
                {customActions?.map((action, i) => (
                  <DropdownMenuItem
                    key={i}
                    onClick={action.onClick}
                    disabled={action.disabled}
                    className={cn(action.destructive && 'text-destructive')}
                  >
                    {action.icon}
                    {action.label}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}

          {/* Maximize button */}
          {canMaximize && onMaximize && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={onMaximize}
              title="Maximize"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </Button>
          )}

          {/* Close button */}
          {onClose && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 hover:bg-destructive/10 hover:text-destructive"
              onClick={onClose}
              title="Close"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </CardHeader>

      {/* Content */}
      <CardContent
        className={cn('flex-1 overflow-auto p-0', contentClassName)}
      >
        {error ? (
          <div className="flex items-center justify-center h-full p-4">
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-destructive/10 mb-3">
                <X className="h-6 w-6 text-destructive" />
              </div>
              <p className="font-medium text-destructive">
                Error loading widget
              </p>
              <p className="text-sm text-muted-foreground mt-1 max-w-xs">
                {error.message || 'An unexpected error occurred'}
              </p>
              {canRefresh && onRefresh && (
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-3"
                  onClick={onRefresh}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry
                </Button>
              )}
            </div>
          </div>
        ) : isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground mx-auto" />
              <p className="text-sm text-muted-foreground mt-2">Loading...</p>
            </div>
          </div>
        ) : (
          children
        )}
      </CardContent>

      {/* US-011: Config dialog */}
      {widgetId && widgetType && (
        <WidgetConfigDialog
          open={configOpen}
          onOpenChange={setConfigOpen}
          widgetId={widgetId}
          widgetType={widgetType}
          currentConfig={currentConfig}
        />
      )}
    </Card>
  )
}
