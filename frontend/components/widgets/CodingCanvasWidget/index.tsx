'use client'

/**
 * CodingCanvasWidget — Thin wrapper around WorkspaceExplorer for chat embedding
 *
 * Delegates all file browsing/editing to WorkspaceExplorer.
 * This widget adds WidgetBase chrome (close, maximize, refresh) and
 * forwards task streaming events.
 */

import { useCallback } from 'react'
import { Code2 } from 'lucide-react'

import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import type {
  WidgetBaseProps,
  WidgetDefinition,
  CodingCanvasWidgetData,
} from '../types'

import { WorkspaceExplorer } from '../../workspace/WorkspaceExplorer'
import { useWorkspaceFiles } from './useWorkspaceFiles'

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CodingCanvasWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading: externalLoading,
  error: externalError,
  onClose,
  onMaximize,
}: WidgetBaseProps<CodingCanvasWidgetData>) {
  const { workspaceId } = data

  const { invalidateCache, fetchDirectory } = useWorkspaceFiles(workspaceId)

  const handleRefresh = useCallback(() => {
    invalidateCache()
    fetchDirectory('.')
  }, [invalidateCache, fetchDirectory])

  return (
    <WidgetBase
      title={title}
      icon={<Code2 className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={externalLoading}
      error={externalError}
      onClose={onClose}
      onMaximize={onMaximize}
      canRefresh
      onRefresh={handleRefresh}
      canMaximize
      widgetId={id}
      widgetType="coding_canvas"
      contentClassName="p-0"
    >
      <WorkspaceExplorer
        workspaceId={workspaceId}
        lastEvent={data.lastEvent}
        className="h-full min-h-[300px]"
      />
    </WidgetBase>
  )
}

// ---------------------------------------------------------------------------
// Widget Definition & Registration
// ---------------------------------------------------------------------------

export const CodingCanvasWidgetDef: WidgetDefinition<CodingCanvasWidgetData> = {
  type: 'coding_canvas',
  displayName: 'Code Canvas',
  description: 'Browse and view workspace files with Monaco editor',
  icon: Code2,
  component: CodingCanvasWidget,
  defaultSize: { width: 8, height: 6 },
  minSize: { width: 4, height: 3 },
  capabilities: ['fullscreen', 'refreshable', 'resizable'],
}

registerWidget(CodingCanvasWidgetDef)
