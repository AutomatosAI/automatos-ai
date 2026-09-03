'use client'

/**
 * CodingCanvasWidget — Code Canvas: workspace files + a live Auto (SDK) session
 * PRD-170
 *
 * Left: WorkspaceExplorer (file tree + Monaco editor). Right: the streamed Auto
 * session panel (S3 turns, S4 diff approvals, auto-accept). The session's file
 * edits live-refresh the tree (S3 AC): a `file_edit` turn is bridged to
 * WorkspaceExplorer's `lastEvent` (a `file_write` with a changing timestamp).
 */

import { useCallback, useMemo } from 'react'
import { RootPicker } from './RootPicker'
import { WORKSPACE_ROOT, canvasTitleFor } from './code-root'
import { useWorkspaceStore } from '@/stores/workspace-store'
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
import { useCanvasSession } from './useCanvasSession'
import { CanvasSessionPanel } from './CanvasSessionPanel'

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
  // PRD-235 W2: the Canvas is rooted at a folder — a ticket's session directory or
  // a repository under projects/ — and the chat's page context follows it.
  const root = data.rootPath || WORKSPACE_ROOT
  const { invalidateCache, fetchDirectory } = useWorkspaceFiles(workspaceId, root)
  const session = useCanvasSession(workspaceId)

  const handleRefresh = useCallback(() => {
    invalidateCache()
    fetchDirectory(root)
  }, [invalidateCache, fetchDirectory, root])
  const handleRootChange = useCallback((next: string) => {
    useWorkspaceStore.getState().updateWidget(id, { data: { ...data, rootPath: next }, title: canvasTitleFor(next) })
  }, [id, data])

  // Bridge the session's latest file edit to WorkspaceExplorer's refresh signal:
  // the tree re-fetches when this `timestamp` (the session's tree-refresh tick)
  // changes on an agent file edit (S3 live-refresh AC). Falls back to any
  // externally-provided event (task streaming) when the session is idle. Both
  // are normalised to WorkspaceExplorer's `{ path; type; timestamp? }` shape
  // (its timestamp is numeric; the widget-data event carries a string one).
  const lastEvent = useMemo<
    { path: string; type: string; timestamp?: number } | null
  >(() => {
    if (session.treeRefreshTick > 0) {
      const lastEdit = [...session.ui.turns].reverse().find((t) => t.kind === 'file_edit')
      if (lastEdit?.path) {
        return { path: lastEdit.path, type: 'file_write', timestamp: session.treeRefreshTick }
      }
    }
    if (data.lastEvent?.path) {
      return {
        path: data.lastEvent.path,
        type: data.lastEvent.type,
        timestamp: Date.parse(data.lastEvent.timestamp) || undefined,
      }
    }
    return null
  }, [session.treeRefreshTick, session.ui.turns, data.lastEvent])

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
      <div className="flex h-full min-h-[300px] flex-col">
      <RootPicker workspaceId={workspaceId} value={root} onChange={handleRootChange} />
      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[1fr_360px]">
        <WorkspaceExplorer
          workspaceId={workspaceId}
          rootPath={root}
          lastEvent={lastEvent}
          className="h-full min-h-[300px]"
        />
        <CanvasSessionPanel session={session} workspaceId={workspaceId} />
      </div>
      </div>
    </WidgetBase>
  )
}

// ---------------------------------------------------------------------------
// Widget Definition & Registration
// ---------------------------------------------------------------------------

export const CodingCanvasWidgetDef: WidgetDefinition<CodingCanvasWidgetData> = {
  type: 'coding_canvas',
  displayName: 'Code Canvas',
  description: 'Code with Auto: browse files, stream a live SDK session, approve diffs, commit',
  icon: Code2,
  component: CodingCanvasWidget,
  defaultSize: { width: 8, height: 6 },
  minSize: { width: 4, height: 3 },
  capabilities: ['fullscreen', 'refreshable', 'resizable'],
}

registerWidget(CodingCanvasWidgetDef)
