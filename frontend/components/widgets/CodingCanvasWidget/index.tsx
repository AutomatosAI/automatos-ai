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

import { useCallback, useMemo, useState } from 'react'
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
import dynamic from 'next/dynamic'
import type { CanvasTerminalProps } from './CanvasTerminal'

// PRD-239 S7: xterm.js reads `self` at import time and the chat page is still
// server-rendered, so the terminal pane only ever loads in the browser.
const CanvasTerminal = dynamic<CanvasTerminalProps>(
  () => import('./CanvasTerminal').then((m) => m.CanvasTerminal),
  { ssr: false, loading: () => <div className="p-3 text-xs text-muted-foreground">Loading the terminal…</div> },
)

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
  const session = useCanvasSession(workspaceId, { taskId: data.taskId ?? null })
  // PRD-239 S7: a ticket's Canvas opens on the terminal; the SDK canvas on its session.
  const [rightTab, setRightTab] = useState<'terminal' | 'session'>(session.external ? 'terminal' : 'session')

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
      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[1fr_420px]">
        <WorkspaceExplorer
          workspaceId={workspaceId}
          rootPath={root}
          lastEvent={lastEvent}
          className="h-full min-h-[300px]"
        />
        {/* PRD-239 S7: the right column is a real terminal (your own shell on your
            machine, via the CLI host) or the session view — one at a time. */}
        <div className="flex h-full min-h-0 flex-col border-l border-border">
          <div className="flex items-center gap-1 border-b border-border px-2 py-1" role="tablist" data-testid="canvas-right-tabs">
            <button
              type="button"
              role="tab"
              aria-selected={rightTab === 'terminal'}
              onClick={() => setRightTab('terminal')}
              className={`rounded px-2 py-1 text-xs ${rightTab === 'terminal' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'}`}
            >
              Terminal
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={rightTab === 'session'}
              onClick={() => setRightTab('session')}
              className={`rounded px-2 py-1 text-xs ${rightTab === 'session' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'}`}
            >
              {session.external ? 'Session log' : 'Auto session'}
            </button>
          </div>
          <div className="min-h-0 flex-1">
            {rightTab === 'terminal' ? (
              <CanvasTerminal taskId={data.taskId ?? null} />
            ) : (
              <CanvasSessionPanel session={session} workspaceId={workspaceId} />
            )}
          </div>
        </div>
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
