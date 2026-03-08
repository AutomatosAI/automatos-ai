'use client'

/**
 * CodingCanvasWidget — Monaco-based workspace file browser + code viewer
 * PRD-66 Phase 1: Code Viewer Widget (read-only)
 *
 * Compound layout: FileExplorer (left) | CodeEditor (right)
 * Uses react-resizable-panels for the split.
 */

import { useCallback, useEffect, useState } from 'react'
import { Code2, GitBranch, Terminal } from 'lucide-react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { Button } from '@/components/ui/button'

import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import type {
  WidgetBaseProps,
  WidgetDefinition,
  CodingCanvasWidgetData,
  OpenFileTab,
} from '../types'

import { FileExplorer } from './FileExplorer'
import { CodeEditor } from './CodeEditor'
import { EditorTabs } from './EditorTabs'
import { useWorkspaceFiles } from './useWorkspaceFiles'
import { RepoSelector } from './RepoSelector'
import { InteractiveTerminal } from '../TerminalWidget/InteractiveTerminal'

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

  // File system hook
  const {
    tree,
    isLoadingTree,
    treeError,
    fetchDirectory,
    fetchFileContent,
    invalidateCache,
  } = useWorkspaceFiles(workspaceId)

  // Editor state
  const [openTabs, setOpenTabs] = useState<OpenFileTab[]>(data.openFiles ?? [])
  const [activeTabPath, setActiveTabPath] = useState<string | null>(
    data.activeFilePath ?? null
  )

  // Repo selector dialog
  const [repoSelectorOpen, setRepoSelectorOpen] = useState(false)

  // Terminal panel
  const [showTerminal, setShowTerminal] = useState(false)

  // Ctrl+` to toggle terminal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === '`') {
        e.preventDefault()
        setShowTerminal(prev => !prev)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Fetch root directory on mount
  useEffect(() => {
    if (workspaceId) {
      fetchDirectory('.')
    }
  }, [workspaceId, fetchDirectory])

  // When a new file event comes in from task streaming, auto-open it
  useEffect(() => {
    if (data.lastEvent?.path && data.lastEvent.type === 'file_write') {
      handleFileSelect(data.lastEvent.path)
      invalidateCache()
      fetchDirectory('.')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.lastEvent?.timestamp])

  // ------ Handlers ------

  const handleFileSelect = useCallback(
    async (filePath: string) => {
      // If already open AND has content, just activate it
      const existing = openTabs.find((t) => t.path === filePath)
      if (existing && !existing.isLoading && existing.content != null) {
        setActiveTabPath(filePath)
        return
      }

      // Create a loading tab (or keep existing one)
      if (!existing) {
        const name = filePath.split('/').pop() || filePath
        const loadingTab: OpenFileTab = {
          path: filePath,
          name,
          language: 'plaintext',
          isLoading: true,
        }
        setOpenTabs((prev) => [...prev, loadingTab])
      }
      setActiveTabPath(filePath)

      // Fetch content (also re-fetches if tab was stuck loading)
      const result = await fetchFileContent(filePath)
      if (result) {
        setOpenTabs((prev) =>
          prev.map((t) => (t.path === filePath ? result : t))
        )
      } else {
        // Remove the tab on failure
        setOpenTabs((prev) => prev.filter((t) => t.path !== filePath))
        setActiveTabPath((prev) =>
          prev === filePath ? null : prev
        )
      }
    },
    [openTabs, fetchFileContent]
  )

  const handleDirectoryToggle = useCallback(
    (dirPath: string) => {
      fetchDirectory(dirPath)
    },
    [fetchDirectory]
  )

  const handleCloseTab = useCallback(
    (path: string) => {
      setOpenTabs((prev) => {
        const remaining = prev.filter((t) => t.path !== path)
        // If we're closing the active tab, switch to the last remaining tab
        setActiveTabPath((active) =>
          active === path
            ? remaining.length > 0 ? remaining[remaining.length - 1].path : null
            : active
        )
        return remaining
      })
    },
    []
  )

  const handleRefresh = useCallback(() => {
    invalidateCache()
    fetchDirectory('.')
  }, [invalidateCache, fetchDirectory])

  const handleCloneStarted = useCallback(
    (_taskId: string) => {
      // Refresh the file tree after a short delay to allow clone to start
      setTimeout(() => {
        invalidateCache()
        fetchDirectory('.')
      }, 3000)
    },
    [invalidateCache, fetchDirectory]
  )

  // Active file for the editor
  const activeFile = openTabs.find((t) => t.path === activeTabPath) ?? null

  // Determine if workspace is empty (no files/dirs loaded, no error, not loading)
  const isWorkspaceEmpty = !isLoadingTree && !treeError && tree.length === 0

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
      <PanelGroup direction="horizontal" className="h-full min-h-[300px]">
        {/* File Explorer Panel */}
        <Panel defaultSize={25} minSize={15} maxSize={50}>
          <div className="h-full border-r border-border/30 bg-muted/10 overflow-hidden flex flex-col">
            <div className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground border-b border-border/20">
              Explorer
            </div>
            <div className="flex-1 overflow-y-auto">
              {isWorkspaceEmpty ? (
                <div className="flex flex-col items-center justify-center h-full gap-3 p-4 text-center">
                  <GitBranch className="h-8 w-8 text-muted-foreground/50" />
                  <p className="text-xs text-muted-foreground">No files in workspace</p>
                  <button
                    onClick={() => setRepoSelectorOpen(true)}
                    className="px-3 py-1.5 text-xs font-medium rounded-md border border-border hover:bg-secondary transition-colors text-foreground"
                  >
                    Connect Repo
                  </button>
                </div>
              ) : (
                <FileExplorer
                  entries={tree}
                  isLoading={isLoadingTree}
                  error={treeError}
                  onFileSelect={handleFileSelect}
                  onDirectoryToggle={handleDirectoryToggle}
                  selectedPath={activeTabPath}
                />
              )}
            </div>
          </div>
        </Panel>

        {/* Repo Selector Dialog */}
        <RepoSelector
          workspaceId={workspaceId}
          open={repoSelectorOpen}
          onOpenChange={setRepoSelectorOpen}
          onCloneStarted={handleCloneStarted}
        />

        {/* Resize Handle */}
        <PanelResizeHandle className="w-[3px] bg-border/30 hover:bg-primary/40 transition-colors cursor-col-resize" />

        {/* Editor + Terminal Panel */}
        <Panel defaultSize={75} minSize={40}>
          <PanelGroup direction="vertical" className="h-full">
            {/* Editor section */}
            <Panel defaultSize={showTerminal ? 70 : 100} minSize={30}>
              <div className="h-full flex flex-col overflow-hidden">
                {/* Tab bar with terminal toggle */}
                <div className="flex items-center">
                  <div className="flex-1 min-w-0">
                    <EditorTabs
                      tabs={openTabs}
                      activeTabPath={activeTabPath}
                      onSelectTab={setActiveTabPath}
                      onCloseTab={handleCloseTab}
                    />
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 mr-1"
                    onClick={() => setShowTerminal(prev => !prev)}
                    title={`${showTerminal ? 'Hide' : 'Show'} Terminal (Ctrl+\`)`}
                  >
                    <Terminal className={`h-3.5 w-3.5 ${showTerminal ? 'text-primary' : 'text-muted-foreground'}`} />
                  </Button>
                </div>

                {/* Monaco editor */}
                <div className="flex-1 min-h-0">
                  <CodeEditor file={activeFile} />
                </div>
              </div>
            </Panel>

            {/* Terminal panel (conditional) */}
            {showTerminal && (
              <>
                <PanelResizeHandle className="h-[3px] bg-border/30 hover:bg-primary/40 transition-colors cursor-row-resize" />
                <Panel defaultSize={30} minSize={10}>
                  <InteractiveTerminal workspaceId={workspaceId} className="h-full" />
                </Panel>
              </>
            )}
          </PanelGroup>
        </Panel>
      </PanelGroup>
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
