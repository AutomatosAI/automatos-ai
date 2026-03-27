'use client'

/**
 * WorkspaceExplorer — Standalone workspace file browser + code viewer + terminal
 *
 * Extracted from CodingCanvasWidget (PRD-66) so it can be used both as:
 * 1. A standalone page at /workspace
 * 2. Embedded in chat via CodingCanvasWidget (WidgetBase wrapper)
 *
 * Compound layout: FileExplorer (left) | CodeEditor + Terminal (right)
 * Uses react-resizable-panels for the split.
 */

import { useCallback, useEffect, useState } from 'react'
import { GitBranch, Terminal } from 'lucide-react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { Button } from '@/components/ui/button'

import { FileExplorer } from '../widgets/CodingCanvasWidget/FileExplorer'
import { CodeEditor } from '../widgets/CodingCanvasWidget/CodeEditor'
import { EditorTabs } from '../widgets/CodingCanvasWidget/EditorTabs'
import { useWorkspaceFiles } from '../widgets/CodingCanvasWidget/useWorkspaceFiles'
import { RepoSelector } from '../widgets/CodingCanvasWidget/RepoSelector'
import { InteractiveTerminal } from '../widgets/TerminalWidget/InteractiveTerminal'
import type { OpenFileTab } from '../widgets/types'

export interface WorkspaceExplorerProps {
  workspaceId: string
  /** File event from task streaming — auto-opens the file when set */
  lastEvent?: { path: string; type: string; timestamp?: number } | null
  /** Minimum height CSS class (default: h-full) */
  className?: string
}

export function WorkspaceExplorer({
  workspaceId,
  lastEvent,
  className = 'h-full',
}: WorkspaceExplorerProps) {
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
  const [openTabs, setOpenTabs] = useState<OpenFileTab[]>([])
  const [activeTabPath, setActiveTabPath] = useState<string | null>(null)

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
    if (lastEvent?.path && lastEvent.type === 'file_write') {
      handleFileSelect(lastEvent.path, { forceReload: true })
      invalidateCache()
      fetchDirectory('.')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastEvent?.timestamp])

  // ------ Handlers ------

  const handleFileSelect = useCallback(
    async (filePath: string, options?: { forceReload?: boolean }) => {
      const existing = openTabs.find((t) => t.path === filePath)
      if (!options?.forceReload && existing && !existing.isLoading && existing.content != null) {
        setActiveTabPath(filePath)
        return
      }

      setOpenTabs((prev) => {
        if (prev.find((t) => t.path === filePath)) return prev
        const name = filePath.split('/').pop() || filePath
        return [...prev, { path: filePath, name, language: 'plaintext', isLoading: true }]
      })
      setActiveTabPath(filePath)

      const result = await fetchFileContent(filePath)
      if (result) {
        setOpenTabs((prev) =>
          prev.map((t) => (t.path === filePath ? result : t))
        )
      } else {
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

  const handleCloneStarted = useCallback(
    (_taskId: string) => {
      setTimeout(() => {
        invalidateCache()
        fetchDirectory('.')
      }, 3000)
    },
    [invalidateCache, fetchDirectory]
  )

  const activeFile = openTabs.find((t) => t.path === activeTabPath) ?? null
  const isWorkspaceEmpty = !isLoadingTree && !treeError && tree.length === 0

  return (
    <div className={className}>
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
    </div>
  )
}
