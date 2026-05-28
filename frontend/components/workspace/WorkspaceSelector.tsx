'use client'

/**
 * WorkspaceSelector Component -- US-014
 *
 * Dropdown in the app header that shows the current workspace name,
 * recent workspaces, navigation shortcuts, and an unsaved-changes indicator.
 * Also registers a Ctrl/Cmd+S keyboard shortcut to open the save dialog.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useWorkspaceStore } from '@/stores/workspace-store'
import type { SavedWorkspaceSummary } from '@/stores/workspace-store'

import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Separator } from '@/components/ui/separator'
import { WorkspaceSaveDialog } from './WorkspaceSaveDialog'
import { cn } from '@/lib/utils'

import {
  ChevronDown,
  Plus,
  LayoutTemplate,
  FolderOpen,
  Save,
} from 'lucide-react'

// ── Props ────────────────────────────────────────────────────────────────

export interface WorkspaceSelectorProps {
  /** Called when the user clicks "All Workspaces" */
  onOpenManager?: () => void
  /** Called when the user clicks "Browse Templates" */
  onOpenTemplates?: () => void
  className?: string
}

// ── Component ────────────────────────────────────────────────────────────

export function WorkspaceSelector({
  onOpenManager,
  onOpenTemplates,
  className,
}: WorkspaceSelectorProps) {
  const currentWorkspaceId = useWorkspaceStore((s) => s.currentWorkspaceId)
  const savedWorkspaces = useWorkspaceStore((s) => s.savedWorkspaces)
  const hasUnsavedChanges = useWorkspaceStore((s) => s.hasUnsavedChanges)
  const isLoading = useWorkspaceStore((s) => s.isLoading)
  const loadWorkspaces = useWorkspaceStore((s) => s.loadWorkspaces)
  const loadWorkspace = useWorkspaceStore((s) => s.loadWorkspace)
  const clearWidgets = useWorkspaceStore((s) => s.clearWidgets)

  const [popoverOpen, setPopoverOpen] = useState(false)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)

  // ── Load workspaces list on mount ────────────────────────────────────

  useEffect(() => {
    loadWorkspaces()
  }, [loadWorkspaces])

  // ── Auto-load last workspace from localStorage on mount ──────────────

  useEffect(() => {
    if (currentWorkspaceId) {
      loadWorkspace(currentWorkspaceId)
    }
    // Run only once on mount -- currentWorkspaceId is persisted via Zustand
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Ctrl/Cmd+S keyboard shortcut ────────────────────────────────────

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault()
        setSaveDialogOpen(true)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // ── Derived data ─────────────────────────────────────────────────────

  const currentWorkspace = useMemo(() => {
    if (!currentWorkspaceId) return null
    return savedWorkspaces.find((ws) => ws.id === currentWorkspaceId) ?? null
  }, [currentWorkspaceId, savedWorkspaces])

  const recentWorkspaces = useMemo(() => {
    const sorted = [...savedWorkspaces].sort((a, b) => {
      const aDate = a.updated_at ? new Date(a.updated_at).getTime() : 0
      const bDate = b.updated_at ? new Date(b.updated_at).getTime() : 0
      return bDate - aDate
    })
    return sorted.slice(0, 5)
  }, [savedWorkspaces])

  const currentName = currentWorkspace?.name ?? 'Untitled Workspace'

  // ── Handlers ─────────────────────────────────────────────────────────

  const handleSelectWorkspace = useCallback(
    (ws: SavedWorkspaceSummary) => {
      loadWorkspace(ws.id)
      setPopoverOpen(false)
    },
    [loadWorkspace]
  )

  const handleNewWorkspace = useCallback(() => {
    // Reset to a blank workspace
    clearWidgets()
    useWorkspaceStore.setState({
      currentWorkspaceId: null,
      hasUnsavedChanges: false,
      isDirty: false,
    })
    setPopoverOpen(false)
  }, [clearWidgets])

  const handleAllWorkspaces = useCallback(() => {
    setPopoverOpen(false)
    onOpenManager?.()
  }, [onOpenManager])

  const handleBrowseTemplates = useCallback(() => {
    setPopoverOpen(false)
    onOpenTemplates?.()
  }, [onOpenTemplates])

  // ── Render ───────────────────────────────────────────────────────────

  return (
    <>
      <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className={cn(
              'gap-1.5 max-w-[220px] text-foreground/80 hover:text-foreground',
              className
            )}
          >
            {/* Unsaved changes indicator */}
            {hasUnsavedChanges && (
              <span
                className="h-2 w-2 rounded-full bg-primary flex-shrink-0"
                title="Unsaved changes"
              />
            )}

            <span className="truncate text-sm font-medium">{currentName}</span>
            <ChevronDown className="h-3.5 w-3.5 flex-shrink-0 opacity-60" />
          </Button>
        </PopoverTrigger>

        <PopoverContent align="start" className="w-64 p-0">
          {/* Recent workspaces */}
          {recentWorkspaces.length > 0 && (
            <div className="px-1 pt-1 pb-0">
              <div className="px-2 py-1.5">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Recent
                </span>
              </div>
              {recentWorkspaces.map((ws) => {
                const isActive = ws.id === currentWorkspaceId
                return (
                  <button
                    key={ws.id}
                    onClick={() => handleSelectWorkspace(ws)}
                    className={cn(
                      'flex items-center gap-2 w-full rounded-lg px-2 py-1.5 text-left text-sm',
                      'hover:bg-muted/50 transition-colors',
                      isActive && 'bg-primary/5 text-primary font-medium'
                    )}
                  >
                    <span className="truncate">{ws.name}</span>
                    {isActive && (
                      <span className="ml-auto h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                    )}
                  </button>
                )
              })}
            </div>
          )}

          {recentWorkspaces.length > 0 && (
            <Separator className="my-1" />
          )}

          {/* Action items */}
          <div className="px-1 pb-1">
            <button
              onClick={handleAllWorkspaces}
              className="flex items-center gap-2 w-full rounded-lg px-2 py-1.5 text-sm hover:bg-muted/50 transition-colors"
            >
              <FolderOpen className="h-4 w-4 text-muted-foreground" />
              All Workspaces
            </button>

            <button
              onClick={handleNewWorkspace}
              className="flex items-center gap-2 w-full rounded-lg px-2 py-1.5 text-sm hover:bg-muted/50 transition-colors"
            >
              <Plus className="h-4 w-4 text-muted-foreground" />
              New Workspace
            </button>

            <button
              onClick={handleBrowseTemplates}
              className="flex items-center gap-2 w-full rounded-lg px-2 py-1.5 text-sm hover:bg-muted/50 transition-colors"
            >
              <LayoutTemplate className="h-4 w-4 text-muted-foreground" />
              Browse Templates
            </button>

            <Separator className="my-1" />

            <button
              onClick={() => {
                setPopoverOpen(false)
                setSaveDialogOpen(true)
              }}
              className="flex items-center gap-2 w-full rounded-lg px-2 py-1.5 text-sm hover:bg-muted/50 transition-colors"
            >
              <Save className="h-4 w-4 text-muted-foreground" />
              Save Workspace
              <kbd className="ml-auto text-[10px] font-mono text-muted-foreground bg-muted/50 px-1.5 py-0.5 rounded">
                {typeof navigator !== 'undefined' &&
                navigator.platform?.includes('Mac')
                  ? '\u2318S'
                  : 'Ctrl+S'}
              </kbd>
            </button>
          </div>
        </PopoverContent>
      </Popover>

      {/* Save dialog (shared with Ctrl+S shortcut) */}
      <WorkspaceSaveDialog
        open={saveDialogOpen}
        onOpenChange={setSaveDialogOpen}
        currentName={currentWorkspace?.name}
        currentDescription={currentWorkspace?.description ?? undefined}
      />
    </>
  )
}
