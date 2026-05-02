'use client'

/**
 * WorkspaceManager Component — US-007
 *
 * Lists saved workspaces grouped by Recent and My Workspaces.
 * Provides create, rename, delete, and load actions.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useWorkspaceStore } from '@/stores/workspace-store'
import type { SavedWorkspaceSummary } from '@/stores/workspace-store'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import {
  LayoutGrid,
  Columns,
  Maximize2,
  Rows,
  MoreHorizontal,
  Plus,
  Trash2,
  Pencil,
  FolderOpen,
  Inbox,
} from 'lucide-react'

// ── Layout mode icon mapping ─────────────────────────────────────────

const LAYOUT_MODE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  grid: LayoutGrid,
  split: Columns,
  focus: Maximize2,
  freeform: Rows,
}

function LayoutModeIcon({ mode, className }: { mode: string; className?: string }) {
  const Icon = LAYOUT_MODE_ICONS[mode] || LayoutGrid
  return <Icon className={className} />
}

// ── Relative time helper ─────────────────────────────────────────────

function relativeTime(dateStr: string | null | undefined): string {
  if (!dateStr) return ''

  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSec = Math.floor(diffMs / 1000)
  const diffMin = Math.floor(diffSec / 60)
  const diffHr = Math.floor(diffMin / 60)
  const diffDay = Math.floor(diffHr / 24)

  if (diffSec < 60) return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHr < 24) return `${diffHr}h ago`
  if (diffDay < 7) return `${diffDay}d ago`
  if (diffDay < 30) return `${Math.floor(diffDay / 7)}w ago`
  return date.toLocaleDateString()
}

// ── WorkspaceManager ─────────────────────────────────────────────────

interface WorkspaceManagerProps {
  className?: string
}

export function WorkspaceManager({ className }: WorkspaceManagerProps) {
  const savedWorkspaces = useWorkspaceStore((s) => s.savedWorkspaces)
  const isLoading = useWorkspaceStore((s) => s.isLoading)
  const isSaving = useWorkspaceStore((s) => s.isSaving)
  const currentWorkspaceId = useWorkspaceStore((s) => s.currentWorkspaceId)
  const loadWorkspaces = useWorkspaceStore((s) => s.loadWorkspaces)
  const loadWorkspace = useWorkspaceStore((s) => s.loadWorkspace)
  const saveWorkspace = useWorkspaceStore((s) => s.saveWorkspace)
  const deleteWorkspace = useWorkspaceStore((s) => s.deleteWorkspace)

  // Dialog state
  const [isNewDialogOpen, setIsNewDialogOpen] = useState(false)
  const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false)
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false)

  // Form state
  const [newName, setNewName] = useState('')
  const [newDescription, setNewDescription] = useState('')

  // Target workspace for rename/delete
  const [targetWorkspace, setTargetWorkspace] = useState<SavedWorkspaceSummary | null>(null)

  // Fetch workspaces on mount
  useEffect(() => {
    loadWorkspaces()
  }, [loadWorkspaces])

  // ── Grouped workspaces ───────────────────────────────────────────

  const { recentWorkspaces, allWorkspaces } = useMemo(() => {
    // Sort by updated_at descending for "all"
    const sorted = [...savedWorkspaces].sort((a, b) => {
      const aDate = a.updated_at ? new Date(a.updated_at).getTime() : 0
      const bDate = b.updated_at ? new Date(b.updated_at).getTime() : 0
      return bDate - aDate
    })

    // Recent: last 5 by updated_at (proxy for last_opened_at)
    const recent = sorted.slice(0, 5)

    return { recentWorkspaces: recent, allWorkspaces: sorted }
  }, [savedWorkspaces])

  // ── Handlers ─────────────────────────────────────────────────────

  const handleCreateWorkspace = useCallback(async () => {
    if (!newName.trim()) return
    try {
      await saveWorkspace(newName.trim(), newDescription.trim() || undefined)
      setIsNewDialogOpen(false)
      setNewName('')
      setNewDescription('')
    } catch {
      // Error is logged in the store
    }
  }, [newName, newDescription, saveWorkspace])

  const handleRenameWorkspace = useCallback(async () => {
    if (!targetWorkspace || !newName.trim()) return
    // Load, rename via save with updated name, then refresh
    // For now we re-save which updates the name
    try {
      await loadWorkspace(targetWorkspace.id)
      await saveWorkspace(newName.trim(), newDescription.trim() || undefined)
      setIsRenameDialogOpen(false)
      setTargetWorkspace(null)
      setNewName('')
      setNewDescription('')
    } catch {
      // Error is logged in the store
    }
  }, [targetWorkspace, newName, newDescription, loadWorkspace, saveWorkspace])

  const handleDeleteWorkspace = useCallback(async () => {
    if (!targetWorkspace) return
    try {
      await deleteWorkspace(targetWorkspace.id)
      setIsDeleteDialogOpen(false)
      setTargetWorkspace(null)
    } catch {
      // Error is logged in the store
    }
  }, [targetWorkspace, deleteWorkspace])

  const handleOpenWorkspace = useCallback(
    (id: string) => {
      loadWorkspace(id)
    },
    [loadWorkspace]
  )

  const openRenameDialog = useCallback((ws: SavedWorkspaceSummary) => {
    setTargetWorkspace(ws)
    setNewName(ws.name)
    setNewDescription(ws.description ?? '')
    setIsRenameDialogOpen(true)
  }, [])

  const openDeleteDialog = useCallback((ws: SavedWorkspaceSummary) => {
    setTargetWorkspace(ws)
    setIsDeleteDialogOpen(true)
  }, [])

  // ── Render helpers ───────────────────────────────────────────────

  function renderWorkspaceRow(ws: SavedWorkspaceSummary) {
    const isActive = ws.id === currentWorkspaceId

    return (
      <div
        key={ws.id}
        className={cn(
          'group flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer',
          'hover:bg-muted/50 transition-colors',
          isActive && 'bg-primary/5 ring-1 ring-primary/20'
        )}
        onClick={() => handleOpenWorkspace(ws.id)}
      >
        {/* Layout mode icon */}
        <div
          className={cn(
            'flex h-8 w-8 items-center justify-center rounded-lg bg-muted/50 flex-shrink-0',
            isActive && 'bg-primary/10'
          )}
        >
          <LayoutModeIcon
            mode={ws.layout_mode}
            className={cn(
              'h-4 w-4 text-muted-foreground',
              isActive && 'text-primary'
            )}
          />
        </div>

        {/* Name + description */}
        <div className="flex-1 min-w-0">
          <div
            className={cn(
              'text-sm font-medium truncate',
              isActive ? 'text-foreground' : 'text-foreground/90'
            )}
          >
            {ws.name}
          </div>
          {ws.description && (
            <div className="text-xs text-muted-foreground truncate mt-0.5">
              {ws.description.length > 60
                ? ws.description.slice(0, 60) + '...'
                : ws.description}
            </div>
          )}
        </div>

        {/* Relative time */}
        <span className="text-xs text-muted-foreground flex-shrink-0 hidden sm:block">
          {relativeTime(ws.updated_at)}
        </span>

        {/* Action menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-7 w-7 flex-shrink-0 opacity-0 group-hover:opacity-100',
                'transition-opacity'
              )}
              onClick={(e) => e.stopPropagation()}
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-40">
            <DropdownMenuItem
              onClick={(e) => {
                e.stopPropagation()
                openRenameDialog(ws)
              }}
            >
              <Pencil className="h-4 w-4 mr-2" />
              Rename
            </DropdownMenuItem>
            <DropdownMenuItem
              className="text-destructive focus:text-destructive"
              onClick={(e) => {
                e.stopPropagation()
                openDeleteDialog(ws)
              }}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    )
  }

  // ── Loading state ────────────────────────────────────────────────

  if (isLoading && savedWorkspaces.length === 0) {
    return (
      <div className={cn('flex flex-col gap-3 p-4', className)}>
        <div className="flex items-center justify-between mb-2">
          <Skeleton className="h-5 w-32" />
          <Skeleton className="h-9 w-36" />
        </div>
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    )
  }

  // ── Empty state ──────────────────────────────────────────────────

  if (savedWorkspaces.length === 0 && !isLoading) {
    return (
      <div className={cn('flex flex-col', className)}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-4 pb-2">
          <h2 className="text-sm font-semibold text-foreground">Workspaces</h2>
          <Button
            size="sm"
            className="gap-1.5"
            onClick={() => {
              setNewName('')
              setNewDescription('')
              setIsNewDialogOpen(true)
            }}
          >
            <Plus className="h-4 w-4" />
            New Workspace
          </Button>
        </div>

        <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-muted/50 mb-4">
            <Inbox className="h-8 w-8 text-muted-foreground/50" />
          </div>
          <h3 className="text-base font-medium text-foreground mb-1">
            No workspaces yet
          </h3>
          <p className="text-sm text-muted-foreground max-w-xs">
            Create your first workspace to start organizing widgets, layouts,
            and conversations.
          </p>
          <Button
            className="mt-4 gap-1.5"
            onClick={() => {
              setNewName('')
              setNewDescription('')
              setIsNewDialogOpen(true)
            }}
          >
            <Plus className="h-4 w-4" />
            Create Workspace
          </Button>
        </div>

        {/* New workspace dialog */}
        {renderNewWorkspaceDialog()}
      </div>
    )
  }

  // ── Main list ────────────────────────────────────────────────────

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-4 pb-2 flex-shrink-0">
        <div className="flex items-center gap-2">
          <FolderOpen className="h-4 w-4 text-muted-foreground" />
          <h2 className="text-sm font-semibold text-foreground">Workspaces</h2>
          <span className="text-xs text-muted-foreground bg-muted/50 px-2 py-0.5 rounded-full">
            {savedWorkspaces.length}
          </span>
        </div>
        <Button
          size="sm"
          className="gap-1.5"
          disabled={isSaving}
          onClick={() => {
            setNewName('')
            setNewDescription('')
            setIsNewDialogOpen(true)
          }}
        >
          <Plus className="h-4 w-4" />
          New Workspace
        </Button>
      </div>

      {/* Workspace list */}
      <ScrollArea className="flex-1 px-2">
        <div className="flex flex-col gap-1 pb-4">
          {/* Recent section */}
          {recentWorkspaces.length > 0 && (
            <>
              <div className="px-2 pt-3 pb-1">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Recent
                </span>
              </div>
              {recentWorkspaces.map(renderWorkspaceRow)}
            </>
          )}

          {/* All workspaces section */}
          {allWorkspaces.length > 5 && (
            <>
              <div className="px-2 pt-4 pb-1">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  My Workspaces
                </span>
              </div>
              {allWorkspaces.slice(5).map(renderWorkspaceRow)}
            </>
          )}
        </div>
      </ScrollArea>

      {/* Dialogs */}
      {renderNewWorkspaceDialog()}
      {renderRenameDialog()}
      {renderDeleteDialog()}
    </div>
  )

  // ── Dialog renderers ─────────────────────────────────────────────

  function renderNewWorkspaceDialog() {
    return (
      <Dialog open={isNewDialogOpen} onOpenChange={setIsNewDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Workspace</DialogTitle>
            <DialogDescription>
              Create a new workspace to organize your widgets and layouts.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-4 py-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="ws-name">Name</Label>
              <Input
                id="ws-name"
                placeholder="My Workspace"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && newName.trim()) {
                    handleCreateWorkspace()
                  }
                }}
                autoFocus
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="ws-desc">Description (optional)</Label>
              <Textarea
                id="ws-desc"
                placeholder="What is this workspace for?"
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                rows={2}
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsNewDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              disabled={!newName.trim() || isSaving}
              onClick={handleCreateWorkspace}
            >
              {isSaving ? 'Creating...' : 'Create'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    )
  }

  function renderRenameDialog() {
    return (
      <Dialog open={isRenameDialogOpen} onOpenChange={setIsRenameDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename Workspace</DialogTitle>
            <DialogDescription>
              Update the name and description for this workspace.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-4 py-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="rename-name">Name</Label>
              <Input
                id="rename-name"
                placeholder="Workspace name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && newName.trim()) {
                    handleRenameWorkspace()
                  }
                }}
                autoFocus
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="rename-desc">Description (optional)</Label>
              <Textarea
                id="rename-desc"
                placeholder="What is this workspace for?"
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                rows={2}
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsRenameDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              disabled={!newName.trim() || isSaving}
              onClick={handleRenameWorkspace}
            >
              {isSaving ? 'Saving...' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    )
  }

  function renderDeleteDialog() {
    return (
      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Workspace</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &ldquo;{targetWorkspace?.name}&rdquo;?
              This action cannot be undone. All widgets and layout data in this
              workspace will be permanently removed.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setTargetWorkspace(null)}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={handleDeleteWorkspace}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    )
  }
}
