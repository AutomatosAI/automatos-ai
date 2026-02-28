'use client'

import * as React from 'react'
import { Loader2 } from 'lucide-react'
import { toast } from 'sonner'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { useWorkspaceStore } from '@/stores/workspace-store'

// ── Constants ───────────────────────────────────────────────────────────
const NAME_MAX = 255
const DESC_MAX = 500

// ── Props ───────────────────────────────────────────────────────────────
export interface WorkspaceSaveDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  currentName?: string
  currentDescription?: string
}

/**
 * US-008 — WorkspaceSaveDialog
 *
 * Modal for saving the current workspace. When an existing workspace is loaded
 * (currentWorkspaceId is set), the user can either overwrite it ("Save") or
 * create a brand-new copy ("Save as New"). For fresh workspaces the dialog
 * shows a single "Save" button that creates a new workspace.
 */
export function WorkspaceSaveDialog({
  open,
  onOpenChange,
  currentName = '',
  currentDescription = '',
}: WorkspaceSaveDialogProps) {
  const currentWorkspaceId = useWorkspaceStore((s) => s.currentWorkspaceId)
  const saveWorkspace = useWorkspaceStore((s) => s.saveWorkspace)
  const isSaving = useWorkspaceStore((s) => s.isSaving)

  const [name, setName] = React.useState(currentName)
  const [description, setDescription] = React.useState(currentDescription)

  // Keep fields in sync when the dialog opens with new prop values
  React.useEffect(() => {
    if (open) {
      setName(currentName)
      setDescription(currentDescription)
    }
  }, [open, currentName, currentDescription])

  const nameError = name.trim().length === 0

  // ── Save helpers ────────────────────────────────────────────────────

  /** Overwrite the current workspace (PUT) */
  async function handleSave() {
    if (nameError) return
    try {
      await saveWorkspace(name.trim(), description.trim() || undefined)
      toast.success('Workspace saved')
      onOpenChange(false)
    } catch {
      toast.error('Failed to save workspace')
    }
  }

  /** Create a brand-new workspace (POST), clearing currentWorkspaceId first */
  async function handleSaveAsNew() {
    if (nameError) return
    // Temporarily clear the id so saveWorkspace() issues a POST
    useWorkspaceStore.setState({ currentWorkspaceId: null })
    try {
      await saveWorkspace(name.trim(), description.trim() || undefined)
      toast.success('Workspace saved as new')
      onOpenChange(false)
    } catch {
      toast.error('Failed to save workspace')
    }
  }

  const isExisting = Boolean(currentWorkspaceId)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>
            {isExisting ? 'Save Workspace' : 'Save New Workspace'}
          </DialogTitle>
          <DialogDescription>
            {isExisting
              ? 'Update the current workspace or save a copy with a new name.'
              : 'Give your workspace a name and optional description.'}
          </DialogDescription>
        </DialogHeader>

        {/* ── Form fields ──────────────────────────────────────────── */}
        <div className="grid gap-4 py-2">
          {/* Name */}
          <div className="grid gap-2">
            <Label htmlFor="ws-name">
              Name <span className="text-destructive">*</span>
            </Label>
            <Input
              id="ws-name"
              placeholder="My Workspace"
              maxLength={NAME_MAX}
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isSaving}
              autoFocus
            />
            <p className="text-xs text-muted-foreground text-right">
              {name.length}/{NAME_MAX}
            </p>
          </div>

          {/* Description */}
          <div className="grid gap-2">
            <Label htmlFor="ws-desc">Description</Label>
            <Textarea
              id="ws-desc"
              placeholder="Optional description..."
              maxLength={DESC_MAX}
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              disabled={isSaving}
            />
            <p className="text-xs text-muted-foreground text-right">
              {description.length}/{DESC_MAX}
            </p>
          </div>
        </div>

        {/* ── Actions ──────────────────────────────────────────────── */}
        <DialogFooter>
          {isExisting ? (
            <>
              <Button
                variant="outline"
                onClick={handleSaveAsNew}
                disabled={isSaving || nameError}
              >
                {isSaving ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : null}
                Save as New
              </Button>
              <Button
                onClick={handleSave}
                disabled={isSaving || nameError}
              >
                {isSaving ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : null}
                Save
              </Button>
            </>
          ) : (
            <Button
              onClick={handleSave}
              disabled={isSaving || nameError}
            >
              {isSaving ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : null}
              Save
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
