'use client'

/**
 * US-013 — WorkspaceShareDialog
 *
 * Modal for sharing a workspace with other users. Allows the workspace owner
 * to search for users by email, assign view/edit permissions, and manage
 * existing shares (update permission or revoke access).
 */

import { useCallback, useEffect, useState } from 'react'
import { Share2, UserPlus, X, Users, Loader2 } from 'lucide-react'
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
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import apiClient from '@/lib/api-client'

// ── Types ──────────────────────────────────────────────────────────────

type SharePermission = 'view' | 'edit'

interface ShareEntry {
  id: string
  workspace_id: string
  user_id: string
  permission: string
  user_email: string | null
  user_name: string | null
  created_at: string | null
}

/** Tracks a pending change (add, update, or remove) before the user saves. */
interface PendingShare {
  user_id: string
  email: string
  name: string | null
  permission: SharePermission
  /** 'existing' shares already persisted, 'new' are about to be created */
  origin: 'existing' | 'new'
  /** Mark for removal on save */
  markedForRemoval: boolean
  /** Original permission (to detect changes) */
  originalPermission?: string
}

// ── Props ──────────────────────────────────────────────────────────────

export interface WorkspaceShareDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  workspaceId: string
}

// ── Component ──────────────────────────────────────────────────────────

export function WorkspaceShareDialog({
  open,
  onOpenChange,
  workspaceId,
}: WorkspaceShareDialogProps) {
  const [shares, setShares] = useState<PendingShare[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  // Add-user form state
  const [newEmail, setNewEmail] = useState('')
  const [newPermission, setNewPermission] = useState<SharePermission>('view')
  const [addError, setAddError] = useState<string | null>(null)

  // ── Fetch current shares on open ───────────────────────────────────

  const fetchShares = useCallback(async () => {
    if (!workspaceId) return
    setIsLoading(true)
    try {
      const data = await apiClient.get<ShareEntry[]>(
        `/api/workspaces/${workspaceId}/shares`
      )
      const mapped: PendingShare[] = (data ?? []).map((s) => ({
        user_id: s.user_id,
        email: s.user_email ?? s.user_id,
        name: s.user_name ?? null,
        permission: (s.permission === 'edit' ? 'edit' : 'view') as SharePermission,
        origin: 'existing' as const,
        markedForRemoval: false,
        originalPermission: s.permission,
      }))
      setShares(mapped)
    } catch (err) {
      console.error('[WorkspaceShareDialog] Failed to fetch shares:', err)
      // Non-fatal: the dialog can still be used to add new shares
      setShares([])
    } finally {
      setIsLoading(false)
    }
  }, [workspaceId])

  useEffect(() => {
    if (open) {
      fetchShares()
      setNewEmail('')
      setNewPermission('view')
      setAddError(null)
    }
  }, [open, fetchShares])

  // ── Add user to pending list ───────────────────────────────────────

  function handleAddUser() {
    setAddError(null)

    const trimmed = newEmail.trim()
    if (!trimmed) {
      setAddError('Enter a user email or ID')
      return
    }

    // Check duplicates
    const exists = shares.some(
      (s) =>
        !s.markedForRemoval &&
        (s.email.toLowerCase() === trimmed.toLowerCase() ||
          s.user_id === trimmed)
    )
    if (exists) {
      setAddError('This user already has access')
      return
    }

    setShares((prev) => [
      ...prev,
      {
        user_id: trimmed, // Backend will resolve; we pass email as user_id
        email: trimmed,
        name: null,
        permission: newPermission,
        origin: 'new',
        markedForRemoval: false,
      },
    ])
    setNewEmail('')
    setNewPermission('view')
  }

  // ── Update permission in pending list ──────────────────────────────

  function handlePermissionChange(index: number, perm: SharePermission) {
    setShares((prev) =>
      prev.map((s, i) => (i === index ? { ...s, permission: perm } : s))
    )
  }

  // ── Mark for removal ───────────────────────────────────────────────

  function handleRemove(index: number) {
    setShares((prev) =>
      prev.map((s, i) => {
        if (i !== index) return s
        if (s.origin === 'new') {
          // New entries can just be removed from the list entirely
          return { ...s, markedForRemoval: true }
        }
        return { ...s, markedForRemoval: true }
      })
    )
  }

  // ── Undo removal ──────────────────────────────────────────────────

  function handleUndoRemove(index: number) {
    setShares((prev) =>
      prev.map((s, i) => (i === index ? { ...s, markedForRemoval: false } : s))
    )
  }

  // ── Save changes ──────────────────────────────────────────────────

  async function handleSave() {
    setIsSaving(true)

    try {
      const promises: Promise<unknown>[] = []

      for (const share of shares) {
        if (share.markedForRemoval && share.origin === 'existing') {
          // DELETE existing share
          promises.push(
            apiClient.delete(
              `/api/workspaces/${workspaceId}/share/${share.user_id}`
            )
          )
        } else if (share.origin === 'new' && !share.markedForRemoval) {
          // POST new share
          promises.push(
            apiClient.post(`/api/workspaces/${workspaceId}/share`, {
              user_id: share.user_id,
              permission: share.permission,
            })
          )
        } else if (
          share.origin === 'existing' &&
          !share.markedForRemoval &&
          share.permission !== share.originalPermission
        ) {
          // POST to update permission (backend upserts)
          promises.push(
            apiClient.post(`/api/workspaces/${workspaceId}/share`, {
              user_id: share.user_id,
              permission: share.permission,
            })
          )
        }
      }

      await Promise.all(promises)
      toast.success('Sharing settings saved')
      onOpenChange(false)
    } catch (err) {
      console.error('[WorkspaceShareDialog] Save failed:', err)
      toast.error('Failed to save sharing settings')
    } finally {
      setIsSaving(false)
    }
  }

  // ── Derived state ──────────────────────────────────────────────────

  const visibleShares = shares.filter(
    (s) => !(s.origin === 'new' && s.markedForRemoval)
  )

  const hasChanges = shares.some(
    (s) =>
      s.origin === 'new' ||
      s.markedForRemoval ||
      (s.origin === 'existing' && s.permission !== s.originalPermission)
  )

  // ── Render ────────────────────────────────────────────────────────

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Share2 className="h-5 w-5" />
            Share Workspace
          </DialogTitle>
          <DialogDescription>
            Invite people to collaborate on this workspace. They will receive
            access based on the permission level you assign.
          </DialogDescription>
        </DialogHeader>

        {/* ── Add user section ──────────────────────────────────────── */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium">Add people</label>
          <div className="flex gap-2">
            <Input
              placeholder="Email or user ID"
              value={newEmail}
              onChange={(e) => {
                setNewEmail(e.target.value)
                setAddError(null)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  handleAddUser()
                }
              }}
              disabled={isSaving}
              className="flex-1"
            />
            <Select
              value={newPermission}
              onValueChange={(v) => setNewPermission(v as SharePermission)}
            >
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="view">Can view</SelectItem>
                <SelectItem value="edit">Can edit</SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant="outline"
              size="icon"
              onClick={handleAddUser}
              disabled={isSaving || !newEmail.trim()}
              title="Add user"
            >
              <UserPlus className="h-4 w-4" />
            </Button>
          </div>
          {addError && (
            <p className="text-xs text-destructive">{addError}</p>
          )}
        </div>

        {/* ── Current shares list ──────────────────────────────────── */}
        <div className="flex flex-col gap-1 mt-2">
          <div className="flex items-center gap-2 mb-1">
            <Users className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">People with access</span>
            {visibleShares.length > 0 && (
              <Badge variant="secondary" className="text-xs">
                {visibleShares.filter((s) => !s.markedForRemoval).length}
              </Badge>
            )}
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : visibleShares.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">
              No one else has access yet. Add people above.
            </p>
          ) : (
            <div className="flex flex-col gap-1.5 max-h-[240px] overflow-y-auto pr-1">
              {visibleShares.map((share, idx) => (
                <div
                  key={`${share.user_id}-${idx}`}
                  className={`flex items-center gap-3 rounded-xl px-3 py-2 transition-colors ${
                    share.markedForRemoval
                      ? 'bg-destructive/5 opacity-60'
                      : 'bg-muted/30 hover:bg-muted/50'
                  }`}
                >
                  {/* Avatar placeholder */}
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-semibold flex-shrink-0">
                    {(share.name ?? share.email ?? '?')
                      .charAt(0)
                      .toUpperCase()}
                  </div>

                  {/* Name / Email */}
                  <div className="flex-1 min-w-0">
                    {share.name && (
                      <div className="text-sm font-medium truncate">
                        {share.name}
                      </div>
                    )}
                    <div className="text-xs text-muted-foreground truncate">
                      {share.email}
                    </div>
                  </div>

                  {/* Permission dropdown or "removed" badge */}
                  {share.markedForRemoval ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs h-7"
                      onClick={() => handleUndoRemove(idx)}
                      disabled={isSaving}
                    >
                      Undo
                    </Button>
                  ) : (
                    <>
                      <Select
                        value={share.permission}
                        onValueChange={(v) =>
                          handlePermissionChange(idx, v as SharePermission)
                        }
                        disabled={isSaving}
                      >
                        <SelectTrigger className="w-[110px] h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="view">Can view</SelectItem>
                          <SelectItem value="edit">Can edit</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 flex-shrink-0 text-muted-foreground hover:text-destructive"
                        onClick={() => handleRemove(idx)}
                        disabled={isSaving}
                        title="Remove access"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Footer ────────────────────────────────────────────────── */}
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={isSaving || !hasChanges}
          >
            {isSaving ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : null}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
