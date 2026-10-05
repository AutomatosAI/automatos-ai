'use client'

/** Confirmation dialog for removing a BYOK API key (PRD-54). */
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import type { ApiKeyOut } from './api-keys-types'

interface DeleteApiKeyDialogProps {
  target: ApiKeyOut | null
  isPending: boolean
  onClose: () => void
  onConfirm: (id: number) => void
}

export function DeleteApiKeyDialog({ target, isPending, onClose, onConfirm }: DeleteApiKeyDialogProps) {
  return (
    <Dialog open={!!target} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Delete API Key</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete{' '}
            <span className="font-medium text-foreground">
              {target?.display_name || target?.masked_key}
            </span>
            ? This action cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={isPending}
            onClick={() => target && onConfirm(target.id)}
          >
            {isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
