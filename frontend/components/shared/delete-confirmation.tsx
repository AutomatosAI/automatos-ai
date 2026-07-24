'use client'

import * as React from 'react'
import { Loader2 } from 'lucide-react'
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogCancel,
  AlertDialogAction,
} from '@/components/ui/alert-dialog'
import { buttonVariants } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export interface DeleteConfirmationProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title?: string
  description?: string
  /** When set, builds a default description: "This permanently deletes <itemName>." */
  itemName?: string
  confirmLabel?: string
  cancelLabel?: string
  /** Sync or async. Modal closes on resolve, stays open (so caller can toast) on throw. */
  onConfirm: () => void | Promise<void>
  /** Controlled busy state; when omitted the modal tracks its own pending state. */
  loading?: boolean
  /** Destructive (red) confirm styling. Default true — this is a delete primitive. */
  destructive?: boolean
}

/**
 * Canonical destructive-confirmation modal (PRD-169 S2). Replaces every
 * `window.confirm(...)` / `window.location.reload()` delete flow with one
 * accessible, themed dialog.
 */
export function DeleteConfirmation({
  open,
  onOpenChange,
  title = 'Delete this item?',
  description,
  itemName,
  confirmLabel = 'Delete',
  cancelLabel = 'Cancel',
  onConfirm,
  loading,
  destructive = true,
}: DeleteConfirmationProps) {
  const [pending, setPending] = React.useState(false)
  const busy = loading ?? pending

  const body =
    description ??
    (itemName
      ? `This permanently deletes ${itemName}. This action cannot be undone.`
      : 'This action cannot be undone.')

  async function handleConfirm() {
    try {
      setPending(true)
      await onConfirm()
      onOpenChange(false)
    } finally {
      setPending(false)
    }
  }

  return (
    <AlertDialog open={open} onOpenChange={(next) => !busy && onOpenChange(next)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{body}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={busy}>{cancelLabel}</AlertDialogCancel>
          <AlertDialogAction
            disabled={busy}
            onClick={(e) => {
              e.preventDefault()
              void handleConfirm()
            }}
            className={cn(destructive && buttonVariants({ variant: 'destructive' }))}
          >
            {busy && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
