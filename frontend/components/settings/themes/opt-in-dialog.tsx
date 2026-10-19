'use client'

/**
 * PRD-261 A-07d — the opt-in to run one private theme version (Q6, PRD §6).
 *
 * Unreviewed code runs in the member's browser with their session, so it runs only after this
 * member says yes to this exact package. The dialog shows the server's wording (`optIn.wording`,
 * never a copy) and the version, and posts the `packageId` it SHOWED: `install` is the row as it
 * was when the dialog opened, so a list refetched behind an open dialog can't swap the version
 * the member is agreeing to. A 409 (the install moved on) shows the server's sentence and
 * refetches the list (hooks/use-installed-themes.ts).
 */
import { ShieldAlert } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useOptInTheme } from '@/hooks/use-installed-themes'
import type { InstalledTheme } from '@/lib/api-client'

import { themeName } from './theme-labels'

export const OPT_IN_CONFIRM = 'Opt in and use'

interface OptInDialogProps {
  /** The row as it was when the member asked to use it; null closes the dialog. */
  readonly install: InstalledTheme | null
  readonly onClose: () => void
  /** Called once the server recorded the opt-in for the package shown. */
  readonly onOptedIn: (install: InstalledTheme) => void
}

export function OptInDialog({ install, onClose, onOptedIn }: OptInDialogProps) {
  const optIn = useOptInTheme()

  const confirm = async () => {
    if (!install) return
    try {
      await optIn.mutateAsync({ installId: install.installId, packageId: install.optIn.packageId })
    } catch {
      return // the hook showed the server's sentence
    }
    onOptedIn(install)
    onClose()
  }

  return (
    <Dialog open={install !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent data-testid="theme-opt-in-dialog">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ShieldAlert className="h-5 w-5 text-warning" aria-hidden />
            Run {install ? themeName(install) : 'this theme'} {install?.version}?
          </DialogTitle>
          <DialogDescription data-testid="theme-opt-in-wording">{install?.optIn.wording}</DialogDescription>
        </DialogHeader>
        <p className="text-xs text-muted-foreground">
          You opt in to this version only. A new version asks again. Other members decide for themselves.
        </p>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={confirm} disabled={optIn.isLoading}>
            {OPT_IN_CONFIRM}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
