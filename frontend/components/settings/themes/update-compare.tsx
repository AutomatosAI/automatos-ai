'use client'

/**
 * PRD-261 A-07d — "update available": what moving an install to its pending version changes
 * (`GET /api/themes/installs/{id}/compare/{packageId}`: the versions, the changed files from the
 * two file manifests, the scan diff), and Apply, which posts the package the compare SHOWED
 * (`POST …/update`, manage + step-up). A private version is a new opt-in for every member: opt-ins
 * are keyed by package (A-07b).
 */
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useThemeUpdateComparison, useUpdateThemeInstall } from '@/hooks/use-installed-themes'
import type { InstalledTheme, ThemeUpdateComparison } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { themeName } from './theme-labels'

export const APPLY_UPDATE = 'Apply update'
/** The most file paths listed per group; the rest are counted. */
const MAX_LISTED_FILES = 12

function FileGroup({ label, paths }: { label: string; paths: readonly string[] }) {
  if (paths.length === 0) return null
  const listed = paths.slice(0, MAX_LISTED_FILES)
  return (
    <div>
      <p className="text-xs font-medium text-foreground">
        {label} ({paths.length})
      </p>
      <ul className="mt-1 space-y-0.5 font-mono text-xs text-muted-foreground">
        {listed.map((filePath) => (
          <li key={filePath}>{filePath}</li>
        ))}
        {paths.length > listed.length && <li>and {paths.length - listed.length} more</li>}
      </ul>
    </div>
  )
}

function ComparisonBody({ comparison }: { comparison: ThemeUpdateComparison }) {
  const { files, scan } = comparison
  const scanKeys = Object.keys(scan.changed)
  const noFiles = files.added.length + files.removed.length + files.changed.length === 0
  return (
    <div className="space-y-3" data-testid="theme-update-compare">
      <p className="text-sm">
        Version {comparison.from.version} → {comparison.to.version}
        {!comparison.to.reviewed && <span className="ml-2 text-xs text-muted-foreground">not reviewed</span>}
      </p>
      {noFiles ? <p className="text-xs text-muted-foreground">No file changed.</p> : null}
      <FileGroup label="Added" paths={files.added} />
      <FileGroup label="Changed" paths={files.changed} />
      <FileGroup label="Removed" paths={files.removed} />
      <p className="text-xs text-muted-foreground">
        Scan verdict: {scan.verdict.from ?? 'not run'} → {scan.verdict.to ?? 'not run'}
        {scanKeys.length > 0 && <> · changed: {scanKeys.join(', ')}</>}
      </p>
    </div>
  )
}

interface UpdateCompareProps {
  /** The install whose pending version is compared; null closes the dialog. */
  readonly install: InstalledTheme | null
  readonly onClose: () => void
}

export function UpdateCompare({ install, onClose }: UpdateCompareProps) {
  return (
    <Dialog open={install !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent>{install && <UpdateCompareContent install={install} onClose={onClose} />}</DialogContent>
    </Dialog>
  )
}

function UpdateCompareContent({ install, onClose }: { install: InstalledTheme; onClose: () => void }) {
  const comparison = useThemeUpdateComparison(install, true)
  const update = useUpdateThemeInstall()
  const shown = comparison.data

  const apply = async () => {
    if (!shown) return
    try {
      await update.mutateAsync({ installId: install.installId, packageId: shown.to.packageId })
    } catch {
      return // the hook showed the server's sentence
    }
    onClose()
  }

  return (
    <>
      <DialogHeader>
        <DialogTitle>Update {themeName(install)}</DialogTitle>
        <DialogDescription>Members opt in again to a new private version before it runs for them.</DialogDescription>
      </DialogHeader>
      {comparison.isLoading && <p className="text-sm text-muted-foreground">Comparing the two versions…</p>}
      {comparison.isError && <p className="text-sm text-destructive">{themeErrorMessage(comparison.error)}</p>}
      {shown && <ComparisonBody comparison={shown} />}
      <DialogFooter>
        <Button variant="outline" onClick={onClose}>
          Cancel
        </Button>
        <Button onClick={apply} disabled={!shown || update.isLoading}>
          {APPLY_UPDATE}
        </Button>
      </DialogFooter>
    </>
  )
}
