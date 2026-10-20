'use client'

/**
 * PRD-261 A-07d — one installed theme: its badges (PRD §5: "Private · not reviewed", default,
 * update available, "needs update" for a clone whose core page changed, pinned, revoked), "Use
 * this theme" for every member, and the manage actions (default, update, rollback, uninstall)
 * only for `workspace:manage`. The server refuses what the caller may not do; this decides which
 * buttons to show.
 */
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useRollbackTheme, useSetDefaultTheme, useUninstallTheme } from '@/hooks/use-installed-themes'
import type { InstalledTheme } from '@/lib/api-client'

import {
  DEFAULT_BADGE,
  IN_USE,
  NEEDS_UPDATE,
  NEEDS_UPDATE_HINT,
  PRIVATE_NOT_REVIEWED,
  UNREVIEWED_NOT_DEFAULT,
  UPDATE_AVAILABLE,
  isDisabled,
  isPrivateNotReviewed,
  themeName,
} from './theme-labels'

export const USE_THEME = 'Use this theme'

function RowBadges({ install }: { install: InstalledTheme }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {isPrivateNotReviewed(install) && <Badge variant="secondary">{PRIVATE_NOT_REVIEWED}</Badge>}
      {install.isDefault && <Badge>{DEFAULT_BADGE}</Badge>}
      {install.pendingPackageId && (
        <Badge variant="outline">
          {UPDATE_AVAILABLE}
          {install.pendingVersion ? ` · ${install.pendingVersion}` : ''}
        </Badge>
      )}
      {install.needsUpdate && (
        <Badge variant="destructive" title={NEEDS_UPDATE_HINT} data-testid="theme-needs-update">
          {NEEDS_UPDATE}
        </Badge>
      )}
      {install.pinnedVersion && <Badge variant="outline">Pinned to {install.pinnedVersion}</Badge>}
    </div>
  )
}

function ManageActions({ install, onReviewUpdate }: { install: InstalledTheme; onReviewUpdate: () => void }) {
  const setDefault = useSetDefaultTheme()
  const rollback = useRollbackTheme()
  const uninstall = useUninstallTheme()
  const [confirmingUninstall, setConfirmingUninstall] = useState(false)
  const busy = setDefault.isLoading || rollback.isLoading || uninstall.isLoading

  return (
    <div className="flex flex-wrap gap-2" data-testid="theme-manage-actions">
      {install.isDefault ? (
        <Button size="sm" variant="outline" disabled={busy} onClick={() => setDefault.mutate(null)}>
          Clear default
        </Button>
      ) : (
        <Button
          size="sm"
          variant="outline"
          disabled={busy || !install.reviewed}
          title={install.reviewed ? undefined : UNREVIEWED_NOT_DEFAULT}
          onClick={() => setDefault.mutate(install.installId)}
        >
          Make default
        </Button>
      )}
      {install.pendingPackageId && (
        <Button size="sm" variant="outline" disabled={busy} onClick={onReviewUpdate}>
          Review update
        </Button>
      )}
      {install.previousPackageId && (
        <Button size="sm" variant="outline" disabled={busy} onClick={() => rollback.mutate(install.installId)}>
          Roll back
        </Button>
      )}
      <Button
        size="sm"
        variant={confirmingUninstall ? 'destructive' : 'ghost'}
        disabled={busy}
        onClick={() => (confirmingUninstall ? uninstall.mutate(install.installId) : setConfirmingUninstall(true))}
      >
        {confirmingUninstall ? 'Confirm uninstall' : 'Uninstall'}
      </Button>
    </div>
  )
}

interface InstalledRowProps {
  readonly install: InstalledTheme
  readonly canManage: boolean
  /** The member's current choice is this install. */
  readonly inUse: boolean
  readonly choosing: boolean
  readonly onUse: (install: InstalledTheme) => void
  readonly onReviewUpdate: (install: InstalledTheme) => void
}

export function InstalledRow({ install, canManage, inUse, choosing, onUse, onReviewUpdate }: InstalledRowProps) {
  return (
    <li className="space-y-3 rounded-xl border border-border/50 px-4 py-3" data-testid={`installed-theme-${install.installId}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <p className="font-medium text-foreground">
            {themeName(install)} <span className="text-xs text-muted-foreground">{install.version}</span>
          </p>
          <p className="truncate font-mono text-xs text-muted-foreground">{install.themeId}</p>
          <RowBadges install={install} />
          {install.revoked && <p className="text-xs text-destructive">{install.revoked.reason}</p>}
        </div>
        {inUse ? (
          <Badge>{IN_USE}</Badge>
        ) : (
          <Button size="sm" disabled={choosing || isDisabled(install)} onClick={() => onUse(install)}>
            {USE_THEME}
          </Button>
        )}
      </div>
      {canManage && <ManageActions install={install} onReviewUpdate={() => onReviewUpdate(install)} />}
    </li>
  )
}
