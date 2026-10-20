'use client'

/**
 * PRD-261 C-04b — "Add to my workspace" on a listing's detail (C-04a install by reference).
 *
 * Owners and admins (`workspace:manage`) add the newest version this host runs, unless the
 * workspace's "who installs" leaves them out (`ThemeSettings.canInstall`, the route's
 * `require_install_role`): then the button gives way to the same sentence A-07d's Install theme
 * shows. The server still checks both and asks for step-up. Once the theme is in the workspace every member
 * gets "Use this theme" (`UseInstalledTheme`: A-07d's opt-in dialog and choice, reused). Update,
 * rollback and uninstall of a listing install are A-07d's installed list, on the same tab.
 */
import { Download } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { installRoleRefusal } from '@/components/settings/themes/install-theme'
import { useThemeSettings } from '@/hooks/use-installed-themes'
import type { ThemeListing } from '@/lib/api-client'

import { UseInstalledTheme } from './install-use'
import {
  ADDING,
  ADD_TO_WORKSPACE,
  INSTALLED_BADGE,
  MANAGERS_ADD,
  NOT_COMPATIBLE_HINT,
  OWN_UPLOAD_NOTE,
  UPDATE_NOTE,
  runsOwnUpload,
} from './listing-format'
import { useInstallThemeListing } from './listing-hooks'

interface AddToWorkspaceProps {
  readonly listing: ThemeListing
  readonly canManage: boolean
}

function InstalledState({ listing, installId }: { listing: ThemeListing; installId: string }) {
  const own = runsOwnUpload(listing)
  return (
    <div className="space-y-2" data-testid="theme-listing-installed">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">
          {INSTALLED_BADGE}
          {listing.installed ? ` · ${listing.installed.version}` : ''}
        </Badge>
        <UseInstalledTheme installId={installId} />
      </div>
      {own && <p className="text-xs text-muted-foreground">{OWN_UPLOAD_NOTE}</p>}
      {listing.installed?.updatePackageId && <p className="text-xs text-muted-foreground">{UPDATE_NOTE}</p>}
    </div>
  )
}

export function AddToWorkspace({ listing, canManage }: AddToWorkspaceProps) {
  const install = useInstallThemeListing()
  const refusal = installRoleRefusal(useThemeSettings().data)
  const installId = listing.installed?.installId ?? install.data?.install.installId ?? null

  if (installId) return <InstalledState listing={listing} installId={installId} />
  if (!canManage) return <p className="text-xs text-muted-foreground">{MANAGERS_ADD}</p>
  if (refusal) {
    return (
      <p className="text-xs text-muted-foreground" data-testid="theme-listing-install-refused">
        {refusal}
      </p>
    )
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Button
        disabled={!listing.compatible || install.isLoading}
        title={listing.compatible ? undefined : NOT_COMPATIBLE_HINT}
        onClick={() => install.mutate({ listingId: listing.listingId })}
      >
        <Download className="mr-2 h-4 w-4" aria-hidden />
        {install.isLoading ? ADDING : ADD_TO_WORKSPACE}
      </Button>
      {!listing.compatible && <p className="text-xs text-muted-foreground">{NOT_COMPATIBLE_HINT}</p>}
    </div>
  )
}
