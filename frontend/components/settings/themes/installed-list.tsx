'use client'

/**
 * PRD-261 A-07d — the workspace's installed themes and the member's choice among them.
 *
 * "Use this theme" on a private version this member has not opted in to opens the opt-in dialog
 * first (with the row as it was when clicked; a 404/409 closes it so a retry starts from the
 * refetched row); once the server recorded it, the choice is stored
 * and the page reloads into the theme (hooks/use-installed-themes.ts `useChooseTheme`). Every
 * member sees the list and the choice; the manage actions show only for `workspace:manage`.
 */
import { useState } from 'react'

import { useActiveThemeChoice, useChooseTheme, useInstalledThemes } from '@/hooks/use-installed-themes'
import type { InstalledTheme } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { InstalledRow } from './installed-row'
import { OptInDialog } from './opt-in-dialog'
import { MEMBERS_SEE_ONLY, NO_INSTALLS } from './theme-labels'
import { UpdateCompare } from './update-compare'

export function InstalledList({ canManage }: { canManage: boolean }) {
  const installed = useInstalledThemes()
  const active = useActiveThemeChoice()
  const choose = useChooseTheme()
  const [optInFor, setOptInFor] = useState<InstalledTheme | null>(null)
  const [comparing, setComparing] = useState<InstalledTheme | null>(null)

  const use = (install: InstalledTheme) => {
    if (install.optIn.required) setOptInFor(install)
    else choose.mutate(install.installId)
  }

  if (installed.isLoading) return <p className="text-sm text-muted-foreground">Loading installed themes…</p>
  if (installed.isError) return <p className="text-sm text-destructive">{themeErrorMessage(installed.error)}</p>
  const installs = installed.data ?? []

  return (
    <section aria-label="Installed themes" className="space-y-3">
      {installs.length === 0 ? (
        <p className="text-sm text-muted-foreground">{NO_INSTALLS}</p>
      ) : (
        <ul className="space-y-2">
          {installs.map((install) => (
            <InstalledRow
              key={install.installId}
              install={install}
              canManage={canManage}
              inUse={active.data === install.installId}
              choosing={choose.isLoading}
              onUse={use}
              onReviewUpdate={setComparing}
            />
          ))}
        </ul>
      )}
      {!canManage && <p className="text-xs text-muted-foreground">{MEMBERS_SEE_ONLY}</p>}
      <OptInDialog
        install={optInFor}
        onClose={() => setOptInFor(null)}
        onOptedIn={(install) => choose.mutate(install.installId)}
      />
      {canManage && <UpdateCompare install={comparing} onClose={() => setComparing(null)} />}
    </section>
  )
}
