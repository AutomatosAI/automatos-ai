'use client'

/**
 * PRD-261 A-07d — the workspace's installed themes and the member's choice among them.
 *
 * "Use this theme" chooses through the one chooser the theme menu and ⌘K share (choose-install.tsx):
 * a private version this member has not opted in to opens the opt-in dialog first (with the row as
 * it was when clicked; a 404/409 closes it so a retry starts from the refetched row); once the
 * server recorded it, the choice is stored and the page reloads into the theme. Every member sees
 * the list and the choice; the manage actions show only for `workspace:manage`.
 */
import { useState } from 'react'

import { useActiveThemeChoice, useInstalledThemes } from '@/hooks/use-installed-themes'
import type { InstalledTheme } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { useChooseInstall } from './choose-install'
import { InstalledRow } from './installed-row'
import { MEMBERS_SEE_ONLY, NO_INSTALLS } from './theme-labels'
import { UpdateCompare } from './update-compare'

export function InstalledList({ canManage }: { canManage: boolean }) {
  const installed = useInstalledThemes()
  const active = useActiveThemeChoice()
  const { chooseInstall, choosing } = useChooseInstall()
  const [comparing, setComparing] = useState<InstalledTheme | null>(null)

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
              choosing={choosing}
              onUse={chooseInstall}
              onReviewUpdate={setComparing}
            />
          ))}
        </ul>
      )}
      {!canManage && <p className="text-xs text-muted-foreground">{MEMBERS_SEE_ONLY}</p>}
      {canManage && <UpdateCompare install={comparing} onClose={() => setComparing(null)} />}
    </section>
  )
}
